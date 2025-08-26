import os
os.environ['USER_AGENT'] = 'myagent'

from typing import List, Dict, Any, Optional
import pandas as pd

from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ---------------------------
# 1) Config
# ---------------------------
CSV_PATH = "aig-10k.csv"   # change if needed
COMPANY_NAME = "American International Group, Inc."
FORM = "10-K"
PERSIST_DIR = "./chroma_db_edgar"
EMB_MODEL = "all-minilm"         # local Ollama embedding
GEN_MODEL = "llama3.2:3b"        # local Ollama LLM

# Canonical EDGAR tag mapping (extend as needed)
FACT_ALIASES = {
    # numeric
    "revenue": "Revenues",
    "total revenue": "Revenues",
    "total revenues": "Revenues",
    "net income": "NetIncomeLoss",
    "long term debt": "LongTermDebt",
    "assets": "Assets",
    # categorical (CSV-only path fallbacks)
    "form": "form",                 # categorical from CSV
    "fiscal period": "fp",          # categorical from CSV (FY)
    # add more if needed (e.g., 'industry' once you add 10-K text pages)
}

# ---------------------------
# 2) Read & prepare data (pandas)
# ---------------------------
df = pd.read_csv(CSV_PATH)

# Make types consistent
if df["fy"].dtype != int:
    df["fy"] = df["fy"].astype("Int64")  # FY year sometimes floats in raw dumps

# De-duplicate to one value per fact/year/form (keep the latest "filed")
df_sorted = df.sort_values(["companyFact", "fy", "form", "filed"], ascending=[True, True, True, True])
df_dedup = df_sorted.drop_duplicates(subset=["companyFact", "fy", "form"], keep="last")

# Limit to our company and filing type
df_aig_10k = df_dedup[(df_dedup["entityName"] == COMPANY_NAME) & (df_dedup["form"] == FORM)]

#df_aig_10k.to_csv('df_dedup_10k.csv', index=False)
# ---------------------------
# 3) Build Documents for Chroma
# ---------------------------
def row_to_doc(row: pd.Series) -> Document:
    # Short, dense content per row; rich metadata for filtering
    content = (
        f"entityName: {row.entityName}; cik: {row.cik}; form: {row.form}; "
        f"fy: {row.fy}; fp: {row.fp}; fact: {row.companyFact}; "
        f"value: {row.val} {row.units}; end: {row.end}; filed: {row.filed}; accn: {row.accn}"
    )
    meta = {
        "entityName": row.entityName,
        "cik": int(row.cik),
        "form": row.form,
        "fy": int(row.fy) if pd.notnull(row.fy) else None,
        "fp": row.fp,
        "companyFact": row.companyFact,
        "accn": row.accn,
        "filed": row.filed,
        "units": row.units,
        "end": row.end,
    }
    # Use a stable id per (fact, fy, form, accn)
    doc_id = f"{row.companyFact}|{row.fy}|{row.form}|{row.accn}"
    return Document(page_content=content, metadata=meta, id=doc_id)

docs = [row_to_doc(r) for _, r in df_aig_10k.iterrows()]

# ---------------------------
# 4) Vector store (Chroma)
# ---------------------------
emb = OllamaEmbeddings(model=EMB_MODEL, show_progress=True)
chroma = Chroma(persist_directory=PERSIST_DIR, embedding_function=emb)

# Idempotent add (skip if already present)
def ensure_index(populate_docs: List[Document]):
    try:
        existing = chroma.get(include=[])  # returns dict with 'ids'
        existing_ids = set(existing.get("ids", []))
    except Exception:
        existing_ids = set()
    new_docs = [d for d in populate_docs if d.id not in existing_ids]
    if new_docs:
        chroma.add_documents(new_docs)

ensure_index(docs)

# ---------------------------
# 5) Retriever with metadata filter
# ---------------------------
# def make_retriever(year: Optional[int] = None):
#     base_filter: Dict[str, Any] = {"form": FORM, "entityName": COMPANY_NAME}
#     if year is not None:
#         base_filter["fy"] = int(year)
#     return chroma.as_retriever(search_kwargs={"k": 6, "filter": base_filter})
def _build_where(year: int | None):
    clauses = [
        {"form": {"$eq": "10-K"}},
        {"entityName": {"$eq": "American International Group, Inc."}},
    ]
    if year is not None:
        clauses.append({"fy": {"$eq": int(year)}})  # ensure int type
    return {"$and": clauses}

def make_retriever(year: int | None = None):
    where = _build_where(year)
    return chroma.as_retriever(search_kwargs={"k": 6, "filter": where})


# ---------------------------
# 6) Normalization helpers
# ---------------------------
def canonical_fact(name: str) -> str:
    key = name.strip().lower()
    return FACT_ALIASES.get(key, name)

# ---------------------------
# 7) Prompt -> JSON
# ---------------------------
schema_hint = """
Return JSON with keys: 
- "variable" (canonical EDGAR fact or CSV column),
- "value",
- "units",
- "fy",
- "form",
- "accn",
- "filed",
- "source" (short string describing the row).
"""

prompt = PromptTemplate(
    template=(
        "You are a precise data extraction assistant for EDGAR facts already normalized from CSV rows.\n"
        "Use ONLY the given rows to answer. If the value isn't present, say you don't know.\n"
        f"{schema_hint}\n\n"
        "Question: {question}\n\n"
        "Rows:\n{documents}\n\n"
        "Answer in JSON only:\n"
        "Do not add any newline characters in or around the JSON or add any irrelevant details\n"
    ),
    input_variables=["question", "documents"],
)
llm = ChatOllama(model=GEN_MODEL, temperature=0)
rag_chain = prompt | llm | StrOutputParser()

# ---------------------------
# 8) RAG application
# ---------------------------
class EdgarCSV_RAG:
    def __init__(self, retriever_maker):
        self.retriever_maker = retriever_maker

    def ask(self, question: str, fy: Optional[int] = None):
        r = self.retriever_maker(fy)
        hits = r.invoke(question)
        text_block = "\n".join([h.page_content for h in hits])
        return rag_chain.invoke({"question": question, "documents": text_block})

rag = EdgarCSV_RAG(make_retriever)

# ---------------------------
# 9) Deterministic structured lookup (ground truth)
# ---------------------------
def ground_truth_value(fact: str, year: int) -> Optional[Dict[str, Any]]:
    """
    Return the ground truth row for an exact companyFact and fiscal year.
    Always picks the row with the latest 'filed' date.
    """
    # restrict to that fact + year
    sub = df_aig_10k[
        (df_aig_10k["companyFact"] == fact) &
        (df_aig_10k["fy"].astype("Int64") == int(year))
    ]
    if sub.empty:
        return None

    # pick the most recent filed
    sub = sub.copy()
    sub["__filed_dt"] = pd.to_datetime(sub["filed"], errors="coerce")
    row = sub.sort_values("__filed_dt").iloc[-1]

    return {
        "variable": fact,
        "value": row["val"],
        "units": row["units"],
        "fy": int(row["fy"]),
        "form": row["form"],
        "accn": row["accn"],
        "filed": row["filed"],
    }


# ---------------------------
# 10) Example usage
# ---------------------------
# Example Q: "What was AIG total revenue in FY 2021?"
# q = "AIG total revenue in FY 2022"
# print(rag.ask(q, fy=2022))
import json
def simple_evaluation():
    # 5 example queries (you can tweak fiscal years)
    test_queries = [
        ("AIG total revenue in FY 2022", "Revenues", 2022),
        ("AIG cash in FY 2022", "Cash", 2022),
        ("AIG assets in FY 2022", "Assets", 2022)
    ]

    results = []
    for q, var, year in test_queries:
        # run LLM
        llm_output = rag.ask(q, fy=year)
        print(llm_output)

        # try to parse JSON from LLM
        try:
            llm_json = json.loads(llm_output)
            llm_answer = llm_json.get("value", llm_output)
        except Exception:
            llm_answer = llm_output

        # ground truth
        gt = ground_truth_value(var, year)
        gt_answer = gt["value"] if gt else None

        results.append({
            "question": q,
            "llm_answer": llm_answer,
            "ground_truth": gt_answer,
        })

    df_eval = pd.DataFrame(results)
    out_path = "simple_eval_results.csv"
    df_eval.to_csv(out_path, index=False)
    print(f"Saved evaluation to {out_path}")
    print(df_eval)

simple_evaluation()