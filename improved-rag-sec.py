import os
import json
import pandas as pd
from typing import List, Dict, Any, Optional
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from rapidfuzz import process, fuzz

# -------------------------------
# 1. Configuration
# -------------------------------
CSV_PATH = "aig-10k-key-facts.csv"
COMPANY_NAME = "American International Group, Inc."
FORM = "10-K"
PERSIST_DIR = "./chroma_db_edgar"
EMB_MODEL = "all-minilm"
GEN_MODEL = "llama3.2:3b"
INDEX_MODE = True # Set to False to use existing index

# -------------------------------
# 2. Read and Clean CSV
# -------------------------------
df = pd.read_csv(CSV_PATH)
df["fy"] = df["fy"].astype("Int64")
df["companyFact_lower"] = df["companyFact"].str.lower()

df_sorted = df.sort_values(["companyFact", "fy", "form", "filed"], ascending=True)
df_dedup = df_sorted.drop_duplicates(subset=["companyFact", "fy", "form"], keep="last")
df_filtered = df_dedup[(df_dedup["entityName"] == COMPANY_NAME) & (df_dedup["form"] == FORM)]

# -------------------------------
# 3. Build Document Store
# -------------------------------
llm = ChatOllama(model=GEN_MODEL, temperature=0, random_seed=42)

if INDEX_MODE:
    def summarize_row(row: pd.Series) -> str:
        prompt = f"""Convert the following EDGAR financial row into a readable English sentence without any irrelevant commentary.
        There should be no identifiers to signal the beginning or end of the summary, just the summary is needed.
        entityName: {row.entityName}
        cik: {row.cik}
        form: {row.form}
        fiscal year: {row.fy}
        fact: {row.companyFact}
        value: {row.val} {row.units}
        end date: {row.end}

        """
        print(prompt)
        return llm.invoke(prompt)

    def row_to_doc(row: pd.Series) -> Document:
        try:
            # print("Summarizing row:", row, "\n")
            content = summarize_row(row).content
            print("Summarized content:",content,"\n","**********")
        except:
            content = f"{row.companyFact} was {row.val} {row.units} for FY {row.fy}."
            print("exception")
        meta = {
            "entityName": row.entityName,
            "cik": int(row.cik),
            "form": row.form,
            "fy": int(row.fy) if pd.notnull(row.fy) else None,
            "fp": row.fp,
            "companyFact": row.companyFact,
            "companyFact_lower": row.companyFact.lower(),
            "accn": row.accn,
            "filed": row.filed,
            "units": row.units,
            "end": row.end,
        }
        doc_id = f"{row.companyFact}|{row.fy}|{row.form}|{row.accn}"
        return Document(page_content=content, metadata=meta, id=doc_id)

    for _, r in df_filtered.iterrows():
        row_to_doc(r)

    docs = [row_to_doc(r) for _, r in df_filtered.iterrows()]

    def ensure_index(populate_docs: List[Document]):
        try:
            existing_ids = set(chroma.get(include=[]).get("ids", []))
        except:
            existing_ids = set()
        new_docs = [d for d in populate_docs if d.id not in existing_ids]
        if new_docs:
            chroma.add_documents(new_docs)
    
    # # -------------------------------
    # # 4. Index in Chroma
    # # -------------------------------
    ensure_index(docs)
else:
    emb = OllamaEmbeddings(model=EMB_MODEL, show_progress=True)
    chroma = Chroma(persist_directory=PERSIST_DIR, embedding_function=emb)


# -------------------------------
# 5. Fact Resolution Logic
# -------------------------------
company_facts = df_filtered["companyFact"].dropna().unique()#.tolist()
print(company_facts)
pd.DataFrame(company_facts).to_csv("company_facts.csv", index=False)
company_facts = company_facts.tolist()

def resolve_fact_from_query(user_query: str) -> Optional[str]:
    query_lower = user_query.lower()
    match, score, _ = process.extractOne(query_lower, company_facts, scorer=fuzz.token_sort_ratio)
    return match if score >= 85 else None

# -------------------------------
# 6. Retriever Factory
# -------------------------------
def _build_where(fact: Optional[str], year: Optional[int]):
    clauses = [
        {"form": {"$eq": "10-K"}},
        {"entityName": {"$eq": COMPANY_NAME}},
    ]
    if year is not None:
        clauses.append({"fy": {"$eq": int(year)}})
    if fact:
        clauses.append({"companyFact_lower": {"$contains": fact.lower()}})
    return {"$and": clauses}

def make_retriever(fact: Optional[str] = None, year: Optional[int] = None):
    return chroma.as_retriever(search_type="mmr", search_kwargs={"k": 1, "filter": _build_where(fact, year)})

# -------------------------------
# 7. Prompt Template and Chain
# -------------------------------
schema_hint = """
Return JSON with keys: 
- "variable" (canonical EDGAR fact or CSV column),
- "value",
- "units",
- "fy",
- "form",
- "accn",
- "filed",
- "source"
"""

prompt = PromptTemplate(
    template=(
        "You are a precise data extraction assistant for EDGAR facts already normalized from CSV rows.\n"
        "Use ONLY the given rows to answer. If the value isn't present, say you don't know.\n"
        f"{schema_hint}\n\n"
        "Question: {question}\n\n"
        "Rows:\n{documents}\n\n"
        "Answer in JSON only:\n"
    ),
    input_variables=["question", "documents"],
)

rag_chain = prompt | llm | StrOutputParser()

# -------------------------------
# 8. Main RAG Application
# -------------------------------
class EdgarCSV_RAG:
    def __init__(self, retriever_maker):
        self.retriever_maker = retriever_maker

    def ask(self, question: str, fy: Optional[int] = None):
        resolved_fact = resolve_fact_from_query(question)
        retriever = self.retriever_maker(resolved_fact, fy)
        hits = retriever.invoke(question)
        text_block = "\n".join([doc.page_content for doc in hits])
        return rag_chain.invoke({"question": question, "documents": text_block})

rag = EdgarCSV_RAG(make_retriever)

response = rag.ask("How much were AIG’s assets worth in 2022?", fy=2022)
print("response",response)



def ground_truth_lookup(fact: str, year: int) -> Optional[Dict[str, Any]]:
    """
    Get the ground truth value for a given fact and year.
    """
    sub = df_filtered[
        (df_filtered["companyFact"].str.lower() == fact.lower()) &
        (df_filtered["fy"].astype("Int64") == int(year))
    ]
    if sub.empty:
        return None
    row = sub.sort_values("filed").iloc[-1]
    return {
        "variable": row["companyFact"],
        "value": row["val"],
        "units": row["units"],
        "fy": int(row["fy"]),
        "form": row["form"],
        "accn": row["accn"],
        "filed": row["filed"]
    }

def evaluate_rag_queries(rag_app: EdgarCSV_RAG):
    # Test queries (you can expand this list)
    test_queries = [
        ("How much were AIG’s assets worth in 2022?", "Assets", 2022),
        ("What was AIG's cash in 2022?", "Cash", 2022),
        ("Total revenue for AIG in 2022?", "Revenues", 2022),
        ("What were the liabilities reported by AIG for FY 2022?", "Liabilities", 2022),
        ("What were the earnings per share of AIG in 2022?", "EarningsPerShareBasic", 2022),
    ]

    results = []
    for q, expected_fact, year in test_queries:
        llm_response = rag_app.ask(q, fy=year)

        try:
            parsed = json.loads(llm_response)
            llm_val = parsed.get("value")
        except Exception:
            parsed = {}
            llm_val = llm_response

        gt = ground_truth_lookup(expected_fact, year)
        gt_val = gt["value"] if gt else None

        results.append({
            "question": q,
            "expected_fact": expected_fact,
            "fy": year,
            "llm_response_value": llm_val,
            "ground_truth_value": gt_val,
            "llm_full_response": parsed,
            "ground_truth_full": gt
        })

    df_results = pd.DataFrame(results)
    df_results.to_csv("rag_eval_results.csv", index=False)
    print("Saved evaluation to 'rag_eval_results.csv'")
    return df_results

# Run the evaluation
df_eval_results = evaluate_rag_queries(rag)
print(df_eval_results[["question", "llm_response_value", "ground_truth_value"]])
df_eval_results.to_csv("rag_eval_results.csv", index=False)
