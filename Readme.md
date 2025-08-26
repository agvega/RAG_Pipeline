RAG over SEC Company Facts (CSV) with Chroma + Ollama

This prototype implements a compact Retrieval-Augmented Generation (RAG) flow over SEC company facts stored in a CSV. It uses LangChain, ChromaDB (vector store), and Ollama (local LLM + embeddings). The script indexes SEC facts for American International Group, Inc. (AIG) 10-K rows and runs a small evaluation.

Scope (current state)
Retrieval operates over CSV rows, not page-level 10-K text. The code constrains to AIG + 10-K via metadata filters and evaluates three demo questions. PySpark and page-level retrieval are intentionally out of scope for this prototype.

Executive summary & timebox note

Per the original prompt, the exercise is framed as taking “about an hour.” Under that timebox—and without AI assistance during implementation—this repository delivers a working prototype that: (1) ingests an AIG 10-K CSV, (2) builds a Chroma vector index with metadata filters, (3) performs metadata-constrained retrieval, (4) uses an Ollama-hosted model to produce JSON answers, and (5) exports a small evaluation file. This reflects what was feasible in the allowed timeframe and conditions. Further enhancements (e.g., page-level retrieval from a single filing, PySpark orchestration, and broader evaluation) are listed for future work but not included here.

Contents

rag-sec-filings.py — end-to-end script: load CSV → de-dupe → Chroma index → metadata-filtered retrieval → LLM JSON extraction → simple evaluation to CSV

requirements.txt — Python dependencies

aig-10k.csv — sample input (AIG 10-K company facts)

df_dedup_10k.csv — sample deduplicated snapshot (not required at runtime)

Outputs (runtime):

./chroma_db_edgar/ — persisted Chroma index

simple_eval_results.csv — evaluation results

Quickstart
1) Prerequisites

Python 3.10+

Ollama installed and running (local LLM runtime). Pull models with ollama pull, list with ollama list. 
LangChain

Models used by this script:

LLM: llama3.2:3b (or another small local model)

Embeddings: all-minilm (Ollama’s MiniLM embedding model) 
ollama.com

# Start Ollama (if needed) and pull models
ollama serve
ollama pull llama3.2:3b
ollama pull all-minilm

2) Install Python deps
pip install -r requirements.txt

3) Run
python rag-sec-filings.py


What happens:

Loads aig-10k.csv

De-duplicates to one value per (companyFact, fy, form) (keeping the latest filed)

Builds/updates a Chroma index at ./chroma_db_edgar

Runs three demo questions and writes simple_eval_results.csv

LangChain × Ollama (ChatOllama) is used for local inference; Chroma metadata filters are used to constrain retrieval by fields like form, entityName, and fy. 
LangChain
Chroma Docs

Configuration (inside rag-sec-filings.py)
CSV_PATH     = "aig-10k.csv"                    # input CSV
COMPANY_NAME = "American International Group, Inc."
FORM         = "10-K"
PERSIST_DIR  = "./chroma_db_edgar"

EMB_MODEL    = "all-minilm"                     # Ollama embeddings
GEN_MODEL    = "llama3.2:3b"                    # Ollama LLM


Fact aliases: FACT_ALIASES maps friendly names to EDGAR tags (e.g., "total revenues" → "Revenues", "net income" → "NetIncomeLoss").

Retrieval: k=6 with a Chroma where filter like {"$and":[{"form":{"$eq":"10-K"}}, {"entityName":{"$eq":"American International Group, Inc."}}, {"fy":{"$eq": 2022}}]} (operator style per Chroma docs). 
Chroma Docs
Chroma Cookbook

Input data schema

aig-10k.csv columns (from sample):

cik, entityName, companyFact, end, val, accn, fy, fp, form, filed, units


Example (abridged):
cik=5272, entityName="American International Group, Inc.", companyFact="Assets", fy=2022, fp="FY", form="10-K", filed=2023-02-17, val=..., units=USD

Results (from simple_eval_results.csv)

File columns: question, llm_answer, ground_truth
Summary: 3 queries → 1 exact, 1 numeric diff, 1 formatting error.

#	question	model answer (parsed)	ground truth	abs. error	% error	status
1	AIG total revenue in FY 2022	56,437,000,000	56,437,000,000	0	0.00%	exact
2	AIG cash in FY 2022	2,216,000,000	2,043,000,000	173,000,000	8.47%	numeric_diff
3	AIG assets in FY 2022	(JSON array returned; not a scalar)	526,634,000,000	—	—	formatting_error

Note on row 3 (formatting): regarding a formatting error I get,  I am working on its fix, but I leave it as it is to meet the deadline. The LLM returned a JSON array of objects instead of the expected single numeric scalar. This is tracked as a known issue (see below).

Known issues (current state)

Row-level formatting (evaluation row #3): The LLM response is an array instead of a scalar; downstream parsing treats it as a formatting error. A targeted prompt tweak (or post-parse) is pending to normalize numeric answers to a single scalar.

CSV-row retrieval unit: Retrieval is over CSV rows (facts), not page-level 10-K text. Page-level retrieval and single-filing enforcement by accession are out of scope here.

Limited evaluation: Only three demo questions are run; retrieval metrics (P@k/Recall@k) are not computed in this prototype.

How it works (brief)

Load & de-duplicate
Sorts by companyFact, fy, form, filed; keeps the latest filed per (companyFact, fy, form).

Build Documents
Each CSV row becomes a LangChain Document with compact text and rich metadata (entityName, cik, form, fy, fp, companyFact, accn, filed, units, …).

Vector index
Uses Chroma via LangChain’s integration with Ollama embeddings (all-minilm) and persists to ./chroma_db_edgar. You can apply metadata filters at query time. 
Chroma Docs
ollama.com

Retrieval
A retriever is created with as_retriever(search_kwargs={"k": 6, "filter": where_dict}), where where_dict uses Chroma’s $eq/$and style. 
Chroma Docs

LLM extraction
The LLM (llama3.2:3b via ChatOllama) is prompted to return JSON only with keys like:
variable, value, units, fy, form, accn, filed, source. 
LangChain

Ground truth lookup
A deterministic function returns the latest filed row for (companyFact, fy) from the same CSV.

Simple evaluation
Writes simple_eval_results.csv with columns: question, llm_answer, ground_truth.

Notes & future work (not included here)

Single-filing, page-level retrieval from a 10-K (by accession), PySpark orchestration, and expanded evaluation (P@k/Recall@k, numeric error stats across 15 labels) are natural next steps.

Swapping in different Ollama models is supported; update GEN_MODEL / EMB_MODEL and pull with ollama pull. 
LangChain

References

LangChain × Ollama (ChatOllama) — integration & usage. 
LangChain

Chroma metadata filtering — where, $and, $eq, etc. 
Chroma Docs
Chroma Cookbook

Ollama embedding model all-minilm — embedding-only model for sentence vectors.
