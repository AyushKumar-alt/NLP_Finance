# Domain-Specific Text Analysis and Retrieval System
## Financial and Economic Documents

**Course:** Natural Language Processing (NLP Assessment-1)  
**Institution:** Vidyashilp University — Bangalore  
**Project Status:** Complete, Validated, and Frozen (Phases 1–4, Dual-Pipeline IR Engine, Evaluation, REST API & Next.js GUI)

---

## 1. Project Overview

This project implements a production-grade, domain-specific Natural Language Processing (NLP) and Information Retrieval (IR) system engineered specifically for Indian financial and economic reports. The system addresses the complex linguistic structure of regulatory frameworks, central bank stability reports, and national economic surveys.

The architecture spans four core phases:
1. **Phase 1: Ingestion & Unit Extraction** — Deterministic PDF parsing, structural content-unit extraction, and metadata registry.
2. **Phase 2: Preprocessing & Representation Experiments** — Comparative tokenization, domain stop-word filtering, stemming vs. lemmatization, POS tagging, NER, n-grams, and BPE.
3. **Phase 3: Information Retrieval Engine** — Dual positional inverted index construction, Abstract Syntax Tree (AST) Boolean query parsing, and phrase search execution.
4. **Phase 4: Comparative Evaluation & Web Application** — Pooled Top-10 relevance evaluation across 15 canonical queries, comprehensive IR metrics (Precision, Recall, F1, MAP, nDCG@K, MRR), and an interactive full-stack web UI.

### Engineering Methodology
The project strictly enforces the systematic engineering control flow:
$$\mathbf{SELECT} \longrightarrow \mathbf{ORDER} \\longrightarrow \\mathbf{IMPLEMENT} \\longrightarrow \\mathbf{COMPARE} \\longrightarrow \\mathbf{EVALUATE} \\longrightarrow \\mathbf{JUSTIFY}$$

---

## 2. Problem Statement

General-purpose NLP pipelines and generic web search engines perform poorly on domain-specific financial corpora due to several structural challenges:
* **Domain Phrase Ambiguity**: Single terms like *"monetary"* or *"repo"* carry high entropy, whereas multi-word phrases like *"monetary policy transmission"* or *"repo rate"* represent atomic economic concepts.
* **Symbolic & Fiscal Tokens**: Standard tokenizers strip critical symbols (`₹`, `$`, `%`, decimal rates) and alter fiscal abbreviations (`FY2025`, `Q3FY24`, `CRAR`, `LCR`, `GVA`).
* **Morphological Mutation**: Aggressive suffix stripping (stemming) can collapse distinct financial terms into ambiguous root stems.
* **Document Structure**: Regulatory reports contain paragraphs, footnotes, tables, figures, boxes, and headings that require strict hierarchical traceability.

---

## 3. Assignment Objective

The primary objective of NLP Assessment-1 is to design, implement, empirically evaluate, and deploy a complete domain-specific IR system that compares **Lemmatization vs. Stemming** within two controlled retrieval pipelines while demonstrating high precision, transparent provenance, and clean web application integration.

---

## 4. Corpus Characterization

The corpus consists of **31 financial and economic reports** collected across three primary institutional source families (with supplementary metadata for PRS Union Budget materials):

1. **Economic Survey of India 2025–26** (`SRC01`)
2. **RBI Financial Stability Report (FSR)** (`SRC02`)
3. **SEBI Annual Report 2025–26** (`SRC03`)

### Canonical Corpus Metrics
* **Total Documents**: `31`
* **Parent Sources**: `3`
* **Total Pages**: `885`
* **Extracted Content Units**: `8,311`
* **Searchable Units**: `6,134`
* **Total Word Count**: `326,589`
* **Total Character Count**: `2,313,647`
* **Baseline Vocabulary**: `25,910`

### Content Unit Distribution
* Paragraphs: `5,760`
* Footnotes: `1,040`
* Headings: `412`
* Other: `382`
* Figures: `340`
* Tables: `220`
* Boxes: `153`
* References: `4`

### Hierarchical Traceability
Every content unit maintains explicit structural provenance:
$$\text{SOURCE FAMILY} \longrightarrow \text{DOCUMENT ID} \longrightarrow \text{PAGE} \longrightarrow \text{SECTION} \longrightarrow \text{CONTENT UNIT} \longrightarrow \text{TEXT}$$

---

## 5. System Architecture

The project connects offline NLP indexing pipelines with a live full-stack web application:

```
[Phase 1 Corpus PDFs]
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│ Phase 1: Ingestion & Unit Segmentation                  │
│ Output: data/corpus/metadata/document_registry.csv      │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ Phase 2: NLP Experiments & Preprocessing                │
│ (Custom Tokenizer, Domain Stopwords, POS, NER, N-Grams) │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ Phase 3: Dual Positional Inverted Index Engine          │
│ Pipeline A (Lemmatization)  │  Pipeline B (Stemming)   │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ Phase 4: Pooled Evaluation & REST API / Web App          │
│ FastAPI Backend (Port 8000) ◄─REST─► Next.js GUI (3000) │
└─────────────────────────────────────────────────────────┘
```

---

## 6. NLP Processing & Component Experiments

### Conservative Financial Text Cleaning
* **Currency Symbols**: `₹`, `$`, `€`, `£` retained for monetary expression context.
* **Rates & Percentages**: `5.5%`, `25bps` preserved as atomic tokens.
* **Fiscal Codes**: `FY2025`, `Q3FY24`, `CRAR`, `LCR`, `GVA` preserved.

### Tokenization Strategy Comparison
* **NLTK Tokenizer**: 349,344 tokens | 18,035 vocabulary (strips symbols)
* **spaCy Tokenizer**: 365,068 tokens | 15,899 vocabulary (splits compounds)
* **Custom Finance Tokenizer**: 345,677 tokens | 18,628 vocabulary (preserves currency/rates)
* **Hybrid Tokenizer**: 344,141 tokens | 19,463 vocabulary (multi-pass fallback)

### Domain-Aware Stop-Word Processing
* **Standard Stopwords (NLTK)**: 250,215 remaining tokens (strips critical terms like *'rate'*, *'tax'*, *'bank'*).
* **Domain-Aware Stopwords**: 249,744 remaining tokens (retains 85 protected financial terms while pruning report administrative noise).

### Stemming vs. Lemmatization
* **Stemmers**: Porter (14,243 vocab), Snowball English (14,250 vocab — selected for Pipeline B), Lancaster (13,023 vocab).
* **Lemmatizers**: spaCy (13,295 vocab), LemmInflect / WordNet (13,135 vocab — selected for Pipeline A).

### Part-of-Speech Tagging
* Fine-Grained Penn Treebank tags (`NN`, `NNP`, `NNS`, `JJ`, `IN`) used for accurate syntactic context.
* **ML POS Classifier Performance**: Accuracy = **87.69%**, Macro F1 = **0.7764**.

### Named Entity Recognition (NER) & N-Grams
* **General Entities**: `26,717` | **Domain Entities**: `4,968` (`MONEY`, `PERCENT`, `ORG`, `DATE`).
* **Key Domain Phrases**: *'monetary policy'*, *'repo rate'*, *'financial stability'*, *'current account deficit'*.
* **BPE Subword Tokenization**: Target vocabulary = 8,000 | Resulting total tokens = 468,203.

---

## 7. Pipeline A vs. Pipeline B Architectures

We construct two controlled primary retrieval pipelines to isolate the impact of normalization:

```
PIPELINE A (pipeline_a_lemma):
  Custom Finance Tokenizer → Date/Number Preservation → Domain Stopwords → Lemmatization (LemmInflect/WordNet) → Positional Index

PIPELINE B (pipeline_b_stem):
  Custom Finance Tokenizer → Date/Number Preservation → Domain Stopwords → Snowball English Stemming → Positional Index
```

### Final Index Statistics (6,134 Searchable Units Each)
* **Pipeline A (`pipeline_a_lemma`)**:
  * Phase 2 Post-Stopword Tokens: `250,794`
  * Phase 3/4 Positional Index Tokens: `249,489`
  * Total Index Terms: `26,155`
  * Unigram Terms: `24,110`
  * Phrase Terms: `2,045`
  * Total Postings: `195,579`
* **Pipeline B (`pipeline_b_stem`)**:
  * Phase 2 Post-Stopword Tokens: `250,794`
  * Phase 3/4 Positional Index Tokens: `249,489`
  * Total Index Terms: `21,403`
  * Unigram Terms: `18,763`
  * Phrase Terms: `2,640`
  * Total Postings: `195,567`

---

## 8. Information Retrieval Engine

The Phase 3 engine implements:
* **Positional Postings**: `(unit_id, tf, [pos_1, pos_2, ...])` enabling exact phrase distance verification ($pos_{i+1} = pos_i + 1$).
* **Abstract Syntax Tree (AST) Boolean Parser**: A recursive descent parser supporting single terms, exact phrases (enclosed in quotation marks), `AND`, `OR`, `NOT`, and nested parenthetical expressions.
* **Phrase Positional Normalization**: Cleanly strips outer quotation marks so adjacent tokens query positional postings directly.
* **Strict Boolean Operator Syntax**:
  * Supported operators: `AND`, `OR`, `NOT`.
  * Standalone unquoted `&` is detected and blocked across lexer, backend, and frontend validation with: `"& is not a supported Boolean operator. Use AND instead. (e.g. "monetary policy" AND "repo rate")"`.
* **Deterministic Custom Retrieval Ranking Score**:
  $$\text{Score} = 1.0 \cdot N_{\text{matched}} + 2.0 \cdot \mathbb{I}_{\text{phrase}} + 0.25 \cdot \log_{10}(1 + \text{TF}_{\text{matched}})$$
  * Transparently combines query-term coverage (1.0 per distinct positive term), exact phrase bonuses (2.0), and logarithmic term frequency ($0.25 \times \log_{10}(1 + \text{TF})$). Not a probability or confidence score.
* **Multi-Term Evidence Snippet Generation**:
  * Sliding window algorithm maximizes the number of distinct positive matched terms captured in the snippet or centers on exact matched phrases.
  * When terms are far apart, appends an explicit continuity note: `[Additional matched term occurs elsewhere in this content unit.]`.

---

## 9. Evaluation Methodology

Following TREC/Cranfield evaluation standards, evaluation is conducted using a **pooled Top-10 candidate relevance set**:
* **15 Canonical Benchmark Queries** (`Q01`–`Q15`)
* **166 Unique Query-Unit Candidate Pairs**
* **166 Judged Pairs** (`155` relevant, `11` non-relevant, `0` unjudged)
* **Binary Relevance Rubric** (`1` = Substantive financial discussion, `0` = Passing mention / table artifact)
* **Recall Terminology**: Reported as **Pooled Recall** based on dual-pipeline candidate union.

---

## 10. Final Evaluation Results & Selection

### Overall Comparative Metrics

| Metric | Pipeline A (`pipeline_a_lemma`) | Pipeline B (`pipeline_b_stem`) | Delta ($\Delta$) | Winner |
| :--- | :-: | :-: | :-: | :-: |
| **Precision** | **0.9764** | 0.9571 | +0.0193 | **Pipeline A** |
| **Pooled Recall** | **0.8820** | 0.8723 | +0.0097 | **Pipeline A** |
| **F1-Score** | **0.9183** | 0.9053 | +0.0130 | **Pipeline A** |
| **MAP** | **0.9328** | 0.9231 | +0.0097 | **Pipeline A** |
| **MRR** | **1.0000** | 1.0000 | 0.0000 | Tie |
| **P@1** | **1.0000** | 1.0000 | 0.0000 | Tie |
| **P@3** | **0.9778** | 0.9556 | +0.0222 | **Pipeline A** |
| **P@5** | **0.9600** | 0.9200 | +0.0400 | **Pipeline A** |
| **P@10** | **0.8467** | 0.8400 | +0.0067 | **Pipeline A** |
| **R@1** | **0.1651** | 0.1651 | 0.0000 | Tie |
| **R@3** | **0.3702** | 0.3546 | +0.0156 | **Pipeline A** |
| **R@5** | **0.5130** | 0.4923 | +0.0207 | **Pipeline A** |
| **R@10** | **0.8820** | 0.8723 | +0.0097 | **Pipeline A** |
| **nDCG@1** | **1.0000** | 1.0000 | 0.0000 | Tie |
| **nDCG@3** | **0.9754** | 0.9559 | +0.0195 | **Pipeline A** |
| **nDCG@5** | **0.9663** | 0.9472 | +0.0191 | **Pipeline A** |
| **nDCG@10** | **0.9632** | 0.9554 | +0.0078 | **Pipeline A** |

### Per-Query Breakdown Summary
* **Pipeline A Wins**: `3` queries (`Q09` *banking AND credit*, `Q14` *GDP AND investment*, `Q15` *(GDP OR GVA) AND policy*)
* **Pipeline B Wins**: `0` queries
* **Ties**: `12` queries

### FINAL SELECTED PIPELINE: **Pipeline A — Lemmatization**
* **Primary Criterion**: Mean Average Precision (**MAP = 0.9328** vs **0.9231**).
* **Rationale**: Lemmatization preserves domain semantics for multi-term queries, avoiding stem-collision noise caused by aggressive suffix stripping.

---

## 11. Canonical 15-Query Registry

1. `Q01`: `inflation` (keyword)
2. `Q02`: `GDP` (keyword)
3. `Q03`: `FDI` (keyword)
4. `Q04`: `"monetary policy"` (phrase)
5. `Q05`: `"financial stability"` (phrase)
6. `Q06`: `"current account deficit"` (phrase)
7. `Q07`: `GDP AND inflation` (Boolean AND)
8. `Q08`: `RBI AND "repo rate"` (Boolean phrase + AND)
9. `Q09`: `banking AND credit` (Boolean AND)
10. `Q10`: `GDP OR GVA` (Boolean OR)
11. `Q11`: `inflation OR disinflation` (Boolean OR)
12. `Q12`: `inflation AND NOT food` (Boolean NOT)
13. `Q13`: `banking AND NOT insurance` (Boolean NOT)
14. `Q14`: `GDP AND investment` (Boolean AND)
15. `Q15`: `(GDP OR GVA) AND policy` (Grouped Boolean)

---

## 12. Full-Stack Web Application Architecture

* **Frontend**: Next.js 16 (React 19, Tailwind CSS, TypeScript)
* **Backend API**: FastAPI (Python REST API serving `Phase3Runtime`)
* **Capabilities**:
  * **Search Page User Journey**: Streamlined query builder featuring real-time syntax validation, examples, auto-detect vs manual query type selection, top-K selection, and pipeline toggling.
  * **Explainable Ranking Score**: Interactive `Ranking Score ⓘ` tooltip providing mathematical breakdowns of term, phrase, and TF components.
  * **Unambiguous Result Counts**: Clearly distinguishes `Unique Documents`, `Matching Content Units`, and `Showing` (with notice for Phase 3 50-candidate cap).
  * **Per-Result Matched Badges**: Individual chips for terms matching each specific content unit, properly handling `OR` and grouped satisfaction.
  * **Pipeline Isolation**: Instant switching between Pipeline A (`pipeline_a_lemma`) and Pipeline B (`pipeline_b_stem`) on independent indexes.
  * **Provenance & Evaluation**: Hierarchical document/page/section provenance display, corpus statistics explorer, and interactive Cranfield evaluation metrics dashboard.
* **Test Suite**: Phase 3/4 tests and dedicated search explainability regression test suite (`tests/test_search_explainability_fixes.py`) passing.

---

## 13. Repository Structure

```
NLP_Domain_Text_Analysis/
├── backend/            # FastAPI backend server & REST endpoint schemas
│   ├── main.py         # FastAPI application entrypoint
│   └── schemas/        # Request/Response Pydantic models
├── config/             # YAML configuration files for Phase 1–4
├── data/               # Corpus metadata, manifests, and structured units
├── docs/               # Architecture documents and methodology specifications
├── frontend/           # Next.js 16 web application
│   ├── src/app/        # App router pages (search, evaluation, pipelines, etc.)
│   ├── src/components/ # Shared layout & UI components
│   └── src/lib/        # API client & TypeScript interfaces
├── notebooks/          # Master academic submission Jupyter notebook
│   └── NLP_Domain_Text_Analysis_Assessment.ipynb
├── results/            # Persisted canonical Phase 1–4 artifacts & index files
│   ├── phase1/         # Extraction manifests
│   ├── phase2/         # Preprocessing, POS, NER, BPE, N-gram artifacts
│   ├── phase3/         # Inverted index JSON & retrieval summaries
│   └── phase4/         # Relevance judgments & evaluation metrics CSVs
├── src/                # Modular Python source code for Phase 1–4 engines
│   ├── phase1/         # Ingestion engine
│   ├── phase2/         # NLP experiment runners
│   ├── phase3/         # Inverted index & retrieval engine
│   └── phase4/         # Evaluator & runtime engine factory
└── tests/              # Pytest test suite for Phase 1–4
```

---

## 14. Installation & Requirements

### System Requirements
* Python 3.10+
* Node.js 18+ & npm

### Python Setup
```powershell
# Create & activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Frontend Setup
```powershell
npm --prefix frontend install
```

---

## 15. Running the Application

### 1. Start FastAPI Backend API (Port 8000)
```powershell
python -m uvicorn backend.main:app --reload --port 8000
```

### 2. Start Next.js Frontend GUI (Port 3000)
```powershell
npm --prefix frontend run dev
```

Open your web browser to **[http://localhost:3000](http://localhost:3000)**.

---

## 16. Running the Academic Notebook

To execute the master academic submission notebook top-to-bottom:

```powershell
jupyter nbconvert --to notebook --execute notebooks/NLP_Domain_Text_Analysis_Assessment.ipynb --output NLP_Domain_Text_Analysis_Assessment_Executed.ipynb
```

Or open directly in VS Code / Jupyter Lab:
`notebooks/NLP_Domain_Text_Analysis_Assessment.ipynb`

---

## 17. Running Tests

Run the complete Phase 4 pytest suite:

```powershell
python -m pytest tests/phase4/ -v
```

---

## 18. Limitations & Scope Boundaries

1. **Pooled Candidate Relevance**: Evaluation uses Top-10 pooled candidate union labeling rather than exhaustive corpus-wide annotation.
2. **Benchmark Query Size**: Evaluation is conducted over 15 canonical domain queries.
3. **Lexical Ranking Model**: Uses TF-IDF / BM25 positional matching without dense neural embeddings.
4. **Static Corpus Boundary**: Content extraction is optimized for structured financial PDFs.
5. **No Statistical Significance Test**: Pipeline A superiority is reported based on empirical metrics without hypothesis testing.

---

## 19. Final Conclusion

This project successfully constructs and evaluates a domain-specific Information Retrieval system for Indian financial text. **Pipeline A (Lemmatization)** demonstrates superior empirical retrieval performance across all key evaluation metrics (**MAP = 0.9328**, **nDCG@10 = 0.9632**, **Precision = 0.9764**, **Pooled Recall = 0.8820**, **F1 = 0.9183**, **MRR = 1.0000**), establishing that domain-aware tokenization and lemmatization effectively preserve structural economic semantics.