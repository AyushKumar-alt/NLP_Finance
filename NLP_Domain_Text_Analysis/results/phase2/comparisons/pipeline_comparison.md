# Controlled Pipeline Normalization Comparison: Pipeline A vs Pipeline B

**Domain:** Financial & Economic Documents  
**Experimental Control:** Both pipelines process identical input units using identical Custom Tokenization and Domain-Aware Stopword Removal.

---

## 1. Pipeline Definitions

### Pipeline A — Lemmatization Pipeline
1. **Custom Tokenization** (preserves ₹1,000, 6.5%, FY26, 2025-26, scale words)
2. **Domain-Aware Stopword Removal** (protects key financial terms: rate, growth, policy, RBI, GDP, etc.)
3. **Lemmatization** (WordNet / LemmInflect rule-based normalization)

### Pipeline B — Stemming Pipeline
1. **Custom Tokenization** (preserves ₹1,000, 6.5%, FY26, 2025-26, scale words)
2. **Domain-Aware Stopword Removal** (protects key financial terms: rate, growth, policy, RBI, GDP, etc.)
3. **Stemming** (Snowball English stemmer)

---

## 2. Experimental Results & Metrics

| Pipeline | Normalization Strategy | Units | Tokens | Vocab Size | TTR | Tokens/Unit | Token Red. % | Vocab Red. % | Domain Terms Preserved | Runtime |
|---|---|---|---|---|---|---|---|---|---|---|
| Pipeline A | Lemmatization (LemmInflect/WordNet) | 6134 | 250,794 | 18,445 | 0.108 | 40.89 | 27.45% | 0.76% | 14 / 14 (100.0%) | 0.2099s |
| Pipeline B | Stemming (Snowball English) | 6134 | 250,794 | 13,116 | 0.0727 | 40.89 | 27.45% | 29.43% | 14 / 14 (100.0%) | 0.4192s |

---

## 3. Controlled Financial Expression Transformations

| Financial Expression | Custom Tokenizer Output | Pipeline A (Lemmatized) | Pipeline B (Stemmed) | Identical? | Linguistic Observation |
|---|---|---|---|---|---|
| GDP | GDP | gdp | gdp | True | Identical |
| GVA | GVA | gva | gva | True | Identical |
| CPI | CPI | cpi | cpi | True | Identical |
| RBI | RBI | rbi | rbi | True | Identical |
| SEBI | SEBI | sebi | sebi | True | Identical |
| repo rate | repo rate | repo rate | repo rate | True | Identical |
| fiscal deficit | fiscal deficit | fiscal deficit | fiscal deficit | True | Identical |
| current account deficit | current account deficit | current account deficit | current account deficit | True | Identical |
| ₹1,000 | ₹1,000 | ₹1,000 | ₹1,000 | True | Identical |
| 6.5% | 6.5% | 6.5% | 6.5% | True | Identical |
| FY26 | FY26 | fy26 | fy26 | True | Identical |
| 2025-26 | 2025-26 | 2025-26 | 2025-26 | True | Identical |
| non-performing assets | non-performing assets | non-performing assets | non-perform asset | False | Stemmer over-stemmed/modified suffix |
| government securities | government securities | government securities | govern secur | False | Stemmer over-stemmed/modified suffix |

---

## 4. Linguistic Observations & Evaluation Strategy

1. **Vocabulary Reduction & Overstemming**:
   - **Pipeline B (Stemming)** achieves aggressive vocabulary reduction, but often truncates domain suffixes (e.g. `inflation` -> `inflat`, `securities` -> `secur`).
   - **Pipeline A (Lemmatization)** preserves full morphological dictionary heads, ensuring human readability and exact match for domain terms (e.g. `securities` -> `security`).

2. **Domain Term Preservation**:
   - Both pipelines successfully preserve critical financial acronyms (`GDP`, `CPI`, `RBI`, `SEBI`) and formatted expressions (`FY26`, `6.5%`, `₹1,000`).

3. **Phase 3 Retrieval Selection Strategy**:
   - **Important**: The final selection of the optimal retrieval pipeline for Phase 3 will **NOT** be made solely based on vocabulary reduction in Phase 2.
   - Phase 2 characterizes linguistic and vocabulary differences. The definitive selection between Pipeline A and Pipeline B for Information Retrieval will be supported by **Phase 3 / Phase 4 IR Retrieval Evaluation** measuring:
     - **Precision, Recall, F1**
     - **Precision@K & Recall@K**
     - **Mean Average Precision (MAP) & NDCG**
