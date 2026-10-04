# Phase 4 Manual Relevance Annotation Guidelines & Instructions

## Overview
This document provides instructions for human relevance annotation on the pooled retrieval candidate set for the 15 domain evaluation queries (Q01–Q15).

Evaluation candidate pools are constructed by taking the **UNION of Top-10 retrieved content units** from:
1. **Pipeline A**: Lemmatization (`pipeline_a_lemma`)
2. **Pipeline B**: Stemming (`pipeline_b_stem`)

The candidate file is saved at:
[`results/phase4/manual_relevance_judgments.csv`](file:///c:/Users/calpo/OneDrive/Documents/NLP/NLP_Domain_Text_Analysis/results/phase4/manual_relevance_judgments.csv)

---

## Relevance Rubric

Each candidate row presents a single `(query_id, unit_id)` pair. Annotators must assign a binary relevance score (`relevance`) in the CSV file:

### RELEVANT (`1`)
Assign `relevance = 1` when the content unit's text:
- Directly answers, substantially discusses, or provides material quantitative/qualitative evidence for the information need expressed by the query.
- Would serve as a helpful, substantive answer to a financial analyst asking *"What does this corpus say about <query>?"*.

### NOT RELEVANT (`0`)
Assign `relevance = 0` when the content unit:
- Contains a query word only lexically or incidentally (e.g., in a boilerplate footer, generic index listing, table footnote, or chart label) without materially addressing the topic.
- Discusses an unrelated topic where the query word appears out of context.
- Is an extraction artifact or incomplete text fragment.

---

## Query-Type Guidelines

1. **Keyword Queries** (e.g., `inflation`, `GDP`, `securities`):
   - The unit text must develop or report data regarding the financial concept, not merely list the word in passing.

2. **Phrase Queries** (e.g., `"fiscal deficit"`, `"repo rate"`, `"non-performing assets"`):
   - Semantic relevance to the specific phrase matters. Exact literal phrase occurrence is not sufficient if the surrounding context does not substantively discuss the phrase.

3. **Boolean Queries** (e.g., `inflation AND growth`, `GDP OR GVA`, `inflation AND NOT food`, `(RBI OR SEBI) AND regulation`):
   - Judge according to the **intended financial information need** represented by the Boolean expression (e.g., for `inflation AND NOT food`, the unit must discuss core/services inflation while excluding units focused solely on food inflation).

---

## Blind & Fair Annotation Rules

- **Do NOT label units based on which pipeline retrieved them.** The candidate list presents `pipeline_a_rank` and `pipeline_b_rank` for reference, but annotation decisions must rely strictly on the unit text content.
- Unjudged candidates are represented by a blank `relevance` field in `manual_relevance_judgments.csv`.
- Once human annotations are entered, save the CSV file to enable Phase 4 retrieval metric calculation.
