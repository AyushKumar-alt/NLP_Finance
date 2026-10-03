"""Phase 2 - NLP experimentation layer over the Phase 1 structured corpus.

Phase 2 consumes ``data/corpus/structured/corpus.jsonl`` produced by Phase 1 and
runs tokenization, preprocessing, stemming, lemmatization, POS, NER, n-gram and
BPE experiments. It never modifies Phase 1 outputs and never touches the source
PDFs.
"""

__all__ = [
    "config",
    "load_corpus",
    "tokenizers",
    "custom_tokenizer",
    "date_number_tokenizer",
    "stopwords",
    "stemming",
    "lemmatization",
    "pos_tagging",
    "custom_pos",
    "ml_pos",
    "ner",
    "ngrams",
    "bpe",
    "comparisons",
    "statistics",
    "figures",
    "validation",
]
