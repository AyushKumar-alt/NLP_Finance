"""Statistics tests - baseline counters must be labelled as baseline."""


def test_sentence_counter_respects_economic_abbreviations(pipeline):
    from src.statistics.stats import SentenceCounter

    counter = SentenceCounter(pipeline["config"])
    text = (
        "Real GDP grew 7.4 per cent in FY26. The U.S. economy slowed. "
        "RBI. IMF. SEBI and MoSPI published data. IIP rose 3 per cent."
    )
    sentences = counter.split(text)
    assert 3 <= len(sentences) <= 5
    assert counter.method.startswith("nltk") or counter.method.startswith("regex")
    assert "7.4 per cent in FY26." in sentences[0]


def test_baseline_tokenizer_keeps_numbers_and_currency(pipeline):
    from src.statistics.stats import build_tokenizer

    tokenizer = build_tokenizer(pipeline["config"])
    tokens = tokenizer.tokens("GDP rose 7.4 per cent to \u20b9346 lakh crore in FY26 (Q3).")
    joined = " ".join(tokens)
    for token in ("7.4", "per cent", "₹346", "FY26"):
        assert token in joined, f"{token} missing from {joined}"


def test_statistics_are_labelled_baseline_not_final(pipeline):
    pipeline["runner"].run()
    stats = pipeline["config"].results_path("document_statistics")
    header = stats.read_text(encoding="utf-8").splitlines()[0]
    assert "baseline_token_count" in header
    assert "sentences" in header
    assert "vocabulary_size" in header
    assert "raw_vocabulary_size" in header


def test_term_dictionary_is_not_called_an_index(pipeline):
    pipeline["runner"].run()
    metadata = pipeline["config"].metadata_path("baseline_term_dictionary")
    assert metadata.exists()
    header = metadata.read_text(encoding="utf-8").splitlines()[0]
    assert header.startswith("term,frequency,document_frequency,documents")


def test_vocabulary_file_contains_domain_abbreviations(pipeline):
    pipeline["runner"].run()
    vocabulary = pipeline["config"].metadata_path("vocabulary_baseline").read_text(encoding="utf-8")
    assert "gdp" in vocabulary.lower()
