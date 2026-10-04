"""Phase 2 orchestrator.

Single entry point::

    python -m src.phase2.run

Pipeline (each stage writes its own artefacts and is independently recoverable):

 1.  load the Phase 1 structured corpus (read only)
 2.  freeze the common experiment sample + representative sample
 3.  tokenization: nltk / spacy / custom / hybrid (+ rule dump, examples)
 4.  date & number aware tokenization
 5.  one shared spaCy pass (tokens, tags, POS, lemmas, entities)
 6.  stopword strategies
 7.  stemming (+ order experiment)
 8.  lemmatization
 9.  POS: default, custom dictionary, custom ML (gold-annotated evaluation)
 10.  NER: general, domain dictionary, error analysis
 11.  n-grams 1..5 + domain phrase mining
 12.  BPE training and analysis
 13.  comparison tables, master comparison, summary CSV/JSON
 14.  charts
 15.  validation report
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

if __package__ in (None, ""):  # allow `python src/phase2/run.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.utils import write_csv, write_json

from . import bpe as bpe_module
from . import comparisons, custom_pos, custom_tokenizer, date_number_tokenizer, figures, lemmatization as lemmatization_module
from . import ml_pos, ner, ngrams
from . import pos_tagging, reporting, statistics, stemming, stopwords, tokenizers, validation
from .config import Phase2Config, clean_managed_output, load_config, log_event, seed_everything, setup_logging
from .lemmatization import run_lemmatization
from .load_corpus import build_experiment_sample, experiment_manifest, load_corpus, representative_sample
from .spacy_pipeline import annotate
from .statistics import TokenizedCorpus, build_tokenized

BANNER = "=" * 50


def _section(logger, title: str) -> None:
    print(f"\n{BANNER}\n{title}\n{BANNER}", flush=True)
    log_event(logger, "INFO", "run", title)


def run(config: Optional[Phase2Config] = None, verbose: bool = False) -> Dict[str, Any]:
    started = time.time()
    config = config or load_config()
    logger = setup_logging(config, verbose=verbose)
    seed_everything(config.random_seed)

    for key in (
        "results_dir", "tokenization_dir", "preprocessing_dir", "stemming_dir",
        "lemmatization_dir", "pos_dir", "ner_dir", "ngrams_dir", "bpe_dir",
        "comparisons_dir", "summaries_dir", "figures_dir", "annotations_dir",
    ):
        config.out_dir(key)
    if bool(config.get("run.clean_phase2_output_before_write", True)):
        clean_managed_output(config, logger)

    # ---------------------------------------------------------------- 1. corpus
    _section(logger, "PHASE 2: NLP EXPERIMENTS")
    corpus = load_corpus(config, logger)
    sample = build_experiment_sample(corpus)
    # Optional smoke-test switch (0 = use the whole policy-selected corpus).
    sample_limit = int(config.get("run.debug_sample_limit_units", 0) or 0)
    if sample_limit and sample_limit < len(sample.units):
        sample = type(sample)(
            policy=sample.policy,
            units=sample.units[:sample_limit],
            documents=sample.documents,
        )
        log_event(logger, "WARNING", "run", f"DEBUG LIMIT: experiment sample truncated to {sample_limit} units")
    summary_corpus = corpus.summary()
    print(
        f"\nCorpus:\n  Documents: {summary_corpus['documents']}\n"
        f"  Pages: {summary_corpus['pages']}\n"
        f"  Units (all types): {summary_corpus['units_total']}\n"
        f"  Units in policy '{summary_corpus['text_selection_policy']}': {summary_corpus['selected_units']}\n"
        f"  Characters: {summary_corpus['selected_characters']:,}",
        flush=True,
    )

    results_dir = config.out_dir("results_dir")
    tokenization_dir = config.out_dir("tokenization_dir")
    preprocessing_dir = config.out_dir("preprocessing_dir")
    stemming_dir = config.out_dir("stemming_dir")
    lemmatization_dir = config.out_dir("lemmatization_dir")
    pos_dir = config.out_dir("pos_dir")
    ner_dir = config.out_dir("ner_dir")
    ngrams_dir = config.out_dir("ngrams_dir")
    bpe_dir = config.out_dir("bpe_dir")
    comparisons_dir = config.out_dir("comparisons_dir")
    summaries_dir = config.out_dir("summaries_dir")

    metrics: Dict[str, Any] = {"corpus_path": str(config.corpus_jsonl)}

    # ---------------------------------------------------------------- 2. sample
    print("\n[1/10] Building the common experiment sample and manifests...", flush=True)
    representative = representative_sample(corpus, logger)
    representative = [
        {**row, "text": row["text"][:1200]}
        for row in representative
    ]
    # index each representative unit inside the sample so the four tokenizers can
    # be compared on exactly that text
    sample_index_by_unit = {unit.unit_id: index for index, unit in enumerate(sample.units)}
    for row in representative:
        row["sample_index"] = sample_index_by_unit.get(str(row["unit_id"]), -1)

    manifest_rows = experiment_manifest(
        sample,
        experiments=["tokenization_nltk", "tokenization_spacy", "tokenization_custom", "tokenization_hybrid", "bpe"],
        policy=corpus.policy,
        max_rows_per_experiment=int(config.get("tokenization.common_manifest_max_rows_per_experiment", 0)),
    )
    write_csv(
        comparisons_dir / "experiment_manifest.csv",
        manifest_rows,
        [
            "experiment_id", "document_id", "source_id", "source_type", "page_number", "unit_id",
            "unit_type", "section_id", "section_number", "section_title", "text_selection_policy",
            "text_length", "unit_index_in_sample",
        ],
    )
    write_csv(
        comparisons_dir / "representative_sample.csv",
        representative,
        ["category", "document_id", "source_id", "source_type", "page_number", "section_id",
         "section_number", "section_title", "unit_id", "unit_type", "char_count", "text"],
    )
    log_event(
        logger, "INFO", "sample",
        f"common sample: {len(sample.units)} units / {sample.character_count} characters; "
        f"representative categories filled: {len(representative)}/10",
    )

    # ---------------------------------------------------------------- 3. tokenizers
    print("[2/10] Tokenization (nltk, spacy, custom, hybrid)...", flush=True)
    tokenizer_runs = tokenizers.run_tokenizers(sample, config, logger)
    tokenization_rows = [run_.metrics() for run_ in tokenizer_runs]
    write_csv(
        tokenization_dir / "tokenization_comparison.csv",
        tokenization_rows,
        list(tokenization_rows[0].keys()),
    )
    metrics["tokenization_rows"] = tokenization_rows

    example_size = int(config.get("tokenization.example_sample_size", 12))
    for run_ in tokenizer_runs:
        rows = tokenizers.example_rows(run_, sample, example_size)
        write_csv(
            tokenization_dir / f"{run_.name}_examples.csv",
            rows,
            ["document_id", "source_id", "page_number", "unit_id", "section_id", "section_number",
             "section_title", "unit_type", "tokenizer", "original_text", "tokens", "token_count"],
        )

    write_csv(
        tokenization_dir / "custom_tokenizer_rules.csv",
        custom_tokenizer.rule_rows(),
        ["rule_id", "priority", "pattern", "description", "example", "expected_behavior", "actual_behavior"],
    )

    tokens_by_name = {run_.name: run_.tokens_per_unit for run_ in tokenizer_runs}
    comparison_examples = reporting.tokenization_examples_comparison(
        representative, tokens_by_name, {u.unit_id: u for u in sample.units}, config
    )
    write_csv(
        tokenization_dir / "tokenization_examples_comparison.csv",
        comparison_examples,
        ["document_id", "source_id", "page_number", "unit_id", "section_id", "section_number",
         "section_title", "example_category", "original_text", "nltk_tokens", "spacy_tokens",
         "custom_tokens", "hybrid_tokens", "nltk_token_count", "spacy_token_count",
         "custom_token_count", "hybrid_token_count"],
    )

    # ---------------------------------------------------------------- 4. date/number
    print("[3/10] Date and number aware tokenization...", flush=True)
    date_number_rows = tokenizers.date_number_comparison(sample, logger)
    write_csv(
        tokenization_dir / "date_number_comparison.csv",
        date_number_rows,
        ["category", "class", "expressions_detected", "expressions_intact_after_standard_tokenization",
         "intact_rate_percent", "typed_component_tokens", "tokens_equal_to_whole_expression"],
    )
    write_csv(
        tokenization_dir / "date_number_patterns.csv",
        date_number_tokenizer.pattern_rows(),
        ["category", "pattern", "description"],
    )

    # ---------------------------------------------------------------- 5. spaCy pass
    print("[4/10] spaCy analysis pass (tokens, tags, POS, lemmas, entities)...", flush=True)
    annotations = annotate(sample.units, config, logger)
    spacy_tokens = [a.tokens for a in annotations.annotations]
    write_csv(
        tokenization_dir / "spacy_examples.csv",
        tokenizers.example_rows(
            tokenizers.TokenizerRun(
                name="spacy",
                tokens_per_unit=spacy_tokens,
                sentences=sum(a.sentence_count for a in annotations.annotations),
                documents_processed=len({u.document_id for u in sample.units}),
                seconds=annotations.seconds,
                description="spaCy en_core_web_sm",
            ),
            sample,
            example_size,
        ),
        ["document_id", "source_id", "page_number", "unit_id", "section_id", "section_number",
         "section_title", "unit_type", "tokenizer", "original_text", "tokens", "token_count"],
    )

    # ---------------------------------------------------------------- 6. stopwords
    print("[5/10] Stopword strategies...", flush=True)
    hybrid_name = "hybrid"
    hybrid_view = build_tokenized(hybrid_name, sample.units, tokens_by_name[hybrid_name])
    stopword_result = stopwords.run_stopword_experiments(hybrid_view, config, logger)
    write_csv(
        preprocessing_dir / "stopword_comparison.csv",
        stopword_result["comparison"],
        list(stopword_result["comparison"][0].keys()),
    )
    write_csv(
        preprocessing_dir / "domain_stopword_analysis.csv",
        stopword_result["domain_analysis"],
        ["term", "in_standard_stopword_list", "in_domain_aware_stopword_list", "corpus_occurrences",
         "occurrences_removed_by_standard", "occurrences_removed_by_domain_aware", "documents_affected",
         "outcome", "rationale"],
    )
    (preprocessing_dir / "final_stopwords.txt").write_text(
        "\n".join(stopword_result["final_stopwords"]) + "\n", encoding="utf-8"
    )
    write_csv(
        preprocessing_dir / "stopwords_protected_terms.csv",
        stopword_result["domain_analysis"],
        ["term", "in_standard_stopword_list", "in_domain_aware_stopword_list", "corpus_occurrences",
         "occurrences_removed_by_standard", "occurrences_removed_by_domain_aware", "outcome"],
    )
    metrics["stopword_rows"] = stopword_result["comparison"]
    stopword_view = stopword_result["views"]["domain_aware"]
    standard_stopword_view = stopword_result["views"]["standard_english"]

    # ---------------------------------------------------------------- 7. stemming
    print("[6/10] Stemming (Porter, Snowball, Lancaster) and ordering...", flush=True)
    stemming_result = stemming.run_stemming(hybrid_view, config, logger)
    write_csv(
        stemming_dir / "stemming_comparison.csv",
        stemming_result["comparison"],
        list(stemming_result["comparison"][0].keys()),
    )
    write_csv(
        stemming_dir / "stemming_examples.csv",
        stemming_result["examples"],
        ["original_word", "porter", "snowball", "lancaster", "algorithms_agreeing", "distinct_stems", "notes"],
    )
    write_csv(
        stemming_dir / "stemming_families.csv",
        stemming_result["families"],
        ["original_word", "stemmer", "stem", "changed", "collision_group", "collision_size",
         "residual_suffixes", "observation"],
    )
    order_rows = stemming.run_ordering_experiment(
        hybrid_view, stopword_result["strategies"]["domain_aware"].words, "porter", logger
    )
    write_csv(
        stemming_dir / "stopword_stemming_order_comparison.csv",
        order_rows,
        list(order_rows[0].keys()),
    )
    metrics["stemming_rows"] = stemming_result["comparison"]

    # ---------------------------------------------------------------- 8. lemmatization
    print("[7/10] Lemmatization (WordNet lookup, spaCy, lemminflect)...", flush=True)
    spacy_view = build_tokenized("spacy", sample.units, spacy_tokens)
    lemmatization_result = run_lemmatization(spacy_view, annotations, config, logger)
    write_csv(
        lemmatization_dir / "lemmatization_comparison.csv",
        lemmatization_result["comparison"],
        list(lemmatization_result["comparison"][0].keys()),
    )
    write_csv(
        lemmatization_dir / "lemmatization_examples.csv",
        lemmatization_result["examples"],
        ["original_word", "observed_pos_in_corpus", "wordnet_lookup_pos_agnostic",
         "wordnet_lookup_pos_aware", "spacy_rule_based", "lemminflect_rulebased",
         "from_probe_list", "notes"],
    )
    metrics["lemmatization_rows"] = lemmatization_result["comparison"]

    # ---------------------------------------------------------------- 9. POS
    print("[8/10] POS tagging (default, custom rules, custom ML)...", flush=True)
    default_pos = pos_tagging.run_default_pos(annotations, config, logger)
    write_csv(
        pos_dir / "default_pos_distribution.csv",
        default_pos["distribution"],
        ["tag", "coarse_category", "tag_description", "count", "percent_of_tokens",
         "documents_containing_tag"],
    )
    write_csv(
        pos_dir / "default_pos_coarse.csv",
        default_pos["coarse"],
        ["coarse_category", "count", "percent_of_tokens"],
    )
    write_csv(
        pos_dir / "default_pos_examples.csv",
        default_pos["examples"],
        ["document_id", "source_id", "page_number", "unit_id", "section_id", "section_number",
         "section_title", "unit_type", "text", "tagged_tokens", "pos_sequence", "token_count"],
    )

    candidates = custom_pos.discover_candidates(annotations, config, logger)
    dictionary = custom_pos.build_dictionary(candidates, config, logger)
    write_csv(
        pos_dir / "custom_pos_candidates.csv",
        custom_pos.candidate_rows(candidates[:400]),
        ["term", "occurrences", "documents", "dominant_tag", "dominant_share", "distinct_tags",
         "observed_tag_distribution", "rule", "expected_tag", "correction_applies", "example_document_id",
         "example_page", "example_unit_id"],
    )
    write_csv(
        pos_dir / "custom_pos_dictionary.csv",
        custom_pos.dictionary_rows(dictionary),
        ["term", "default_tag", "default_tag_share", "distinct_default_tags",
         "observed_tag_distribution", "custom_tag", "rule", "reason", "occurrences_in_corpus",
         "documents_affected", "example", "example_document_id", "example_page", "example_unit_id"],
    )
    retag_result = custom_pos.retag_corpus(annotations, dictionary)
    write_csv(
        pos_dir / "custom_pos_changes.csv",
        retag_result["changed_rows"],
        ["document_id", "page_number", "unit_id", "section_id", "section_title", "token",
         "token_index", "default_tag", "custom_tag", "sentence"],
    )

    gold_file = (config.project_root / str(config.get("pos.ml.annotation.gold_file"))).resolve()
    gold_rows = ml_pos.load_gold(gold_file, logger)
    gold_aligned, gold_problems = ml_pos.align_gold_with_spacy(gold_rows, annotations, logger)
    ml_failure = ""
    try:
        ml_result = ml_pos.run_ml_pos(gold_aligned, annotations, config, logger)
        ml_comparison_row: Dict[str, Any] = dict(ml_result.comparison_row)
        ml_classification_rows: List[Dict[str, Any]] = ml_result.classification_rows
        ml_misclassifications: List[Dict[str, Any]] = ml_result.misclassifications
    except Exception as exc:  # never abort the whole run for one experiment
        ml_failure = str(exc)
        ml_comparison_row = {
            "method": "custom_ml_pos",
            "status": "not_available",
            "reason": ml_failure,
            "note": "no accuracy is reported because the ML experiment could not run",
        }
        ml_classification_rows = []
        ml_misclassifications = []
        log_event(logger, "ERROR", "ml_pos", f"ML POS experiment failed: {exc}")

    write_csv(pos_dir / "ml_pos_results.csv", [ml_comparison_row], list(ml_comparison_row.keys()))
    write_csv(
        pos_dir / "ml_pos_classification_report.csv",
        ml_classification_rows,
        ["method", "label", "precision", "recall", "f1", "support",
         "default_spacy_precision", "default_spacy_recall", "default_spacy_f1"],
    )
    write_csv(
        pos_dir / "ml_pos_misclassifications.csv",
        ml_misclassifications,
        ["method", "token", "gold_pos", "predicted_pos", "default_spacy_pos", "error_type"],
    )
    alignment_rows = [
        {
            "sentence_id": row.sentence_id,
            "document_id": row.document_id,
            "page_number": row.page_number,
            "unit_id": row.unit_id,
            "sentence_token_index": row.token_index,
            "unit_token_index": row.corpus_token_index,
            "token": row.token,
            "gold_upos": row.gold_pos,
            "spacy_tag": row.gold_tag,
            "status": "aligned",
        }
        for row in gold_aligned
    ]
    alignment_rows.extend(
        {
            "sentence_id": problem.get("sentence_id", ""),
            "document_id": "",
            "page_number": "",
            "unit_id": problem.get("unit_id", ""),
            "sentence_token_index": "",
            "unit_token_index": "",
            "token": "",
            "gold_upos": "",
            "spacy_tag": "",
            "status": f"problem: {problem.get('issue', '')}",
        }
        for problem in gold_problems
    )
    write_csv(
        pos_dir / "gold_annotation_alignment.csv",
        alignment_rows,
        ["sentence_id", "document_id", "page_number", "unit_id", "sentence_token_index",
         "unit_token_index", "token", "gold_upos", "spacy_tag", "status"],
    )

    # score all three methods on the identical gold set
    default_upos: Dict[Tuple[str, int], str] = {}
    rule_upos: Dict[Tuple[str, int], str] = {}
    for index, (annotation, unit) in enumerate(zip(annotations.annotations, annotations.units)):
        for position, (token, tag, pos) in enumerate(zip(annotation.tokens, annotation.tags, annotation.pos)):
            default_upos[(unit.unit_id, position)] = pos
        custom_tags = retag_result["custom_tags"][index]
        for position, (token, tag) in enumerate(zip(annotation.tokens, custom_tags)):
            rule_upos[(unit.unit_id, position)] = pos_tagging._upos_of(tag)

    default_summary, default_errors = ml_pos.score_method_on_gold("default_pos", gold_aligned, default_upos)
    rule_summary, rule_errors = ml_pos.score_method_on_gold("custom_rule_pos", gold_aligned, rule_upos)
    ml_summary = dict(ml_comparison_row)
    ml_summary["gold_sentences"] = ml_summary.get("gold_sentences_test", 0)

    pos_comparison = comparisons.pos_comparison_rows(
        default_summary, rule_summary, ml_summary,
        {
            "changed_tokens": retag_result["changed_tokens"],
            "total_tokens": retag_result["total_tokens"],
            "changed_percent": retag_result["changed_percent"],
        },
    )
    write_csv(comparisons_dir / "pos_comparison.csv", pos_comparison, list(pos_comparison[0].keys()))
    write_csv(
        pos_dir / "pos_gold_errors.csv",
        default_errors + rule_errors,
        ["method", "sentence_id", "document_id", "page_number", "unit_id", "token", "token_index",
         "gold_pos", "predicted_pos", "error_type"],
    )
    metrics["pos_comparison"] = pos_comparison

    # ---------------------------------------------------------------- 10. NER
    print("[9/10] NER (general model, domain dictionary, error analysis)...", flush=True)
    general_ner = ner.run_general_ner(annotations, config, logger)
    write_csv(
        ner_dir / "ner_entities.csv",
        general_ner["entities"],
        ["entity_id", "document_id", "source_id", "page_number", "unit_id", "section_id",
         "section_number", "section_title", "unit_type", "entity_text", "entity_label",
         "token_start", "token_end", "context"],
    )
    write_csv(
        ner_dir / "ner_label_distribution.csv",
        general_ner["label_distribution"],
        ["entity_label", "count", "percent_of_entities"],
    )
    domain_ner = ner.run_domain_ner(annotations, config, logger)
    write_csv(
        ner_dir / "custom_ner_results.csv",
        domain_ner["entities"],
        ["entity_id", "document_id", "source_id", "page_number", "unit_id", "section_id",
         "section_number", "section_title", "entity_text", "domain_category",
         "also_tagged_by_general_ner", "context"],
    )
    dictionary_rows = ner.domain_dictionary_rows(
        _domain_corpus_counts(domain_ner), _domain_document_counts(domain_ner)
    )
    write_csv(
        ner_dir / "domain_entity_dictionary.csv",
        dictionary_rows,
        ["term", "domain_category", "token_length", "corpus_mentions", "documents_mentioned",
         "general_ner_equivalent_labels", "justification"],
    )
    error_rows = ner.build_error_analysis(annotations, general_ner, domain_ner, config, logger)
    write_csv(
        ner_dir / "ner_error_analysis.csv",
        error_rows,
        ["text", "default_label", "expected_or_interpreted_label", "error_type", "context",
         "document_id", "page_number", "unit_id", "notes"],
    )
    metrics["ner_totals"] = {
        "total_entities": general_ner["total_entities"],
        "distinct_entity_types": len(general_ner["label_distribution"]),
        "top_labels": ", ".join(
            f"{r['entity_label']}:{r['count']}" for r in general_ner["label_distribution"][:6]
        ),
        "domain_mentions": domain_ner["total_mentions"],
        "domain_terms": len([t for t in ner.DOMAIN_ENTITY_TERMS.values() for _ in t]),
        "error_rows": len(error_rows),
    }

    # ---------------------------------------------------------------- 11. n-grams
    print("[10/10] N-grams (1-5) and BPE...", flush=True)
    sizes = list(config.get("ngrams.sizes", [1, 2, 3, 4, 5]))
    ngram_summary: List[Dict[str, object]] = []
    ngram_top: Dict[int, List[Dict[str, object]]] = {}
    ngram_filenames = {
        1: "unigram_results.csv", 2: "bigram_results.csv", 3: "trigram_results.csv",
        4: "fourgram_results.csv", 5: "fivegram_results.csv",
    }
    for n in sizes:
        rows, summary_row = ngrams.ngram_rows(hybrid_view, n, config)
        write_csv(
            ngrams_dir / ngram_filenames[n],
            rows,
            ["ngram", "n", "frequency", "document_frequency", "example_document", "example_page",
             "example_unit_id", "contains_domain_term", "is_function_word_phrase",
             "has_content_token", "is_repeated_token"],
        )
        ngram_summary.append(summary_row)
        ngram_top[n] = rows
        log_event(logger, "INFO", "ngrams",
                  f"{n}-gram: {summary_row['total_ngrams']} total, {summary_row['unique_ngrams']} unique")
    domain_phrases = ngrams.mine_domain_phrases(hybrid_view, config, logger)
    write_csv(
        ngrams_dir / "domain_phrases.csv",
        domain_phrases,
        ["ngram", "n", "frequency", "document_frequency", "domain_terms_in_phrase",
         "domain_relevance_reason", "function_word_ratio", "example_context", "example_document",
         "example_page", "example_unit_id"],
    )
    ngram_comparison_rows = []
    for summary_row in ngram_summary:
        n = int(summary_row["n"])
        ngram_comparison_rows.append(
            {
                "n": n,
                "total_ngrams": summary_row["total_ngrams"],
                "unique_ngrams": summary_row["unique_ngrams"],
                "top_ngram": summary_row["top_ngram"],
                "top_ngram_frequency": summary_row["top_ngram_frequency"],
                "singletons": summary_row["singletons"],
                "meaningful_domain_phrases": sum(1 for p in domain_phrases if int(p["n"]) == n),
            }
        )
    for row in ngrams.preprocessing_effect_rows(hybrid_view, stopword_view, config):
        ngram_comparison_rows.append({"n": f"stopword_effect_n{row['n']}", **row})
    write_csv(
        comparisons_dir / "ngram_comparison.csv",
        ngram_comparison_rows,
        _union_fieldnames(ngram_comparison_rows),
    )
    metrics["ngram_summary"] = [r for r in ngram_summary]

    # ---------------------------------------------------------------- 12. BPE
    bpe_artifacts = bpe_module.train_bpe(sample, config, logger)
    write_csv(bpe_dir / "bpe_vocabulary.csv", bpe_module.vocabulary_rows(bpe_artifacts),
              ["token_id", "token", "is_special"])
    write_csv(bpe_dir / "bpe_examples.csv", bpe_artifacts.example_rows,
              ["example", "category", "bpe_tokens", "bpe_token_count", "is_single_token",
               "pieces_rejoined", "byte_level_marked"])
    write_csv(bpe_dir / "bpe_statistics.csv", bpe_artifacts.stats_rows, ["metric", "value", "detail"])
    metrics["bpe_vocab_size"] = bpe_artifacts.vocab_size
    metrics["bpe_corpus_tokens"] = bpe_artifacts.corpus_token_count

    # ---------------------------------------------------------------- 13. comparisons
    print("Generating comparisons and summaries...", flush=True)
    corpus_text = "\n".join(unit.text for unit in sample.units)
    testset_rows = reporting.build_domain_testset(corpus_text, {}, config)
    testset_rows = _testset_provenance(testset_rows, sample.units)
    write_csv(
        comparisons_dir / "domain_tokenization_testset.csv",
        testset_rows,
        list(testset_rows[0].keys()),
    )
    bpe_rows = reporting.bpe_comparison_rows(testset_rows, bpe_artifacts.tokenizer, config)
    write_csv(
        comparisons_dir / "bpe_tokenization_comparison.csv",
        bpe_rows,
        list(bpe_rows[0].keys()),
    )

    method_rows = comparisons.tokenization_method_comparison(
        tokenization_rows,
        {str(r["tokenizer"]): int(r["total_tokens"]) for r in tokenization_rows},  # type: ignore[arg-type]
        {str(r["tokenizer"]): int(r["vocabulary_size"]) for r in tokenization_rows},  # type: ignore[arg-type]
        {
            str(r["example_category"]): str(r["nltk_tokens"])[:120] + " | custom: " + str(r["custom_tokens"])[:120]
            for r in comparison_examples
        },
    )
    method_rows.append(
        {
            "method": "bpe",
            "description": "byte-level BPE trained on this corpus",
            "token_count": bpe_artifacts.corpus_token_count,
            "unique_tokens": bpe_artifacts.vocab_size,
            "vocabulary_size": bpe_artifacts.vocab_size,
            "avg_tokens_per_sentence": 0,
            "numeric_tokens": 0, "date_tokens": 0, "percentage_tokens": 0, "currency_tokens": 0,
            "strengths": comparisons.TOKENIZATION_STRENGTHS["bpe"][0],
            "limitations": comparisons.TOKENIZATION_STRENGTHS["bpe"][1],
            "financial_domain_behavior": comparisons.TOKENIZATION_STRENGTHS["bpe"][2],
            "example_cases": "; ".join(
                f"{r['example']} -> {r['bpe_tokens']}" for r in bpe_rows[:5]
            ),
        }
    )
    write_csv(
        comparisons_dir / "tokenization_method_comparison.csv",
        method_rows,
        list(method_rows[0].keys()),
    )

    # preprocessing pipeline stages
    # preprocessing pipeline stages (one consistent token basis: the hybrid
    # domain-aware tokenizer, so every stage is comparable)
    porter_stopword_view = stemming.stem_view(stopword_view, "porter")
    wordnet_lemma_view = lemmatization_module.lemmatize_view(stopword_view, logger=logger)
    stages = [
        ("1_tokenized_baseline", "hybrid domain-aware tokenizer (no filtering)", hybrid_view),
        ("2_stopword_removed", "domain-aware stopword strategy", stopword_view),
        ("3_stemmed", "Porter stemmer applied after stopword removal", porter_stopword_view),
        ("4_lemmatized", "WordNet lookup lemmatizer applied after stopword removal", wordnet_lemma_view),
        ("5_standard_stopwords", "standard English stopword list (for comparison)", standard_stopword_view),
    ]
    preprocessing_rows = comparisons.preprocessing_comparison_rows(stages)
    write_csv(
        preprocessing_dir / "preprocessing_comparison.csv",
        preprocessing_rows,
        list(preprocessing_rows[0].keys()),
    )
    before_after = comparisons.before_after_rows(sample.units[:400], {
        "original": hybrid_view,
        "tokenized_hybrid": hybrid_view,
        "stopword_removed": stopword_view,
        "stemmed_after_stopwords": porter_stopword_view,
        "lemmatized_after_stopwords": wordnet_lemma_view,
    })
    write_csv(
        preprocessing_dir / "before_after_examples.csv",
        before_after,
        list(before_after[0].keys()),
    )

    ab_result = comparisons.run_pipeline_ab_comparison(sample, config, logger)
    write_csv(
        comparisons_dir / "pipeline_comparison.csv",
        ab_result["rows"],
        list(ab_result["rows"][0].keys()),
    )
    (comparisons_dir / "pipeline_comparison.md").write_text(
        reporting.pipeline_comparison_markdown(ab_result),
        encoding=config.encoding,
    )
    metrics["pipeline_ab_comparison"] = ab_result["rows"]

    ner_rows_for_master = [
        {
            "total_entities": general_ner["total_entities"],
            "distinct_entity_types": len(general_ner["label_distribution"]),
        },
        {
            "total_mentions": domain_ner["total_mentions"],
            "distinct_terms": len({r["term"] for r in dictionary_rows}),
        },
    ]
    master_rows = comparisons.method_comparison_rows(
        tokenization_rows, stopword_result["comparison"], stemming_result["comparison"],
        lemmatization_result["comparison"], pos_comparison, ner_rows_for_master,
        ngram_summary, bpe_rows,
        {str(r["expression"]): str(r["custom_tokens"]) for r in testset_rows},
    )
    write_csv(
        comparisons_dir / "phase2_master_comparison.csv",
        master_rows,
        list(master_rows[0].keys()),
    )

    summary_rows = comparisons.summary_rows(
        tokenization_rows, stopword_result["comparison"], stemming_result["comparison"],
        lemmatization_result["comparison"], preprocessing_rows, ngram_summary,
        bpe_artifacts.stats_rows, pos_comparison, metrics["ner_totals"], date_number_rows,
    )
    write_csv(results_dir / "phase2_summary.csv", summary_rows, list(summary_rows[0].keys()))

    # ---------------------------------------------------------------- 14. figures
    print("Generating charts...", flush=True)
    tokenization_vs_bpe = [
        {"method": str(r["tokenizer"]), "total_tokens": int(r["total_tokens"]), "vocabulary_size": int(r["vocabulary_size"])}
        for r in tokenization_rows
    ] + [
        {"method": "bpe", "total_tokens": bpe_artifacts.corpus_token_count, "vocabulary_size": bpe_artifacts.vocab_size}
    ]
    figure_files = figures.build_all(
        config,
        logger,
        tokenization_rows=tokenization_rows,
        stopword_rows=stopword_result["comparison"],
        stemming_rows=stemming_result["comparison"],
        lemmatization_rows=lemmatization_result["comparison"],
        pos_rows=default_pos["coarse"],
        ngram_top=ngram_top,
        ngram_summary=ngram_summary,
        bpe_stats=bpe_artifacts.stats_rows,
        tokenization_vs_bpe=tokenization_vs_bpe,
    )

    # ---------------------------------------------------------------- 15. documents
    # The narrative artefacts are written *before* validation so that the
    # "required outputs exist" rule checks the complete deliverable set.
    print("Writing methodology and README...", flush=True)
    summary_corpus["gold_tokens"] = len(gold_aligned)
    summary_corpus["gold_sentences"] = len({r.sentence_id for r in gold_aligned})
    summary_corpus["bpe_vocabulary_size"] = bpe_artifacts.vocab_size
    summary_corpus["bpe_corpus_tokens"] = bpe_artifacts.corpus_token_count
    summary_corpus["figures"] = figure_files
    summary_corpus["runtime_seconds"] = round(time.time() - started, 2)
    summary_corpus["unit_type_distribution"] = corpus.summary()["units_by_type"]
    summary_corpus["spacy_max_chars_per_chunk"] = annotations.max_chars_per_chunk

    readme_results: Dict[str, Any] = {
        "tokenization": tokenization_rows,
        "stopwords": stopword_result["comparison"],
        "stemming": stemming_result["comparison"],
        "lemmatization": lemmatization_result["comparison"],
        "pos": pos_comparison,
        "ner": {
            "total_entities": general_ner["total_entities"],
            "label_count": len(general_ner["label_distribution"]),
            "domain_mentions": domain_ner["total_mentions"],
            "domain_category_count": len(domain_ner["category_counts"]),
            "error_rows": len(error_rows),
        },
        "ngrams": ngram_summary,
        "bpe": {
            "vocabulary_size": bpe_artifacts.vocab_size,
            "corpus_token_count": bpe_artifacts.corpus_token_count,
            "bytes_per_token": next(
                (r["value"] for r in bpe_artifacts.stats_rows if r["metric"] == "bytes_per_token"), 0
            ),
        },
        "date_number": [r for r in date_number_rows if str(r["category"]) == "TOTAL"],
    }
    outputs = sorted(
        str(path.relative_to(results_dir)).replace("\\", "/")
        for path in results_dir.rglob("*")
        if path.is_file()
    )
    config.out_path("readme").write_text(
        reporting.readme_markdown(config, summary_corpus, outputs, figure_files, readme_results),
        encoding="utf-8",
    )

    config.out_path("methodology").write_text(
        reporting.methodology_markdown(
            config,
            summary_corpus,
            {
                "tokenization": tokenization_rows,
                "stopwords": stopword_result["comparison"],
                "stemming": stemming_result["comparison"],
                "lemmatization": lemmatization_result["comparison"],
                "pos": {"comparison": pos_comparison},
                "ner": {
                    "total_entities": general_ner["total_entities"],
                    "domain_mentions": domain_ner["total_mentions"],
                    "error_rows": len(error_rows),
                },
                "ngrams": ngram_summary,
                "bpe": {
                    "vocabulary_size": bpe_artifacts.vocab_size,
                    "merge_operations": next(
                        (r["value"] for r in bpe_artifacts.stats_rows if r["metric"] == "merge_operations"), 0
                    ),
                    "corpus_token_count": bpe_artifacts.corpus_token_count,
                    "bytes_per_token": next(
                        (r["value"] for r in bpe_artifacts.stats_rows if r["metric"] == "bytes_per_token"), 0
                    ),
                },
                "gold": {
                    "gold_tokens": len(gold_aligned),
                    "gold_sentences": len({r.sentence_id for r in gold_aligned}),
                },
            },
        ),
        encoding="utf-8",
    )
    # ---------------------------------------------------------------- 16. validation
    print("Validating outputs...", flush=True)
    metrics["pos_rows"] = default_pos["distribution"]
    def _summary_payload(results: List[Any]) -> Dict[str, Any]:
        return {
            "corpus": summary_corpus,
            "outputs": list(outputs),
            "tokenization": tokenization_rows,
            "stopwords": stopword_result["comparison"],
            "stemming": stemming_result["comparison"],
            "lemmatization": lemmatization_result["comparison"],
            "preprocessing": preprocessing_rows,
            "pos": {
                "distribution_top": default_pos["distribution"][:25],
                "comparison": pos_comparison,
                "gold_set": {
                    "file": str(gold_file.relative_to(config.project_root)).replace("\\", "/"),
                    "tokens": len(gold_aligned),
                    "sentences": len({r.sentence_id for r in gold_aligned}),
                    "annotation": "manual, single annotator, UPOS-17",
                },
            },
            "ner": {
                "total_entities": general_ner["total_entities"],
                "labels": general_ner["label_distribution"][:25],
                "domain_mentions": domain_ner["total_mentions"],
                "domain_categories": dict(domain_ner["category_counts"].most_common()),
                "error_rows": len(error_rows),
            },
            "ngrams": ngram_summary,
            "bpe": {
                "vocabulary_size": bpe_artifacts.vocab_size,
                "corpus_token_count": bpe_artifacts.corpus_token_count,
                "statistics": bpe_artifacts.stats_rows[:12],
            },
            "date_number": [r for r in date_number_rows if str(r["category"]) == "TOTAL"],
            "validation": [result.as_row() for result in results],
        }

    # first write so that validation (and any reader) sees a summary file;
    # rewritten below with the validation results included
    write_json(results_dir / "phase2_summary.json", _summary_payload([]))

    validation_results = validation.run_validation(config, corpus, metrics, logger)
    write_csv(
        results_dir / "phase2_validation_report.csv",
        [result.as_row() for result in validation_results],
        ["rule_id", "rule", "status", "detail", "evidence"],
    )
    summary_corpus["validation_passed"] = sum(1 for r in validation_results if r.status == "PASS")
    summary_corpus["validation_total"] = len(validation_results)

    write_json(results_dir / "phase2_summary.json", _summary_payload(validation_results))

    # ---------------------------------------------------------------- final report
    # the output list is recomputed after every artefact exists, so the README and
    # the summary JSON list the same complete set of files
    outputs = sorted(
        str(path.relative_to(results_dir)).replace("\\", "/")
        for path in results_dir.rglob("*")
        if path.is_file()
    )
    config.out_path("readme").write_text(
        reporting.readme_markdown(config, summary_corpus, outputs, figure_files, readme_results),
        encoding="utf-8",
    )
    write_json(results_dir / "phase2_summary.json", _summary_payload(validation_results))
    log_event(
        logger,
        "INFO",
        "run",
        f"{len(outputs)} output files listed in README.md and phase2_summary.json",
    )

    # ---------------------------------------------------------------- final report
    _print_summary(
        corpus=corpus,
        tokenization_rows=tokenization_rows,
        stopword_rows=stopword_result["comparison"],
        stemming_rows=stemming_result["comparison"],
        lemmatization_rows=lemmatization_result["comparison"],
        pos_comparison=pos_comparison,
        ner_totals=metrics["ner_totals"],
        ngram_summary=ngram_summary,
        bpe_artifacts=bpe_artifacts,
        date_number_rows=date_number_rows,
        validation_results=validation_results,
        results_dir=results_dir,
        runtime=time.time() - started,
        outputs=outputs,
    )
    return {
        "validation": [result.as_row() for result in validation_results],
        "outputs": outputs,
        "corpus": summary_corpus,
    }


def _testset_provenance(rows: Sequence[Dict[str, Any]], units: Sequence[Any]) -> List[Dict[str, Any]]:
    """Attach corpus provenance to every domain tokenization test expression.

    Each expression is traced to the first corpus unit that contains it, so the
    test set keeps the Phase 1 traceability chain
    ``source_id -> document_id -> page_number -> section_id -> unit_id``.
    """
    needles: Dict[str, str] = {}
    for row in rows:
        expression = str(row.get("expression", "")).strip()
        if expression:
            needles.setdefault(expression.lower(), expression)

    found: Dict[str, Dict[str, Any]] = {}
    counts: Dict[str, int] = {needle: 0 for needle in needles}
    if needles:
        texts = [(str(u.text).lower(), u) for u in units]
        for lowered, unit in texts:
            for needle in needles:
                if needle in lowered:
                    counts[needle] += 1
                    if needle not in found:
                        found[needle] = {
                            "source_id": unit.source_id,
                            "document_id": unit.document_id,
                            "page_number": unit.page_number,
                            "section_id": unit.section_id,
                            "section_number": unit.section_number,
                            "section_title": unit.section_title,
                            "unit_id": unit.unit_id,
                            "corpus_units_containing": 0,
                        }
    for needle, meta in found.items():
        meta["corpus_units_containing"] = counts[needle]

    enriched: List[Dict[str, Any]] = []
    for row in rows:
        expression = str(row.get("expression", "")).strip()
        meta = found.get(expression.lower(), {})
        enriched.append(
            {
                "expression": row.get("expression", ""),
                "category": row.get("category", ""),
                **{
                    "source_id": meta.get("source_id", ""),
                    "document_id": meta.get("document_id", ""),
                    "page_number": meta.get("page_number", ""),
                    "section_id": meta.get("section_id", ""),
                    "section_number": meta.get("section_number", ""),
                    "section_title": meta.get("section_title", ""),
                    "unit_id": meta.get("unit_id", ""),
                    "corpus_units_containing": meta.get("corpus_units_containing", 0),
                },
                **{k: v for k, v in row.items() if k not in {"expression", "category"}},
            }
        )
    return enriched


def _union_fieldnames(rows: Sequence[Dict[str, Any]]) -> List[str]:
    """Column order for tables whose rows come from more than one shape."""
    names: List[str] = []
    for row in rows:
        for key in row:
            if key not in names:
                names.append(key)
    return names


def _domain_corpus_counts(domain_ner: Dict[str, Any]) -> Counter:
    """Per-term corpus mention counts from the domain NER pass."""
    from collections import Counter

    counts: Counter = Counter()
    for row in domain_ner["entities"]:
        counts[str(row["entity_text"])] += 1
    return counts


def _domain_document_counts(domain_ner: Dict[str, Any]) -> Counter:
    from collections import Counter

    counts: Counter = Counter()
    for row in domain_ner["entities"]:
        counts[f"__{row['domain_category']}__{row['entity_text']}"] += 1
    return counts


def _print_summary(**kwargs) -> None:
    results_dir = kwargs["results_dir"]
    print("\n" + BANNER)
    print("PHASE 2 COMPLETE")
    print(BANNER)
    print(f"\nRuntime: {kwargs['runtime']:.1f}s")
    print(f"Outputs written: {len(kwargs['outputs'])} files under {results_dir}")
    print("\nTokenization:")
    for row in kwargs["tokenization_rows"]:
        print(
            f"  {row['tokenizer']:8} tokens={row['total_tokens']:>8}  vocab={row['vocabulary_size']:>7}  "
            f"date={row['date_tokens']:>6}  pct={row['percentage_tokens']:>5}  cur={row['currency_tokens']:>5}"
        )
    print("\nStopwords:")
    for row in kwargs["stopword_rows"]:
        print(f"  {row['strategy']:18} tokens={row['total_tokens']:>8}  reduction={row['token_reduction_percent']}%")
    print("\nStemming:")
    for row in kwargs["stemming_rows"]:
        print(
            f"  {row['algorithm']:18} stems={row['unique_stems']:>7}  "
            f"reduction={row['vocabulary_reduction_percent']}%  collisions={row['colliding_stems']}"
        )
    print("\nLemmatization:")
    for row in kwargs["lemmatization_rows"]:
        print(
            f"  {row['method']:28} lemmas={row['unique_lemmas']:>7}  reduction={row['vocabulary_reduction_percent']}%"
        )
    print("\nPOS (manually annotated sample only):")
    for row in kwargs["pos_comparison"]:
        print(
            f"  {row['method']:18} n={row['dataset_size']}  accuracy={row['accuracy_if_available']}  "
            f"macro_f1={row['f1_if_available']}"
        )
    print("\nNER:")
    print(f"  general entities: {kwargs['ner_totals']['total_entities']}")
    print(f"  domain mentions:  {kwargs['ner_totals']['domain_mentions']}")
    print("\nTop n-grams (highest-frequency n-gram containing words):")
    for row in kwargs["ngram_summary"]:
        print(f"  {row['n']}-gram: {row['top_content_ngram']} (x{row['top_content_ngram_frequency']})")
    print("\nBPE:")
    print(f"  vocabulary: {kwargs['bpe_artifacts'].vocab_size}")
    print(f"  corpus tokens: {kwargs['bpe_artifacts'].corpus_token_count}")
    print("\nDate/number:")
    for row in kwargs["date_number_rows"]:
        if str(row["category"]) == "TOTAL":
            print(
                f"  {row['expressions_detected']} typed expressions; "
                f"{row['expressions_intact_after_standard_tokenization']} survive standard tokenization "
                f"({row['intact_rate_percent']}%)"
            )
    passed = sum(1 for r in kwargs["validation_results"] if r.status == "PASS")
    print(f"\nValidation: {passed}/{len(kwargs['validation_results'])} rules PASS")
    for result in kwargs["validation_results"]:
        if result.status != "PASS":
            print(f"  {result.status} {result.rule_id}: {result.detail}")
    print("\n" + BANNER)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Phase 2 NLP experiments.")
    parser.add_argument("--config", default=None, help="path to phase2_config.yaml")
    parser.add_argument("--verbose", action="store_true", help="verbose console logging")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    outcome = run(config, verbose=args.verbose)
    failures = [row for row in outcome["validation"] if row["status"] == "FAIL"]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
