"""Report figures. Every chart is drawn from a computed result table.

No chart is produced from hard-coded numbers: each function takes the rows that
were just written to CSV and plots them. If a table is empty the chart is
skipped and the skip is logged, so the figures folder never contains a
meaningless plot.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .config import Phase2Config, log_event


def _style(config: Phase2Config) -> None:
    try:
        plt.style.use(str(config.get("reproducibility.charts.style", "seaborn-v0_8-whitegrid")))
    except Exception:  # pragma: no cover - style name depends on matplotlib version
        plt.style.use("default")
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["savefig.dpi"] = int(config.get("reproducibility.charts.dpi", 140))


def _save(fig, path: Path, config: Phase2Config, logger: logging.Logger, name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = str(config.get("reproducibility.charts.format", "png"))
    fig.savefig(path.with_suffix(f".{fmt}"), bbox_inches="tight")
    plt.close(fig)
    log_event(logger, "INFO", "figures", f"chart written: {path.name}")


def _bar(
    labels: Sequence[str],
    values: Sequence[float],
    title: str,
    ylabel: str,
    path: Path,
    config: Phase2Config,
    logger: logging.Logger,
    name: str,
    rotate: int = 20,
    value_fmt: str = "{:,.0f}",
) -> None:
    if not labels or not values:
        log_event(logger, "WARNING", "figures", f"skipped '{name}': no data")
        return
    _style(config)
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.4), 4.6))
    bars = ax.bar(list(labels), list(values), color="#33628c")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=rotate, labelsize=8)
    ax.margins(x=0.02)
    top = max(values) if values else 0
    if top:
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + top * 0.015,
                value_fmt.format(value),
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90 if top > 1e6 else 0,
            )
    _save(fig, path, config, logger, name)


def build_all(
    config: Phase2Config,
    logger: logging.Logger,
    tokenization_rows: List[Dict[str, object]],
    stopword_rows: List[Dict[str, object]],
    stemming_rows: List[Dict[str, object]],
    lemmatization_rows: List[Dict[str, object]],
    pos_rows: List[Dict[str, object]],
    ngram_top: Dict[int, List[Dict[str, object]]],
    ngram_summary: List[Dict[str, object]],
    bpe_stats: List[Dict[str, object]],
    tokenization_vs_bpe: List[Dict[str, object]],
) -> List[str]:
    """Create every chart; returns the file names written."""
    figures_dir = config.out_dir("figures_dir")
    written: List[str] = []

    # 1. token count by tokenizer
    _bar(
        [str(r["tokenizer"]) for r in tokenization_rows],
        [float(r["total_tokens"]) for r in tokenization_rows],
        "Total tokens by tokenizer",
        "tokens",
        figures_dir / "01_token_count_by_tokenizer",
        config,
        logger,
        "token_count_by_tokenizer",
    )
    written.append("01_token_count_by_tokenizer")

    # 2. vocabulary size by tokenizer
    _bar(
        [str(r["tokenizer"]) for r in tokenization_rows],
        [float(r["vocabulary_size"]) for r in tokenization_rows],
        "Vocabulary size by tokenizer (word-bearing tokens, case-folded)",
        "vocabulary",
        figures_dir / "02_vocabulary_by_tokenizer",
        config,
        logger,
        "vocabulary_by_tokenizer",
    )
    written.append("02_vocabulary_by_tokenizer")

    # 3. stopword removal reduction
    _bar(
        [str(r["strategy"]) for r in stopword_rows],
        [float(r["token_reduction_percent"]) for r in stopword_rows],
        "Token reduction by stopword strategy",
        "% of tokens removed",
        figures_dir / "03_stopword_reduction",
        config,
        logger,
        "stopword_reduction",
        value_fmt="{:.1f}%",
    )
    written.append("03_stopword_reduction")

    # 4. stemming vocabulary reduction
    _bar(
        [str(r["algorithm"]) for r in stemming_rows],
        [float(r["vocabulary_reduction_percent"]) for r in stemming_rows],
        "Vocabulary reduction by stemmer",
        "% reduction",
        figures_dir / "04_stemming_vocabulary_reduction",
        config,
        logger,
        "stemming_vocabulary_reduction",
        value_fmt="{:.1f}%",
    )
    written.append("04_stemming_vocabulary_reduction")

    # 5. lemmatization vocabulary reduction
    _bar(
        [str(r["method"]) for r in lemmatization_rows],
        [float(r["vocabulary_reduction_percent"]) for r in lemmatization_rows],
        "Vocabulary reduction by lemmatizer",
        "% reduction",
        figures_dir / "05_lemmatization_vocabulary_reduction",
        config,
        logger,
        "lemmatization_vocabulary_reduction",
        value_fmt="{:.1f}%",
        rotate=25,
    )
    written.append("05_lemmatization_vocabulary_reduction")

    # 6. POS distribution
    if pos_rows:
        top = pos_rows[:12]
        _style(config)
        fig, ax = plt.subplots(figsize=(9, 4.6))
        labels = [str(r["coarse_category"]) for r in top]
        values = [float(r["count"]) for r in top]
        ax.barh(labels[::-1], values[::-1], color="#2f6f4f")
        ax.set_title("POS distribution (universal categories, top 12)")
        ax.set_xlabel("tokens")
        ax.tick_params(axis="y", labelsize=8)
        _save(fig, figures_dir / "06_pos_distribution", config, logger, "pos_distribution")
        written.append("06_pos_distribution")

    # 7-9. top n-grams (content n-grams only: n-grams made purely of punctuation
    # are kept in the result tables but would make the chart unreadable)
    for n, filename in ((1, "07_top_unigrams"), (2, "08_top_bigrams"), (3, "09_top_trigrams")):
        all_rows = ngram_top.get(n) or []
        rows = [
            r
            for r in all_rows
            if str(r.get("has_content_token", "true")).lower() == "true"
            and str(r.get("is_repeated_token", "false")).lower() != "true"
        ][:15]
        if not rows:
            log_event(logger, "WARNING", "figures", f"skipped {filename}: no content n-grams")
            continue
        _style(config)
        fig, ax = plt.subplots(figsize=(8.5, 5.6))
        labels = [str(r["ngram"]) for r in rows][::-1]
        values = [float(r["frequency"]) for r in rows][::-1]
        ax.barh(labels, values, color="#8c5a2b")
        ax.set_title(f"Top {len(rows)} {n}-grams containing words, by frequency")
        ax.set_xlabel("frequency")
        ax.tick_params(axis="y", labelsize=8)
        _save(fig, figures_dir / filename, config, logger, filename)
        written.append(filename)

    # 10. n-gram vocabulary growth
    if ngram_summary:
        _style(config)
        fig, ax = plt.subplots(figsize=(7.5, 4.4))
        ns = [int(r["n"]) for r in ngram_summary]
        totals = [int(r["total_ngrams"]) for r in ngram_summary]
        uniques = [int(r["unique_ngrams"]) for r in ngram_summary]
        ax.plot(ns, uniques, marker="o", label="unique n-grams", color="#33628c")
        ax.plot(ns, totals, marker="s", label="total n-grams", color="#b5651d")
        ax.set_yscale("log")
        ax.set_xlabel("n")
        ax.set_ylabel("count (log scale)")
        ax.set_title("N-gram vocabulary growth")
        ax.legend()
        ax.set_xticks(ns)
        _save(fig, figures_dir / "10_ngram_vocabulary_growth", config, logger, "ngram_growth")
        written.append("10_ngram_vocabulary_growth")

    # 11. BPE vs word tokenizers
    if tokenization_vs_bpe:
        _style(config)
        fig, ax = plt.subplots(figsize=(9, 4.4))
        labels = [str(r["method"]) for r in tokenization_vs_bpe]
        token_values = [float(r["total_tokens"]) for r in tokenization_vs_bpe]
        vocab_values = [float(r["vocabulary_size"]) for r in tokenization_vs_bpe]
        import numpy as np

        positions = np.arange(len(labels))
        ax.bar(positions - 0.2, token_values, width=0.4, label="total tokens", color="#33628c")
        ax2 = ax.twinx()
        ax2.bar(positions + 0.2, vocab_values, width=0.4, label="vocabulary", color="#7a9a4f")
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=20, fontsize=8)
        ax.set_ylabel("total tokens")
        ax2.set_ylabel("vocabulary")
        ax.set_title("Tokenizers incl. BPE: token count and vocabulary size")
        handles = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
        labels_legend = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
        ax.legend(handles, labels_legend, loc="upper right", fontsize=8)
        _save(fig, figures_dir / "11_bpe_token_vocabulary_comparison", config, logger, "bpe_comparison")
        written.append("11_bpe_token_vocabulary_comparison")

    return written
