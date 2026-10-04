/** Preprocessing page: cleaning, stopwords, and order experiments. */

"use client";

import { useEffect, useState, Suspense } from "react";
import { Layout, PageContainer, SectionCard, DataTable, Loading, ErrorState, Badge, MetricCard } from "@/components";
import { api } from "@/lib/api";

const PREPROCESSING_PIPELINE_DEMO = {
  sample_text:
    "In FY2025, the Reserve Bank of India maintained the repo rate at 6.50% while bank credit to the MSME sector grew by 21.8 per cent with ₹50,000 crore liquidity support.",
  stages: [
    {
      stage: "1. Raw Source Text",
      tokens: [
        "In FY2025, the Reserve Bank of India maintained the repo rate at 6.50% while bank credit to the MSME sector grew by 21.8 per cent with ₹50,000 crore liquidity support.",
      ],
      note: "Raw regulatory / survey paragraph containing fiscal years, currencies, percentages, and institutions.",
    },
    {
      stage: "2. Custom Financial Tokenization",
      tokens: [
        "In", "FY2025", ",", "the", "Reserve", "Bank", "of", "India", "maintained", "the", "repo", "rate", "at", "6.50%", "while", "bank", "credit", "to", "the", "MSME", "sector", "grew", "by", "21.8 per cent", "with", "₹50,000 crore", "liquidity", "support", "."
      ],
      note: "Preserves atomic constructs: FY2025, 6.50%, ₹50,000 crore, 21.8 per cent.",
    },
    {
      stage: "3. Domain-Aware Stop-word Removal",
      tokens: [
        "FY2025", "Reserve", "Bank", "India", "maintained", "repo", "rate", "6.50%", "bank", "credit", "MSME", "sector", "grew", "21.8 per cent", "₹50,000 crore", "liquidity", "support"
      ],
      note: "Standard stopwords (in, the, of, at, while, to, by, with) pruned; 85 domain-critical terms ('bank', 'rate', 'credit') protected.",
    },
    {
      stage: "4. Pipeline A Normalization (LemmInflect/WordNet)",
      tokens: [
        "fy2025", "reserve", "bank", "india", "maintain", "repo", "rate", "6.50%", "bank", "credit", "msme", "sector", "grow", "21.8 per cent", "₹50,000 crore", "liquidity", "support"
      ],
      note: "Lemmatization preserves genuine grammatical root forms ('maintained' -> 'maintain', 'grew' -> 'grow').",
    },
    {
      stage: "5. Pipeline B Normalization (Snowball English Stemming)",
      tokens: [
        "fy2025", "reserv", "bank", "india", "maintain", "repo", "rate", "6.50%", "bank", "credit", "msme", "sector", "grew", "21.8 per cent", "₹50,000 crore", "liquid", "support"
      ],
      note: "Stemming truncates suffixes ('reserve' -> 'reserv', 'liquidity' -> 'liquid'), creating common root stems.",
    },
  ],
};

function PreprocessingContent() {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    api.preprocessing().then(setData).catch(setError).finally(() => setLoading(false));
  }, []);

  if (loading) return <Loading />;
  if (error) return <ErrorState error={error} />;
  if (!data) return null;

  return (
    <PageContainer
      title="Preprocessing & Normalization"
      description="Cleaning stages, domain-aware stop-words, and Pipeline A (Lemmatization) vs. Pipeline B (Stemming)."
    >
      {/* Preprocessing Demonstration */}
      <SectionCard title="End-to-End Preprocessing Demonstration">
        <div className="mb-4 p-4 bg-gray-50 border border-gray-200 rounded-lg">
          <div className="text-xs font-bold text-gray-500 uppercase tracking-wide mb-1">
            Standard Processing Flow:
          </div>
          <div className="font-mono text-xs sm:text-sm text-blue-900 font-semibold flex flex-wrap items-center gap-1.5">
            <span className="bg-white px-2 py-1 rounded border shadow-xs">Original Text</span>
            <span>→</span>
            <span className="bg-white px-2 py-1 rounded border shadow-xs">Custom Financial Tokenization</span>
            <span>→</span>
            <span className="bg-white px-2 py-1 rounded border shadow-xs">Domain Stop-word Removal</span>
            <span>→</span>
            <span className="bg-green-100 text-green-900 px-2 py-1 rounded border border-green-300 font-bold">Pipeline A: Lemmatization</span>
            <span className="text-gray-400">/</span>
            <span className="bg-blue-100 text-blue-900 px-2 py-1 rounded border border-blue-300 font-bold">Pipeline B: Stemming</span>
          </div>
        </div>

        <div className="space-y-4">
          {PREPROCESSING_PIPELINE_DEMO.stages.map((st, idx) => (
            <div key={idx} className="border border-gray-200 rounded-lg p-3.5 bg-white">
              <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
                <span className="font-bold text-sm text-gray-900">{st.stage}</span>
                <span className="text-xs text-gray-500 italic">{st.note}</span>
              </div>
              <div className="flex flex-wrap gap-1.5 p-2 bg-gray-50 rounded border border-gray-200">
                {st.tokens.map((tok, ti) => (
                  <code
                    key={ti}
                    className={`text-xs px-2 py-0.5 rounded font-mono ${
                      tok.includes("₹") || tok.includes("%") || tok.includes("FY")
                        ? "bg-amber-100 text-amber-900 font-semibold border border-amber-300"
                        : idx === 3
                        ? "bg-green-100 text-green-900 font-medium"
                        : idx === 4
                        ? "bg-blue-100 text-blue-900 font-medium"
                        : "bg-white text-gray-800 border border-gray-200"
                    }`}
                  >
                    {tok}
                  </code>
                ))}
              </div>
            </div>
          ))}
        </div>

        {/* Pipeline A vs B comparison callout */}
        <div className="mt-4 grid md:grid-cols-2 gap-4">
          <div className="p-3.5 bg-green-50/70 border border-green-200 rounded-lg">
            <div className="flex items-center justify-between mb-1">
              <span className="font-bold text-sm text-green-950">Pipeline A Output (Selected)</span>
              <Badge variant="success">LemmInflect / WordNet</Badge>
            </div>
            <p className="text-xs text-green-900 leading-relaxed">
              Maintains full grammatical vocabulary (26,155 index terms across 6,134 units). Retains distinct inflections for compound concepts without false root collisions.
            </p>
          </div>
          <div className="p-3.5 bg-blue-50/70 border border-blue-200 rounded-lg">
            <div className="flex items-center justify-between mb-1">
              <span className="font-bold text-sm text-blue-950">Pipeline B Output (Comparison)</span>
              <Badge variant="info">Snowball Stemming</Badge>
            </div>
            <p className="text-xs text-blue-900 leading-relaxed">
              Reduces terms to shared stems (21,403 index terms across 6,134 units). Yields slightly lower precision due to stem collisions on multi-term and Boolean financial queries.
            </p>
          </div>
        </div>
      </SectionCard>

      {/* Cleaning Stages */}
      <SectionCard title="Corpus Cleaning Stages">
        <DataTable
          columns={[
            { key: "stage", header: "Stage" },
            { key: "method", header: "Method / Operation" },
            { key: "total_tokens", header: "Total Tokens" },
            { key: "unique_tokens", header: "Unique Tokens" },
            { key: "vocabulary_size", header: "Vocabulary" },
            { key: "token_change_vs_baseline_percent", header: "% Change" },
          ]}
          rows={data.stages ?? []}
          keyField="stage"
          renderCell={(row, col) => {
            if (col === "total_tokens" || col === "unique_tokens" || col === "vocabulary_size") {
              return Number(row[col])?.toLocaleString() ?? "—";
            }
            if (col === "token_change_vs_baseline_percent") {
              const v = Number(row[col]);
              return isNaN(v) ? "—" : `${v >= 0 ? "+" : ""}${v.toFixed(2)}%`;
            }
            if (col === "stage") return <span className="font-mono font-semibold text-xs text-blue-900">{row.stage ?? "—"}</span>;
            return row[col] ?? "—";
          }}
        />
      </SectionCard>

      {/* Stopword Strategy Comparison */}
      {data.stopwords && data.stopwords.length > 0 && (
        <SectionCard title="Stop-word Strategy Comparison">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
            <MetricCard label="Standard Remaining" value="250,215" />
            <MetricCard label="Domain-Aware Remaining" value="249,744" />
            <MetricCard label="Protected Terms" value="85" />
            <MetricCard label="Strategy Selected" value="Domain-Aware" />
          </div>
          <DataTable
            columns={[
              { key: "strategy", header: "Strategy" },
              { key: "description", header: "Description" },
              { key: "total_tokens", header: "Total Tokens" },
              { key: "tokens_removed", header: "Tokens Removed" },
              { key: "token_reduction_percent", header: "Token Reduction %" },
              { key: "vocabulary_size", header: "Vocab Size" },
              { key: "vocabulary_reduction_percent", header: "Vocab Reduction %" },
            ]}
            rows={data.stopwords}
            keyField="strategy"
            renderCell={(row, col) => {
              if (col === "strategy") return <Badge variant={row.strategy === "domain_aware" ? "success" : "default"}>{row.strategy ?? "—"}</Badge>;
              if (col === "total_tokens" || col === "tokens_removed" || col === "vocabulary_size") {
                return Number(row[col])?.toLocaleString() ?? "—";
              }
              if (col === "token_reduction_percent" || col === "vocabulary_reduction_percent") {
                const v = Number(row[col]);
                return isNaN(v) ? "—" : `${v.toFixed(2)}%`;
              }
              return row[col] ?? "—";
            }}
          />
        </SectionCard>
      )}

      {/* Protected Terms */}
      {data.protected_terms && data.protected_terms.length > 0 && (
        <SectionCard title="Protected Financial Terms (Preserved in Preprocessing)">
          <p className="text-xs text-gray-600 mb-3">
            Core economic terms protected from standard stop-word pruning to prevent critical signal loss (e.g. &ldquo;interest rate&rdquo;, &ldquo;bank credit&rdquo;, &ldquo;bond yield&rdquo;):
          </p>
          <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-6 gap-2 max-h-56 overflow-y-auto p-2 bg-gray-50 rounded border">
            {data.protected_terms.map((pt: any, i: number) => (
              <div key={i} className="text-xs bg-white p-1.5 rounded border border-gray-200">
                <span className="font-semibold text-blue-900">{pt.term}</span>
              </div>
            ))}
          </div>
        </SectionCard>
      )}

      {/* Before / After Examples */}
      {data.before_after_examples && data.before_after_examples.length > 0 && (
        <SectionCard title="Before / After Cleaning Examples">
          <DataTable
            columns={[
              { key: "unit_id", header: "Unit ID", width: "160px" },
              { key: "original_text", header: "Original Text" },
              { key: "stopword_removed", header: "After Stopwords" },
              { key: "lemmatized_after_stopwords", header: "Pipeline A (Lemmatized)" },
              { key: "stemmed_after_stopwords", header: "Pipeline B (Stemmed)" },
            ]}
            rows={data.before_after_examples.slice(0, 15)}
            keyField="unit_id"
            renderCell={(row, col) => {
              if (col === "stopword_removed") return <code className="text-xs bg-amber-50 text-amber-900 border border-amber-200 px-1.5 py-0.5 rounded font-mono">{row[col] ?? "—"}</code>;
              if (col === "lemmatized_after_stopwords") return <code className="text-xs bg-green-50 text-green-900 border border-green-200 px-1.5 py-0.5 rounded font-mono">{row[col] ?? "—"}</code>;
              if (col === "stemmed_after_stopwords") return <code className="text-xs bg-blue-50 text-blue-900 border border-blue-200 px-1.5 py-0.5 rounded font-mono">{row[col] ?? "—"}</code>;
              if (col === "unit_id") return <span className="font-mono text-xs text-gray-700">{row.unit_id ?? "—"}</span>;
              return <span className="text-xs text-gray-800">{row[col] ?? "—"}</span>;
            }}
          />
        </SectionCard>
      )}
    </PageContainer>
  );
}

export default function PreprocessingPage() {
  return (
    <Layout>
      <Suspense fallback={<Loading />}>
        <PreprocessingContent />
      </Suspense>
    </Layout>
  );
}