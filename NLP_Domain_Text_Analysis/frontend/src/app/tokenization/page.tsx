/** Tokenization page: Phase 2 tokenizer comparison and interactive tokenization. */

"use client";

import { useEffect, useState, Suspense } from "react";
import { Layout, PageContainer, SectionCard, DataTable, Loading, ErrorState, Badge, MetricCard } from "@/components";
import { api } from "@/lib/api";

const PRESET_SAMPLES = [
  {
    label: "Fiscal Year & Policy Rate",
    text: "In FY2025, RBI maintained the repo rate at 6.50% while CPI inflation moderated to 4.2% with ₹50,000 crore liquidity support.",
  },
  {
    label: "Securities & Currency Amounts",
    text: "From April to November 2025, ₹13,893 crore was raised by listed REITs and InvITs, alongside $2.4 billion in foreign commercial borrowings.",
  },
  {
    label: "Growth & Sectoral Ratios",
    text: "Bank credit to MSMEs increased by 21.8 per cent YoY in November 2025 compared to 13 per cent in FY24, keeping CRAR above 16.5%.",
  },
  {
    label: "Multi-Word Economic Concepts",
    text: "The Monetary Policy Committee noted that current account deficit narrowed while financial stability indicators remained resilient.",
  },
];

const CANONICAL_CORPUS_TOKENIZATION = [
  {
    tokenizer: "NLTK Word Tokenizer",
    tokens: 349344,
    vocabulary: 18035,
    role: "Academic Baseline",
    preservation: "Splits decimals (6.50 -> 6.50 or 6 . 50), strips currency symbols, detaches %",
  },
  {
    tokenizer: "spaCy (en_core_web_sm)",
    tokens: 365068,
    vocabulary: 15899,
    role: "Statistical Baseline",
    preservation: "Separates hyphenated terms, keeps decimals, splits compound currency",
  },
  {
    tokenizer: "Custom Financial Tokenizer",
    tokens: 345677,
    vocabulary: 18628,
    role: "SELECTED FOR RETRIEVAL PIPELINE",
    preservation: "Preserves ₹, $, %, FY2025, decimal rates, and fiscal periods atomically",
  },
  {
    tokenizer: "Hybrid Tokenizer",
    tokens: 344141,
    vocabulary: 19463,
    role: "Advanced Multi-Pass",
    preservation: "Rule-based expression hold-out + NLTK fallback + token repair",
  },
];

function TokenizationContent() {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  // Live tokenizer playground state
  const [inputText, setInputText] = useState(PRESET_SAMPLES[0].text);
  const [liveTokens, setLiveTokens] = useState<any>(null);
  const [liveLoading, setLiveLoading] = useState(false);

  useEffect(() => {
    api.tokenization().then(setData).catch(setError).finally(() => setLoading(false));
    runLiveTokenize(PRESET_SAMPLES[0].text);
  }, []);

  const runLiveTokenize = (text: string) => {
    if (!text.trim()) return;
    setLiveLoading(true);
    api.tokenizeLive(text)
      .then((res) => {
        setLiveTokens(res);
        setLiveLoading(false);
      })
      .catch((err) => {
        console.error("Live tokenize error:", err);
        setLiveLoading(false);
      });
  };

  if (loading) return <Loading />;
  if (error) return <ErrorState error={error} />;
  if (!data) return null;

  return (
    <PageContainer
      title="Tokenization Analysis"
      description="Comparison of standard vs. domain-specific tokenizers on Indian financial and economic text."
    >
      {/* Canonical Corpus Level Stored Comparison */}
      <SectionCard title="Corpus-Level Stored Comparison: NLTK / spaCy / Custom / Hybrid">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <MetricCard label="NLTK Tokens / Vocab" value="349,344 / 18,035" />
          <MetricCard label="spaCy Tokens / Vocab" value="365,068 / 15,899" />
          <MetricCard label="Custom (Selected)" value="345,677 / 18,628" />
          <MetricCard label="Hybrid Tokens / Vocab" value="344,141 / 19,463" />
        </div>

        <DataTable
          columns={[
            { key: "tokenizer" as const, header: "Tokenizer" },
            { key: "tokens" as const, header: "Total Tokens" },
            { key: "vocabulary" as const, header: "Vocabulary" },
            { key: "role" as const, header: "Pipeline Status" },
            { key: "preservation" as const, header: "Domain Token Handling" },
          ]}
          rows={CANONICAL_CORPUS_TOKENIZATION}
          keyField="tokenizer"
          renderCell={(row, col) => {
            if (col === "tokens" || col === "vocabulary") return (row as any)[col]?.toLocaleString() ?? "—";
            if (col === "role") {
              const isSelected = row.role.includes("SELECTED");
              return (
                <Badge variant={isSelected ? "success" : "default"}>
                  {row.role}
                </Badge>
              );
            }
            return (row as any)[col] ?? "—";
          }}
        />

        <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded text-xs text-blue-900 leading-relaxed">
          <strong>Why Custom Tokenization was Selected for Final Pipelines:</strong> Standard tokenizers split critical financial constructs like <code>6.50%</code> into <code>6.50</code> and <code>%</code>, detach <code>₹</code> from monetary amounts, and decompose fiscal abbreviations like <code>FY2025</code>. The Custom Financial Tokenizer preserves these economic metrics atomically, preventing false-positive matches and loss of numerical context during indexing.
        </div>
      </SectionCard>

      {/* Interactive Live Tokenization Playground */}
      <SectionCard title="Live Tokenization: NLTK / spaCy / Custom / Hybrid">
        <p className="text-xs text-gray-600 mb-3">
          Execute real-time tokenization on arbitrary input text across all four tokenizers (Custom Financial, Hybrid, spaCy, and NLTK).
        </p>
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-1 text-gray-800">
              Sample Financial Sentence or Custom Input:
            </label>
            <div className="flex flex-wrap gap-2 mb-2">
              {PRESET_SAMPLES.map((sample, i) => (
                <button
                  key={i}
                  onClick={() => {
                    setInputText(sample.text);
                    runLiveTokenize(sample.text);
                  }}
                  className="px-2.5 py-1 text-xs bg-gray-100 hover:bg-gray-200 rounded border border-gray-300 text-gray-800 transition-colors font-medium"
                >
                  {sample.label}
                </button>
              ))}
            </div>
            <textarea
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              rows={3}
              className="w-full px-3 py-2 border border-gray-300 rounded font-mono text-sm focus:ring-2 focus:ring-blue-500"
              placeholder="Enter custom financial or economic text..."
            />
            <div className="flex justify-end mt-2">
              <button
                onClick={() => runLiveTokenize(inputText)}
                disabled={liveLoading || !inputText.trim()}
                className="px-5 py-2 bg-blue-600 text-white rounded font-medium hover:bg-blue-700 disabled:opacity-50 transition-colors text-sm"
              >
                {liveLoading ? "Tokenizing..." : "Tokenize Text"}
              </button>
            </div>
          </div>

          {/* Side-by-Side Tokenizer Output */}
          {liveTokens && liveTokens.tokens && (
            <div className="mt-4 pt-4 border-t border-gray-200">
              <h4 className="text-sm font-bold text-gray-800 mb-3">
                Live Tokenizer Comparison (Side-by-Side):
              </h4>
              <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
                {/* Custom Tokenizer */}
                <div className="border-2 border-green-300 bg-green-50/40 p-4 rounded-lg">
                  <div className="flex justify-between items-center mb-2">
                    <span className="font-bold text-sm text-green-950">
                      Custom Financial (Live)
                    </span>
                    <Badge variant="success">{liveTokens.counts.custom} tokens</Badge>
                  </div>
                  <div className="flex flex-wrap gap-1.5 max-h-48 overflow-y-auto p-2 bg-white rounded border border-green-200">
                    {liveTokens.tokens.custom.map((tok: string, idx: number) => (
                      <code key={idx} className="text-xs bg-green-100 text-green-900 px-1.5 py-0.5 rounded font-mono">
                        {tok}
                      </code>
                    ))}
                  </div>
                </div>

                {/* Hybrid Tokenizer */}
                <div className="border border-blue-200 bg-blue-50/40 p-4 rounded-lg">
                  <div className="flex justify-between items-center mb-2">
                    <span className="font-bold text-sm text-blue-950">
                      Hybrid Tokenizer (Live)
                    </span>
                    <Badge variant="info">{liveTokens.counts.hybrid} tokens</Badge>
                  </div>
                  <div className="flex flex-wrap gap-1.5 max-h-48 overflow-y-auto p-2 bg-white rounded border border-blue-200">
                    {liveTokens.tokens.hybrid.map((tok: string, idx: number) => (
                      <code key={idx} className="text-xs bg-blue-100 text-blue-900 px-1.5 py-0.5 rounded font-mono">
                        {tok}
                      </code>
                    ))}
                  </div>
                </div>

                {/* spaCy Baseline */}
                <div className="border border-gray-200 bg-gray-50 p-4 rounded-lg">
                  <div className="flex justify-between items-center mb-2">
                    <span className="font-bold text-sm text-gray-800">
                      spaCy Tokenizer (Live)
                    </span>
                    <Badge variant="default">{liveTokens.counts.spacy} tokens</Badge>
                  </div>
                  <div className="flex flex-wrap gap-1.5 max-h-48 overflow-y-auto p-2 bg-white rounded border border-gray-200">
                    {liveTokens.tokens.spacy.map((tok: string, idx: number) => (
                      <code key={idx} className="text-xs bg-gray-100 text-gray-800 px-1.5 py-0.5 rounded font-mono">
                        {tok}
                      </code>
                    ))}
                  </div>
                </div>

                {/* NLTK Baseline */}
                <div className="border border-gray-200 bg-gray-50 p-4 rounded-lg">
                  <div className="flex justify-between items-center mb-2">
                    <span className="font-bold text-sm text-gray-800">
                      NLTK Tokenizer (Live)
                    </span>
                    <Badge variant="default">{liveTokens.counts.nltk} tokens</Badge>
                  </div>
                  <div className="flex flex-wrap gap-1.5 max-h-48 overflow-y-auto p-2 bg-white rounded border border-gray-200">
                    {liveTokens.tokens.nltk.map((tok: string, idx: number) => (
                      <code key={idx} className="text-xs bg-gray-100 text-gray-800 px-1.5 py-0.5 rounded font-mono">
                        {tok}
                      </code>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </SectionCard>

      {/* Custom Rules */}
      {data.custom_rules && data.custom_rules.length > 0 && (
        <SectionCard title="Custom Financial Tokenizer Rules">
          <DataTable
            columns={[
              { key: "rule_id", header: "Rule ID", width: "90px" },
              { key: "pattern", header: "Regex Pattern" },
              { key: "description", header: "Financial Concept Preserved" },
              { key: "example", header: "Corpus Example" },
              { key: "expected_behavior", header: "Preservation Behavior" },
            ]}
            rows={data.custom_rules}
            keyField="rule_id"
            renderCell={(row, col) => {
              if (col === "rule_id") return <Badge variant="info">{row.rule_id ?? "—"}</Badge>;
              if (col === "pattern") return <code className="text-xs bg-gray-100 text-gray-800 px-1 py-0.5 rounded font-mono break-all">{row.pattern ?? "—"}</code>;
              if (col === "example") return <span className="font-semibold text-xs text-blue-900 font-mono">{row.example ?? "—"}</span>;
              return <span className="text-xs text-gray-800">{row[col] ?? "—"}</span>;
            }}
          />
        </SectionCard>
      )}

      {/* Date & Number Handling */}
      {data.date_number_comparison && data.date_number_comparison.length > 0 && (
        <SectionCard title="Date & Number Handling Across Methods">
          <DataTable
            columns={[
              { key: "category", header: "Expression Category" },
              { key: "class", header: "Class" },
              { key: "expressions_detected", header: "Detected Expressions" },
              { key: "expressions_intact_after_standard_tokenization", header: "Standard Intact" },
              { key: "intact_rate_percent", header: "Standard Intact %" },
              { key: "typed_component_tokens", header: "Fragmented Tokens" },
            ]}
            rows={data.date_number_comparison}
            keyField="category"
            renderCell={(row, col) => {
              if (col === "category") return <span className="font-medium text-xs text-gray-900">{row.category?.replace(/_/g, " ") ?? "—"}</span>;
              if (col === "class") return <Badge variant="default">{row.class ?? "—"}</Badge>;
              if (col === "expressions_detected" || col === "expressions_intact_after_standard_tokenization" || col === "typed_component_tokens") {
                return Number(row[col])?.toLocaleString() ?? "—";
              }
              if (col === "intact_rate_percent") {
                const v = Number(row[col]);
                return <span className={`text-xs font-semibold ${v < 50 ? "text-amber-700" : "text-green-700"}`}>{isNaN(v) ? "—" : `${v.toFixed(2)}%`}</span>;
              }
              return <span className="text-xs text-gray-800">{row[col] ?? "—"}</span>;
            }}
          />
        </SectionCard>
      )}
    </PageContainer>
  );
}

export default function TokenizationPage() {
  return (
    <Layout>
      <Suspense fallback={<Loading />}>
        <TokenizationContent />
      </Suspense>
    </Layout>
  );
}