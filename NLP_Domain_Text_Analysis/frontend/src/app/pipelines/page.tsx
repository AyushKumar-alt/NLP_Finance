/** Pipeline selection and comparative evaluation page. */

"use client";

import { useEffect, useState, Suspense } from "react";
import { Layout, PageContainer, SectionCard, DataTable, Loading, ErrorState, Badge, MetricCard } from "@/components";
import { api } from "@/lib/api";

const CANONICAL_INDEX_STATS = [
  { metric: "Searchable Content Units", pipeline_a: "6,134", pipeline_b: "6,134", delta: "0", superior: "Equal" },
  { metric: "Phase 2 Post-Stopword Tokens", pipeline_a: "250,794", pipeline_b: "250,794", delta: "0", superior: "Equal (Linguistic Stage)" },
  { metric: "Phase 3/4 Positional Index Tokens", pipeline_a: "249,489", pipeline_b: "249,489", delta: "0", superior: "Equal (Indexing Stage)" },
  { metric: "Total Index Terms", pipeline_a: "26,155", pipeline_b: "21,403", delta: "+4,752", superior: "Pipeline A (Richer Vocab)" },
  { metric: "Unigram Terms", pipeline_a: "24,110", pipeline_b: "18,763", delta: "+5,347", superior: "Pipeline A" },
  { metric: "Phrase Terms", pipeline_a: "2,045", pipeline_b: "2,640", delta: "-595", superior: "Pipeline B" },
  { metric: "Total Postings", pipeline_a: "195,579", pipeline_b: "195,567", delta: "+12", superior: "Comparable" },
];

const CANONICAL_RETRIEVAL_METRICS = [
  { metric: "Precision (Macro)", a: "0.9764", b: "0.9571", delta: "+0.0193", winner: "Pipeline A" },
  { metric: "Pooled Recall", a: "0.8820", b: "0.8723", delta: "+0.0097", winner: "Pipeline A" },
  { metric: "F1-Score", a: "0.9183", b: "0.9053", delta: "+0.0130", winner: "Pipeline A" },
  { metric: "Mean Average Precision (MAP)", a: "0.9328", b: "0.9231", delta: "+0.0097", winner: "Pipeline A" },
  { metric: "Mean Reciprocal Rank (MRR)", a: "1.0000", b: "1.0000", delta: "0.0000", winner: "Tie" },
  { metric: "Precision @ 1 (P@1)", a: "1.0000", b: "1.0000", delta: "0.0000", winner: "Tie" },
  { metric: "Precision @ 3 (P@3)", a: "0.9778", b: "0.9556", delta: "+0.0222", winner: "Pipeline A" },
  { metric: "Precision @ 5 (P@5)", a: "0.9600", b: "0.9200", delta: "+0.0400", winner: "Pipeline A" },
  { metric: "Precision @ 10 (P@10)", a: "0.8467", b: "0.8400", delta: "+0.0067", winner: "Pipeline A" },
  { metric: "Recall @ 1 (R@1)", a: "0.1651", b: "0.1651", delta: "0.0000", winner: "Tie" },
  { metric: "Recall @ 3 (R@3)", a: "0.3702", b: "0.3546", delta: "+0.0156", winner: "Pipeline A" },
  { metric: "Recall @ 5 (R@5)", a: "0.5130", b: "0.4923", delta: "+0.0207", winner: "Pipeline A" },
  { metric: "Recall @ 10 (R@10)", a: "0.8820", b: "0.8723", delta: "+0.0097", winner: "Pipeline A" },
  { metric: "nDCG @ 1", a: "1.0000", b: "1.0000", delta: "0.0000", winner: "Tie" },
  { metric: "nDCG @ 3", a: "0.9754", b: "0.9559", delta: "+0.0195", winner: "Pipeline A" },
  { metric: "nDCG @ 5", a: "0.9663", b: "0.9472", delta: "+0.0191", winner: "Pipeline A" },
  { metric: "nDCG @ 10", a: "0.9632", b: "0.9554", delta: "+0.0078", winner: "Pipeline A" },
];

function PipelinesContent() {
  const [pipelines, setPipelines] = useState<any[] | null>(null);
  const [selection, setSelection] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    api.pipelines().then(setPipelines).catch(setError).finally(() => setLoading(false));
    api.pipelineSelection().then(setSelection).catch(console.error);
  }, []);

  if (loading) return <Loading />;
  if (error) return <ErrorState error={error} />;

  return (
    <PageContainer
      title="Primary Pipeline Comparison & Selection"
      description="Controlled comparative evaluation between Pipeline A (Lemmatization) and Pipeline B (Stemming)."
    >
      {/* Official Selection Banner */}
      <div className="bg-gradient-to-r from-green-50 to-emerald-50 border-2 border-green-300 rounded-lg p-5 mb-6">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
          <div>
            <span className="text-xs font-bold uppercase tracking-wider text-green-800 bg-green-200/80 px-2.5 py-0.5 rounded">
              Official Selection Decision
            </span>
            <h2 className="text-xl font-bold text-green-950 mt-1">
              Selected: Pipeline A — Lemmatization (pipeline_a_lemma)
            </h2>
            <p className="text-xs text-green-800 mt-1">
              Superior measured performance across MAP, nDCG@10, Precision, and Pooled Recall over 15 benchmark queries.
            </p>
          </div>
          <div className="text-right">
            <span className="text-2xl font-black text-green-900">MAP: 0.9328</span>
            <div className="text-xs text-green-700">vs 0.9231 for Pipeline B</div>
          </div>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <div className="bg-white p-3 rounded border border-green-200 text-center">
            <div className="text-xs text-gray-500 font-medium">Pipeline A Wins</div>
            <div className="text-xl font-bold text-green-800 mt-1">3 Queries</div>
            <div className="text-[10px] text-gray-400 mt-0.5">Q09, Q14, Q15</div>
          </div>
          <div className="bg-white p-3 rounded border border-green-200 text-center">
            <div className="text-xs text-gray-500 font-medium">Pipeline B Wins</div>
            <div className="text-xl font-bold text-gray-700 mt-1">0 Queries</div>
            <div className="text-[10px] text-gray-400 mt-0.5">None</div>
          </div>
          <div className="bg-white p-3 rounded border border-green-200 text-center">
            <div className="text-xs text-gray-500 font-medium">Tied Queries</div>
            <div className="text-xl font-bold text-gray-700 mt-1">12 Queries</div>
            <div className="text-[10px] text-gray-400 mt-0.5">Equal rank precision</div>
          </div>
          <div className="bg-white p-3 rounded border border-green-200 text-center">
            <div className="text-xs text-gray-500 font-medium">Statistical Claim</div>
            <div className="text-xs font-bold text-gray-700 mt-2">Empirical Benchmark</div>
            <div className="text-[10px] text-gray-400 mt-0.5">No claim of universality</div>
          </div>
        </div>
      </div>

      {/* Architecture Flow */}
      <div className="grid md:grid-cols-2 gap-6 mb-6">
        <SectionCard title="Pipeline A Architecture (Selected)">
          <div className="space-y-2 font-mono text-xs">
            <div className="p-2 bg-green-50 text-green-950 rounded border border-green-200 font-semibold">
              1. Custom Financial Tokenizer
            </div>
            <div className="text-center text-gray-400">↓</div>
            <div className="p-2 bg-green-50 text-green-950 rounded border border-green-200 font-semibold">
              2. Date / Numerical / Currency Preservation
            </div>
            <div className="text-center text-gray-400">↓</div>
            <div className="p-2 bg-green-50 text-green-950 rounded border border-green-200 font-semibold">
              3. Domain-Aware Stop-word Pruning (85 protected)
            </div>
            <div className="text-center text-gray-400">↓</div>
            <div className="p-2 bg-green-100 text-green-950 rounded border border-green-300 font-bold">
              4. LemmInflect / WordNet Lemmatization
            </div>
            <div className="text-center text-gray-400">↓</div>
            <div className="p-2 bg-green-50 text-green-950 rounded border border-green-200 font-semibold">
              5. Positional Inverted Index (26,155 terms)
            </div>
          </div>
        </SectionCard>

        <SectionCard title="Pipeline B Architecture (Comparison)">
          <div className="space-y-2 font-mono text-xs">
            <div className="p-2 bg-blue-50 text-blue-950 rounded border border-blue-200 font-semibold">
              1. Custom Financial Tokenizer
            </div>
            <div className="text-center text-gray-400">↓</div>
            <div className="p-2 bg-blue-50 text-blue-950 rounded border border-blue-200 font-semibold">
              2. Date / Numerical / Currency Preservation
            </div>
            <div className="text-center text-gray-400">↓</div>
            <div className="p-2 bg-blue-50 text-blue-950 rounded border border-blue-200 font-semibold">
              3. Domain-Aware Stop-word Pruning (85 protected)
            </div>
            <div className="text-center text-gray-400">↓</div>
            <div className="p-2 bg-blue-100 text-blue-950 rounded border border-blue-300 font-bold">
              4. Snowball English Stemming
            </div>
            <div className="text-center text-gray-400">↓</div>
            <div className="p-2 bg-blue-50 text-blue-950 rounded border border-blue-200 font-semibold">
              5. Positional Inverted Index (21,403 terms)
            </div>
          </div>
        </SectionCard>
      </div>

      {/* Index Statistics */}
      <SectionCard title="Positional Inverted Index Statistics (6,134 Searchable Units)">
        <DataTable
          columns={[
            { key: "metric" as const, header: "Metric" },
            { key: "pipeline_a" as const, header: "Pipeline A (Lemmatization)" },
            { key: "pipeline_b" as const, header: "Pipeline B (Stemming)" },
            { key: "delta" as const, header: "Delta (A - B)" },
            { key: "superior" as const, header: "Assessment" },
          ]}
          rows={CANONICAL_INDEX_STATS}
          keyField="metric"
          renderCell={(row, col) => {
            if (col === "pipeline_a") return <span className="font-semibold text-green-900">{row.pipeline_a}</span>;
            if (col === "pipeline_b") return <span className="font-semibold text-blue-900">{row.pipeline_b}</span>;
            return (row as any)[col] ?? "—";
          }}
        />
      </SectionCard>

      {/* Retrieval Evaluation Comparison */}
      <SectionCard title="Information Retrieval Evaluation (Top-10 Pooled Relevance)">
        <DataTable
          columns={[
            { key: "metric" as const, header: "IR Evaluation Metric" },
            { key: "a" as const, header: "Pipeline A (Lemma)" },
            { key: "b" as const, header: "Pipeline B (Stem)" },
            { key: "delta" as const, header: "Delta (Δ)" },
            { key: "winner" as const, header: "Higher Scoring" },
          ]}
          rows={CANONICAL_RETRIEVAL_METRICS}
          keyField="metric"
          renderCell={(row, col) => {
            if (col === "a") return <span className="font-bold text-green-800">{row.a}</span>;
            if (col === "b") return <span className="text-gray-700">{row.b}</span>;
            if (col === "winner") {
              const isA = row.winner === "Pipeline A";
              return <Badge variant={isA ? "success" : "default"}>{row.winner}</Badge>;
            }
            return (row as any)[col] ?? "—";
          }}
        />

        <div className="mt-4 p-4 bg-gray-50 border border-gray-200 rounded text-xs text-gray-700 leading-relaxed">
          <strong>Academic Discussion:</strong> In financial and economic corpora, multi-word queries such as <em>&ldquo;banking AND credit&rdquo;</em> (Q09), <em>&ldquo;GDP AND investment&rdquo;</em> (Q14), and <em>&ldquo;(GDP OR GVA) AND policy&rdquo;</em> (Q15) require precise semantic boundaries. Stemming merges related but distinct morphological suffixes (e.g. <em>creditor</em>, <em>credited</em>, <em>credit</em>) into broad root stems, leading to occasional false-positive postings. Lemmatization preserves distinct grammatical word forms, enabling Pipeline A to achieve superior precision (0.9764 vs 0.9571) and higher MAP (0.9328 vs 0.9231).
        </div>
      </SectionCard>
    </PageContainer>
  );
}

export default function PipelinesPage() {
  return (
    <Layout>
      <Suspense fallback={<Loading />}>
        <PipelinesContent />
      </Suspense>
    </Layout>
  );
}