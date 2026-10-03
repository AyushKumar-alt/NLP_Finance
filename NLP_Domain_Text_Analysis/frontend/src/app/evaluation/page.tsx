/** Evaluation page: Phase 4 report with charts and judgment review. */

"use client";

import { useEffect, useState, Suspense } from "react";
import { Layout, PageContainer, SectionCard, DataTable, Pagination, Loading, ErrorState, Badge, MetricCard } from "@/components";
import { api } from "@/lib/api";

function BarChart({ data, title }: { data: { label: string; value: number }[]; title: string }) {
  const values = data.map((d) => d.value);
  const max = Math.max(...values, 0.0001);
  return (
    <SectionCard title={title}>
      <div className="h-64 overflow-y-auto">
        {data.length === 0 ? (
          <p className="text-gray-500 text-sm">No data at this cut-off.</p>
        ) : (
          <div className="space-y-2">
{data.map((row, i) => (
                <div key={`${row.label}-${i}`}>
                <div className="flex items-center gap-2">
                  <span className="w-10 text-xs text-gray-500">{row.label}</span>
                  <div className="relative flex-1 bg-gray-100 rounded h-5">
                    <div
                      className="bg-blue-600 h-5 rounded transition-all"
                      style={{ width: `${(row.value / max) * 100}%` }}
                    />
                  </div>
                  <span className="w-12 text-right text-xs text-gray-700">{row.value.toFixed(3)}</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </SectionCard>
  );
}

function EvaluationContent() {
  const [report, setReport] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const [judgmentParams, setJudgmentParams] = useState({ query_id: "", relevance: undefined as boolean | undefined, limit: 100, offset: 0 });
  const [judgments, setJudgments] = useState<any>(null);

  useEffect(() => {
    api.evaluationReport().then(setReport).catch(setError).finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    api.judgments({ ...judgmentParams })
      .then(setJudgments)
      .catch(console.error);
  }, [judgmentParams]);

  if (loading) return <Loading />;
  if (error) return <ErrorState error={error} retry={() => window.location.reload()} />;
  if (!report) return null;

  const unit = report.unit_level ?? {};
  const document = report.document_level ?? {};
  const atK = report.at_k ?? {};
  const perQuery = report.per_query ?? { items: [] };

  return (
    <PageContainer title="Evaluation" description="Phase 4 relevance evaluation of Pipeline B.">
      {/* Methodology callouts */}
      <SectionCard title="Methodology Notes">
        <ul className="space-y-2 text-sm list-disc list-inside">
          {report.notes.map((n: string, i: number) => <li key={i}>{n}</li>)}
        </ul>
        {report.judgment_summary?.per_query && (
          <p className="mt-2 text-xs text-gray-500">
            Judgment distribution: {Object.entries(report.judgment_summary.per_query)
              .map(([q, v]: [string, any]) => `${q}: ${v.relevant}/${v.judged}`)
              .join(" | ")}
          </p>
        )}
      </SectionCard>

      {/* Core metrics */}
      <div className="grid md:grid-cols-3 gap-6">
        <SectionCard title="Unit-Level Metrics (Content Units)">
          <div className="grid grid-cols-2 gap-3">
            <MetricCard label="Precision (macro)" value={unit.precision_macro?.toFixed(4) ?? "—"} />
            <MetricCard label="Recall (macro)" value={unit.recall_macro?.toFixed(4) ?? "—"} />
            <MetricCard label="F1 (macro)" value={unit.f1_macro?.toFixed(4) ?? "—"} />
            <MetricCard label="Precision (micro)" value={unit.precision_micro?.toFixed(4) ?? "—"} />
            <MetricCard label="Recall (micro)" value={unit.recall_micro?.toFixed(4) ?? "—"} />
            <MetricCard label="F1 (micro)" value={unit.f1_micro?.toFixed(4) ?? "—"} />
            <MetricCard label="P@5 (macro)" value={unit.precision_at_5_macro?.toFixed(4) ?? "—"} />
            <MetricCard label="R@5 (macro)" value={unit.recall_at_5_macro?.toFixed(4) ?? "—"} />
            <MetricCard label="P@10 (macro)" value={unit.precision_at_10_macro?.toFixed(4) ?? "—"} />
            <MetricCard label="R@10 (macro)" value={unit.recall_at_10_macro?.toFixed(4) ?? "—"} />
            <MetricCard label="Judged Pairs" value={unit.judged_pairs?.toLocaleString() ?? "—"} />
            <MetricCard label="Pool Depth" value={unit.judged_depth} unit={<Badge variant="warning">pool-bounded recall</Badge>} />
          </div>
          <p className="mt-4 text-xs text-gray-500">
            Recall is pool-bounded: measured against the judgment pool, not the corpus.
          </p>
        </SectionCard>

        <SectionCard title="Document-Level Metrics">
          <div className="grid grid-cols-2 gap-3">
            <MetricCard label="P (macro)" value={document.precision_macro?.toFixed(4) ?? "—"} />
            <MetricCard label="R (macro)" value={document.recall_macro?.toFixed(4) ?? "—"} />
            <MetricCard label="F1 (macro)" value={document.f1_macro?.toFixed(4) ?? "—"} />
            <MetricCard label="P (micro)" value={document.precision_micro?.toFixed(4) ?? "—"} />
            <MetricCard label="R (micro)" value={document.recall_micro?.toFixed(4) ?? "—"} />
            <MetricCard label="F1 (micro)" value={document.f1_micro?.toFixed(4) ?? "—"} />
          </div>
          <p className="mt-4 text-xs text-gray-500">
            A document is relevant if it contains at least one judged relevant content unit.
          </p>
        </SectionCard>

        <SectionCard title="Judgment Summary">
          <div className="grid grid-cols-2 gap-3">
            <MetricCard label="Pairs" value={report.judgment_summary?.pairs?.toLocaleString() ?? "—"} />
            <MetricCard label="Relevant" value={report.judgment_summary?.relevant?.toLocaleString() ?? "—"} unit={<Badge variant="success">+</Badge>} />
            <MetricCard label="Not Relevant" value={report.judgment_summary?.not_relevant?.toLocaleString() ?? "—"} unit={<Badge variant="error">−</Badge>} />
            <MetricCard label="Annotator" value={report.judgment_summary?.annotator ?? "—"} />
          </div>
        </SectionCard>
      </div>

      {/* P@K / R@K charts */}
      <div className="grid md:grid-cols-2 gap-6">
        {atK[5] && (
          <BarChart
            title="Precision@5 by Query"
            data={atK[5].queries?.map((q: any) => ({ label: q.query_id, value: q.precision_at_5 })) ?? []}
          />
        )}
        {atK[10] && (
          <BarChart
            title="Recall@10 by Query"
            data={atK[10].queries?.map((q: any) => ({ label: q.query_id, value: q.recall_at_10 })) ?? []}
          />
        )}
      </div>

      {/* Per-query table */}
      <SectionCard title="Per-Query Results">
        <DataTable
          columns={[
            { key: "query_id", header: "Query" },
            { key: "query", header: "Text" },
            { key: "precision", header: "P", width: "60px" },
            { key: "recall", header: "R", width: "60px" },
            { key: "f1", header: "F1", width: "60px" },
            { key: "average_precision", header: "AP", width: "60px" },
            { key: "precision_at_10", header: "P@10", width: "60px" },
            { key: "recall_at_10", header: "R@10", width: "60px" },
            { key: "retrieved_units", header: "Retrieved", width: "90px" },
            { key: "execution_time_ms", header: "ms", width: "70px" },
          ]}
          rows={perQuery.items ?? []}
          keyField="query_id"
          renderCell={(row, col) => {
            if (["precision", "recall", "f1", "average_precision", "precision_at_10", "recall_at_10"].includes(col)) {
              const v = Number(row[col]);
              return <Badge variant={v >= 0.8 ? "success" : v >= 0.5 ? "warning" : "error"}>{isNaN(v) ? "—" : v.toFixed(4)}</Badge>;
            }
            if (col === "retrieved_units" || col === "execution_time_ms") return Number(row[col])?.toLocaleString() ?? "—";
            if (col === "query") return <code className="text-sm">{row.query ?? "—"}</code>;
            return row[col] ?? "—";
          }}
        />
        <Pagination
          total={perQuery.total ?? 0}
          offset={perQuery.offset ?? 0}
          limit={perQuery.limit ?? 50}
          onChange={(offset) => setJudgmentParams((p) => ({ ...p, offset }))}
        />
      </SectionCard>

      {/* Comparison */}
      <SectionCard title="Pipeline A vs Pipeline B">
        <DataTable
          columns={[
            { key: "metric", header: "Metric" },
            { key: "pipeline_a", header: "Pipeline A" },
            { key: "pipeline_b", header: "Pipeline B" },
            { key: "selected", header: "Selected" },
            { key: "note", header: "Note" },
          ]}
          rows={(report.comparison ?? []).filter((r: any) => r.available)}
          keyField="metric"
          renderCell={(row, col) => {
            if (col === "pipeline_a" || col === "pipeline_b") return row[col] === "N/A" ? <Badge variant="error">N/A</Badge> : row[col];
            if (col === "selected") return <Badge variant={row.selected === "pipeline_b" ? "success" : "default"}>{row.selected}</Badge>;
            if (col === "note") return <p className="text-xs text-gray-500">{row.note || "—"}</p>;
            return row[col] ?? "—";
          }}
        />
        {(report.comparison ?? []).some((r: any) => !r.available) && (
          <p className="text-xs text-gray-600 mt-3">
            Pipeline A relevance rows are marked N/A because scoring an alternate index against a
            judgment pool built from Pipeline B's rankings would be invalid.
          </p>
        )}
      </SectionCard>

      {/* Validation */}
      <SectionCard title="Phase 4 Validation">
        <DataTable
          columns={[
            { key: "rule_id", header: "Rule" },
            { key: "status", header: "Status" },
            { key: "detail", header: "Detail" },
          ]}
          rows={report.validation ?? []}
          keyField="rule_id"
          renderCell={(row, col) => {
            if (col === "status") return <Badge variant={String(row.status).toUpperCase() === "PASS" ? "success" : "error"}>{row.status}</Badge>;
            return row[col] ?? "—";
          }}
        />
      </SectionCard>
    </PageContainer>
  );
}

export default function EvaluationPage() {
  return (
    <Layout>
      <Suspense fallback={<Loading />}>
        <EvaluationContent />
      </Suspense>
    </Layout>
  );
}