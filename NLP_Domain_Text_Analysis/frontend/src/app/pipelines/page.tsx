/** Pipeline selection page. */

"use client";

import { useEffect, useState, Suspense } from "react";
import { Layout, PageContainer, SectionCard, DataTable, Loading, ErrorState, Badge, MetricCard } from "@/components";
import { api } from "@/lib/api";

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
  if (!pipelines) return null;

  return (
    <PageContainer title="Pipelines" description="Phase 2 normalization choices carried into Phase 3's two retrieval pipelines.">
      <SectionCard title="Selection Decision">
        <div className="grid md:grid-cols-2 gap-6">
          <MetricCard label="Selected" value={selection?.final_pipeline ?? "—"} />
          <MetricCard label="Score" value={selection?.score?.toFixed(6) ?? "—"} />
          <MetricCard label="Margin over runner-up" value={selection?.margin_over_runner_up?.toFixed(6) ?? "—"} />
          <MetricCard label="Anchored to Phase 3" value={selection?.final_pipeline_name ?? "—"} />
        </div>
        {selection?.reason && (
          <p className="mt-4 text-sm text-gray-700 bg-gray-50 p-4 rounded leading-relaxed">
            {selection.reason}
          </p>
        )}
        {selection?.criteria_weights && (
          <div className="mt-4">
            <h4 className="font-medium text-gray-700 mb-2">Selection Criteria Weights</h4>
            <dl className="grid grid-cols-2 gap-2 text-sm">
              {Object.entries(selection.criteria_weights).map(([k, v]) => (
                <div key={k} className="flex justify-between"><dt className="text-gray-500">{k}</dt><dd>{(v as number) * 100}%</dd></div>
              ))}
            </dl>
          </div>
        )}
      </SectionCard>

      <SectionCard title="Pipeline Comparison">
        <DataTable
          columns={[
            { key: "pipeline", header: "Pipeline" },
            { key: "name", header: "Name" },
            { key: "is_final", header: "Final" },
            { key: "total_score", header: "Score" },
            { key: "index_terms", header: "Index Terms" },
            { key: "vocabulary_size", header: "Vocabulary" },
            { key: "total_postings", header: "Postings" },
            { key: "processing_time_seconds", header: "Build Time (s)" },
          ]}
          rows={pipelines}
          keyField="pipeline"
          renderCell={(row, col) => {
            if (col === "is_final") return <Badge variant={row.is_final ? "success" : "default"}>{row.is_final ? "Selected" : "Alternative"}</Badge>;
            if (col === "total_score") return row.total_score?.toFixed(6) ?? "—";
            if (col === "index_terms" || col === "vocabulary_size" || col === "total_postings") return row[col]?.toLocaleString() ?? "—";
            if (col === "processing_time_seconds") return row[col]?.toFixed(2) ?? "—";
            return row[col] ?? "—";
          }}
        />
      </SectionCard>

      <SectionCard title="Criteria Breakdown">
        <DataTable
          columns={[
            { key: "pipeline", header: "Pipeline" },
            { key: "criterion", header: "Criterion" },
            { key: "rate", header: "Rate" },
            { key: "weighted", header: "Weighted" },
            { key: "detail", header: "Evidence" },
          ]}
          rows={pipelines.flatMap((p) =>
            Object.entries(p.criteria ?? {}).map(([criterion, value]: [string, any]) => ({
              pipeline: p.pipeline,
              criterion,
              rate: value.rate,
              weighted: value.weighted,
              detail: `${value.numerator}/${value.denominator}`,
            }))
          )}
          keyField="pipeline"
          renderCell={(row, col) => {
            if (col === "rate" || col === "weighted") return (row as any)[col]?.toFixed(4) ?? "—";
            return (row as any)[col] ?? "—";
          }}
        />
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