/** Statistics page: aggregated view across all four phases. */

"use client";

import { useEffect, useState, Suspense } from "react";
import { Layout, PageContainer, MetricCard, SectionCard, Loading, ErrorState, Badge, DataTable } from "@/components";
import { api } from "@/lib/api";

function StatisticsContent() {
  const [stats, setStats] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    api.statistics().then(setStats).catch(setError).finally(() => setLoading(false));
  }, []);

  if (loading) return <Loading />;
  if (error) return <ErrorState error={error} />;
  if (!stats) return null;

  const c = stats.corpus;
  const i = stats.index;
  const p3 = stats.phase3;
  const p4 = stats.phase4;
  const val = stats.validation;

  return (
    <PageContainer title="Statistics" description="Aggregated metrics across all four phases.">
      {/* Corpus */}
      <SectionCard title="Corpus Overview">
        <div className="grid grid-cols-2 md:grid-cols-6 gap-4 mb-6">
          <MetricCard label="Documents" value={c.documents} />
          <MetricCard label="Sources" value={c.sources} />
          <MetricCard label="Pages" value={c.pages.toLocaleString()} />
          <MetricCard label="Units" value={c.units.toLocaleString()} />
          <MetricCard label="Selected" value={c.selected_units?.toLocaleString() ?? "—"} />
          <MetricCard label="Baseline Tokens" value={c.baseline_tokens?.toLocaleString() ?? "—"} />
        </div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <MetricCard label="Paragraphs" value={c.paragraphs?.toLocaleString() ?? "—"} />
          <MetricCard label="Tables" value={c.tables?.toLocaleString() ?? "—"} />
          <MetricCard label="Figures" value={c.figures?.toLocaleString() ?? "—"} />
          <MetricCard label="Footnotes" value={c.footnotes?.toLocaleString() ?? "—"} />
        </div>
        {c.units_by_type && (
          <div className="mt-4">
            <h4 className="font-medium text-gray-700 mb-2">Units by Type</h4>
            <DataTable
              columns={[{ key: "type", header: "Type" }, { key: "count", header: "Count" }]}
              rows={Object.entries(c.units_by_type).map(([type, count]) => ({ type, count }))}
              keyField="type"
            />
          </div>
        )}
      </SectionCard>

      {/* Index */}
      <SectionCard title="Inverted Index">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <MetricCard label="Final Pipeline" value={i.pipeline_name ?? "—"} />
          <MetricCard label="Terms" value={i.terms?.toLocaleString() ?? "—"} />
          <MetricCard label="Postings" value={i.postings?.toLocaleString() ?? "—"} />
          <MetricCard label="Indexed Units" value={i.units?.toLocaleString() ?? "—"} />
        </div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4">
          <MetricCard label="Postings/Term" value={i.postings_per_term?.toFixed(2) ?? "—"} />
          <MetricCard label="Singleton Terms" value={i.singleton_terms?.toLocaleString() ?? "—"} />
          <MetricCard label="Hapax Ratio" value={i.hapax_ratio?.toFixed(4) ?? "—"} />
          <MetricCard label="Index Size" value={(i.index_bytes / 1024 / 1024).toFixed(1)} unit="MB" />
        </div>
      </SectionCard>

      {/* Pipelines */}
      <SectionCard title="Pipeline Comparison">
        {stats.pipelines?.pipelines && (
          <DataTable
            columns={[
              { key: "pipeline", header: "Pipeline" },
              { key: "name", header: "Name" },
              { key: "total_score", header: "Score" },
              { key: "index_terms", header: "Index Terms" },
              { key: "vocabulary_size", header: "Vocab" },
              { key: "total_postings", header: "Postings" },
              { key: "is_final", header: "Final" },
            ]}
            rows={stats.pipelines.pipelines}
            keyField="pipeline"
            renderCell={(row, col) => {
              if (col === "is_final") return <Badge variant={row.is_final ? "success" : "default"}>{row.is_final ? "Yes" : "No"}</Badge>;
              if (col === "total_score") return row.total_score?.toFixed(6) ?? "—";
              if (col === "index_terms" || col === "vocabulary_size" || col === "total_postings") return row[col]?.toLocaleString() ?? "—";
              return row[col] ?? "—";
            }}
          />
        )}
      </SectionCard>

      {/* Phase 4 */}
      {p4.available && (
        <SectionCard title="Evaluation (Phase 4)">
          <div className="grid grid-cols-2 md:grid-cols-6 gap-4">
            <MetricCard label="Judgment Pairs" value={p4.judgments?.pairs?.toLocaleString() ?? "—"} />
            <MetricCard label="Relevant" value={p4.judgments?.relevant?.toLocaleString() ?? "—"} unit={<Badge variant="success">+</Badge>} />
            <MetricCard label="Not Relevant" value={p4.judgments?.not_relevant?.toLocaleString() ?? "—"} unit={<Badge variant="error">−</Badge>} />
            <MetricCard label="Pool Depth" value={p4.judgments?.pool_depth ?? "—"} />
            <MetricCard label="Queries Evaluated" value={p4.queries_evaluated ?? "—"} />
            <MetricCard label="Judged Depth" value={p4.judged_depth ?? "—"} />
          </div>
          <div className="grid grid-cols-2 md:grid-cols-6 gap-4 mt-4">
            <MetricCard label="P Macro" value={p4.unit_level?.precision_macro?.toFixed(4) ?? "—"} />
            <MetricCard label="R Macro" value={p4.unit_level?.recall_macro?.toFixed(4) ?? "—"} />
            <MetricCard label="F1 Macro" value={p4.unit_level?.f1_macro?.toFixed(4) ?? "—"} />
            <MetricCard label="P@10 Macro" value={p4.unit_level?.precision_at_10_macro?.toFixed(4) ?? "—"} />
            <MetricCard label="R@10 Macro" value={p4.unit_level?.recall_at_10_macro?.toFixed(4) ?? "—"} />
            <MetricCard label="Validation" value={`${val.passed} / ${val.total}`} unit={val.all_passed ? <Badge variant="success">All passed</Badge> : <Badge variant="warning">Issues</Badge>} />
          </div>
        </SectionCard>
      )}

      {/* Validation summary */}
      <SectionCard title="Validation Rules">
        <DataTable
          columns={[
            { key: "phase", header: "Phase" },
            { key: "total", header: "Rules" },
            { key: "passed", header: "Passed" },
            { key: "all_passed", header: "Status" },
          ]}
          rows={Object.entries(val.phases).map(([phase, block]: [string, any]) => ({
            phase: phase.toUpperCase(),
            total: block.total,
            passed: block.passed,
            all_passed: block.all_passed,
          }))}
          keyField="phase"
          renderCell={(row, col) => {
            if (col === "all_passed") return <Badge variant={row.all_passed ? "success" : "error"}>{row.all_passed ? "All passed" : `${row.total - row.passed} failed`}</Badge>;
            return (row as any)[col];
          }}
        />
      </SectionCard>
    </PageContainer>
  );
}

export default function StatisticsPage() {
  return (
    <Layout>
      <Suspense fallback={<Loading />}>
        <StatisticsContent />
      </Suspense>
    </Layout>
  );
}