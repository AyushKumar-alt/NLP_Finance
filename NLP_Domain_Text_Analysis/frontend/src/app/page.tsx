/** Home page: project overview and quick links. */

"use client";

import React, { Suspense, useEffect, useState } from "react";
import { Layout, PageContainer, MetricCard, SectionCard, Loading, ErrorState } from "@/components";
import { api } from "@/lib/api";

function HomeContent() {
  const [health, setHealth] = useState<any>(null);
  const [stats, setStats] = useState<any>(null);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    let cancelled = false;
    Promise.all([api.health(), api.statistics()])
      .then(([h, s]) => {
        if (!cancelled) {
          setHealth(h);
          setStats(s);
        }
      })
      .catch((e) => !cancelled && setError(e));
    return () => { cancelled = true; };
  }, []);

  if (error) return <ErrorState error={error} retry={() => window.location.reload()} />;
  if (!health || !stats) return <Loading message="Loading project overview…" />;

  const phases = health.phases;
  const phase4 = stats.phase4;

  return (
    <PageContainer title="NLP Domain Text Analysis" description="Four-phase pipeline for Indian financial and economic document analysis.">
      {/* Phase status */}
      <SectionCard title="Pipeline Status">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            { key: "phase1", label: "Phase 1: Ingestion" },
            { key: "phase2", label: "Phase 2: Experiments" },
            { key: "phase3", label: "Phase 3: Retrieval" },
            { key: "phase4", label: "Phase 4: Evaluation" },
          ].map((p) => (
            <div key={p.key} className="text-center p-4 rounded-lg bg-gray-50">
              <div className={`text-3xl font-bold ${phases[p.key] ? "text-green-600" : "text-gray-300"}`}>
                {phases[p.key] ? "✓" : "○"}
              </div>
              <div className="text-sm text-gray-600 mt-1">{p.label}</div>
            </div>
          ))}
        </div>
      </SectionCard>

      {/* Corpus metrics */}
      <SectionCard title="Corpus at a Glance">
        <div className="grid grid-cols-2 md:grid-cols-6 gap-4">
          <MetricCard label="Documents" value={stats.corpus.documents} />
          <MetricCard label="Pages" value={stats.corpus.pages.toLocaleString()} />
          <MetricCard label="Units" value={stats.corpus.units.toLocaleString()} />
          <MetricCard label="Selected Units" value={stats.corpus.selected_units?.toLocaleString() ?? "—"} />
          <MetricCard label="Baseline Tokens" value={stats.corpus.baseline_tokens?.toLocaleString() ?? "—"} />
          <MetricCard label="Sources" value={stats.corpus.sources} />
        </div>
      </SectionCard>

      {/* Phase 3 & 4 highlights */}
      <div className="grid md:grid-cols-2 gap-6">
        <SectionCard title="Retrieval (Phase 3)">
          <dl className="space-y-3 text-sm">
            <div className="flex justify-between">
              <dt className="text-gray-500">Final pipeline</dt>
              <dd className="font-medium">{stats.pipelines.final_pipeline_name ?? "—"}</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-gray-500">Index terms</dt>
              <dd className="font-medium">{stats.index.terms?.toLocaleString() ?? "—"}</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-gray-500">Postings</dt>
              <dd className="font-medium">{stats.index.postings?.toLocaleString() ?? "—"}</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-gray-500">Queries answered</dt>
              <dd className="font-medium">{stats.phase3.queries_answered ?? "—"} / {stats.phase3.queries_defined ?? "—"}</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-gray-500">Avg. latency</dt>
              <dd className="font-medium">{stats.phase3.mean_execution_time_ms?.toFixed(1) ?? "—"} ms</dd>
            </div>
          </dl>
        </SectionCard>

        <SectionCard title="Evaluation (Phase 4)">
          {phase4.available ? (
            <dl className="space-y-3 text-sm">
              <div className="flex justify-between">
                <dt className="text-gray-500">Judgment pairs</dt>
                <dd className="font-medium">{phase4.judgments?.pairs?.toLocaleString() ?? "—"}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-gray-500">Pool depth</dt>
                <dd className="font-medium">{phase4.judgments?.pool_depth ?? "—"}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-gray-500">Unit-level F1 (macro)</dt>
                <dd className="font-medium text-green-600">
                  {(phase4.unit_level?.f1_macro ?? 0).toFixed(4)}
                </dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-gray-500">P@10 (macro)</dt>
                <dd className="font-medium text-blue-600">
                  {(phase4.unit_level?.precision_at_10_macro ?? 0).toFixed(4)}
                </dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-gray-500">Validation</dt>
                <dd className="font-medium">
                  {phase4.validation?.passed ?? 0} / {phase4.validation?.rules ?? 0} passed
                </dd>
              </div>
            </dl>
          ) : (
            <p className="text-gray-500">Phase 4 artefacts not available. Run <code className="bg-gray-100 px-1 rounded">python -m src.phase4.run</code>.</p>
          )}
        </SectionCard>
      </div>

      {/* Quick links */}
      <SectionCard title="Quick Links">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {[
            { href: "/documents", label: "Browse Documents" },
            { href: "/search", label: "Search Corpus" },
            { href: "/index", label: "Explore Index" },
            { href: "/evaluation", label: "Evaluation Report" },
          ].map((l) => (
            <a
              key={l.href}
              href={l.href}
              className="block p-4 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors text-center"
            >
              <div className="font-medium text-gray-900">{l.label}</div>
            </a>
          ))}
        </div>
      </SectionCard>
    </PageContainer>
  );
}

export default function HomePage() {
  return (
    <Layout>
      <Suspense fallback={<Loading />}>
        <HomeContent />
      </Suspense>
    </Layout>
  );
}