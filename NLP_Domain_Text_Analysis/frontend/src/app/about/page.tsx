/** About page: project summary. */

"use client";

import { useEffect, useState, Suspense } from "react";
import { Layout, PageContainer, SectionCard, Loading, ErrorState, Badge } from "@/components";
import { api } from "@/lib/api";

function AboutContent() {
  const [stats, setStats] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    api.statistics().then(setStats).catch(setError).finally(() => setLoading(false));
  }, []);

  if (loading) return <Loading />;
  if (error) return <ErrorState error={error} />;
  if (!stats) return null;

  const val = stats.validation;
  const p3 = stats.phase3;
  const p4 = stats.phase4;

  return (
    <PageContainer title="About" description="Project overview, methodology, and provenance.">
      <SectionCard title="Project">
        <dl className="space-y-3 text-sm">
          <div className="flex justify-between"><dt className="text-gray-500">Name</dt><dd>NLP Domain Text Analysis</dd></div>
          <div className="flex justify-between"><dt className="text-gray-500">Domain</dt><dd>Indian financial and economic documents</dd></div>
          <div className="flex justify-between"><dt className="text-gray-500">Phases</dt><dd>1 (Ingestion) → 2 (Experiments) → 3 (Retrieval) → 4 (Evaluation)</dd></div>
          <div className="flex justify-between"><dt className="text-gray-500">Data sources</dt><dd>{stats.sources.length} sources</dd></div>
        </dl>
      </SectionCard>

      <SectionCard title="Architecture">
        <ol className="space-y-3 text-sm list-decimal list-inside">
          <li><strong>Python is the source of truth.</strong> Each phase writes CSV/JSON artefacts to results/. No metric is recomputed in TypeScript.</li>
          <li><strong>FastAPI backend serves the artefacts.</strong> POST /api/search calls src.phase3.retrieval.RetrievalEngine directly, so the API and the Phase 3 CLI cannot disagree on rankings.</li>
          <li><strong>Next.js is presentation only.</strong> It renders the API response, caches nothing, and never implements tokenization, search, or evaluation itself.</li>
          <li><strong>One write path.</strong> The only mutation is POST /api/evaluation/judgments, which appends a human label through src.phase4.relevance.RelevanceStore.</li>
        </ol>
      </SectionCard>

      <SectionCard title="Phase 3 Selection">
        <dl className="space-y-3 text-sm">
          <div className="flex justify-between"><dt className="text-gray-500">Final pipeline</dt><dd><Badge variant="success">{p3.final_pipeline_name}</Badge></dd></div>
          <div className="flex justify-between"><dt className="text-gray-500">Decision rule</dt><dd>Weighted score over four criteria (financial-expression preservation, domain-term recall, variant-collapse rate, query answerability)</dd></div>
        </dl>
      </SectionCard>

      <SectionCard title="Phase 4 Evaluation Design">
        <dl className="space-y-3 text-sm">
          <div className="flex justify-between"><dt className="text-gray-500">Judged pipeline</dt><dd>Pipeline B only</dd></div>
          <div className="flex justify-between"><dt className="text-gray-500">Judgment type</dt><dd><Badge variant="info">pooled manual judgment</Badge></dd></div>
          <div className="flex justify-between"><dt className="text-gray-500">Pool depth</dt><dd>{p4.judgments?.pool_depth}</dd></div>
          <div className="flex justify-between"><dt className="text-gray-500">Pairs</dt><dd>{p4.judgments?.relevant} relevant / {p4.judgments?.not_relevant} not relevant / {p4.judgments?.pairs} total</dd></div>
          <div className="flex justify-between"><dt className="text-gray-500">Recall is</dt><dd>Pool-bounded (measured against the judgment pool, not the full corpus)</dd></div>
          <div className="flex justify-between"><dt className="text-gray-500">Pipeline A</dt><dd>Retrieval statistics only; relevance metrics are N/A</dd></div>
        </dl>
      </SectionCard>

      <SectionCard title="Validation Status">
        <div className="flex gap-4">
          <Badge variant={val.all_passed ? "success" : "error"}>
            {val.passed}/{val.total} rules passed
          </Badge>
        </div>
      </SectionCard>

      <SectionCard title="Reproducibility">
        <p className="text-sm text-gray-600">To regenerate all artefacts:</p>
        <pre className="bg-gray-100 p-4 rounded text-sm overflow-x-auto">
          python -m src.phase1.run{`\n`}
          python -m src.phase2.run{`\n`}
          python -m src.phase3.run{`\n`}
          python -m src.phase4.run
        </pre>
        <p className="text-sm text-gray-600 mt-2">
          To start the backend: `python -m uvicorn backend.main:app --reload --port 8000`
        </p>
        <p className="text-sm text-gray-600">
          The frontend reads the API from <code className="bg-gray-100 px-1 rounded">NEXT_PUBLIC_API_BASE_URL</code>.
        </p>
      </SectionCard>

      {p4.judgments?.rubric && (
        <SectionCard title="Relevance Judgment Rubric">
          <pre className="bg-gray-50 p-4 rounded text-sm whitespace-pre-wrap leading-relaxed">
            {p4.judgments.rubric}
          </pre>
        </SectionCard>
      )}
    </PageContainer>
  );
}

export default function AboutPage() {
  return (
    <Layout>
      <Suspense fallback={<Loading />}>
        <AboutContent />
      </Suspense>
    </Layout>
  );
}