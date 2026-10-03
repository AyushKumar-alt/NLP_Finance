/** Preprocessing page: cleaning, stopwords, and order experiments. */

"use client";

import { useEffect, useState, Suspense } from "react";
import { Layout, PageContainer, SectionCard, DataTable, Loading, ErrorState, Badge } from "@/components";
import { api } from "@/lib/api";

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
    <PageContainer title="Preprocessing" description="Cleaning stages, stopword strategies, and stopword/morphology order comparison.">
      <SectionCard title="Cleaning Stages">
        <DataTable
          columns={[
            { key: "stage", header: "Stage" },
            { key: "description", header: "Description" },
            { key: "tokens_before", header: "Tokens Before" },
            { key: "tokens_after", header: "Tokens After" },
            { key: "reduction_pct", header: "Reduction %" },
          ]}
          rows={data.stages ?? []}
          keyField="stage"
          renderCell={(row, col) => {
            if (col === "tokens_before" || col === "tokens_after") return row[col]?.toLocaleString() ?? "—";
            if (col === "reduction_pct") return row[col]?.toFixed(2) ?? "—";
            return row[col] ?? "—";
          }}
        />
      </SectionCard>

      {data.stopwords && data.stopwords.length > 0 && (
        <SectionCard title="Stopword Strategy Comparison">
          <DataTable
            columns={[
              { key: "strategy", header: "Strategy" },
              { key: "tokens_removed", header: "Removed" },
              { key: "vocabulary_reduction", header: "Vocab Reduction" },
              { key: "domain_terms_preserved", header: "Domain Preserved" },
            ]}
            rows={data.stopwords}
            keyField="strategy"
            renderCell={(row, col) => {
              if (col === "tokens_removed" || col === "vocabulary_reduction") return row[col]?.toLocaleString() ?? "—";
              return row[col] ?? "—";
            }}
          />
        </SectionCard>
      )}

      {data.before_after_examples && data.before_after_examples.length > 0 && (
        <SectionCard title="Before / After Examples">
          <DataTable
            columns={[
              { key: "stage", header: "Stage" },
              { key: "before", header: "Before" },
              { key: "after", header: "After" },
            ]}
            rows={data.before_after_examples.slice(0, 20)}
            keyField="stage"
            renderCell={(row, col) => {
              if (col === "before" || col === "after") return <code className="text-sm bg-gray-100 px-1 rounded">{row[col] ?? "—"}</code>;
              return row[col] ?? "—";
            }}
          />
        </SectionCard>
      )}

      {data.domain_stopword_analysis && data.domain_stopword_analysis.length > 0 && (
        <SectionCard title="Domain Stopword Analysis">
          <DataTable
            columns={[
              { key: "term", header: "Term" },
              { key: "frequency", header: "Frequency" },
              { key: "domain_score", header: "Domain Score" },
              { key: "decision", header: "Decision" },
            ]}
            rows={data.domain_stopword_analysis}
            keyField="term"
            renderCell={(row, col) => {
              if (col === "decision") return <Badge variant={row.decision === "keep" ? "success" : row.decision === "remove" ? "error" : "warning"}>{row.decision}</Badge>;
              if (col === "frequency") return row[col]?.toLocaleString() ?? "—";
              if (col === "domain_score") return row[col]?.toFixed(4) ?? "—";
              return row[col] ?? "—";
            }}
          />
        </SectionCard>
      )}

      {data.protected_terms && data.protected_terms.length > 0 && (
        <SectionCard title="Protected Terms (Never Removed)">
          <DataTable
            columns={[
              { key: "term", header: "Term" },
              { key: "reason", header: "Reason" },
            ]}
            rows={data.protected_terms}
            keyField="term"
          />
        </SectionCard>
      )}

      {data.final_stopwords_text && (
        <SectionCard title="Final Stopword List">
          <pre className="bg-gray-100 p-4 rounded text-sm overflow-x-auto max-h-96">
            {data.final_stopwords_text}
          </pre>
        </SectionCard>
      )}

      <SectionCard title="Stopword / Morphology Order Explanation">
        <p className="text-gray-700 whitespace-pre-wrap">{data.order_explanation ?? "Phase 2 ran the order experiment and Phase 3 turned it into Pipeline A and Pipeline B: Pipeline A removes domain-aware stopwords before lemmatizing, Pipeline B stems first and matches stopwords on stems."}</p>
      </SectionCard>
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