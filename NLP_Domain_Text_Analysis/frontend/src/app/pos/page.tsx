/** POS tagging page. */

"use client";

import { useEffect, useState, Suspense } from "react";
import { Layout, PageContainer, SectionCard, DataTable, Loading, ErrorState, Badge, MetricCard } from "@/components";
import { api } from "@/lib/api";

function POSContent() {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    api.pos().then(setData).catch(setError).finally(() => setLoading(false));
  }, []);

  if (loading) return <Loading />;
  if (error) return <ErrorState error={error} />;
  if (!data) return null;

  const dist = data.distribution ?? [];
  const coarse = data.coarse ?? [];
  const gold = data.gold_set ?? {};

  return (
    <PageContainer title="POS Tagging" description="Default POS distribution, custom dictionary, and ML comparison against gold annotations.">
      <div className="grid md:grid-cols-2 gap-6 mb-6">
        <SectionCard title="Default POS Distribution (Fine-Grained)">
          <DataTable
            columns={[
              { key: "pos", header: "Tag" },
              { key: "count", header: "Count" },
              { key: "pct", header: "%" },
            ]}
            rows={dist}
            keyField="pos"
            renderCell={(row, col) => {
              if (col === "count") return row.count?.toLocaleString() ?? "—";
              if (col === "pct") return row.pct?.toFixed(2) ?? "—";
              return <Badge variant="default">{row[col] ?? "—"}</Badge>;
            }}
          />
        </SectionCard>

        <SectionCard title="Coarse POS Distribution">
          <DataTable
            columns={[
              { key: "coarse_pos", header: "Coarse Tag" },
              { key: "count", header: "Count" },
              { key: "pct", header: "%" },
            ]}
            rows={coarse}
            keyField="coarse_pos"
            renderCell={(row, col) => {
              if (col === "count") return row.count?.toLocaleString() ?? "—";
              if (col === "pct") return row.pct?.toFixed(2) ?? "—";
              return <Badge variant="info">{row[col] ?? "—"}</Badge>;
            }}
          />
        </SectionCard>
      </div>

      {data.examples && data.examples.length > 0 && (
        <SectionCard title="POS Tagging Examples">
          <DataTable
            columns={[
              { key: "text", header: "Text" },
              { key: "tokens", header: "Tokens" },
              { key: "tags", header: "POS Tags" },
            ]}
            rows={data.examples.slice(0, 20)}
            keyField="text"
            renderCell={(row, col) => {
              if (col === "tokens" || col === "tags") return <code className="text-sm bg-gray-100 px-1 rounded">{row[col]?.join(" ") ?? "—"}</code>;
              return <code className="text-sm">{row[col] ?? "—"}</code>;
            }}
          />
        </SectionCard>
      )}

      {data.dictionary && data.dictionary.length > 0 && (
        <SectionCard title="Custom POS Dictionary">
          <DataTable
            columns={[
              { key: "token", header: "Token" },
              { key: "pos", header: "Assigned POS" },
              { key: "source", header: "Source" },
            ]}
            rows={data.dictionary}
            keyField="token"
            renderCell={(row, col) => {
              if (col === "pos") return <Badge variant="info">{row.pos ?? "—"}</Badge>;
              return row[col] ?? "—";
            }}
          />
        </SectionCard>
      )}

      {data.changes && data.changes.length > 0 && (
        <SectionCard title="Custom POS Changes (vs Default)">
          <DataTable
            columns={[
              { key: "token", header: "Token" },
              { key: "default_pos", header: "Default POS" },
              { key: "custom_pos", header: "Custom POS" },
              { key: "reason", header: "Reason" },
            ]}
            rows={data.changes}
            keyField="token"
            renderCell={(row, col) => {
              if (col === "default_pos" || col === "custom_pos") return <Badge variant="default">{row[col] ?? "—"}</Badge>;
              return row[col] ?? "—";
            }}
          />
        </SectionCard>
      )}

      {data.ml_results && data.ml_results.length > 0 && (
        <SectionCard title="ML POS Tagger Results">
          <DataTable
            columns={[
              { key: "model", header: "Model" },
              { key: "accuracy", header: "Accuracy" },
              { key: "f1_macro", header: "F1 Macro" },
              { key: "f1_micro", header: "F1 Micro" },
            ]}
            rows={data.ml_results}
            keyField="model"
            renderCell={(row, col) => {
              if (col === "accuracy" || col === "f1_macro" || col === "f1_micro") {
                const v = Number(row[col]);
                return isNaN(v) ? "—" : v.toFixed(4);
              }
              return row[col] ?? "—";
            }}
          />
        </SectionCard>
      )}

      {data.ml_report && data.ml_report.length > 0 && (
        <SectionCard title="ML Classification Report">
          <DataTable
            columns={[
              { key: "class", header: "Class" },
              { key: "precision", header: "Precision" },
              { key: "recall", header: "Recall" },
              { key: "f1", header: "F1" },
              { key: "support", header: "Support" },
            ]}
            rows={data.ml_report}
            keyField="class"
            renderCell={(row, col) => {
              if (col === "precision" || col === "recall" || col === "f1") {
                const v = Number(row[col]);
                return isNaN(v) ? "—" : v.toFixed(4);
              }
              if (col === "support") return Number(row[col])?.toLocaleString() ?? "—";
              return row[col] ?? "—";
            }}
          />
        </SectionCard>
      )}

      {data.ml_misclassifications && data.ml_misclassifications.length > 0 && (
        <SectionCard title="ML Misclassifications">
          <DataTable
            columns={[
              { key: "token", header: "Token" },
              { key: "true_pos", header: "True POS" },
              { key: "pred_pos", header: "Predicted POS" },
              { key: "context", header: "Context" },
            ]}
            rows={data.ml_misclassifications.slice(0, 30)}
            keyField="token"
            renderCell={(row, col) => {
              if (col === "true_pos" || col === "pred_pos") return <Badge variant="default">{row[col] ?? "—"}</Badge>;
              if (col === "context") return <code className="text-sm">{row[col] ?? "—"}</code>;
              return row[col] ?? "—";
            }}
          />
        </SectionCard>
      )}

      {data.gold_alignment && data.gold_alignment.length > 0 && (
        <SectionCard title="Gold Annotation Alignment">
          <DataTable
            columns={[
              { key: "metric", header: "Metric" },
              { key: "value", header: "Value" },
            ]}
            rows={data.gold_alignment}
            keyField="metric"
          />
        </SectionCard>
      )}

      {gold.total_tokens && (
        <SectionCard title="Gold Standard Summary">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <MetricCard label="Total Tokens" value={gold.total_tokens?.toLocaleString() ?? "—"} />
            <MetricCard label="Annotated" value={gold.annotated_tokens?.toLocaleString() ?? "—"} />
            <MetricCard label="Agreement" value={gold.agreement_rate?.toFixed(4) ?? "—"} />
            <MetricCard label="Annotators" value={gold.annotator_count ?? "—"} />
          </div>
        </SectionCard>
      )}

      {data.comparison && data.comparison.length > 0 && (
        <SectionCard title="POS Tagger Comparison">
          <DataTable
            columns={[
              { key: "tagger", header: "Tagger" },
              { key: "accuracy", header: "Accuracy" },
              { key: "speed", header: "Speed (tokens/s)" },
            ]}
            rows={data.comparison}
            keyField="tagger"
            renderCell={(row, col) => {
              if (col === "accuracy") return row[col]?.toFixed(4) ?? "—";
              if (col === "speed") return row[col]?.toLocaleString() ?? "—";
              return row[col] ?? "—";
            }}
          />
        </SectionCard>
      )}
    </PageContainer>
  );
}

export default function POSPage() {
  return (
    <Layout>
      <Suspense fallback={<Loading />}>
        <POSContent />
      </Suspense>
    </Layout>
  );
}