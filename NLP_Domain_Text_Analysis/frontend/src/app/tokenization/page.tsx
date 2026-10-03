/** Tokenization page: Phase 2 tokenizer comparison. */

"use client";

import { useEffect, useState, Suspense } from "react";
import { Layout, PageContainer, SectionCard, DataTable, Loading, ErrorState, Badge, MetricCard } from "@/components";
import { api } from "@/lib/api";

function TokenizationContent() {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    api.tokenization().then(setData).catch(setError).finally(() => setLoading(false));
  }, []);

  if (loading) return <Loading />;
  if (error) return <ErrorState error={error} />;
  if (!data) return null;

  const tokenizers = data.tokenizers ?? [];

  return (
    <PageContainer title="Tokenization" description="Comparison of four tokenizers on the financial corpus.">
      <SectionCard title="Tokenizer Summary">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <MetricCard label="Tokenizers Compared" value={tokenizers.length} />
          <MetricCard label="Most Compact" value={data.most_compact_tokenizer ?? "—"} />
        </div>
        <DataTable
          columns={[
            { key: "tokenizer", header: "Tokenizer" },
            { key: "total_tokens", header: "Total Tokens" },
            { key: "vocabulary_size", header: "Vocabulary" },
            { key: "avg_tokens_per_document", header: "Avg Tokens/Doc" },
            { key: "execution_time_seconds", header: "Time (s)" },
            { key: "numeric_tokens", header: "Numeric" },
            { key: "date_tokens", header: "Dates" },
            { key: "financial_expression_preservation", header: "Fin. Expr." },
          ]}
          rows={tokenizers}
          keyField="tokenizer"
          renderCell={(row, col) => {
            if (col === "total_tokens" || col === "vocabulary_size") return row[col]?.toLocaleString() ?? "—";
            if (col === "avg_tokens_per_document" || col === "execution_time_seconds") return row[col]?.toFixed(2) ?? "—";
            if (col === "financial_expression_preservation") return <Badge variant={row.financial_expression_preservation === "1" ? "success" : "default"}>{row.financial_expression_preservation ?? "—"}</Badge>;
            return row[col] ?? "—";
          }}
        />
      </SectionCard>

      {data.examples && data.examples.length > 0 && (
        <SectionCard title="Representative Examples">
          <DataTable
            columns={[
              { key: "tokenizer", header: "Tokenizer" },
              { key: "text", header: "Original Text" },
              { key: "tokens", header: "Tokenized" },
            ]}
            rows={data.examples.slice(0, 20)}
            keyField="tokenizer"
            renderCell={(row, col) => {
              if (col === "tokens") return <code className="text-sm bg-gray-100 px-1 rounded">{row.tokens?.join(" | ") ?? "—"}</code>;
              if (col === "text") return <code className="text-sm">{row.text ?? "—"}</code>;
              return row[col] ?? "—";
            }}
          />
        </SectionCard>
      )}

      {data.custom_rules && data.custom_rules.length > 0 && (
        <SectionCard title="Custom Tokenizer Rules">
          <DataTable
            columns={[
              { key: "rule_name", header: "Rule" },
              { key: "pattern", header: "Pattern" },
              { key: "description", header: "Description" },
            ]}
            rows={data.custom_rules}
            keyField="rule_name"
          />
        </SectionCard>
      )}

      {data.date_number_comparison && data.date_number_comparison.length > 0 && (
        <SectionCard title="Date & Number Handling">
          <DataTable
            columns={[
              { key: "tokenizer", header: "Tokenizer" },
              { key: "category", header: "Category" },
              { key: "preserved", header: "Preserved" },
              { key: "lost", header: "Lost" },
            ]}
            rows={data.date_number_comparison}
            keyField="tokenizer"
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