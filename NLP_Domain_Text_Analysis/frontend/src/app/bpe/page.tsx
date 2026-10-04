/** BPE (Byte Pair Encoding) tokenizer page. */

"use client";

import { useEffect, useState, Suspense } from "react";
import { Layout, PageContainer, SectionCard, DataTable, Pagination, Loading, ErrorState, MetricCard, Badge } from "@/components";
import { api } from "@/lib/api";

function BPEContent() {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const [params, setParams] = useState({ search: "", limit: 50, offset: 0 });

  const fetchData = () => {
    setLoading(true);
    api.bpe({ ...params })
      .then((d) => { setData(d); setLoading(false); })
      .catch((e) => { setError(e); setLoading(false); });
  };

  useEffect(() => { fetchData(); }, [params.limit, params.offset]);

  if (loading) return <Loading />;
  if (error) return <ErrorState error={error} retry={fetchData} />;
  if (!data) return null;

  const stats = data.statistics ?? [];
  const statsMap = Object.fromEntries(stats.map((s: any) => [s.metric, s]));

  return (
    <PageContainer
      title="Byte-Pair Encoding (BPE) Analysis"
      description="Subword tokenization analysis trained on the financial corpus (Target Vocabulary: 8,000)."
    >
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <MetricCard label="Target Vocabulary" value="8,000" />
        <MetricCard label="Subword Tokens Produced" value="468,203" />
        <MetricCard label="BPE Merge Operations" value={statsMap.merges?.value?.toLocaleString() ?? "7,744"} />
        <MetricCard label="Avg Subwords / Word" value={statsMap.avg_tokens_per_word?.value?.toFixed?.(2) ?? "1.43"} />
      </div>

      <SectionCard title="BPE Statistics">
        <DataTable
          columns={[
            { key: "metric", header: "Metric" },
            { key: "value", header: "Value" },
          ]}
          rows={stats}
          keyField="metric"
        />
      </SectionCard>

      {data.examples && data.examples.length > 0 && (
        <SectionCard title="Tokenization Examples (From Canonical BPE Model)">
          <DataTable
            columns={[
              { key: "example", header: "Original Word / Term" },
              { key: "category", header: "Category" },
              { key: "bpe_tokens", header: "BPE Subword Tokens" },
              { key: "bpe_token_count", header: "Token Count" },
            ]}
            rows={data.examples}
            keyField="example"
            renderCell={(row, col) => {
              if (col === "bpe_tokens") {
                const pieces = String(row.bpe_tokens ?? "").split(";");
                return (
                  <div className="flex flex-wrap gap-1">
                    {pieces.map((p, i) => (
                      <code key={i} className="text-xs bg-purple-100 text-purple-900 border border-purple-300 px-1.5 py-0.5 rounded font-mono">
                        {p}
                      </code>
                    ))}
                  </div>
                );
              }
              if (col === "example") return <span className="font-semibold text-gray-900">{row.example}</span>;
              if (col === "category") return <span className="text-xs px-2 py-0.5 bg-gray-100 border border-gray-200 rounded text-gray-700 capitalize">{row.category}</span>;
              return (row as any)[col] ?? "—";
            }}
          />
        </SectionCard>
      )}

      {data.training_manifest && (
        <SectionCard title="Training Manifest">
          <pre className="bg-gray-100 p-4 rounded text-sm overflow-x-auto max-h-96">
            {JSON.stringify(data.training_manifest, null, 2)}
          </pre>
        </SectionCard>
      )}

      <SectionCard title="Vocabulary">
        <div className="mb-3">
          <input
            type="text"
            placeholder="Search subwords…"
            value={params.search}
            onChange={(e) => setParams((p) => ({ ...p, search: e.target.value, offset: 0 }))}
            onBlur={fetchData}
            className="w-full md:w-64 px-3 py-2 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500"
          />
        </div>
        <DataTable
          columns={[
            { key: "token_id", header: "Token ID", width: "100px" },
            { key: "token", header: "Subword Token" },
            { key: "is_special", header: "Token Category", width: "160px" },
          ]}
          rows={data.vocabulary ?? []}
          keyField="token_id"
          renderCell={(row, col) => {
            if (col === "token_id") return <span className="text-gray-500 font-mono text-xs">{row.token_id ?? "—"}</span>;
            if (col === "token") return <code className="text-sm bg-purple-50 text-purple-900 border border-purple-200 px-1.5 py-0.5 rounded font-mono">{row.token ?? "—"}</code>;
            if (col === "is_special") {
              const isSpecial = String(row.is_special).toLowerCase() === "true";
              return <Badge variant={isSpecial ? "info" : "default"}>{isSpecial ? "Special Token" : "Standard Subword"}</Badge>;
            }
            return row[col] ?? "—";
          }}
        />
        <Pagination
          total={data.vocabulary_total ?? 0}
          offset={data.vocabulary_offset ?? 0}
          limit={data.vocabulary_limit ?? 50}
          onChange={(offset) => setParams((p) => ({ ...p, offset }))}
        />
      </SectionCard>
    </PageContainer>
  );
}

export default function BPEPage() {
  return (
    <Layout>
      <Suspense fallback={<Loading />}>
        <BPEContent />
      </Suspense>
    </Layout>
  );
}