/** BPE (Byte Pair Encoding) tokenizer page. */

"use client";

import { useEffect, useState, Suspense } from "react";
import { Layout, PageContainer, SectionCard, DataTable, Pagination, Loading, ErrorState, MetricCard } from "@/components";
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
    <PageContainer title="BPE Tokenizer" description="Byte Pair Encoding trained on the corpus, with statistics and vocabulary.">
      <div className="grid md:grid-cols-4 gap-6 mb-6">
        <MetricCard label="Vocabulary Size" value={statsMap.vocabulary_size?.value?.toLocaleString() ?? "—"} />
        <MetricCard label="Merges" value={statsMap.merges?.value?.toLocaleString() ?? "—"} />
        <MetricCard label="Tokens Produced" value={statsMap.total_tokens?.value?.toLocaleString() ?? "—"} />
        <MetricCard label="Avg Tokens/Word" value={statsMap.avg_tokens_per_word?.value?.toFixed?.(2) ?? "—"} />
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
        <SectionCard title="Tokenization Examples">
          <DataTable
            columns={[
              { key: "text", header: "Original" },
              { key: "tokens", header: "BPE Tokens" },
              { key: "token_count", header: "Token Count" },
            ]}
            rows={data.examples.slice(0, 20)}
            keyField="text"
            renderCell={(row, col) => {
              if (col === "tokens") return <code className="text-sm bg-gray-100 px-1 rounded">{row.tokens?.join(" ") ?? "—"}</code>;
              if (col === "text") return <code className="text-sm">{row.text ?? "—"}</code>;
              return row[col] ?? "—";
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
            { key: "token", header: "Subword" },
            { key: "rank", header: "Rank" },
            { key: "frequency", header: "Frequency" },
          ]}
          rows={data.vocabulary ?? []}
          keyField="token"
          renderCell={(row, col) => {
            if (col === "token") return <code className="text-sm bg-gray-50 px-1 rounded">{row.token ?? "—"}</code>;
            if (col === "frequency" || col === "rank") return row[col]?.toLocaleString() ?? "—";
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