/** N-grams page. */

"use client";

import { useEffect, useState, Suspense } from "react";
import { Layout, PageContainer, SectionCard, DataTable, Pagination, Loading, ErrorState, MetricCard } from "@/components";
import { api } from "@/lib/api";

const N_SIZES = [1, 2, 3, 4, 5] as const;

function NgramsContent() {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const [params, setParams] = useState({ n: 2, search: "", domain_only: false, limit: 50, offset: 0 });

  useEffect(() => {
    fetchData();
  }, [params.n, params.domain_only, params.limit, params.offset]);

  const fetchData = () => {
    setLoading(true);
    api.ngrams({ ...params })
      .then((d) => { setData(d); setLoading(false); })
      .catch((e) => { setError(e); setLoading(false); });
  };

  if (loading) return <Loading />;
  if (error) return <ErrorState error={error} retry={fetchData} />;
  if (!data) return null;

  const overview = data.all_sizes ?? [];

  return (
    <PageContainer title="N-grams" description="Unigram to five-gram frequencies, including domain phrases ">
      <SectionCard title="N-gram Overview">
        <DataTable
          columns={[
            { key: "n", header: "N" },
            { key: "total_ngrams", header: "Total" },
            { key: "unique_ngrams", header: "Unique" },
            { key: "singletons", header: "Singletons" },
            { key: "top_ngram", header: "Top N-gram" },
            { key: "top_ngram_frequency", header: "Top Freq." },
            { key: "top_content_ngram", header: "Top Content" },
            { key: "top_content_ngram_frequency", header: "Top Content Freq." },
          ]}
          rows={overview}
          keyField="n"
          renderCell={(row, col) => {
            if (col === "total_ngrams" || col === "unique_ngrams" || col === "singletons" || col === "top_ngram_frequency" || col === "top_content_ngram_frequency") return row[col]?.toLocaleString() ?? "—";
            return row[col] ?? "—";
          }}
        />
      </SectionCard>

      <SectionCard title={`Controls — viewing n=${params.n}`}>
        <div className="flex flex-wrap gap-6 items-end">
          <div className="space-y-1">
            <label className="text-sm font-medium">N</label>
            <div className="flex gap-2">
              {N_SIZES.map((n) => (
                <button
                  key={n}
                  onClick={() => setParams((p) => ({ ...p, n, offset: 0 }))}
                  className={`px-3 py-1 rounded border ${params.n === n ? "bg-blue-600 text-white" : "bg-gray-100"}`}
                >
                  {n}-gram
                </button>
              ))}
            </div>
          </div>
          <div className="space-y-1 flex-1 min-w-64">
            <label className="text-sm font-medium">Search</label>
            <input
              type="text"
              value={params.search}
              onChange={(e) => setParams((p) => ({ ...p, search: e.target.value, offset: 0 }))}
              onBlur={fetchData}
              placeholder="Filter n-grams containing this text…"
              className="w-full px-3 py-2 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              id="domain_only"
              checked={params.domain_only}
              onChange={() => setParams((p) => ({ ...p, domain_only: !p.domain_only, offset: 0 }))}
              className="text-blue-600"
            />
            <label htmlFor="domain_only" className="text-sm">Domain phrases only</label>
          </div>
        </div>
      </SectionCard>

      <SectionCard title={`Top ${params.n}-grams (${data.total?.toLocaleString() ?? 0} total)`}>
        <DataTable
          columns={[
            { key: "rank", header: "#" },
            { key: "ngram", header: "N-gram" },
            { key: "frequency", header: "Frequency" },
            { key: "document_frequency", header: "Doc Freq" },
          ]}
          rows={data.items ?? []}
          keyField="ngram"
          renderCell={(row, col) => {
            if (col === "frequency" || col === "document_frequency") return row[col]?.toLocaleString() ?? "—";
            if (col === "ngram") return <code className="text-sm">{row.ngram ?? "—"}</code>;
            return row[col] ?? "—";
          }}
        />
        <Pagination
          total={data.total ?? 0}
          offset={data.offset ?? 0}
          limit={data.limit ?? 50}
          onChange={(offset) => {
            setParams((p) => ({ ...p, offset }));
            fetchData();
          }}
        />
      </SectionCard>

      {data.comparison && data.comparison.length > 0 && (
        <SectionCard title="N-gram Comparison">
          <DataTable
            columns={[
              { key: "n", header: "N" },
              { key: "metric", header: "Metric" },
              { key: "value", header: "Value" },
            ]}
            rows={data.comparison}
            keyField="n"
            renderCell={(row, col) => {
              if (col === "value") return row[col]?.toFixed?.(4) ?? row[col] ?? "—";
              return row[col] ?? "—";
            }}
          />
        </SectionCard>
      )}
    </PageContainer>
  );
}

export default function NgramsPage() {
  return (
    <Layout>
      <Suspense fallback={<Loading />}>
        <NgramsContent />
      </Suspense>
    </Layout>
  );
}