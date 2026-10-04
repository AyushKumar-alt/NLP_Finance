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

      {/* Domain Phrases Highlight */}
      <SectionCard title="Representative Financial & Economic Domain Phrases">
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
          {[
            { phrase: "monetary policy", category: "Central Banking", n: 2 },
            { phrase: "repo rate", category: "Policy Interest Rate", n: 2 },
            { phrase: "current account deficit", category: "External Sector", n: 3 },
            { phrase: "financial stability", category: "Macroprudential", n: 2 },
            { phrase: "gross domestic product", category: "National Accounts", n: 3 },
            { phrase: "capital adequacy ratio", category: "Banking Health", n: 3 },
            { phrase: "foreign direct investment", category: "Capital Inflow", n: 3 },
            { phrase: "scheduled commercial banks", category: "Financial Sector", n: 3 },
          ].map((item, idx) => (
            <div
              key={idx}
              onClick={() => {
                setParams((p) => ({ ...p, n: item.n, search: item.phrase, offset: 0 }));
              }}
              className="p-3 bg-blue-50/60 hover:bg-blue-100/70 border border-blue-200 rounded-lg cursor-pointer transition-colors"
            >
              <div className="font-mono text-xs font-bold text-blue-900 truncate">{item.phrase}</div>
              <div className="flex justify-between items-center mt-1.5 text-[10px] text-gray-500">
                <span>{item.category}</span>
                <span className="bg-white px-1.5 py-0.5 rounded border border-blue-200 font-semibold">{item.n}-gram</span>
              </div>
            </div>
          ))}
        </div>
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
            { key: "rank", header: "#", width: "60px" },
            { key: "ngram", header: "N-gram" },
            { key: "frequency", header: "Frequency" },
            { key: "document_frequency", header: "Doc Freq" },
          ]}
          rows={(data.items ?? []).map((item: any, idx: number) => ({
            ...item,
            rank: (params.offset ?? 0) + idx + 1,
          }))}
          keyField="ngram"
          renderCell={(row, col) => {
            if (col === "rank") return <span className="text-gray-500 font-mono text-xs">{row.rank}</span>;
            if (col === "frequency" || col === "document_frequency") return Number(row[col])?.toLocaleString() ?? "—";
            if (col === "ngram") return <code className="text-sm font-semibold text-blue-900 bg-blue-50 px-1 py-0.5 rounded">{row.ngram ?? "—"}</code>;
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
        <SectionCard title="N-gram Comparison (Raw vs Stopword-Filtered)">
          <DataTable
            columns={[
              { key: "n", header: "N", width: "60px" },
              { key: "total_ngrams", header: "Total N-grams" },
              { key: "unique_ngrams", header: "Unique N-grams" },
              { key: "singletons", header: "Singletons" },
              { key: "top_ngram", header: "Top N-gram" },
              { key: "top_ngram_frequency", header: "Top Frequency" },
              { key: "meaningful_domain_phrases", header: "Domain Phrases" },
            ]}
            rows={data.comparison}
            keyField={(row: any, i: number) => `${row.n}_${i}`}
            renderCell={(row, col) => {
              if (col === "total_ngrams" || col === "unique_ngrams" || col === "singletons" || col === "top_ngram_frequency" || col === "meaningful_domain_phrases") {
                return Number(row[col])?.toLocaleString() ?? "—";
              }
              if (col === "top_ngram") return <code className="text-xs bg-gray-100 px-1 py-0.5 rounded font-mono">{row.top_ngram ?? "—"}</code>;
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