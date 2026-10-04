/** Inverted Index Explorer page: inspect postings, term frequencies, and positions. */

"use client";

import { useEffect, useState, Suspense } from "react";
import { Layout, PageContainer, SectionCard, DataTable, Loading, ErrorState, Badge, MetricCard } from "@/components";
import { api, type TermLookupResponse } from "@/lib/api";

const SAMPLE_TERMS = [
  "inflation",
  "gdp",
  "policy",
  "rate",
  "repo",
  "credit",
  "stability",
  "deficit",
  "bank",
  "investment",
];

function IndexExplorerContent() {
  const [term, setTerm] = useState("inflation");
  const [activeTerm, setActiveTerm] = useState("inflation");
  const [data, setData] = useState<TermLookupResponse | null>(null);
  const [topTerms, setTopTerms] = useState<any[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const lookup = (targetTerm: string) => {
    const trimmed = targetTerm.trim();
    if (!trimmed) return;
    setLoading(true);
    setError(null);
    setActiveTerm(trimmed);
    api.term(trimmed, 50)
      .then((res) => {
        setData(res);
        setLoading(false);
      })
      .catch((err) => {
        setError(err);
        setLoading(false);
      });
  };

  useEffect(() => {
    lookup("inflation");
    api.terms({ limit: 25, order: "posting_frequency" })
      .then((res) => {
        if (res && (res.items || res.terms)) setTopTerms(res.items || res.terms);
      })
      .catch(console.error);
  }, []);

  return (
    <PageContainer
      title="Inverted Index Explorer"
      description="Inspect postings, frequencies, and token positions in the Phase 3 Positional Inverted Index."
    >
      {/* Lookup controls */}
      <SectionCard title="Term Lookup">
        <div className="space-y-4">
          <div className="flex flex-col md:flex-row gap-3">
            <input
              type="text"
              value={term}
              onChange={(e) => setTerm(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && lookup(term)}
              placeholder="Enter a financial term (e.g. inflation, gdp, repo, credit)..."
              className="flex-1 px-4 py-2 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500 font-mono text-sm"
            />
            <button
              onClick={() => lookup(term)}
              disabled={loading || !term.trim()}
              className="px-6 py-2 bg-blue-600 text-white rounded font-medium hover:bg-blue-700 disabled:opacity-50 transition-colors"
            >
              {loading ? "Looking up..." : "Lookup Term"}
            </button>
          </div>

          <div>
            <span className="text-xs text-gray-500 mr-2 font-medium">Quick Samples:</span>
            <div className="inline-flex flex-wrap gap-1.5 mt-1">
              {SAMPLE_TERMS.map((sample) => (
                <button
                  key={sample}
                  onClick={() => {
                    setTerm(sample);
                    lookup(sample);
                  }}
                  className={`px-2 py-0.5 text-xs rounded border transition-colors ${
                    activeTerm === sample
                      ? "bg-blue-100 text-blue-800 border-blue-300 font-semibold"
                      : "bg-gray-50 text-gray-700 border-gray-200 hover:bg-gray-100"
                  }`}
                >
                  {sample}
                </button>
              ))}
            </div>
          </div>
        </div>
      </SectionCard>

      {/* Loading state */}
      {loading && <Loading message={`Loading postings for "${activeTerm}"...`} />}

      {/* Error state */}
      {error && !loading && (
        <SectionCard title="Lookup Error">
          <ErrorState error={error} retry={() => lookup(activeTerm)} />
        </SectionCard>
      )}

      {/* Postings Results */}
      {data && !loading && (
        <>
          <SectionCard title={`Index Statistics for "${data.term}"`}>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-4">
              <MetricCard label="Surface Term" value={data.term} />
              <MetricCard label="Normalized Term" value={data.normalized_terms?.join(" ") || "—"} />
              <MetricCard label="Document Frequency" value={data.document_frequency?.toLocaleString() ?? "—"} />
              <MetricCard label="Posting Count (Units)" value={data.postings_total?.toLocaleString() ?? "—"} />
              <MetricCard label="Total Frequency (TF)" value={data.total_frequency?.toLocaleString() ?? "—"} />
            </div>

            <div className="flex flex-wrap gap-2 text-xs text-gray-600 bg-gray-50 p-3 rounded">
              <span className="font-semibold">Pipeline:</span>
              <Badge variant="success">{data.pipeline}</Badge>
              <span className="ml-3 font-semibold">Searchable:</span>
              <Badge variant={data.searchable ? "success" : "default"}>
                {data.searchable ? "Yes (In Index)" : "No (Out of Vocabulary)"}
              </Badge>
              {data.missing_terms && data.missing_terms.length > 0 && (
                <span className="text-amber-700 ml-2">
                  Missing components: {data.missing_terms.join(", ")}
                </span>
              )}
            </div>
          </SectionCard>

          {data.postings && data.postings.length > 0 ? (
            <SectionCard title={`Postings List (Showing ${data.postings.length} of ${data.postings_total ?? data.postings.length} matching units)`}>
              <div className="space-y-3">
                {data.postings.map((p, idx) => (
                  <div key={p.unit_id} className="border border-gray-200 rounded-lg p-3 hover:bg-gray-50 transition-colors">
                    <div className="flex flex-wrap items-center justify-between gap-2 mb-1.5">
                      <div className="flex flex-wrap items-center gap-2">
                        <Badge variant="default">#{idx + 1}</Badge>
                        <code className="text-sm font-semibold text-blue-900">{p.unit_id}</code>
                        <Badge variant="info">Doc: {p.document_id}</Badge>
                        <Badge variant="default">Source: {p.source_id}</Badge>
                        <Badge variant="default">Page {p.page_number}</Badge>
                        <Badge variant="success">TF: {p.term_frequency}</Badge>
                      </div>
                      {p.section_title && p.section_title !== "UNKNOWN" && (
                        <span className="text-xs text-gray-500 font-medium truncate max-w-xs">
                          {p.section_title}
                        </span>
                      )}
                    </div>

                    {/* Positions */}
                    {p.positions && p.positions.length > 0 && (
                      <div className="text-xs text-gray-600 mb-1.5">
                        <span className="font-semibold text-gray-700">Token Positions:</span>{" "}
                        <span className="font-mono bg-gray-100 px-1.5 py-0.5 rounded text-gray-800">
                          [{p.positions.slice(0, 15).join(", ")}{p.positions.length > 15 ? ` ... +${p.positions.length - 15} more` : ""}]
                        </span>
                      </div>
                    )}

                    {/* Snippet */}
                    {p.snippet && (
                      <p className="text-xs text-gray-700 leading-relaxed bg-white p-2 rounded border border-gray-100">
                        {p.snippet}
                      </p>
                    )}
                  </div>
                ))}
              </div>
            </SectionCard>
          ) : (
            <SectionCard title="Postings List">
              <p className="text-sm text-gray-500 py-4 text-center">
                No indexed postings found for &ldquo;{data.term}&rdquo;.
              </p>
            </SectionCard>
          )}
        </>
      )}

      {/* Top index terms */}
      {topTerms && topTerms.length > 0 && (
        <SectionCard title="Top Vocabulary Terms (by Posting Frequency)">
          <DataTable
            columns={[
              { key: "rank" as const, header: "#", width: "60px" },
              { key: "term" as const, header: "Term" },
              { key: "posting_frequency" as const, header: "Posting Frequency (Units)" },
              { key: "document_frequency" as const, header: "Document Frequency" },
              { key: "term_frequency" as const, header: "Total Occurrences (TF)" },
            ]}
            rows={topTerms.slice(0, 20).map((t, i) => ({ ...t, rank: i + 1 }))}
            keyField="term"
            renderCell={(row, col) => {
              if (col === "term") {
                return (
                  <button
                    onClick={() => {
                      setTerm(row.term);
                      lookup(row.term);
                    }}
                    className="text-blue-600 hover:underline font-mono text-sm font-semibold"
                  >
                    {row.term}
                  </button>
                );
              }
              if (col === "posting_frequency" || col === "document_frequency") {
                return row[col]?.toLocaleString() ?? "—";
              }
              if (col === "term_frequency") {
                return (row.term_frequency ?? row.total_frequency)?.toLocaleString() ?? "—";
              }
              return row[col] ?? "—";
            }}
          />
        </SectionCard>
      )}
    </PageContainer>
  );
}

export default function IndexExplorerPage() {
  return (
    <Layout>
      <Suspense fallback={<Loading />}>
        <IndexExplorerContent />
      </Suspense>
    </Layout>
  );
}
