/** Search page: runs a query through the Phase 3 retrieval engine. */

"use client";

import { useState, Suspense } from "react";
import { Layout, PageContainer, SectionCard, Loading, ErrorState, Badge, Pagination } from "@/components";
import { api, type ParseResponse, type SearchResponse, type QueryType } from "@/lib/api";

const QUERY_TYPE_OPTIONS: Array<{ value: string; label: string }> = [
  { value: "", label: "Auto-detect from query text" },
  { value: "keyword", label: "Keyword (all terms)" },
  { value: "phrase", label: "Phrase (exact order)" },
  { value: "boolean_and", label: "Boolean AND" },
  { value: "boolean_or", label: "Boolean OR" },
  { value: "boolean_not", label: "Boolean AND NOT" },
  { value: "boolean_group", label: "Boolean grouped (parentheses)" },
];

function SearchContent() {
  const [query, setQuery] = useState("monetary policy");
  const [queryType, setQueryType] = useState<QueryType | "">("");
  const [topK, setTopK] = useState(10);
  const [pipeline, setPipeline] = useState<string>("pipeline_a_lemma");
  const [result, setResult] = useState<SearchResponse | null>(null);
  const [parseResult, setParseResult] = useState<ParseResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const runSearch = () => {
    const trimmed = query.trim();
    if (!trimmed) return;
    setError(null);
    setResult(null);
    setLoading(true);

    api.parseQuery(trimmed, queryType || undefined)
      .then((parsed) => {
        setParseResult(parsed);
        if (!parsed.valid) {
          setError(new Error(parsed.error ?? "Query is not valid"));
          setLoading(false);
          return Promise.reject(new Error(parsed.error ?? "Invalid query"));
        }
        return api.search({
          query: trimmed,
          query_type: parsed.query_type as QueryType,
          top_k: topK,
          pipeline,
        });
      })
      .then((response) => {
        setResult(response);
        setLoading(false);
      })
      .catch((e: Error) => {
        if (!(e instanceof Error) || e.message !== (parseResult?.error ?? "Invalid query")) {
          setError(e);
        }
        setLoading(false);
      });
  };

  const handleQueryChange = (value: string) => {
    setQuery(value);
    if (value.trim()) {
      api.parseQuery(value, queryType || undefined)
        .then(setParseResult)
        .catch(() => setParseResult(null));
    }
  };

  const showResults = result && result.returned > 0;
  const showError = error && !loading;

  return (
    <PageContainer title="Search" description="Run a query through the retrieval engine. FINAL SELECTED PIPELINE = Pipeline A (Lemmatization).">
      <SectionCard title="Query Builder">
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-1">Query Text</label>
            <textarea
              value={query}
              onChange={(e) => handleQueryChange(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && (e.preventDefault(), runSearch())}
              rows={2}
              className="w-full px-3 py-2 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500 font-mono text-sm"
              placeholder='E.g.: GDP AND NOT inflation, "repo rate", RBI AND credit'
            />
            <p className="text-xs text-gray-500 mt-1">
              Use &ldquo;&rdquo; for phrases, AND / OR / NOT for Boolean logic. NOT must follow an AND (e.g. &ldquo;GDP AND NOT inflation&rdquo;).
            </p>
          </div>

          {parseResult && (
            <div className={`p-3 rounded border ${parseResult.valid ? "bg-green-50 border-green-200" : "bg-red-50 border-red-200"}`}>
              <div className="flex items-center gap-2">
                <Badge variant={parseResult.valid ? "success" : "error"}>
                  {parseResult.valid ? "Valid" : "Invalid"}
                </Badge>
                <span className="text-sm text-gray-700">Detected type: <code>{parseResult.query_type}</code></span>
                {parseResult.normalized && (
                  <span className="text-sm text-gray-500">normalized: <code>{parseResult.normalized}</code></span>
                )}
              </div>
              {parseResult.error && <p className="text-sm text-red-700 mt-1">{parseResult.error}</p>}
            </div>
          )}

          <div className="grid md:grid-cols-4 gap-4">
            <div>
              <label className="block text-sm font-medium mb-1">Query Type</label>
              <select
                value={queryType}
                onChange={(e) => setQueryType(e.target.value as QueryType | "")}
                className="w-full px-3 py-2 border border-gray-300 rounded"
              >
                {QUERY_TYPE_OPTIONS.map((o) => (
                  <option key={o.value} value={o.value}>{o.label}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">Top K</label>
              <input
                type="number"
                min={1}
                max={200}
                value={topK}
                onChange={(e) => setTopK(Math.min(200, Math.max(1, parseInt(e.target.value) || 10)))}
                className="w-full px-3 py-2 border border-gray-300 rounded"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">Pipeline</label>
              <select
                value={pipeline}
                onChange={(e) => setPipeline(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded font-medium"
              >
                <option value="pipeline_a_lemma">Pipeline A — Lemmatization (FINAL SELECTED WINNER)</option>
                <option value="pipeline_b_stem">Pipeline B — Snowball Stemming (Comparison Option)</option>
              </select>
            </div>
            <div className="flex items-end">
              <button
                onClick={runSearch}
                disabled={loading || !query.trim() || (parseResult?.valid === false)}
                className="w-full px-4 py-2 bg-blue-600 text-white rounded font-medium disabled:opacity-50 hover:bg-blue-700 transition-colors"
              >
                {loading ? "Searching…" : "Search"}
              </button>
            </div>
          </div>
        </div>
      </SectionCard>

      {/* Results */}
      {loading && <Loading message="Searching the corpus…" />}

      {showError && (
        <SectionCard title="Error">
          <ErrorState error={error!} retry={runSearch} hint="Check the query syntax, or try fewer / more specific terms." />
        </SectionCard>
      )}

      {result && (
        <SectionCard title="Ranked Results">
          {/* Query echo + metadata */}
          <div className="grid md:grid-cols-4 gap-4 mb-4">
            <Badge variant="info">Query type: {result.query_type}</Badge>
            <Badge variant="success">Pipeline: {result.pipeline_name}</Badge>
            <Badge variant="info">Method: {result.retrieval_method}</Badge>
            <Badge variant="info">Docs: {result.result_documents}</Badge>
          </div>

          {result.matched_terms && result.matched_terms.length > 0 && (
            <p className="text-sm mb-2">Matched terms: {result.matched_terms.map((t) => <code key={t} className="bg-gray-100 px-1 rounded mr-1">{t}</code>)}</p>
          )}
          {result.excluded_terms && result.excluded_terms.length > 0 && (
            <p className="text-sm mb-2 text-amber-700">
              Excluded by NOT: {result.excluded_terms.map((t) => <code key={t} className="bg-amber-100 px-1 rounded mr-1">{t}</code>)}
            </p>
          )}
          {result.missing_terms && result.missing_terms.length > 0 && (
            <p className="text-sm mb-2 text-gray-500">
              Not in index: {result.missing_terms.map((t) => <code key={t} className="bg-gray-200 px-1 rounded mr-1">{t}</code>)}
            </p>
          )}

          {/* Timing + count summary */}
          <div className="mb-4 text-sm text-gray-600">
            Returned {result.returned} of {result.total_results} results ({result.execution_time_ms.toFixed(1)} ms)
          </div>

          {/* Hits */}
          <div className="space-y-4">
            {result.results.map((hit, idx) => (
              <div key={hit.unit_id} className="border border-gray-200 rounded-lg p-4 hover:bg-gray-50 transition-colors">
                <div className="flex justify-between items-start mb-2">
                  <div className="flex flex-wrap items-center gap-2">
                    <Badge variant="default">Rank #{idx + 1}</Badge>
                    <code className="text-sm font-semibold">{hit.unit_id}</code>
                    <Badge variant="success">Score: {hit.score.toFixed(4)}</Badge>
                    <Badge variant="info">Type: {hit.unit_type}</Badge>
                    {hit.document_id && <Badge variant="default">Doc: {hit.document_id}</Badge>}
                    {hit.source_id && <Badge variant="default">Source: {hit.source_id}</Badge>}
                    {hit.page_number && <Badge variant="default">Page {hit.page_number}</Badge>}
                  </div>
                  <div className="text-right text-xs text-gray-500 font-mono">
                    {hit.citation}
                  </div>
                </div>
                {hit.section_title && hit.section_title !== "UNKNOWN" && (
                  <p className="text-xs text-gray-500 font-medium mb-1">Section: {hit.section_title}</p>
                )}
                <p className="text-gray-800 mb-2 line-clamp-3 leading-relaxed">{hit.snippet}</p>
                <div className="flex flex-wrap gap-1">
                  {hit.matched_terms_list.map((t) => (
                    <code key={t} className="text-xs bg-blue-50 text-blue-800 px-1.5 py-0.5 rounded">{t}</code>
                  ))}
                </div>
              </div>
            ))}
          </div>

          {result.returned === 0 && result.total_results === 0 && (
            <p className="text-gray-500 text-center py-8">
              No units matched this query. {result.missing_terms?.length > 0 && "The terms you entered are not in the index."}
            </p>
          )}

          {result.returned > 0 && result.total_results > result.returned && (
            <p className="text-sm text-gray-500 mt-4">
              Top {result.top_k} shown. Phase 3 caps results at {result.max_results_available} per query.
            </p>
          )}
        </SectionCard>
      )}
    </PageContainer>
  );
}

export default function SearchPage() {
  return (
    <Layout>
      <Suspense fallback={<Loading />}>
        <SearchContent />
      </Suspense>
    </Layout>
  );
}