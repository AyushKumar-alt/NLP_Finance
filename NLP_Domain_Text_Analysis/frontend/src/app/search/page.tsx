/** Search page: runs a query through the Phase 3 retrieval engine. */

"use client";

import { useState, Suspense } from "react";
import { Layout, PageContainer, SectionCard, Loading, ErrorState, Badge } from "@/components";
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

function checkUnquotedAmpersand(text: string): boolean {
  const withoutQuotes = text.replace(/"[^"]*"|'[^']*'/g, "");
  return /(?:^|\s)&(?:\s|$)/.test(withoutQuotes);
}

function SearchContent() {
  const [query, setQuery] = useState("monetary policy");
  const [queryType, setQueryType] = useState<QueryType | "">("");
  const [topK, setTopK] = useState(10);
  const [pipeline, setPipeline] = useState<string>("pipeline_a_lemma");
  const [result, setResult] = useState<SearchResponse | null>(null);
  const [parseResult, setParseResult] = useState<ParseResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [activeScoreTooltip, setActiveScoreTooltip] = useState<string | null>(null);

  const hasAmpersandWarning = checkUnquotedAmpersand(query);

  const runSearch = () => {
    const trimmed = query.trim();
    if (!trimmed) return;
    if (checkUnquotedAmpersand(trimmed)) {
      setError(new Error("& is not a supported Boolean operator. Use AND instead."));
      return;
    }
    setError(null);
    setResult(null);
    setLoading(true);

    const effectiveQueryType = queryType || undefined;
    api.parseQuery(trimmed, effectiveQueryType)
      .then((parsed) => {
        setParseResult(parsed);
        if (!parsed.valid) {
          setError(new Error(parsed.error ?? "Query is not valid"));
          setLoading(false);
          return Promise.reject(new Error(parsed.error ?? "Invalid query"));
        }
        return api.search({
          query: trimmed,
          query_type: (queryType ? queryType : parsed.query_type) as QueryType,
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
    } else {
      setParseResult(null);
    }
  };

  const handleQueryTypeChange = (newType: QueryType | "") => {
    setQueryType(newType);
    if (query.trim()) {
      api.parseQuery(query, newType || undefined)
        .then(setParseResult)
        .catch(() => setParseResult(null));
    }
  };

  const showError = error && !loading;

  return (
    <PageContainer
      title="Search"
      description="Run a query through the retrieval engine. FINAL SELECTED PIPELINE = Pipeline A (Lemmatization)."
    >
      {/* 1. Query Builder */}
      <SectionCard title="Query Builder">
        <div className="space-y-4">
          {/* 2. Query Text */}
          <div>
            <label className="block text-sm font-semibold text-gray-800 mb-1">Query Text</label>
            <textarea
              value={query}
              onChange={(e) => handleQueryChange(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && (e.preventDefault(), runSearch())}
              rows={2}
              className="w-full px-3 py-2 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500 font-mono text-sm shadow-sm"
              placeholder='E.g.: GDP AND NOT inflation, "repo rate", RBI AND credit'
            />

            {/* 3. Syntax help & examples */}
            <p className="text-xs text-gray-600 mt-1">
              Use &ldquo;&rdquo; for exact phrases, and AND / OR / NOT for Boolean logic. NOT must follow an AND (e.g. &ldquo;GDP AND NOT inflation&rdquo;).
            </p>
            <div className="mt-2 flex flex-wrap items-center gap-1.5">
              <span className="text-xs font-semibold text-gray-600 mr-1">Query Examples:</span>
              {[
                { label: "inflation", q: "inflation" },
                { label: '"monetary policy"', q: '"monetary policy"' },
                { label: "GDP AND inflation", q: "GDP AND inflation" },
                { label: "GDP OR GVA", q: "GDP OR GVA" },
                { label: "inflation AND NOT food", q: "inflation AND NOT food" },
                { label: "(GDP OR GVA) AND policy", q: "(GDP OR GVA) AND policy" },
                { label: 'RBI AND "repo rate"', q: 'RBI AND "repo rate"' },
              ].map((item, idx) => (
                <button
                  key={idx}
                  type="button"
                  onClick={() => handleQueryChange(item.q)}
                  className="px-2 py-0.5 text-xs bg-gray-100 hover:bg-blue-50 hover:text-blue-800 text-gray-700 rounded border border-gray-200 transition-colors font-mono"
                >
                  {item.label}
                </button>
              ))}
            </div>
          </div>

          {/* 4. Validation status & Ampersand warning */}
          {hasAmpersandWarning && (
            <div className="p-3 rounded-lg border bg-amber-50 border-amber-300 text-amber-900 text-sm">
              <div className="flex items-center gap-2 font-semibold">
                <span className="text-base">⚠️</span>
                <span>&amp; is not a supported Boolean operator. Use AND instead.</span>
              </div>
              <p className="mt-1 text-xs text-amber-800">
                Example suggestion:{" "}
                <code className="bg-amber-100 px-1 py-0.5 rounded font-mono font-bold text-amber-950">
                  &quot;monetary policy&quot; AND &quot;repo rate&quot;
                </code>
              </p>
              <button
                type="button"
                onClick={() => handleQueryChange(query.replace(/(?:^|\s)&(?:\s|$)/g, " AND "))}
                className="mt-2 px-2.5 py-1 text-xs font-semibold bg-amber-200 hover:bg-amber-300 text-amber-950 rounded transition-colors"
              >
                Replace &amp; with AND
              </button>
            </div>
          )}

          {!hasAmpersandWarning && parseResult && (
            <div
              className={`p-3 rounded-lg border ${
                parseResult.valid ? "bg-green-50 border-green-200" : "bg-red-50 border-red-200"
              }`}
            >
              <div className="flex flex-wrap items-center gap-2">
                <Badge variant={parseResult.valid ? "success" : "error"}>
                  {parseResult.valid ? "Valid Syntax" : "Invalid Syntax"}
                </Badge>
                <span className="text-sm text-gray-700">
                  Detected query type: <code className="font-semibold text-gray-900">{parseResult.query_type}</code>
                </span>
                {parseResult.normalized && (
                  <span className="text-sm text-gray-500">
                    Normalized: <code className="text-gray-700">{parseResult.normalized}</code>
                  </span>
                )}
              </div>
              {parseResult.error && (
                <p className="text-sm text-red-700 font-medium mt-1.5">{parseResult.error}</p>
              )}
            </div>
          )}

          {/* 5. Detected/selected query type, 6. Top K, 7. Pipeline selector, 8. Search Button */}
          <div className="grid md:grid-cols-4 gap-4 pt-1">
            <div>
              <label className="block text-sm font-semibold text-gray-800 mb-1">Query Type</label>
              <select
                value={queryType}
                onChange={(e) => handleQueryTypeChange(e.target.value as QueryType | "")}
                className="w-full px-3 py-2 border border-gray-300 rounded text-sm bg-white shadow-sm focus:ring-2 focus:ring-blue-500"
              >
                {QUERY_TYPE_OPTIONS.map((o) => (
                  <option key={o.value} value={o.value}>
                    {o.label}
                  </option>
                ))}
              </select>
              <p className="text-[11px] text-gray-500 mt-1">
                {queryType === ""
                  ? "Auto-detects keyword, exact phrase, or Boolean operators from query text."
                  : `Explicit override: forces ${queryType} mode.`}
              </p>
            </div>

            <div>
              <label className="block text-sm font-semibold text-gray-800 mb-1">Top K</label>
              <input
                type="number"
                min={1}
                max={200}
                value={topK}
                onChange={(e) => setTopK(Math.min(200, Math.max(1, parseInt(e.target.value) || 10)))}
                className="w-full px-3 py-2 border border-gray-300 rounded text-sm bg-white shadow-sm focus:ring-2 focus:ring-blue-500"
              />
              <p className="text-[11px] text-gray-500 mt-1">Number of top units to display (1–200).</p>
            </div>

            <div>
              <label className="block text-sm font-semibold text-gray-800 mb-1">Pipeline</label>
              <select
                value={pipeline}
                onChange={(e) => setPipeline(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded text-sm bg-white shadow-sm focus:ring-2 focus:ring-blue-500 font-medium"
              >
                <option value="pipeline_a_lemma">Pipeline A — Lemmatization (WINNER)</option>
                <option value="pipeline_b_stem">Pipeline B — Snowball Stemming</option>
              </select>
              <p className="text-[11px] text-gray-500 mt-1">Independent inverted indexes.</p>
            </div>

            <div className="flex flex-col justify-end">
              <button
                onClick={runSearch}
                disabled={loading || !query.trim() || hasAmpersandWarning || parseResult?.valid === false}
                className="w-full px-4 py-2 bg-blue-600 text-white rounded-lg font-semibold text-sm shadow-sm disabled:opacity-50 hover:bg-blue-700 transition-colors"
              >
                {loading ? "Searching…" : "Search"}
              </button>
              <p className="text-[11px] text-gray-400 mt-1 text-center">Press Enter to search</p>
            </div>
          </div>
        </div>
      </SectionCard>

      {/* Loading & Error States */}
      {loading && <Loading message="Executing Phase 3 retrieval engine..." />}

      {showError && (
        <SectionCard title="Error">
          <ErrorState
            error={error!}
            retry={runSearch}
            hint="Check the query syntax, or verify terms against the index vocabulary."
          />
        </SectionCard>
      )}

      {/* Results Section */}
      {result && (
        <SectionCard title="Query Results &amp; Evidence">
          {/* 9. Query interpretation summary */}
          <div className="bg-slate-50 border border-slate-200 rounded-lg p-4 mb-6 shadow-sm">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
              <div>
                <span className="block text-xs text-gray-500 font-medium uppercase tracking-wider">Query</span>
                <span className="font-mono text-sm font-bold text-gray-900 break-words">{result.query}</span>
              </div>
              <div>
                <span className="block text-xs text-gray-500 font-medium uppercase tracking-wider">Interpretation</span>
                <span className="inline-flex items-center mt-0.5 px-2 py-0.5 rounded text-xs font-bold bg-blue-100 text-blue-800 capitalize">
                  {result.query_type.replace("_", " ")}
                </span>
                <span className="text-[11px] text-gray-500 block mt-0.5">Method: {result.retrieval_method}</span>
              </div>
              <div>
                <span className="block text-xs text-gray-500 font-medium uppercase tracking-wider">Pipeline</span>
                <span className="text-sm font-bold text-gray-800 block">{result.pipeline_name}</span>
                <span className="text-[11px] text-gray-500 font-mono">key: {result.pipeline}</span>
              </div>
              <div>
                <span className="block text-xs text-gray-500 font-medium uppercase tracking-wider">Execution Time</span>
                <span className="text-sm font-bold text-emerald-700 block">{result.execution_time_ms.toFixed(1)} ms</span>
                <span className="text-[11px] text-gray-500">Live inverted index lookup</span>
              </div>
            </div>

            {/* 4. Result counts clearly labeled */}
            <div className="grid grid-cols-3 gap-4 py-3 border-t border-slate-200 bg-white rounded-md px-3 border mb-3">
              <div>
                <span className="block text-xs text-gray-500 font-medium">Unique Documents</span>
                <span className="text-xl font-extrabold text-gray-900">{result.result_documents}</span>
              </div>
              <div>
                <span className="block text-xs text-gray-500 font-medium">Matching Content Units</span>
                <span className="text-xl font-extrabold text-gray-900">{result.total_results}</span>
              </div>
              <div>
                <span className="block text-xs text-gray-500 font-medium">Showing</span>
                <span className="text-xl font-extrabold text-blue-600">{result.returned}</span>
              </div>
            </div>

            {result.total_results >= result.max_results_available && (
              <p className="text-xs text-amber-800 bg-amber-50 px-3 py-1.5 rounded border border-amber-200 mb-3">
                Retrieval engine caps candidate results at {result.max_results_available} per query.
              </p>
            )}

            {/* 3. Matched query terms rendered as distinct chips */}
            <div className="space-y-2 pt-2 border-t border-slate-200">
              {result.matched_terms_list && result.matched_terms_list.length > 0 && (
                <div className="flex flex-wrap items-center gap-1.5">
                  <span className="text-xs font-semibold text-gray-700">Positive query terms:</span>
                  {result.matched_terms_list.map((t) => (
                    <span
                      key={t}
                      className="px-2 py-0.5 text-xs font-semibold bg-blue-100 text-blue-800 rounded border border-blue-200 font-mono shadow-sm"
                    >
                      [{t}]
                    </span>
                  ))}
                </div>
              )}
              {result.excluded_terms && result.excluded_terms.length > 0 && (
                <div className="flex flex-wrap items-center gap-1.5">
                  <span className="text-xs font-semibold text-red-700">Excluded by NOT:</span>
                  {result.excluded_terms.map((t) => (
                    <span
                      key={t}
                      className="px-2 py-0.5 text-xs font-semibold bg-red-100 text-red-800 rounded border border-red-200 font-mono"
                    >
                      [{t}]
                    </span>
                  ))}
                </div>
              )}
              {result.missing_terms && result.missing_terms.length > 0 && (
                <div className="flex flex-wrap items-center gap-1.5">
                  <span className="text-xs font-semibold text-gray-500">Not in index vocabulary:</span>
                  {result.missing_terms.map((t) => (
                    <span
                      key={t}
                      className="px-2 py-0.5 text-xs font-medium bg-gray-100 text-gray-600 rounded border border-gray-200 font-mono"
                    >
                      [{t}]
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* 10. Ranked results & 11. Per-result evidence/explanation */}
          <div className="space-y-4">
            {result.results.map((hit, idx) => {
              const tooltipOpen = activeScoreTooltip === hit.unit_id;
              return (
                <div
                  key={hit.unit_id}
                  className="border border-gray-200 rounded-lg p-5 hover:bg-slate-50 transition-colors shadow-sm bg-white"
                >
                  <div className="flex flex-wrap justify-between items-start gap-2 mb-2 pb-2.5 border-b border-gray-100">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="px-2.5 py-0.5 text-xs font-bold bg-gray-900 text-white rounded">
                        Rank #{idx + 1}
                      </span>
                      <code className="text-sm font-bold text-gray-900">{hit.unit_id}</code>

                      {/* 5. Score with explanation tooltip */}
                      <div className="relative inline-block">
                        <button
                          type="button"
                          onClick={() => setActiveScoreTooltip(tooltipOpen ? null : hit.unit_id)}
                          onMouseEnter={() => setActiveScoreTooltip(hit.unit_id)}
                          onMouseLeave={() => setActiveScoreTooltip(null)}
                          className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-bold bg-emerald-100 text-emerald-900 border border-emerald-300 hover:bg-emerald-200 transition-colors"
                          title="Click to view Ranking Score breakdown"
                        >
                          Ranking Score: {hit.score.toFixed(4)} <span className="text-emerald-700 font-black">ⓘ</span>
                        </button>

                        {tooltipOpen && (
                          <div
                            className="absolute z-30 left-0 top-full mt-2 w-80 p-3 bg-gray-900 text-white text-xs rounded-lg shadow-2xl border border-gray-700"
                            onMouseEnter={() => setActiveScoreTooltip(hit.unit_id)}
                            onMouseLeave={() => setActiveScoreTooltip(null)}
                          >
                            <p className="font-bold text-gray-100 mb-1">Custom Retrieval Ranking Score</p>
                            <p className="text-gray-300 text-[11px] leading-relaxed mb-2.5">
                              Custom retrieval score used to order matching content units. It combines query-term coverage,
                              exact phrase matching where applicable, and term frequency. Higher values indicate a stronger
                              match under this ranking function. It is not a probability or confidence score.
                            </p>
                            {hit.score_breakdown && (
                              <div className="border-t border-gray-700 pt-2 space-y-1 font-mono text-[11px]">
                                <div className="flex justify-between text-gray-300">
                                  <span>Matched-term component ({hit.score_breakdown.matched_term_count} × 1.0):</span>
                                  <span className="font-semibold">{hit.score_breakdown.term_component.toFixed(4)}</span>
                                </div>
                                <div className="flex justify-between text-gray-300">
                                  <span>Phrase component ({hit.score_breakdown.phrase_hit ? "2.0 bonus" : "0.0"}):</span>
                                  <span className="font-semibold">{hit.score_breakdown.phrase_component.toFixed(4)}</span>
                                </div>
                                <div className="flex justify-between text-gray-300">
                                  <span>TF component (0.25 × log10(1 + {hit.score_breakdown.total_tf})):</span>
                                  <span className="font-semibold">{hit.score_breakdown.tf_component.toFixed(4)}</span>
                                </div>
                                <div className="flex justify-between font-bold text-emerald-400 border-t border-gray-700 pt-1 text-xs">
                                  <span>Total Ranking Score:</span>
                                  <span>{hit.score_breakdown.total_score.toFixed(4)}</span>
                                </div>
                              </div>
                            )}
                          </div>
                        )}
                      </div>

                      <span className="px-2 py-0.5 text-xs font-medium bg-slate-100 text-slate-700 rounded border border-slate-200">
                        {hit.unit_type}
                      </span>
                      {hit.document_id && (
                        <span className="px-2 py-0.5 text-xs font-medium bg-gray-100 text-gray-700 rounded">
                          Doc: {hit.document_id}
                        </span>
                      )}
                      {hit.source_id && (
                        <span className="px-2 py-0.5 text-xs font-medium bg-gray-100 text-gray-700 rounded">
                          Source: {hit.source_id}
                        </span>
                      )}
                      {hit.page_number && (
                        <span className="px-2 py-0.5 text-xs font-medium bg-gray-100 text-gray-700 rounded">
                          Page {hit.page_number}
                        </span>
                      )}
                    </div>
                    <div className="text-xs text-gray-500 font-mono">{hit.citation}</div>
                  </div>

                  {hit.section_title && hit.section_title !== "UNKNOWN" && (
                    <p className="text-xs text-gray-600 font-medium mb-2">
                      <span className="font-semibold text-gray-700">Section:</span> {hit.section_title}
                    </p>
                  )}

                  {/* 3. Matched in this result: distinct chips for this content unit only */}
                  <div className="flex flex-wrap items-center gap-1.5 mb-2.5">
                    <span className="text-xs font-semibold text-gray-700">Matched in this result:</span>
                    {hit.matched_terms_list && hit.matched_terms_list.length > 0 ? (
                      hit.matched_terms_list.map((term) => (
                        <span
                          key={term}
                          className="px-2 py-0.5 text-xs font-medium bg-emerald-50 text-emerald-800 rounded border border-emerald-200 font-mono"
                        >
                          [{term}]
                        </span>
                      ))
                    ) : (
                      <span className="text-xs text-gray-400">None</span>
                    )}
                    {hit.phrase_match && (
                      <span className="px-2 py-0.5 text-xs font-bold bg-purple-100 text-purple-900 rounded border border-purple-200">
                        Exact Phrase Match
                      </span>
                    )}
                  </div>

                  {/* 6. Evidence: Snippet centered on matched evidence */}
                  <div className="bg-gray-50 rounded-lg p-3.5 border border-gray-200 text-sm text-gray-800 leading-relaxed font-sans">
                    <span className="font-semibold text-xs text-gray-500 block mb-1">Evidence:</span>
                    <p className="italic">{hit.snippet}</p>
                  </div>
                </div>
              );
            })}
          </div>

          {result.returned === 0 && result.total_results === 0 && (
            <p className="text-gray-500 text-center py-10">
              No units matched this query.{" "}
              {result.missing_terms?.length > 0 && "The terms you entered are not in the index vocabulary."}
            </p>
          )}

          {result.returned > 0 && result.total_results > result.returned && (
            <p className="text-xs text-gray-500 mt-4 text-center">
              Showing top {result.returned} of {result.total_results} matching content units.
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