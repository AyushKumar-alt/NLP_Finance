/** Named Entity Recognition page. */

"use client";

import { useEffect, useState, Suspense } from "react";
import { Layout, PageContainer, SectionCard, DataTable, Pagination, Loading, ErrorState, Badge, MetricCard } from "@/components";
import { api } from "@/lib/api";

function NERContent() {
  const [data, setData] = useState<any>(null);
  const [sidebar, setSidebar] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const [params, setParams] = useState({ kind: "general" as "general" | "domain", label: "", document_id: "", search: "", limit: 50, offset: 0 });

  useEffect(() => {
    api.nerSidebar().then(setSidebar).catch(console.error);
    fetchData();
  }, [params.kind]);

  const fetchData = () => {
    setLoading(true);
    api.ner({ ...params })
      .then((d) => { setData(d); setLoading(false); })
      .catch((e) => { setError(e); setLoading(false); });
  };

  if (loading) return <Loading />;
  if (error) return <ErrorState error={error} retry={fetchData} />;
  if (!data) return null;

  const handlePageChange = (offset: number) => {
    setParams((p) => ({ ...p, offset }));
  };

  const handleFilterChange = (updates: Partial<typeof params>) => {
    setParams((p) => ({ ...p, ...updates, offset: 0 }));
  };

  return (
    <PageContainer
      title="Named Entity Recognition (NER) Analysis"
      description="Comparative evaluation of general statistical NER vs. domain-adapted financial NER patterns."
    >
      {/* Corpus-level NER counts */}
      <div className="bg-gradient-to-r from-teal-50 to-emerald-50 border border-teal-200 rounded-lg p-5 mb-6">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
          <div>
            <h2 className="text-lg font-bold text-teal-950">Corpus Entity Extraction Totals</h2>
            <p className="text-xs text-teal-800 mt-0.5">
              Domain-specific financial entity patterns directly complement standard English named entity recognition.
            </p>
          </div>
          <div className="flex gap-2">
            <span className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-semibold bg-blue-100 text-blue-800 border border-blue-300">
              General NER: 26,717
            </span>
            <span className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-semibold bg-emerald-100 text-emerald-800 border border-emerald-300">
              Domain NER: 4,968
            </span>
          </div>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center">
          <div className="bg-white p-3 rounded border border-teal-100">
            <div className="text-xs text-gray-500 font-medium">General NER Entities</div>
            <div className="text-xl font-bold text-teal-950 mt-1">26,717</div>
            <div className="text-xs text-gray-400 mt-0.5">ORG, GPE, PERSON, DATE</div>
          </div>
          <div className="bg-white p-3 rounded border border-teal-100">
            <div className="text-xs text-gray-500 font-medium">Domain Financial Entities</div>
            <div className="text-xl font-bold text-emerald-700 mt-1">4,968</div>
            <div className="text-xs text-gray-400 mt-0.5">MONEY, PERCENT, FISCAL_YEAR</div>
          </div>
          <div className="bg-white p-3 rounded border border-teal-100">
            <div className="text-xs text-gray-500 font-medium">Domain Category Coverage</div>
            <div className="text-xl font-bold text-teal-950 mt-1">7 Categories</div>
            <div className="text-xs text-gray-400 mt-0.5">Rates, Currencies, Policies</div>
          </div>
          <div className="bg-white p-3 rounded border border-teal-100">
            <div className="text-xs text-gray-500 font-medium">Domain Precision</div>
            <div className="text-xl font-bold text-emerald-700 mt-1">High</div>
            <div className="text-xs text-gray-400 mt-0.5">Zero symbol fragmentation</div>
          </div>
        </div>
      </div>

      <div className="grid md:grid-cols-4 gap-6">
        <aside className="md:col-span-1 space-y-6">
          <SectionCard title="Entity Kind">
            <div className="space-y-2">
              {["general", "domain"].map((k) => (
                <label key={k} className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="radio"
                    name="kind"
                    value={k}
                    checked={params.kind === k}
                    onChange={() => handleFilterChange({ kind: k as "general" | "domain" })}
                    className="text-blue-600"
                  />
                  <span className="capitalize">{k}</span>
                  <Badge variant="default" className="ml-auto">
                    {sidebar?.[k === "general" ? "general_total" : "domain_total"] ?? 0}
                  </Badge>
                </label>
              ))}
            </div>
          </SectionCard>

          {sidebar && (
            <SectionCard title="Filter by Label">
              <input
                type="text"
                placeholder="Search labels…"
                value={params.search}
                onChange={(e) => handleFilterChange({ search: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500 mb-3"
              />
              <div className="max-h-64 overflow-y-auto space-y-1">
                {sidebar[params.kind === "general" ? "general_labels" : "domain_categories"]
                  ?.filter((l: any) => l.label.toLowerCase().includes(params.search.toLowerCase()))
                  ?.slice(0, 30)
                  .map((l: any) => (
                    <label key={l.label} className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={params.label === l.label}
                        onChange={() => handleFilterChange({ label: params.label === l.label ? "" : l.label })}
                        className="text-blue-600"
                      />
                      <span className="text-sm truncate">{l.label}</span>
                      <Badge variant="default" className="ml-auto">{l.count}</Badge>
                    </label>
                  ))}
              </div>
            </SectionCard>
          )}

          <SectionCard title="Document Filter">
            <input
              type="text"
              placeholder="Document ID (e.g. D01)"
              value={params.document_id}
              onChange={(e) => handleFilterChange({ document_id: e.target.value })}
              className="w-full px-3 py-2 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500"
            />
          </SectionCard>
        </aside>

        <main className="md:col-span-3 space-y-6">
          <SectionCard title={`Entities (${data.total?.toLocaleString() ?? 0})`}>
            <DataTable
              columns={[
                { key: "entity_text", header: "Entity" },
                { key: "entity_label", header: "Label / Category" },
                { key: "document_id", header: "Document", width: "100px" },
                { key: "page_number", header: "Page", width: "80px" },
                { key: "section_number", header: "Section", width: "100px" },
              ]}
              rows={data.items ?? []}
              keyField={(row: any, i: number) => row.entity_id ?? `${row.document_id}_${row.entity_text}_${i}`}
              renderCell={(row, col) => {
                if (col === "entity_label") {
                  const lbl = row.entity_label || row.domain_category || (data.label_field ? row[data.label_field] : undefined);
                  return <Badge variant="info">{lbl ?? "—"}</Badge>;
                }
                if (col === "entity_text") return <code className="text-sm font-semibold text-blue-900 bg-blue-50 px-1 py-0.5 rounded">{row.entity_text ?? "—"}</code>;
                return row[col] ?? "—";
              }}
            />
            <Pagination
              total={data.total ?? 0}
              offset={data.offset ?? 0}
              limit={data.limit ?? 50}
              onChange={handlePageChange}
            />
          </SectionCard>

          {sidebar && (
            <SectionCard title="Label Distribution">
              <DataTable
                columns={[
                  { key: "entity_label", header: "Label" },
                  { key: "count", header: "Count" },
                  { key: "percent_of_entities", header: "% of Total Entities" },
                ]}
                rows={sidebar.label_distribution ?? []}
                keyField="entity_label"
                renderCell={(row, col) => {
                  if (col === "entity_label") return <Badge variant="default">{row.entity_label ?? "—"}</Badge>;
                  if (col === "count") return Number(row.count)?.toLocaleString() ?? "—";
                  if (col === "percent_of_entities") {
                    const v = Number(row.percent_of_entities);
                    return isNaN(v) ? "—" : `${v.toFixed(2)}%`;
                  }
                  return row[col] ?? "—";
                }}
              />
            </SectionCard>
          )}

          {sidebar?.domain_dictionary && sidebar.domain_dictionary.length > 0 && (
            <SectionCard title="Domain Entity Dictionary">
              <DataTable
                columns={[
                  { key: "term", header: "Domain Entity / Term" },
                  { key: "domain_category", header: "Domain Category" },
                  { key: "corpus_mentions", header: "Mentions" },
                  { key: "documents_mentioned", header: "Documents" },
                  { key: "general_ner_equivalent_labels", header: "General NER Equivalents" },
                ]}
                rows={sidebar.domain_dictionary.slice(0, 30)}
                keyField="term"
                renderCell={(row, col) => {
                  if (col === "term") return <span className="font-bold text-xs font-mono text-blue-950">{row.term ?? "—"}</span>;
                  if (col === "domain_category") return <Badge variant="info">{row.domain_category ?? "—"}</Badge>;
                  if (col === "corpus_mentions" || col === "documents_mentioned") return Number(row[col])?.toLocaleString() ?? "—";
                  if (col === "general_ner_equivalent_labels") return <span className="text-xs text-gray-600 font-mono">{row[col] ?? "—"}</span>;
                  return <span className="text-xs text-gray-800">{row[col] ?? "—"}</span>;
                }}
              />
            </SectionCard>
          )}

          {sidebar?.error_analysis && sidebar.error_analysis.length > 0 && (
            <SectionCard title="Error Analysis">
              <DataTable
                columns={[
                  { key: "text", header: "Entity Text" },
                  { key: "default_label", header: "Default Label" },
                  { key: "expected_or_interpreted_label", header: "Target Category" },
                  { key: "error_type", header: "Error Type" },
                  { key: "context", header: "Corpus Context" },
                  { key: "document_id", header: "Doc" },
                ]}
                rows={sidebar.error_analysis.slice(0, 30)}
                keyField={(row: any, i: number) => `${row.document_id}_${row.text}_${i}`}
                renderCell={(row, col) => {
                  if (col === "text") return <code className="text-xs bg-gray-100 font-semibold text-gray-900 px-1 py-0.5 rounded font-mono">{row.text ?? "—"}</code>;
                  if (col === "default_label") return <Badge variant="warning">{row.default_label ?? "—"}</Badge>;
                  if (col === "expected_or_interpreted_label") return <Badge variant="success">{row.expected_or_interpreted_label ?? "—"}</Badge>;
                  if (col === "error_type") return <Badge variant="default">{row.error_type ?? "—"}</Badge>;
                  if (col === "context") return <span className="text-xs text-gray-700 italic">{row.context ?? "—"}</span>;
                  return <span className="text-xs text-gray-800">{row[col] ?? "—"}</span>;
                }}
              />
            </SectionCard>
          )}
        </main>
      </div>
    </PageContainer>
  );
}

export default function NERPage() {
  return (
    <Layout>
      <Suspense fallback={<Loading />}>
        <NERContent />
      </Suspense>
    </Layout>
  );
}