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
    <PageContainer title="Named Entity Recognition" description="Entity extraction with label filtering and document-level search.">
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
                { key: "entity_label", header: "Label" },
                { key: "document_id", header: "Document", width: "100px" },
                { key: "page_number", header: "Page", width: "80px" },
                { key: "section_number", header: "Section", width: "100px" },
              ]}
              rows={data.items ?? []}
              keyField="entity_text"
              renderCell={(row, col) => {
                if (col === "entity_label") return <Badge variant="info">{row.entity_label ?? "—"}</Badge>;
                if (col === "entity_text") return <code className="text-sm">{row.entity_text ?? "—"}</code>;
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
                  { key: "label", header: "Label" },
                  { key: "count", header: "Count" },
                ]}
                rows={sidebar.label_distribution ?? []}
                keyField="label"
              />
            </SectionCard>
          )}

          {sidebar?.domain_dictionary && sidebar.domain_dictionary.length > 0 && (
            <SectionCard title="Domain Entity Dictionary">
              <DataTable
                columns={[
                  { key: "entity_text", header: "Entity" },
                  { key: "domain_category", header: "Category" },
                  { key: "frequency", header: "Frequency" },
                ]}
                rows={sidebar.domain_dictionary}
                keyField="entity_text"
              />
            </SectionCard>
          )}

          {sidebar?.error_analysis && sidebar.error_analysis.length > 0 && (
            <SectionCard title="Error Analysis">
              <DataTable
                columns={[
                  { key: "error_type", header: "Error Type" },
                  { key: "count", header: "Count" },
                  { key: "examples", header: "Examples" },
                ]}
                rows={sidebar.error_analysis}
                keyField="error_type"
                renderCell={(row, col) => {
                  if (col === "examples") return <code className="text-sm">{row[col] ?? "—"}</code>;
                  return row[col] ?? "—";
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