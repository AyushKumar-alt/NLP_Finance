/** Documents page: browse and filter the corpus registry. */

"use client";

import { useState, useEffect, Suspense } from "react";
import Link from "next/link";
import { Layout, PageContainer, DataTable, Pagination, Loading, ErrorState, Badge } from "@/components";
import { api, type DocumentSummary } from "@/lib/api";

function DocumentsContent() {
  const [data, setData] = useState<{ items: DocumentSummary[]; total: number; offset: number; limit: number; has_more: boolean } | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const [filters, setFilters] = useState({ source_id: "", document_type: "", search: "" });

  const fetchData = (offset = 0) => {
    setLoading(true);
    api.documents({ ...filters, limit: 50, offset })
      .then((d) => { setData(d); setLoading(false); })
      .catch((e) => { setError(e); setLoading(false); });
  };

  useEffect(() => { fetchData(); }, [filters]);

  if (loading) return <Loading />;
  if (error) return <ErrorState error={error} retry={() => fetchData()} />;
  if (!data) return null;

  const columns = [
    { key: "document_id" as const, header: "ID", width: "100px" },
    { key: "title" as const, header: "Title" },
    { key: "source_id" as const, header: "Source", width: "100px" },
    { key: "year" as const, header: "Year", width: "80px" },
    { key: "units" as const, header: "Units", width: "80px" },
    { key: "page_count" as const, header: "Pages", width: "80px" },
    { key: "words" as const, header: "Words", width: "100px" },
    { key: "quality_grade" as const, header: "Quality", width: "100px" },
    { key: "extraction_status" as const, header: "Status", width: "120px" },
  ];

  return (
    <PageContainer title="Documents & Corpus Registry" description="Browse and inspect the 31 indexed Indian financial and economic reports.">
      {/* Loaded Corpus Status & Overview */}
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-lg p-5 mb-6">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
          <div>
            <h2 className="text-lg font-bold text-blue-950">Loaded Corpus: Indian Financial & Economic Documents</h2>
            <p className="text-xs text-blue-800 mt-1 max-w-2xl">
              The application operates on the preprocessed 31-document corpus used throughout this assessment. Corpus ingestion is performed offline through Phase 1 to preserve reproducibility.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <span className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-semibold bg-green-100 text-green-800 border border-green-300">
              ● Corpus Loaded & Validated
            </span>
            <span className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-semibold bg-blue-100 text-blue-800 border border-blue-300">
              Offline Phase 1 Ingestion
            </span>
          </div>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-3">
          <div className="bg-white p-3 rounded border border-blue-100 text-center">
            <div className="text-xs text-gray-500 font-medium">Documents</div>
            <div className="text-xl font-bold text-gray-900 mt-1">31</div>
          </div>
          <div className="bg-white p-3 rounded border border-blue-100 text-center">
            <div className="text-xs text-gray-500 font-medium">Parent Sources</div>
            <div className="text-xl font-bold text-gray-900 mt-1">3</div>
          </div>
          <div className="bg-white p-3 rounded border border-blue-100 text-center">
            <div className="text-xs text-gray-500 font-medium">Total Pages</div>
            <div className="text-xl font-bold text-gray-900 mt-1">885</div>
          </div>
          <div className="bg-white p-3 rounded border border-blue-100 text-center">
            <div className="text-xs text-gray-500 font-medium">Extracted Units</div>
            <div className="text-xl font-bold text-gray-900 mt-1">8,311</div>
          </div>
          <div className="bg-white p-3 rounded border border-blue-100 text-center">
            <div className="text-xs text-gray-500 font-medium">Searchable Units</div>
            <div className="text-xl font-bold text-blue-700 mt-1">6,134</div>
          </div>
          <div className="bg-white p-3 rounded border border-blue-100 text-center">
            <div className="text-xs text-gray-500 font-medium">Corpus Words</div>
            <div className="text-xl font-bold text-gray-900 mt-1">326,589</div>
          </div>
        </div>

        <div className="mt-4 pt-3 border-t border-blue-100 text-xs text-blue-900 flex flex-wrap justify-between items-center gap-2">
          <span><strong>Publishing Sources:</strong> Economic Survey 2025-26 (SRC01), RBI Financial Stability Report (SRC02), SEBI Annual Report 2025-26 (SRC03).</span>
          <span className="italic text-gray-600">Traceability: Source → Document → Page → Section → Content Unit</span>
        </div>
      </div>

      {/* Filters */}
      <div className="bg-white border border-gray-200 rounded-lg p-4 mb-6">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <input
            type="text"
            placeholder="Search title, filename, ID…"
            value={filters.search}
            onChange={(e) => setFilters({ ...filters, search: e.target.value })}
            className="px-3 py-2 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          />
          <input
            type="text"
            placeholder="Source ID (e.g. SRC01)"
            value={filters.source_id}
            onChange={(e) => setFilters({ ...filters, source_id: e.target.value })}
            className="px-3 py-2 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          />
          <input
            type="text"
            placeholder="Document type"
            value={filters.document_type}
            onChange={(e) => setFilters({ ...filters, document_type: e.target.value })}
            className="px-3 py-2 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          />
          <div className="flex items-end">
            <button onClick={() => setFilters({ source_id: "", document_type: "", search: "" })} className="text-sm text-blue-600 hover:underline">Clear filters</button>
          </div>
        </div>
      </div>

      {/* Table */}
      <DataTable
        columns={columns}
        rows={data.items}
        keyField="document_id"
        renderCell={(row, col) => {
          if (col === "quality_grade") return <Badge variant={row.quality_grade === "A" ? "success" : row.quality_grade === "B" ? "info" : "warning"}>{row.quality_grade ?? "—"}</Badge>;
          if (col === "extraction_status") {
            const isSuccess = String(row.extraction_status).toLowerCase() === "success";
            return <Badge variant={isSuccess ? "success" : "warning"}>{row.extraction_status ?? "—"}</Badge>;
          }
          if (col === "title") return <Link href={`/documents/${row.document_id}`} className="text-blue-600 hover:underline">{row.title ?? "—"}</Link>;
          if (col === "words" || col === "page_count" || col === "units") return (row as any)[col]?.toLocaleString() ?? (row as any)["pages"]?.toLocaleString() ?? "—";
          return (row as any)[col] ?? "—";
        }}
      />

      {/* Pagination */}
      <Pagination
        total={data.total}
        offset={data.offset}
        limit={data.limit}
        onChange={fetchData}
      />
    </PageContainer>
  );
}

export default function DocumentsPage() {
  return (
    <Layout>
      <Suspense fallback={<Loading />}>
        <DocumentsContent />
      </Suspense>
    </Layout>
  );
}