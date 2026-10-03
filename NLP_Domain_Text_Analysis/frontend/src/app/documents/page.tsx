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
    { key: "pages" as const, header: "Pages", width: "80px" },
    { key: "words" as const, header: "Words", width: "100px" },
    { key: "quality_grade" as const, header: "Quality", width: "100px" },
    { key: "extraction_status" as const, header: "Status", width: "120px" },
  ];

  return (
    <PageContainer title="Documents" description="Browse the 31 documents in the corpus.">
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
          if (col === "extraction_status") return <Badge variant={row.extraction_status === "success" ? "success" : "warning"}>{row.extraction_status ?? "—"}</Badge>;
          if (col === "title") return <Link href={`/documents/${row.document_id}`} className="text-blue-600 hover:underline">{row.title ?? "—"}</Link>;
          if (col === "words" || col === "pages" || col === "units") return (row as any)[col]?.toLocaleString() ?? "—";
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