/** Single document detail page. */

"use client";

import { useParams } from "next/navigation";
import { useEffect, useState, Suspense } from "react";
import { Layout, PageContainer, Loading, ErrorState, Badge, SectionCard } from "@/components";
import { api, type DocumentDetail } from "@/lib/api";

function DocumentDetailContent() {
  const params = useParams();
  const documentId = params.documentId as string;
  const [doc, setDoc] = useState<DocumentDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    api.document(documentId)
      .then((d) => { setDoc(d.document); setLoading(false); })
      .catch((e) => { setError(e); setLoading(false); });
  }, [documentId]);

  if (loading) return <Loading message={`Loading ${documentId}…`} />;
  if (error) return <ErrorState error={error} />;
  if (!doc) return null;

  return (
    <PageContainer title={`Document: ${doc.document_id}`} description={doc.title ?? undefined}>
      <div className="grid md:grid-cols-3 gap-6">
        <div className="md:col-span-2 space-y-6">
          <SectionCard title="Metadata">
            <dl className="grid grid-cols-2 gap-3 text-sm">
              <dt className="text-gray-500">Source</dt>
              <dd className="font-medium">{doc.source?.source_title ?? doc.source_id ?? "—"}</dd>
              <dt className="text-gray-500">Publisher</dt>
              <dd>{doc.publisher ?? "—"}</dd>
              <dt className="text-gray-500">Year</dt>
              <dd>{doc.year ?? "—"}</dd>
              <dt className="text-gray-500">Document type</dt>
              <dd>{doc.document_type ?? "—"}</dd>
              <dt className="text-gray-500">Ingestion status</dt>
              <dd><Badge variant={doc.ingestion_status === "success" ? "success" : "warning"}>{doc.ingestion_status ?? "—"}</Badge></dd>
              <dt className="text-gray-500">Extraction status</dt>
              <dd><Badge variant={doc.extraction_status === "success" ? "success" : "warning"}>{doc.extraction_status ?? "—"}</Badge></dd>
              <dt className="text-gray-500">Quality</dt>
              <dd><Badge variant={doc.quality_grade === "A" ? "success" : doc.quality_grade === "B" ? "info" : "warning"}>{doc.quality_grade ?? "—"} ({doc.quality_score?.toFixed(1) ?? "—"})</Badge></dd>
              <dt className="text-gray-500">SHA-256</dt>
              <dd className="font-mono text-xs break-all">{doc.sha256 ?? "—"}</dd>
            </dl>
          </SectionCard>

          <SectionCard title="Content Statistics">
            <div className="grid grid-cols-3 gap-4 text-sm">
              <div><span className="text-gray-500">Pages</span><br /><span className="font-bold text-2xl">{doc.pages?.length.toLocaleString() ?? "—"}</span></div>
              <div><span className="text-gray-500">Units</span><br /><span className="font-bold text-2xl">{doc.units?.toLocaleString() ?? "—"}</span></div>
              <div><span className="text-gray-500">Words</span><br /><span className="font-bold text-2xl">{doc.words?.toLocaleString() ?? "—"}</span></div>
              <div><span className="text-gray-500">Characters</span><br /><span className="font-bold text-2xl">{doc.characters?.toLocaleString() ?? "—"}</span></div>
              <div><span className="text-gray-500">Baseline tokens</span><br /><span className="font-bold text-2xl">{doc.baseline_tokens?.toLocaleString() ?? "—"}</span></div>
              <div><span className="text-gray-500">Vocabulary</span><br /><span className="font-bold text-2xl">{doc.vocabulary_size?.toLocaleString() ?? "—"}</span></div>
            </div>
          </SectionCard>

          {doc.pages && doc.pages.length > 0 && (
            <SectionCard title={`Pages (${doc.pages.length})`}>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                {doc.pages.slice(0, 20).map((p: any) => (
                  <div key={p.page_id} className="text-sm p-2 bg-gray-50 rounded">
                    <span className="font-medium">{p.page_id}</span>
                    <span className="text-gray-500 ml-2">p.{p.page_number}</span>
                  </div>
                ))}
                {doc.pages.length > 20 && <span className="text-gray-500 text-sm col-span-full">…and {doc.pages.length - 20} more</span>}
              </div>
            </SectionCard>
          )}

          {doc.sections && doc.sections.length > 0 && (
            <SectionCard title={`Sections (${doc.sections.length})`}>
              <div className="overflow-x-auto">
                <table className="min-w-full text-sm">
                  <thead><tr className="bg-gray-50"><th className="px-3 py-2 text-left">Section</th><th className="px-3 py-2 text-left">Title</th><th className="px-3 py-2 text-left">Type</th><th className="px-3 py-2 text-left">Page</th></tr></thead>
                  <tbody className="divide-y divide-gray-200">
                    {doc.sections.map((s: any) => (
                      <tr key={s.section_id} className="hover:bg-gray-50">
                        <td className="px-3 py-2 font-mono">{s.section_number}</td>
                        <td className="px-3 py-2">{s.section_title}</td>
                        <td className="px-3 py-2"><Badge>{s.section_type ?? "—"}</Badge></td>
                        <td className="px-3 py-2">{s.page_number}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </SectionCard>
          )}
        </div>

        <div className="space-y-6">
          <SectionCard title="Unit Census">
            <dl className="space-y-2 text-sm">
              {Object.entries(doc.units_by_type ?? {}).map(([type, count]) => (
                <div key={type} className="flex justify-between"><dt className="text-gray-500 capitalize">{type}</dt><dd className="font-mono">{count}</dd></div>
              ))}
              <div className="border-t pt-2 flex justify-between font-medium">
                <dt>Total</dt><dd>{doc.unit_count?.toLocaleString()}</dd>
              </div>
            </dl>
          </SectionCard>

          <SectionCard title="Characters / Words">
            <div className="grid grid-cols-2 gap-4 text-center">
              <div className="p-3 bg-gray-50 rounded"><div className="text-2xl font-bold">{doc.unit_characters?.toLocaleString() ?? "—"}</div><div className="text-sm text-gray-500">Characters</div></div>
              <div className="p-3 bg-gray-50 rounded"><div className="text-2xl font-bold">{doc.unit_words?.toLocaleString() ?? "—"}</div><div className="text-sm text-gray-500">Words</div></div>
            </div>
          </SectionCard>
        </div>
      </div>
    </PageContainer>
  );
}

export default function DocumentPage() {
  return (
    <Layout>
      <Suspense fallback={<Loading />}>
        <DocumentDetailContent />
      </Suspense>
    </Layout>
  );
}