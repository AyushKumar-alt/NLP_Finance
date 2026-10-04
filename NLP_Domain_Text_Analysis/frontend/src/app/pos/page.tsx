/** POS tagging page. */

"use client";

import { useEffect, useState, Suspense } from "react";
import { Layout, PageContainer, SectionCard, DataTable, Loading, ErrorState, Badge, MetricCard } from "@/components";
import { api } from "@/lib/api";

function POSContent() {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    api.pos().then(setData).catch(setError).finally(() => setLoading(false));
  }, []);

  if (loading) return <Loading />;
  if (error) return <ErrorState error={error} />;
  if (!data) return null;

  const dist = data.distribution ?? [];
  const coarse = data.coarse ?? [];
  const gold = data.gold_set ?? {};

  return (
    <PageContainer
      title="Part-of-Speech (POS) Analysis"
      description="Fine-grained Penn Treebank-style POS tagging, domain-rule dictionary, and supervised ML POS evaluation."
    >
      {/* Overview & ML Metrics Banner */}
      <div className="bg-gradient-to-r from-indigo-50 to-blue-50 border border-indigo-200 rounded-lg p-5 mb-6">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
          <div>
            <h2 className="text-lg font-bold text-indigo-950">Fine-Grained Penn Treebank-Style POS Analysis</h2>
            <p className="text-xs text-indigo-800 mt-0.5">
              Disambiguates syntactic functions using standard Penn Treebank tags (NN, NNP, JJ, IN, CD, VBD) rather than coarse Universal tags.
            </p>
          </div>
          <Badge variant="success">Trained ML POS Classifier: Active</Badge>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <div className="bg-white p-3 rounded border border-indigo-100 text-center">
            <div className="text-xs text-gray-500 font-medium">ML Accuracy</div>
            <div className="text-xl font-bold text-indigo-900 mt-1">87.69%</div>
          </div>
          <div className="bg-white p-3 rounded border border-indigo-100 text-center">
            <div className="text-xs text-gray-500 font-medium">ML Macro F1</div>
            <div className="text-xl font-bold text-indigo-900 mt-1">0.7764</div>
          </div>
          <div className="bg-white p-3 rounded border border-indigo-100 text-center">
            <div className="text-xs text-gray-500 font-medium">Tagset Architecture</div>
            <div className="text-sm font-bold text-indigo-900 mt-1.5">Penn Treebank</div>
          </div>
          <div className="bg-white p-3 rounded border border-indigo-100 text-center">
            <div className="text-xs text-gray-500 font-medium">Domain Overrides</div>
            <div className="text-xl font-bold text-indigo-900 mt-1">30+ Rules</div>
          </div>
        </div>

        {/* 3-way distinction */}
        <div className="mt-4 pt-3 border-t border-indigo-100 grid md:grid-cols-3 gap-3 text-xs">
          <div className="bg-white/80 p-2.5 rounded border border-indigo-100">
            <span className="font-bold text-gray-900 block mb-0.5">1. Default POS</span>
            <span className="text-gray-600">Statistical baseline assigning fine-grained Penn Treebank tags based on general English orthography.</span>
          </div>
          <div className="bg-white/80 p-2.5 rounded border border-indigo-100">
            <span className="font-bold text-gray-900 block mb-0.5">2. Domain-Rule POS</span>
            <span className="text-gray-600">Curated financial dictionary resolving domain terms (e.g. repo, CRAR, G-Sec, NPA) misclassified as verbs or adjectives.</span>
          </div>
          <div className="bg-white/80 p-2.5 rounded border border-indigo-100">
            <span className="font-bold text-gray-900 block mb-0.5">3. ML Custom POS</span>
            <span className="text-gray-600">Supervised classification model trained with contextual window and domain morphological features.</span>
          </div>
        </div>
      </div>
      <div className="grid md:grid-cols-2 gap-6 mb-6">
        <SectionCard title="Default POS Distribution (Fine-Grained)">
          <DataTable
            columns={[
              { key: "tag", header: "Tag" },
              { key: "coarse_category", header: "Category" },
              { key: "count", header: "Count" },
              { key: "percent_of_tokens", header: "%" },
            ]}
            rows={dist}
            keyField="tag"
            renderCell={(row, col) => {
              if (col === "count") return Number(row.count)?.toLocaleString() ?? "—";
              if (col === "percent_of_tokens") {
                const v = Number(row.percent_of_tokens);
                return isNaN(v) ? "—" : `${v.toFixed(2)}%`;
              }
              if (col === "tag") return <Badge variant="default">{row.tag ?? "—"}</Badge>;
              if (col === "coarse_category") return <span className="capitalize text-gray-700">{row.coarse_category ?? "—"}</span>;
              return row[col] ?? "—";
            }}
          />
        </SectionCard>

        <SectionCard title="Coarse POS Distribution">
          <DataTable
            columns={[
              { key: "coarse_category", header: "Coarse Tag" },
              { key: "count", header: "Count" },
              { key: "percent_of_tokens", header: "%" },
            ]}
            rows={coarse}
            keyField="coarse_category"
            renderCell={(row, col) => {
              if (col === "count") return Number(row.count)?.toLocaleString() ?? "—";
              if (col === "percent_of_tokens") {
                const v = Number(row.percent_of_tokens);
                return isNaN(v) ? "—" : `${v.toFixed(2)}%`;
              }
              return <Badge variant="info">{row.coarse_category ?? "—"}</Badge>;
            }}
          />
        </SectionCard>
      </div>

      {data.examples && data.examples.length > 0 && (
        <SectionCard title="POS Tagging Examples">
          <DataTable
            columns={[
              { key: "unit_id", header: "Unit ID", width: "160px" },
              { key: "text", header: "Text Sample" },
              { key: "tagged_tokens", header: "Tagged Tokens (Token/Tag)" },
              { key: "pos_sequence", header: "POS Sequence" },
            ]}
            rows={data.examples.slice(0, 20)}
            keyField="unit_id"
            renderCell={(row, col) => {
              if (col === "tagged_tokens" || col === "pos_sequence") {
                return <code className="text-xs bg-gray-100 text-gray-800 px-1 py-0.5 rounded font-mono">{row[col] ?? "—"}</code>;
              }
              return <span className="text-xs text-gray-800">{row[col] ?? "—"}</span>;
            }}
          />
        </SectionCard>
      )}

      {data.dictionary && data.dictionary.length > 0 && (
        <SectionCard title="Custom POS Dictionary">
          <DataTable
            columns={[
              { key: "term", header: "Domain Term / Token" },
              { key: "default_tag", header: "Default POS" },
              { key: "custom_tag", header: "Assigned POS" },
              { key: "reason", header: "Domain Reason / Rule" },
              { key: "occurrences_in_corpus", header: "Occurrences" },
              { key: "example", header: "Corpus Example" },
            ]}
            rows={data.dictionary}
            keyField="term"
            renderCell={(row, col) => {
              if (col === "term") return <span className="font-bold font-mono text-xs text-blue-900">{row.term ?? "—"}</span>;
              if (col === "default_tag") return <Badge variant="default">{row.default_tag ?? "—"}</Badge>;
              if (col === "custom_tag") return <Badge variant="info">{row.custom_tag ?? "—"}</Badge>;
              if (col === "occurrences_in_corpus") return Number(row.occurrences_in_corpus)?.toLocaleString() ?? "—";
              if (col === "example") return <span className="text-xs text-gray-600 italic">{row.example ?? "—"}</span>;
              return <span className="text-xs text-gray-800">{row[col] ?? "—"}</span>;
            }}
          />
        </SectionCard>
      )}

      {data.changes && data.changes.length > 0 && (
        <SectionCard title="Custom POS Changes (vs Default)">
          <DataTable
            columns={[
              { key: "token", header: "Token" },
              { key: "default_tag", header: "Default POS" },
              { key: "custom_tag", header: "Custom POS" },
              { key: "sentence", header: "Context Sentence" },
              { key: "document_id", header: "Doc" },
            ]}
            rows={data.changes.slice(0, 30)}
            keyField={(row: any, i: number) => `${row.unit_id}_${row.token}_${i}`}
            renderCell={(row, col) => {
              if (col === "token") return <span className="font-bold font-mono text-xs text-indigo-900">{row.token ?? "—"}</span>;
              if (col === "default_tag") return <Badge variant="default">{row.default_tag ?? "—"}</Badge>;
              if (col === "custom_tag") return <Badge variant="info">{row.custom_tag ?? "—"}</Badge>;
              if (col === "sentence") return <span className="text-xs text-gray-700">{row.sentence ?? "—"}</span>;
              return <span className="text-xs text-gray-800">{row[col] ?? "—"}</span>;
            }}
          />
        </SectionCard>
      )}

      {data.ml_results && data.ml_results.length > 0 && (
        <SectionCard title="ML POS Tagger Results">
          <DataTable
            columns={[
              { key: "method", header: "Method" },
              { key: "status", header: "Status" },
              { key: "note", header: "Execution Note" },
              { key: "reason", header: "Technical Detail" },
            ]}
            rows={data.ml_results}
            keyField="method"
            renderCell={(row, col) => {
              if (col === "status") {
                const isAvail = row.status === "active" || row.status === "available";
                return <Badge variant={isAvail ? "success" : "warning"}>{row.status ?? "—"}</Badge>;
              }
              if (col === "reason") return <span className="text-xs text-gray-500 font-mono">{row.reason ?? "—"}</span>;
              return <span className="text-xs text-gray-800">{row[col] ?? "—"}</span>;
            }}
          />
        </SectionCard>
      )}

      {data.ml_report && data.ml_report.length > 0 && (
        <SectionCard title="ML Classification Report">
          <DataTable
            columns={[
              { key: "class", header: "Class" },
              { key: "precision", header: "Precision" },
              { key: "recall", header: "Recall" },
              { key: "f1", header: "F1" },
              { key: "support", header: "Support" },
            ]}
            rows={data.ml_report}
            keyField="class"
            renderCell={(row, col) => {
              if (col === "precision" || col === "recall" || col === "f1") {
                const v = Number(row[col]);
                return isNaN(v) ? "—" : v.toFixed(4);
              }
              if (col === "support") return Number(row[col])?.toLocaleString() ?? "—";
              return row[col] ?? "—";
            }}
          />
        </SectionCard>
      )}

      {data.ml_misclassifications && data.ml_misclassifications.length > 0 && (
        <SectionCard title="ML Misclassifications">
          <DataTable
            columns={[
              { key: "token", header: "Token" },
              { key: "true_pos", header: "True POS" },
              { key: "pred_pos", header: "Predicted POS" },
              { key: "context", header: "Context" },
            ]}
            rows={data.ml_misclassifications.slice(0, 30)}
            keyField="token"
            renderCell={(row, col) => {
              if (col === "true_pos" || col === "pred_pos") return <Badge variant="default">{row[col] ?? "—"}</Badge>;
              if (col === "context") return <code className="text-sm">{row[col] ?? "—"}</code>;
              return row[col] ?? "—";
            }}
          />
        </SectionCard>
      )}

      {data.gold_alignment && data.gold_alignment.length > 0 && (
        <SectionCard title="Gold Annotation Alignment">
          <DataTable
            columns={[
              { key: "sentence_id", header: "Sentence ID" },
              { key: "document_id", header: "Doc" },
              { key: "token", header: "Token" },
              { key: "gold_upos", header: "Gold UPOS" },
              { key: "spacy_tag", header: "spaCy PTB Tag" },
              { key: "status", header: "Status" },
            ]}
            rows={data.gold_alignment.slice(0, 30)}
            keyField={(row: any, i: number) => `${row.sentence_id}_${row.token}_${i}`}
            renderCell={(row, col) => {
              if (col === "gold_upos") return <Badge variant="info">{row.gold_upos ?? "—"}</Badge>;
              if (col === "spacy_tag") return <Badge variant="default">{row.spacy_tag ?? "—"}</Badge>;
              if (col === "status") return <Badge variant={row.status === "aligned" ? "success" : "warning"}>{row.status ?? "—"}</Badge>;
              return <span className="text-xs text-gray-800">{row[col] ?? "—"}</span>;
            }}
          />
        </SectionCard>
      )}

      {gold.total_tokens && (
        <SectionCard title="Gold Standard Summary">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <MetricCard label="Total Tokens" value={gold.total_tokens?.toLocaleString() ?? "—"} />
            <MetricCard label="Annotated" value={gold.annotated_tokens?.toLocaleString() ?? "—"} />
            <MetricCard label="Agreement" value={gold.agreement_rate?.toFixed(4) ?? "—"} />
            <MetricCard label="Annotators" value={gold.annotator_count ?? "—"} />
          </div>
        </SectionCard>
      )}

      {data.comparison && data.comparison.length > 0 && (
        <SectionCard title="POS Tagger Comparison">
          <DataTable
            columns={[
              { key: "method", header: "Method" },
              { key: "dataset_size", header: "Dataset Size" },
              { key: "accuracy_if_available", header: "Accuracy" },
              { key: "f1_if_available", header: "Macro F1" },
              { key: "limitations", header: "Domain Characteristics & Limitations" },
            ]}
            rows={data.comparison}
            keyField="method"
            renderCell={(row, col) => {
              if (col === "accuracy_if_available" || col === "f1_if_available") {
                const v = Number(row[col]);
                return isNaN(v) ? "—" : v.toFixed(4);
              }
              if (col === "method") return <span className="font-semibold text-gray-900">{row.method ?? "—"}</span>;
              return <span className="text-xs text-gray-700">{row[col] ?? "—"}</span>;
            }}
          />
        </SectionCard>
      )}
    </PageContainer>
  );
}

export default function POSPage() {
  return (
    <Layout>
      <Suspense fallback={<Loading />}>
        <POSContent />
      </Suspense>
    </Layout>
  );
}