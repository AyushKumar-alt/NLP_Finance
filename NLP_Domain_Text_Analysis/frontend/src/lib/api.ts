/** Centralised API client for the Next.js frontend.

Every call returns a typed result or throws an ``ApiError`` that the UI can
display uniformly. The client adds request timeouts and a consistent error
shape so components don't need bespoke handling.
*/

import { API_BASE_URL } from "./config";
import type {
  ApiError,
  DocumentDetail,
  DocumentListResponse,
  DocumentSummary,
  EvaluationReport,
  HealthResponse,
  JudgmentCreate,
  JudgmentResponse,
  JudgmentRow,
  ParseResponse,
  SearchHit,
  SearchRequest,
  SearchResponse,
  StatisticsResponse,
  TermLookupResponse,
  TokenizerComparison,
  Paged,
} from "./types";

export class ApiClientError extends Error {
  constructor(
    public readonly error: string,
    public readonly detail: string,
    public readonly hint?: string,
    public readonly status: number = 500
  ) {
    super(detail);
    this.name = "ApiClientError";
  }

  static fromResponse(response: Response, data: ApiError): ApiClientError {
    return new ApiClientError(data.error, data.detail, data.hint, response.status);
  }
}

async function handle<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let data: ApiError;
    try {
      data = await response.json();
    } catch {
      data = { error: "http_error", detail: response.statusText };
    }
    throw ApiClientError.fromResponse(response, data);
  }
  if (response.status === 204) {
    return undefined as T;
  }
  return response.json();
}

function buildUrl(path: string, params?: Record<string, any>): string {
  const url = new URL(path, API_BASE_URL);
  if (params) {
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== null && value !== "") {
        url.searchParams.append(key, String(value));
      }
    });
  }
  return url.toString();
}

export const api = {
  // System
  health: () => fetch(buildUrl("/api/health")).then((r) => handle<HealthResponse>(r)),

  statistics: () =>
    fetch(buildUrl("/api/statistics")).then((r) => handle<StatisticsResponse>(r)),

  // Corpus
  documents: (params?: {
    source_id?: string;
    document_type?: string;
    ingestion_status?: string;
    search?: string;
    limit?: number;
    offset?: number;
  }) =>
    fetch(buildUrl("/api/documents", params)).then((r) =>
      handle<DocumentListResponse>(r)
    ),

  document: (documentId: string) =>
    fetch(buildUrl(`/api/documents/${encodeURIComponent(documentId)}`)).then((r) =>
      handle<{ document_id: string; document: DocumentDetail }>(r)
    ),

  unit: (unitId: string) =>
    fetch(buildUrl(`/api/units/${encodeURIComponent(unitId)}`)).then((r) =>
      handle<{ unit_id: string; unit: any }>(r)
    ),

  // Experiments (Phase 2)
  experimentsSummary: () =>
    fetch(buildUrl("/api/experiments/summary")).then((r) => handle<any>(r)),

  tokenization: () =>
    fetch(buildUrl("/api/experiments/tokenization")).then((r) =>
      handle<TokenizerComparison>(r)
    ),

  tokenizeLive: (text: string) =>
    fetch(buildUrl("/api/experiments/tokenize"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    }).then((r) => handle<any>(r)),

  preprocessing: () =>
    fetch(buildUrl("/api/experiments/preprocessing")).then((r) => handle<any>(r)),

  stemming: () =>
    fetch(buildUrl("/api/experiments/stemming")).then((r) => handle<any>(r)),

  lemmatization: () =>
    fetch(buildUrl("/api/experiments/lemmatization")).then((r) => handle<any>(r)),

  pos: () =>
    fetch(buildUrl("/api/experiments/pos")).then((r) => handle<any>(r)),

  ner: (params?: {
    kind?: "general" | "domain";
    label?: string;
    document_id?: string;
    search?: string;
    limit?: number;
    offset?: number;
  }) =>
    fetch(buildUrl("/api/experiments/ner", params)).then((r) => handle<any>(r)),

  nerSidebar: () =>
    fetch(buildUrl("/api/experiments/ner/sidebar")).then((r) => handle<any>(r)),

  ngrams: (params?: {
    n?: number;
    search?: string;
    domain_only?: boolean;
    limit?: number;
    offset?: number;
  }) =>
    fetch(buildUrl("/api/experiments/ngrams", params)).then((r) => handle<any>(r)),

  bpe: (params?: { search?: string; limit?: number; offset?: number }) =>
    fetch(buildUrl("/api/experiments/bpe", params)).then((r) => handle<any>(r)),

  // Retrieval (Phase 3)
  search: (request: SearchRequest) =>
    fetch(buildUrl("/api/search"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    }).then((r) => handle<SearchResponse>(r)),

  parseQuery: (query: string, query_type?: string) =>
    fetch(buildUrl("/api/search/parse"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, query_type }),
    }).then((r) => handle<ParseResponse>(r)),

  queries: () =>
    fetch(buildUrl("/api/queries")).then((r) => handle<Array<any>>(r)),

  queryHistory: () =>
    fetch(buildUrl("/api/queries/history")).then((r) => handle<Array<any>>(r)),

  storedResults: (queryId: string, params?: { limit?: number; offset?: number }) =>
    fetch(buildUrl(`/api/queries/${encodeURIComponent(queryId)}/results`, params)).then((r) =>
      handle<Paged<any>>(r)
    ),

  term: (term: string, limit?: number) =>
    fetch(buildUrl("/api/index/term", { term, limit })).then((r) =>
      handle<TermLookupResponse>(r)
    ),

  terms: (params?: { limit?: number; order?: string }) =>
    fetch(buildUrl("/api/index/terms", params)).then((r) => handle<any>(r)),

  indexStatistics: () =>
    fetch(buildUrl("/api/index/statistics")).then((r) => handle<any>(r)),

  pipelines: () =>
    fetch(buildUrl("/api/pipelines")).then((r) => handle<Array<any>>(r)),

  pipelineSelection: () =>
    fetch(buildUrl("/api/pipelines/selection")).then((r) => handle<any>(r)),

  pipeline: (pipeline: string) =>
    fetch(buildUrl(`/api/pipelines/${encodeURIComponent(pipeline)}`)).then((r) =>
      handle<any>(r)
    ),

  phase3Validation: () =>
    fetch(buildUrl("/api/validation")).then((r) => handle<any>(r)),

  // Evaluation (Phase 4)
  evaluationReport: () =>
    fetch(buildUrl("/api/evaluation")).then((r) => handle<EvaluationReport>(r)),

  evaluationSummary: () =>
    fetch(buildUrl("/api/evaluation/summary")).then((r) => handle<any>(r)),

  evaluationResults: (params?: {
    query_id?: string;
    limit?: number;
    offset?: number;
  }) =>
    fetch(buildUrl("/api/evaluation/results", params)).then((r) => handle<Paged<any>>(r)),

  evaluationAggregates: () =>
    fetch(buildUrl("/api/evaluation/aggregates")).then((r) => handle<any>(r)),

  evaluationComparison: () =>
    fetch(buildUrl("/api/evaluation/comparison")).then((r) => handle<any>(r)),

  judgments: (params?: {
    query_id?: string;
    relevance?: boolean;
    limit?: number;
    offset?: number;
  }) =>
    fetch(buildUrl("/api/evaluation/judgments", params)).then((r) =>
      handle<Paged<JudgmentRow>>(r)
    ),

  saveJudgment: (judgment: JudgmentCreate) =>
    fetch(buildUrl("/api/evaluation/judgments"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(judgment),
    }).then((r) => handle<JudgmentResponse>(r)),

  phase4Validation: () =>
    fetch(buildUrl("/api/evaluation/validation")).then((r) => handle<any>(r)),
};

/** Hook-friendly wrapper that returns ``{ data, error, loading }``. */
export function useApi<T>(promise: Promise<T>): Promise<{
  data: T | null;
  error: ApiClientError | null;
  loading: boolean;
}> {
  let loading = true;
  const result = promise
    .then((data) => ({ data, error: null as ApiClientError | null, loading: false }))
    .catch((error: ApiClientError) => ({ data: null as T | null, error, loading: false }));
  return result;
}

/** Type helper for components that want to pattern-match on the error. */
export function isApiError(error: unknown): error is ApiClientError {
  return error instanceof ApiClientError;
}

export type {
  ApiError,
  DocumentDetail,
  DocumentListResponse,
  DocumentSummary,
  EvaluationReport,
  HealthResponse,
  JudgmentCreate,
  JudgmentRow,
  JudgmentResponse,
  PageMeta,
  ParseResponse,
  Paged,
  QueryType,
  SearchHit,
  SearchRequest,
  SearchResponse,
  StatisticsResponse,
  TermLookupResponse,
  TokenizerComparison,
} from "./types";