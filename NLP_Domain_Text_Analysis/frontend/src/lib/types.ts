/** TypeScript equivalents of the FastAPI response shapes.

These mirror the Pydantic models in ``backend/schemas/models.py``. Keep them in
sync: when the API changes, update the Python model, run the test suite, then
update this file so the frontend types remain accurate.
*/

/** A project identifier: letters, digits and _ . : + - only. */
export type Identifier = string;

/** The six query types Phase 3 recognises. */
export type QueryType =
  | "keyword"
  | "phrase"
  | "boolean_and"
  | "boolean_or"
  | "boolean_not"
  | "boolean_group";

/** Envelope for any paged response. */
export interface PageMeta {
  total: number;
  offset: number;
  limit: number;
  returned: number;
  has_more: boolean;
}

export interface Paged<T> extends PageMeta {
  items: T[];
}

/** Every error the API returns has exactly this shape. */
export interface ApiError {
  error: string;
  detail: string;
  hint?: string;
}

/** /api/health */
export interface HealthResponse {
  status: "ok" | "degraded";
  project: string;
  phase: number;
  generated_by: string;
  retrieval_available: boolean;
  retrieval_detail: {
    available: boolean;
    final_pipeline?: string;
    index_terms?: number;
    postings?: number;
    documents?: number;
    units?: number;
    indexed_units?: number;
    load_seconds?: number;
    pipelines?: string[];
    reason?: string;
  };
  phases: { phase1: boolean; phase2: boolean; phase3: boolean; phase4: boolean };
  uptime_seconds: number;
}

/** /api/statistics */
export interface StatisticsResponse {
  project: {
    phases: number[];
    domain: string;
    phases_available: { phase1: boolean; phase2: boolean; phase3: boolean; phase4: boolean };
    generated_by: { phase1?: string; phase3?: string; phase4?: string };
    elapsed_seconds: { phase3?: number; phase4?: number };
  };
  corpus: {
    documents: number;
    sources: number;
    pages: number;
    words: number;
    characters: number;
    baseline_tokens: number;
    sections: number;
    units: number;
    paragraphs: number;
    tables: number;
    figures: number;
    footnotes: number;
    units_by_type: Record<string, number>;
    documents_by_year: Record<string, number>;
    text_selection_policy?: string;
    selected_units?: number;
    per_document: Array<{
      document_id: string;
      title?: string;
      words?: number;
      units?: number;
      pages?: number;
    }>;
    per_source: Array<{
      source_id: string;
      source_title?: string;
      documents?: number;
      pages?: number;
      units?: number;
    }>;
  };
  sources: Array<{
    source_id: string;
    source_title?: string;
    publisher?: string;
    year?: string;
    source_type?: string;
    documents: number;
    observed_pages: number;
    observed_units: number;
    observed_baseline_tokens: number;
    document_ids: string[];
  }>;
  index: {
    available: boolean;
    final_pipeline?: string;
    pipeline_name?: string;
    terms?: number;
    postings?: number;
    units?: number;
    documents?: number;
    index_bytes?: number;
    postings_per_term?: number;
    singleton_terms?: number;
    hapax_ratio?: number;
    mean_term_frequency?: number;
    mean_document_frequency?: number;
    longest_posting_list?: number;
    longest_posting_term?: string;
  };
  pipelines: {
    available: boolean;
    final_pipeline?: string;
    final_pipeline_name?: string;
    score?: number;
    margin_over_runner_up?: number;
    reason?: string;
    criteria_weights?: Record<string, number>;
    pipelines: Array<{
      pipeline: string;
      name?: string;
      is_final: boolean;
      total_score?: number;
      criteria?: Record<string, any>;
      index_terms?: number;
      vocabulary_size?: number;
      total_postings?: number;
      processing_time_seconds?: number;
    }>;
  };
  phase2: {
    available: boolean;
    tokenizer_count?: number;
    tokenizers?: Array<{
      tokenizer: string;
      total_tokens?: number;
      vocabulary_size?: number;
      execution_time_seconds?: number;
    }>;
    most_compact_tokenizer?: string;
  };
  phase3: {
    available: boolean;
    final_pipeline?: string;
    final_pipeline_name?: string;
    queries_defined?: number;
    queries_answered?: number;
    answerability?: number;
    total_units_returned?: number;
    total_documents_returned?: number;
    mean_execution_time_ms?: number;
  };
  phase4: {
    available: boolean;
    judgments?: {
      pairs: number;
      relevant: number;
      not_relevant: number;
      pool_depth: number;
      annotator?: string;
      rubric?: string;
      per_query?: Record<string, any>;
    };
    queries_evaluated?: number;
    judged_depth?: number;
    unjudged_policy?: string;
    unit_level?: Record<string, any>;
    document_level?: Record<string, any>;
    runner_up?: Record<string, any>;
    validation?: { rules: number; passed: number };
  };
  validation: {
    phases: Record<string, { available: boolean; total: number; passed: number; all_passed: boolean }>;
    total: number;
    passed: number;
    all_passed: boolean;
  };
  sources_note: string;
}

/** /api/documents */
export interface DocumentSummary {
  document_id: string;
  source_id?: string;
  title?: string;
  filename?: string;
  publisher?: string;
  year?: string;
  document_type?: string;
  page_count?: number;
  pages?: number;
  file_size_bytes?: number;
  sha256?: string;
  ingestion_status?: string;
  extraction_status?: string;
  extraction_quality?: string;
  title_provenance?: string;
  publisher_provenance?: string;
  year_provenance?: string;
  detected_source?: string;
  sentences?: number;
  words?: number;
  characters?: number;
  baseline_token_count?: number;
  unique_words?: number;
  vocabulary_size?: number;
  raw_vocabulary_size?: number;
  average_document_length?: number;
  sentences_per_page?: number;
  sections?: number;
  units?: number;
  paragraphs?: number;
  tables?: number;
  figures?: number;
  footnotes?: number;
  raw_characters?: number;
  cleaned_characters?: number;
  quality_score?: number;
  quality_grade?: string;
  pages_with_low_text?: number;
  pages_likely_scanned?: number;
  warnings?: string;
}

export interface DocumentListResponse extends Paged<DocumentSummary> {
  filters: {
    source_id?: string;
    document_type?: string;
    ingestion_status?: string;
    search?: string;
  };
}

export interface DocumentDetail {
  document_id: string;
  source_id?: string;
  source?: {
    source_id: string;
    source_title?: string;
    publisher?: string;
    year?: string;
    source_type?: string;
    documents: number;
    observed_pages: number;
    observed_units: number;
    observed_baseline_tokens: number;
    document_ids: string[];
  };
  title?: string;
  filename?: string;
  publisher?: string;
  year?: string;
  document_type?: string;
  page_count?: number;
  pages: Array<{ page_id: string; page_number: number }>;
  sections: Array<{
    document_id: string;
    section_id: string;
    section_number: string;
    section_title: string;
    section_type?: string;
    page_number: number;
  }>;
  unit_count: number;
  units_by_type: Record<string, number>;
  unit_characters: number;
  unit_words: number;
  ingestion_status?: string;
  extraction_status?: string;
  quality_grade?: string;
  quality_score?: number;
  sha256?: string;
  units?: number;
  words?: number;
  characters?: number;
  baseline_token_count?: number;
  baseline_tokens?: number;
  vocabulary_size?: number;
}

/** /api/experiments/tokenization */
export interface TokenizerComparison {
  tokenizers: Array<{
    tokenizer: string;
    description?: string;
    documents_processed?: number;
    units_processed?: number;
    sentences?: number;
    total_tokens?: number;
    unique_tokens?: number;
    vocabulary_size?: number;
    avg_tokens_per_document?: number;
    avg_tokens_per_sentence?: number;
    avg_tokens_per_unit?: number;
    numeric_tokens?: number;
    date_tokens?: number;
    fiscal_year_tokens?: number;
    percentage_tokens?: number;
    currency_tokens?: number;
    abbreviation_tokens?: number;
    hyphenated_tokens?: number;
    special_financial_tokens?: number;
    execution_time_seconds?: number;
    strengths?: string;
    limitations?: string;
    financial_domain_behavior?: string;
  }>;
  examples: Array<any>;
  custom_rules: Array<any>;
  date_number_comparison: Array<any>;
  date_number_patterns: Array<any>;
  representative_sample: Array<any>;
  bpe_comparison: Array<any>;
}

/** /api/search */
export interface SearchRequest {
  query: string;
  query_type?: QueryType;
  top_k?: number;
  pipeline?: string;
}

export interface ScoreBreakdown {
  matched_term_count: number;
  term_component: number;
  phrase_hit: boolean;
  phrase_component: number;
  total_tf: number;
  tf_component: number;
  total_score: number;
}

export interface SearchHit {
  unit_id: string;
  document_id: string;
  source_id: string;
  page_number: number;
  section_id: string;
  section_number: string;
  section_title: string;
  unit_type: string;
  score: number;
  matched_terms: string;
  matched_terms_list: string[];
  matched_term_frequency_map: Record<string, number>;
  phrase_match: boolean;
  snippet: string;
  has_section_title: boolean;
  citation: string;
  score_breakdown?: ScoreBreakdown;
}

export interface SearchResponse {
  query: string;
  requested_query_type?: QueryType;
  query_type: QueryType;
  pipeline: string;
  pipeline_name: string;
  retrieval_method: string;
  total_results: number;
  returned: number;
  top_k: number;
  result_documents: number;
  matched_terms: string[];
  matched_terms_list: string[];
  missing_terms: string[];
  missing_terms_list: string[];
  excluded_terms: string[];
  execution_time_ms: number;
  max_results_available: number;
  results: SearchHit[];
  documents: Array<any>;
}

/** /api/search/parse */
export interface ParseResponse {
  query: string;
  query_type: QueryType;
  normalized?: string;
  valid: boolean;
  error?: string;
}

/** /api/index/term */
export interface TermLookupResponse {
  term: string;
  normalized_terms: string[];
  matched_terms: string[];
  missing_terms: string[];
  pipeline: string;
  searchable: boolean;
  max_results_available: number;
  term_key?: string;
  posting_frequency?: number;
  total_frequency?: number;
  document_frequency?: number;
  postings_total?: number;
  postings_returned?: number;
  document_ids?: string[];
  postings?: Array<{
    unit_id: string;
    document_id: string;
    source_id: string;
    page_number: number;
    section_id: string;
    section_number: string;
    section_title: string;
    unit_type: string;
    term_frequency: number;
    positions: number[];
    snippet: string;
  }>;
}

/** /api/evaluation */
export interface EvaluationReport {
  phase: string;
  selected_pipeline: string;
  judged_pipeline: string;
  inputs: Record<string, any>;
  queries: Array<{
    query_id: string;
    query: string;
    query_type: string;
    description: string;
    expected_domain: string;
  }>;
  judgment_summary: {
    pairs: number;
    relevant: number;
    not_relevant: number;
    pool_depth: number;
    annotator?: string;
    rubric?: string;
    per_query?: Record<string, any>;
  };
  judged_depth?: number;
  unjudged_policy?: string;
  queries_evaluated?: number;
  queries_failed?: number;
  problems?: Array<any>;
  runner_up?: Record<string, any>;
  unit_level?: Record<string, any>;
  document_level?: Record<string, any>;
  at_k?: Record<string, any>;
  per_query?: Paged<any>;
  comparison?: Array<any>;
  validation?: Array<any>;
  notes: string[];
}

/** /api/evaluation/judgments */
export interface JudgmentRow {
  query_id: string;
  query?: string;
  unit_id: string;
  rank?: string;
  document_id?: string;
  page_number?: string;
  section_number?: string;
  relevance: "0" | "1";
  notes?: string;
  annotator?: string;
  judgment_method?: string;
  judged_at?: string;
}

export interface JudgmentCreate {
  query_id: string;
  unit_id: string;
  relevance: boolean;
  annotator: string;
  notes?: string;
  query?: string;
  rank?: number;
  document_id?: string;
}

export interface JudgmentResponse {
  saved: boolean;
  judgment: {
    query_id: string;
    unit_id: string;
    relevance: number;
    relevance_label: string;
    rank?: string;
    notes?: string;
    annotator: string;
    judgment_method: string;
    judged_at?: string;
  };
  judgment_count: number;
  relevant_count: number;
  validation: { rules: number; passed: number; all_passed: boolean };
  message: string;
}