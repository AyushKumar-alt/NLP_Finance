/** Reusable UI components for data display. */

import { ReactNode } from "react";

export function Loading({ message = "Loading…" }: { message?: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-12 text-gray-600">
      <svg
        className="animate-spin h-8 w-8 text-blue-600 mb-3"
        xmlns="http://www.w3.org/2000/svg"
        fill="none"
        viewBox="0 0 24 24"
      >
        <circle
          className="opacity-25"
          cx="12"
          cy="12"
          r="10"
          stroke="currentColor"
          strokeWidth="4"
        />
        <path
          className="opacity-75"
          fill="currentColor"
          d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
        />
      </svg>
      <p>{message}</p>
    </div>
  );
}

export function ErrorState({ error, retry, hint }: { error: Error | { message: string }; retry?: () => void; hint?: string }) {
  return (
    <div className="p-6 bg-red-50 border border-red-200 rounded-lg">
      <div className="flex items-start gap-3">
        <svg
          className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5"
          fill="currentColor"
          viewBox="0 0 20 20"
        >
          <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
        </svg>
        <div className="flex-1">
          <h3 className="text-red-900 font-medium">Failed to load</h3>
          <p className="text-red-700 text-sm mt-1">{error.message}</p>
          {hint && <p className="text-red-600 text-xs mt-2">{hint}</p>}
          {retry && (
            <button
              onClick={retry}
              className="mt-3 text-sm text-red-900 underline hover:text-red-700"
            >
              Try again
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

export function EmptyState({ message, action }: { message: string; action?: ReactNode }) {
  return (
    <div className="text-center py-12 text-gray-500">
      <svg className="mx-auto h-12 w-12 text-gray-300 mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
      </svg>
      <p className="text-lg">{message}</p>
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}

export function DataTable<T extends Record<string, any>>({
  columns,
  rows,
  keyField,
  emptyMessage = "No data to display",
  renderCell,
  className = "",
}: {
  columns: Array<{ key: string; header: string; width?: string }>;
  rows: T[];
  keyField: string;
  emptyMessage?: string;
  renderCell?: (row: T, col: string) => ReactNode;
  className?: string;
}) {
  if (rows.length === 0) {
    return <EmptyState message={emptyMessage} />;
  }
  return (
    <div className={`overflow-x-auto rounded-lg border border-gray-200 ${className}`}>
      <table className="min-w-full divide-y divide-gray-200">
        <thead className="bg-gray-50">
          <tr>
            {columns.map((col) => (
              <th
                key={String(col.key)}
                scope="col"
                className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                style={{ width: col.width }}
              >
                {col.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="bg-white divide-y divide-gray-200">
          {rows.map((row, rowIndex) => (
            <tr key={`${String(row[keyField])}-${rowIndex}`} className="hover:bg-gray-50">
              {columns.map((col) => (
                <td key={`${String(row[keyField])}-${rowIndex}-${String(col.key)}`} className="px-4 py-3 text-sm text-gray-900">
                  {renderCell ? renderCell(row, col.key) : String(row[col.key] ?? "")}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function Pagination({
  total,
  offset,
  limit,
  onChange,
  showTotal = true,
}: {
  total: number;
  offset: number;
  limit: number;
  onChange: (offset: number) => void;
  showTotal?: boolean;
}) {
  const pageCount = Math.ceil(total / limit);
  const currentPage = Math.floor(offset / limit);
  if (pageCount <= 1) return null;

  return (
    <nav className="flex items-center justify-between px-4 py-3 border-t border-gray-200 bg-gray-50">
      {showTotal && (
        <p className="text-sm text-gray-700">
          Showing {offset + 1}–{Math.min(offset + limit, total)} of {total}
        </p>
      )}
      <div className="flex items-center gap-2">
        <button
          onClick={() => onChange(Math.max(0, offset - limit))}
          disabled={offset === 0}
          className="px-3 py-1 text-sm border border-gray-300 rounded hover:bg-gray-100 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Previous
        </button>
        <span className="px-3 text-sm text-gray-700">
          Page {currentPage + 1} of {pageCount}
        </span>
        <button
          onClick={() => onChange(offset + limit)}
          disabled={offset + limit >= total}
          className="px-3 py-1 text-sm border border-gray-300 rounded hover:bg-gray-100 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Next
        </button>
      </div>
    </nav>
  );
}

export function Badge({ children, variant = "default", className = "" }: { children: ReactNode; variant?: "default" | "success" | "warning" | "error" | "info"; className?: string }) {
  const variants = {
    default: "bg-gray-100 text-gray-800",
    success: "bg-green-100 text-green-800",
    warning: "bg-yellow-100 text-yellow-800",
    error: "bg-red-100 text-red-800",
    info: "bg-blue-100 text-blue-800",
  };
  return (
    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${variants[variant]}`}>
      {children}
    </span>
  );
}

export function MetricCard({ label, value, change, unit = "" }: { label: string; value: string | number; change?: string; unit?: string | ReactNode }) {
  return (
    <div className="bg-white border border-gray-200 rounded-lg p-5">
      <p className="text-sm font-medium text-gray-500">{label}</p>
      <p className="mt-2 text-3xl font-bold text-gray-900">
        {value}
        {unit && <span className="text-lg font-normal text-gray-500 ml-1">{unit}</span>}
      </p>
      {change && (
        <p className="mt-2 text-sm text-green-600">{change}</p>
      )}
    </div>
  );
}

export function SectionCard({ title, children, footer }: { title: string; children: ReactNode; footer?: ReactNode }) {
  return (
    <div className="bg-white border border-gray-200 rounded-lg">
      <div className="px-5 py-4 border-b border-gray-200">
        <h3 className="text-lg font-semibold text-gray-900">{title}</h3>
      </div>
      <div className="p-5">{children}</div>
      {footer && <div className="px-5 py-4 border-t border-gray-200 bg-gray-50">{footer}</div>}
    </div>
  );
}