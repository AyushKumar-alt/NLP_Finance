/** Shared layout components. */

import Link from "next/link";
import { usePathname } from "next/navigation";
import { ReactNode } from "react";

const navItems: Array<{ href: string; label: string }> = [
  { href: "/", label: "Home" },
  { href: "/documents", label: "Documents" },
  { href: "/statistics", label: "Statistics" },
  { href: "/tokenization", label: "Tokenization" },
  { href: "/preprocessing", label: "Preprocessing" },
  { href: "/pos", label: "POS" },
  { href: "/ner", label: "NER" },
  { href: "/ngrams", label: "N-grams" },
  { href: "/bpe", label: "BPE" },
  { href: "/index-explorer", label: "Inverted Index" },
  { href: "/pipelines", label: "Pipelines" },
  { href: "/search", label: "Search" },
  { href: "/evaluation", label: "Evaluation" },
  { href: "/about", label: "About" },
];


export function Navigation() {
  const pathname = usePathname();
  return (
    <nav className="w-64 bg-gray-50 border-r border-gray-200 flex flex-col h-screen sticky top-0">
      <div className="p-4 border-b border-gray-200">
        <h1 className="text-xl font-bold text-gray-900">NLP Domain Analysis</h1>
        <p className="text-xs text-gray-500 mt-1">Phase 1–4 Dashboard</p>
      </div>
      <ul className="flex-1 overflow-y-auto p-2 space-y-1">
        {navItems.map((item) => (
          <li key={item.href}>
            <Link
              href={item.href}
              className={`block px-3 py-2 rounded-md text-sm transition-colors ${
                pathname === item.href
                  ? "bg-blue-100 text-blue-900 font-medium"
                  : "text-gray-700 hover:bg-gray-100"
              }`}
            >
              {item.label}
            </Link>
          </li>
        ))}
      </ul>
      <div className="p-4 border-t border-gray-200 text-xs text-gray-500">
        API: {process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000"}
      </div>
    </nav>
  );
}

export function PageContainer({ children, title, description }: { children: ReactNode; title: string; description?: string }) {
  return (
    <main className="flex-1 p-6 lg:p-8 overflow-auto">
      <header className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900">{title}</h1>
        {description && <p className="text-gray-600 mt-1">{description}</p>}
      </header>
      <div className="space-y-6">{children}</div>
    </main>
  );
}

export function Layout({ children }: { children: ReactNode }) {
  return (
    <div className="flex h-screen bg-white">
      <Navigation />
      <div className="flex-1 flex flex-col min-w-0">
        <header className="bg-white border-b border-gray-200 px-6 py-3 sticky top-0 z-10">
          <h1 className="text-lg font-semibold text-gray-900">NLP Domain Text Analysis</h1>
        </header>
        <div className="flex-1 flex overflow-hidden">{children}</div>
      </div>
    </div>
  );
}