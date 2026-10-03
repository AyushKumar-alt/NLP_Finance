/** Root layout: HTML wrapper, fonts, and global styles. */

import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
});

export const metadata: Metadata = {
  title: {
    default: "NLP Domain Text Analysis",
    template: "%s | NLP Domain",
  },
  description: "Four-phase pipeline for Indian financial and economic document analysis. Phases 1-3 remain the source of truth; the Next.js frontend presents their artefacts via a FastAPI backend.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={inter.variable}>
      <body className="font-sans antialiased h-screen overflow-hidden bg-gray-50">
        {children}
      </body>
    </html>
  );
}