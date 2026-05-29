import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "Red Bull Conjoint — Causal Inference Dashboard",
  description:
    "Conjoint-experiment causal analysis of Red Bull purchase preference. DoWhy, EconML, PyMC, CausalNex, Synthesis.",
};

const NAV = [
  { href: "/", label: "Overview" },
  { href: "/about", label: "About" },
  { href: "/dowhy", label: "DoWhy" },
  { href: "/econml", label: "EconML" },
  { href: "/pymc", label: "PyMC" },
  { href: "/causalnex", label: "CausalNex" },
  { href: "/synthesis", label: "Synthesis" },
];

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
        <link
          rel="stylesheet"
          href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap"
        />
      </head>
      <body>
        <div className="rb-header">
          <span className="rb-wordmark">Red Bull Conjoint</span>
          <span className="rb-divider"></span>
          <span className="rb-subtitle">Causal Inference Dashboard</span>
        </div>
        <nav className="rb-nav">
          {NAV.map((item) => (
            <Link key={item.href} href={item.href}>
              {item.label}
            </Link>
          ))}
        </nav>
        <main className="rb-container">{children}</main>
      </body>
    </html>
  );
}
