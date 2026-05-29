import fs from "node:fs";
import path from "node:path";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

export default function About() {
  const md = fs.readFileSync(
    path.join(process.cwd(), "ARCHITECTURE.md"),
    "utf-8",
  );
  return (
    <>
      <section className="rb-tab-intro">
        <div className="rb-tab-intro-head">
          <span className="rb-tab-intro-eyebrow">Architecture reference</span>
          <span className="rb-method-pill">source · ARCHITECTURE.md</span>
        </div>
        <p style={{ margin: 0 }}>
          What the app is, how the pieces fit, and the format of the data,
          layout, visuals, and persistence. Rendered from{" "}
          <code>ARCHITECTURE.md</code> at the repo root.
        </p>
      </section>
      <div className="rb-about">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{md}</ReactMarkdown>
      </div>
    </>
  );
}
