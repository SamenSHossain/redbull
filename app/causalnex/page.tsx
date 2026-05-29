export default function CausalNex() {
  return (
    <>
      <section className="rb-tab-intro">
        <div className="rb-tab-intro-head">
          <span className="rb-tab-intro-eyebrow">Structure learning</span>
          <span className="rb-method-pill">method · causalnex · NOTEARS</span>
        </div>
        <p style={{ margin: 0 }}>
          Instead of assuming a DAG, learn one from the data. NOTEARS returns a
          weighted dependency network; edges are pruned by{" "}
          <code>|weight|</code>. Key finding: after conditioning on actual
          consumption, there&apos;s no direct edge{" "}
          <code>sm_engagement → purchase_intent_rb</code> — social engagement
          is a downstream correlate, not a direct cause of purchase.
        </p>
      </section>

      <div className="rb-stub">
        Interactive structure-learning network with threshold slider — coming
        after Next.js port (NOTEARS is too heavy for serverless; will ship
        pre-computed snapshots at a few thresholds). Use the{" "}
        <a href="https://huggingface.co/spaces/samenh/redbull" target="_blank" rel="noreferrer">
          Hugging Face Space
        </a>{" "}
        for the live version.
      </div>
    </>
  );
}
