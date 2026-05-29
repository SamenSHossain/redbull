export default function EconML() {
  return (
    <>
      <section className="rb-tab-intro">
        <div className="rb-tab-intro-head">
          <span className="rb-tab-intro-eyebrow">Heterogeneity</span>
          <span className="rb-method-pill">method · econml</span>
        </div>
        <p style={{ margin: 0 }}>
          <code>LinearDML</code> for ATE inference,{" "}
          <code>CausalForestDML</code> for CATE. The sidebar sliders describe a
          hypothetical respondent and the page predicts the CATE for{" "}
          <em>that</em> profile with a 90% CI.
        </p>
      </section>

      <div className="rb-card">
        <div className="rb-card-header">ATE on the full sample</div>
        <p style={{ fontSize: 36, fontWeight: 700, color: "var(--rb-blue-2)", margin: "0 0 4px 0" }}>
          +0.485
        </p>
        <p style={{ color: "var(--rb-fg-muted)", margin: 0 }}>
          favorability points, 95% CI [+0.284, +0.685]
        </p>
      </div>

      <div className="rb-stub">
        Slider-driven CATE calculator and CATE-vs-modifier curves — coming
        after Next.js port (will hit a tiny Python serverless route over the
        pickled forest). Use the{" "}
        <a href="https://huggingface.co/spaces/samenh/redbull" target="_blank" rel="noreferrer">
          Hugging Face Space
        </a>{" "}
        for the interactive version.
      </div>
    </>
  );
}
