export default function Overview() {
  return (
    <>
      <section className="rb-tab-intro">
        <div className="rb-tab-intro-head">
          <span className="rb-tab-intro-eyebrow">Causal inference dashboard</span>
          <span className="rb-method-pill">stack · dowhy + econml + pymc + causalnex</span>
        </div>
        <p style={{ margin: 0 }}>
          Conjoint-experiment causal analysis of Red Bull purchase preference —
          64 respondents × 12 product profiles = 768 ratings. Each tab runs the
          same effect through a different causal-inference toolkit so you can
          see where they agree and where they disagree.
        </p>
      </section>

      <div className="rb-card">
        <div className="rb-card-header">Headline finding</div>
        <p>
          High social-media engagement <strong>causes</strong> +0.328 favorability
          points on a 1–7 scale (DoWhy, refutation-robust). EconML and PyMC
          confirm direction and shape; the effect is concentrated in
          already-engaged moderate users and near zero for heavy users
          (saturation) and disengaged users (no leverage).
        </p>
        <p style={{ marginBottom: 0 }}>
          See <a href="/synthesis">Synthesis</a> for the recommendation deltas
          and <a href="/about">About</a> for the full architecture.
        </p>
      </div>

      <div className="rb-stub">
        <strong>Live Shiny version</strong> with sidebar filters, interactive
        sliders, and live re-estimation is available at the{" "}
        <a
          href="https://huggingface.co/spaces/samenh/redbull"
          target="_blank"
          rel="noreferrer"
        >
          Hugging Face Space
        </a>
        . The static per-tab pages here are being ported to React + Plotly.
      </div>
    </>
  );
}
