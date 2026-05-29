export default function DoWhy() {
  return (
    <>
      <section className="rb-tab-intro">
        <div className="rb-tab-intro-head">
          <span className="rb-tab-intro-eyebrow">Classical identification</span>
          <span className="rb-method-pill">method · dowhy</span>
        </div>
        <p style={{ margin: 0 }}>
          Assumes a DAG (rating ← engagement_high, sm_engagement, price, brand,
          rb_brand_image, price_importance), runs{" "}
          <code>CausalModel.identify_effect</code> + linear-regression backdoor
          estimator, then three refutation tests. The point is{" "}
          <em>robustness</em>, not magnitude.
        </p>
      </section>

      <div className="rb-card">
        <div className="rb-card-header">
          Causal effect of High engagement on favorability (full sample)
        </div>
        <p style={{ fontSize: 36, fontWeight: 700, color: "var(--rb-blue-2)", margin: "0 0 4px 0" }}>
          +0.328
        </p>
        <p style={{ color: "var(--rb-fg-muted)", margin: 0 }}>
          favorability points (1–7 scale), 64 respondents / 768 rows
        </p>
      </div>

      <div className="rb-card">
        <div className="rb-card-header">Refutation tests</div>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 14 }}>
          <thead>
            <tr style={{ textAlign: "left", color: "var(--rb-fg-muted)" }}>
              <th style={{ padding: "8px 0" }}>Test</th>
              <th style={{ padding: "8px 0" }}>New effect</th>
            </tr>
          </thead>
          <tbody>
            <tr><td>Original (linear regression backdoor)</td><td><code>0.3281</code></td></tr>
            <tr><td>Add random common cause</td><td><code>0.3284</code></td></tr>
            <tr><td>Placebo (permuted) treatment</td><td><code>−0.0104</code></td></tr>
            <tr><td>80% data subset bootstrap</td><td><code>0.3173</code></td></tr>
          </tbody>
        </table>
      </div>

      <div className="rb-stub">
        Live re-estimation under sidebar filter — coming after Next.js port.
        Use the{" "}
        <a href="https://huggingface.co/spaces/samenh/redbull" target="_blank" rel="noreferrer">
          Hugging Face Space
        </a>{" "}
        for the interactive version.
      </div>
    </>
  );
}
