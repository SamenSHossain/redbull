export default function PyMC() {
  return (
    <>
      <section className="rb-tab-intro">
        <div className="rb-tab-intro-head">
          <span className="rb-tab-intro-eyebrow">Hierarchical Bayesian conjoint</span>
          <span className="rb-method-pill">method · pymc</span>
        </div>
        <p style={{ margin: 0 }}>
          Each respondent gets their own part-worth vector <code>beta_i</code>{" "}
          drawn from a population <code>mu_beta</code> with a diagonal{" "}
          <code>sigma_beta</code>. The posterior is cached because each MCMC
          fit takes ~17 seconds.
        </p>
      </section>

      <div className="rb-card">
        <div className="rb-card-header">Willingness-to-pay — 95% credible intervals</div>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 14 }}>
          <thead>
            <tr style={{ textAlign: "left", color: "var(--rb-fg-muted)" }}>
              <th style={{ padding: "8px 0" }}>Contrast</th>
              <th style={{ padding: "8px 0" }}>Mean $</th>
              <th style={{ padding: "8px 0" }}>CI 2.5%</th>
              <th style={{ padding: "8px 0" }}>CI 97.5%</th>
            </tr>
          </thead>
          <tbody>
            <tr><td>WTP for Medium vs Light engagement</td><td><code>0.313</code></td><td><code>0.162</code></td><td><code>0.471</code></td></tr>
            <tr><td>WTP for High vs Light engagement</td><td><code>0.448</code></td><td><code>0.277</code></td><td><code>0.631</code></td></tr>
          </tbody>
        </table>
      </div>

      <div className="rb-stub">
        Population forest plot and per-respondent histograms — coming after
        Next.js port (the 25 MB NetCDF posterior will be summarized to JSON
        for client-side rendering with Plotly). Use the{" "}
        <a href="https://huggingface.co/spaces/samenh/redbull" target="_blank" rel="noreferrer">
          Hugging Face Space
        </a>{" "}
        for the full plots.
      </div>
    </>
  );
}
