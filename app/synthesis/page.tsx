const ROWS: { claim: string; finding: React.ReactNode; rec: React.ReactNode; kind: "keep" | "strat" | "reframe" }[] = [
  {
    claim: "High engagement adds utility",
    finding: (
      <>
        DoWhy: identified, refutation-robust.
        <br />
        PyMC: WTP 95% CrI <strong>$0.28 – $0.63</strong>
      </>
    ),
    rec: (
      <>
        Keep, but <strong>report the uncertainty band</strong> alongside the
        point estimate.
      </>
    ),
    kind: "keep",
  },
  {
    claim: "Engagement is the 3rd most important attribute (15%)",
    finding: (
      <>
        True on average. But CATE quintiles span{" "}
        <strong>+0.10 → +0.91</strong> — a ~9× spread.
      </>
    ),
    rec: (
      <>
        <strong>Stratify the recommendation</strong> by respondent segment.
      </>
    ),
    kind: "strat",
  },
  {
    claim: "Frictionless conversion is universal",
    finding: (
      <>
        EconML: largest lift for already-engaged users. Heavy users saturated
        (negative moderator, t = −3.3).
      </>
    ),
    rec: (
      <>
        Reframe as a <strong>retention / activation</strong> tool, not
        acquisition.
      </>
    ),
    kind: "reframe",
  },
  {
    claim: "Social media drives the funnel",
    finding: (
      <>
        CausalNex: no direct edge{" "}
        <code>sm_engagement → purchase_intent</code> after conditioning on
        consumption.
      </>
    ),
    rec: (
      <>
        Treat social as <strong>top-of-funnel awareness</strong>, not a
        conversion engine.
      </>
    ),
    kind: "reframe",
  },
];

export default function Synthesis() {
  return (
    <>
      <section className="rb-tab-intro">
        <div className="rb-tab-intro-head">
          <span className="rb-tab-intro-eyebrow">Recommendation deltas</span>
          <span className="rb-method-pill">method · summary</span>
        </div>
        <p style={{ margin: 0 }}>
          Four-row comparison of the marketing-team headline claim vs. what
          the causal analysis actually shows, plus the ordered list of the
          strongest causal levers.
        </p>
      </section>

      <div className="rb-card">
        <div className="rb-card-header">
          How the recommendations change after causal analysis
        </div>
        <div className="rb-synth">
          <div className="rb-synth-head">Headline claim</div>
          <div className="rb-synth-head">Causal finding</div>
          <div className="rb-synth-head">Recommendation change</div>
          {ROWS.map((r, i) => (
            <>
              <div key={`c-${i}`} className={`rb-synth-cell rb-claim rb-${r.kind}`}>{r.claim}</div>
              <div key={`f-${i}`} className="rb-synth-cell">{r.finding}</div>
              <div key={`r-${i}`} className="rb-synth-cell">{r.rec}</div>
            </>
          ))}
        </div>
      </div>

      <div className="rb-card">
        <div className="rb-card-header">Strongest causal levers (in order)</div>
        <ol style={{ fontSize: 15, lineHeight: 1.65, paddingLeft: 20 }}>
          <li>
            <strong>Price</strong> — still dominates 4:1. Subscription /
            bundling is the biggest dial.
          </li>
          <li>
            <strong>Functional repositioning</strong> — addresses CausalNex's
            finding that broader ED-category consumption is{" "}
            <em>negatively</em> correlated with Red Bull intent. Reclaim share
            from competitors directly.
          </li>
          <li>
            <strong>Shoppable social, targeted</strong> — only for the
            already-engaged tail. Wastes impressions on the disengaged.
          </li>
        </ol>
      </div>
    </>
  );
}
