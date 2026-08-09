import React, { useState } from "react";

// ============================================================
// XAI Stability Benchmark — Complete Interactive App
// Yanhong (Linda) Miao
//
// Six sections, one global Simple/Technical toggle:
//   Overview · Q&A · Models · XAI Tools · Explore Data · Finding
//
// Data = final QC'd re-run (verified). Content is Linda's own
// (Q&A) plus accurate model/method explanations.
// Deployable to GitHub Pages / Vercel. All browsers.
// ============================================================

// ---------------- VERIFIED DATA (final re-run) ----------------
const DATASETS = [
  { key: "uavcan",     name: "UAVCAN",             domain: "Drone CAN-bus protocol",       features: 10, classes: 2,
    ess: { rf: 0.448, cnn: 0.478, ae: 0.356, iso: 0.429 }, f1: { rf: 0.908, cnn: 0.887, ae: 0.501, iso: 0.473 },
    role: "Secures the internal CAN bus that drones use to pass commands between flight controller, motors, and sensors.",
    why: "The lowest-dimensional dataset (10 features). Represents attacks on a drone's internal control network — a distinct, low-signal environment that anchors the low end of the dimensionality range." },
  { key: "uavids",     name: "UAVIDS-2025",        domain: "UAV swarm network",            features: 18, classes: 5,
    ess: { rf: 0.369, cnn: 0.444, ae: 0.181, iso: 0.389 }, f1: { rf: 0.952, cnn: 0.925, ae: 0.851, iso: 0.923 },
    role: "Detects five attack types (Blackhole, Flooding, Sybil, Wormhole, plus Normal) across a UAV swarm network.",
    why: "A recent 2025 dataset added as an out-of-sample test. It landed exactly where the dimensionality trend predicted (18 features) — evidence the finding is real, not tuned." },
  { key: "uav_cyber",  name: "UAV-Cyber",          domain: "Drone cyber-physical (T-ITS)", features: 37, classes: 3,
    ess: { rf: 0.619, cnn: 0.587, ae: 0.183, iso: 0.369 }, f1: { rf: 0.993, cnn: 0.975, ae: 0.023, iso: 0.870 },
    role: "Covers cyber-physical attacks where a drone's physical behavior and its network signals are both affected.",
    why: "A mid-dimensional (37 features) cyber-physical case. Notable because the Autoencoder nearly fails here (F1 0.023) — a genuine, honest hard case for anomaly detection." },
  { key: "isot",       name: "ISOT",               domain: "Botnet / drone traffic",       features: 61, classes: 10,
    ess: { rf: 0.389, cnn: 0.214, ae: 0.083, iso: 0.587 }, f1: { rf: 1.000, cnn: 0.996, ae: 0.981, iso: 0.949 },
    role: "Identifies botnet and malicious traffic patterns relevant to networked and autonomous systems.",
    why: "A higher-dimensional (61 features) botnet dataset. Bridges the gap between the drone-specific sets and the largest network datasets, filling the middle-high range." },
  { key: "uav_attack", name: "UAV-Attack",         domain: "Drone attack benchmark",       features: 80, classes: 3,
    ess: { rf: 0.000, cnn: 0.102, ae: 0.150, iso: 0.083 }, f1: { rf: 1.000, cnn: 1.000, ae: 0.903, iso: 0.772 },
    role: "A dedicated drone-attack benchmark covering multiple UAV intrusion scenarios.",
    why: "High-dimensional (80 features) and drone-specific. Produces the starkest result: perfect detection (F1 1.000) with zero explanation agreement (ESS 0.000) — the headline tension of the whole study." },
  { key: "cicids",     name: "CICIDS2017",         domain: "General network IDS",          features: 80, classes: 13,
    ess: { rf: 0.157, cnn: 0.185, ae: 0.210, iso: 0.143 }, f1: { rf: 0.999, cnn: 0.978, ae: 0.571, iso: 0.511 },
    role: "A widely-used general network intrusion dataset with 13 attack categories — the broad security backdrop.",
    why: "The largest and most-cited (80 features, 13 classes). Grounds the benchmark in a standard the whole field knows, and anchors the high-dimensional end alongside UAV-Attack." },
];
const avgEss = (d) => (d.ess.rf + d.ess.cnn + d.ess.ae + d.ess.iso) / 4;
const byFeatures = [...DATASETS].sort((a, b) => a.features - b.features);

// ---------------- MODELS ----------------
const MODELS = [
  { key: "rf", name: "Random Forest", type: "supervised", tag: "RF",
    simple: "A team of decision trees that vote. Each tree learns simple yes/no rules from labeled examples of normal and attack traffic, then they vote on each new sample. Stable and easy to interpret.",
    technical: "An ensemble of 100 decision trees trained on labeled data. Each tree splits on feature thresholds; the ensemble averages their votes. Strong, stable baseline for tabular IDS data; feature importances are readable directly from tree structure.",
    why: "Included as the strong, interpretable baseline. It usually posts the highest detection — yet on UAV-Attack it pairs perfect F1 with zero ESS, making it central to the project's core tension." },
  { key: "cnn", name: "1D-CNN", type: "supervised", tag: "CNN",
    simple: "A neural network that slides a small window across the features, learning patterns among neighboring signals. Can catch subtle combinations a simple rule might miss.",
    technical: "A one-dimensional convolutional network (two Conv1D layers, pooling, dense, softmax). Treats the ordered feature vector as a sequence to learn local feature interactions — a common representation in IDS work, though features are not inherently spatial.",
    why: "Included as the deep-learning supervised contrast to Random Forest, and because it supports gradient-based explanations (Integrated Gradients) that the tree models cannot." },
  { key: "ae", name: "Autoencoder", type: "unsupervised", tag: "AE",
    simple: "Learns what NORMAL traffic looks like by compressing and rebuilding it. When it sees an attack, it rebuilds it badly — that large error flags the anomaly. It never sees attacks during training.",
    technical: "An unsupervised reconstruction model trained only on benign traffic (encoder 32→16→32→output, MSE loss). Anomaly score = reconstruction error; threshold set at the 95th percentile of benign errors. Performance varies widely by dataset separability.",
    why: "Included as the unsupervised, attack-free learner — realistic when labeled attacks are scarce. Its wide performance swings (0.023 to 0.981 F1) show how much anomaly detection depends on the data." },
  { key: "iso", name: "Isolation Forest", type: "unsupervised", tag: "IF",
    simple: "Finds anomalies by how easily a sample can be 'isolated' — unusual points get separated in few steps. Also trained only on normal traffic.",
    technical: "An unsupervised isolation-based detector (100 trees) trained on benign traffic. Anomalies require fewer random splits to isolate, yielding shorter path lengths. Uses the +1/-1 decision output for attack scoring.",
    why: "Added in this study as a second unsupervised detector, giving a fair 2-vs-2 supervised/unsupervised design and a non-neural anomaly baseline to compare against the Autoencoder." },
];

// ---------------- XAI METHODS ----------------
const XAI = [
  { name: "SHAP", tag: "SHAP", applies: "All 4 models",
    simple: "Fairly splits the 'credit' for a decision among all features, based on game theory. Tells you how much each signal pushed the AI toward 'attack'.",
    technical: "Shapley-value attribution. TreeExplainer for RF (exact, fast), GradientExplainer for the neural nets, KernelExplainer for the anomaly detectors. Absolute values averaged to a global per-feature importance.",
    why: "The most theoretically grounded method — a natural reference point. Whether SHAP agrees with the others is a big part of what ESS captures." },
  { name: "LIME", tag: "LIME", applies: "All 4 models",
    simple: "Explains one decision at a time by slightly changing the input and watching what happens, then fits a simple local model. Fast and doesn't need to see inside the AI.",
    technical: "Local Interpretable Model-agnostic Explanations. Perturbs inputs around an instance and fits a sparse linear surrogate; feature indices read from as_map(). Averaged over sampled instances for a global view. Seeded for reproducibility.",
    why: "Model-agnostic and fast — the most deployment-friendly method. Included to test whether a practical, lightweight explainer agrees with heavier ones." },
  { name: "Permutation Importance", tag: "PI", applies: "All 4 models",
    simple: "Shuffles one signal at a time and measures how much detection gets worse. If shuffling a signal hurts a lot, that signal mattered.",
    technical: "Measures the drop in detection performance when a feature's values are randomly permuted. For unsupervised models, computed on downstream detection performance (reconstruction-error thresholding), not reconstruction directly. Seeded generator for reproducibility.",
    why: "A performance-based check independent of any model internals. It grounds ESS in 'what actually changes detection', not just internal attributions." },
  { name: "Integrated Gradients", tag: "IG", applies: "CNN & AE only",
    simple: "Traces the AI's reasoning from a blank baseline up to the real input, adding up how each signal contributed along the way. Only works on the neural-network models.",
    technical: "Attributes a prediction by integrating gradients along a straight path from a zero baseline to the input. Applies only to differentiable models (1D-CNN, Autoencoder); tree-based models provide no gradients. Zero baseline is standard; benign-mean baseline noted as future work.",
    why: "Added in this study as a gradient-based method for the neural models, testing whether a very different mathematical approach lands on the same important features." },
];

// ---------------- EVALUATION (3 types: ML, SE, ESS) ----------------
const EVAL = [
  { name: "ML Metrics", tag: "ML", color: "accent",
    simple: "Measures how well a model catches attacks — the basic 'is it accurate?' question.",
    technical: "Accuracy, Precision, Recall, and F1 on the detection task. This project reports attack-focused (binary) F1 as the primary operational view, alongside a weighted view for comparison with prior work.",
    why: "The standard performance layer. It answers 'does the model detect attacks?' — but, as this project shows, it says nothing about whether the model's explanations are trustworthy." },
  { name: "SE Metrics", tag: "SE", color: "amber",
    simple: "Measures the practical, engineering side of each explanation tool — how fast it runs and how hard it is to integrate.",
    technical: "Software-engineering measures such as runtime, integration effort, wrapper requirement, and explanation type. These capture whether an XAI method is realistic to deploy in an operational IDS, not just whether it is accurate.",
    why: "A real security team cares about cost and deployability, not only accuracy. SE metrics bring an engineering lens that pure ML evaluation ignores." },
  { name: "ESS", tag: "ESS", color: "green",
    simple: "The new score. Measures how much the different explanation tools AGREE on which signals mattered.",
    technical: "Explainability Stability Score = mean pairwise top-5 Jaccard overlap across usable XAI methods, per model per dataset. Methods returning all-zero importances are excluded; ESS is undefined for fewer than two usable methods.",
    why: "The project's core contribution. It adds a stability dimension no standard metric captures: a model can be accurate (high F1) yet have explanations that disagree (low ESS)." },
];

// ---------------- Q&A CONTENT (Linda's own) ----------------
const QA = {
  "Aerospace / Defense": {
    icon: "✈️",
    subtitle: "Why this matters to companies building drones, aircraft, and autonomous systems.",
    questions: {
      "Why does explainability matter for autonomous systems?": {
        simple: `An autonomous system may make an important security decision without a human directly controlling every step.\n\nIf its AI says, "This network traffic is an attack," engineers and operators need to understand why.\n\nThis is especially important when safety, reliability, and accountability matter.`,
        technical: `This research evaluates whether different post-hoc XAI methods produce consistent feature-attribution explanations for the same intrusion detection model.\n\nA model may achieve high detection performance while different XAI methods identify different important features.\n\nThis creates an explanation-stability problem that is not captured by F1 score alone.`,
      },
      "Can a highly accurate AI still have an explanation problem?": {
        simple: `Yes.\n\nAn AI can be very good at detecting attacks while different explanation tools disagree about WHY it made those decisions.\n\nSo accuracy alone does not tell us whether an explanation is stable.`,
        technical: `Yes. The benchmark separates predictive performance from explanation stability.\n\nOn UAV-Attack, Random Forest achieved F1 = 1.000 but ESS = 0.000 — the model detected attacks perfectly, while the usable explanation methods showed no top-5 feature overlap.`,
      },
      "What could this mean for real autonomous systems?": {
        simple: `A real autonomous system may need two things:\n\n1. A model that detects threats accurately.\n2. An explanation that helps people understand the decision.\n\nIf explanation methods disagree, engineers may want to cross-check the explanation before using it for an important operational decision.`,
        technical: `The results suggest that XAI evaluation should accompany model performance evaluation in security-sensitive autonomous systems.\n\nESS can provide a stability-oriented measurement alongside F1, rather than treating predictive performance as the only evaluation criterion.\n\nThis is a research benchmark, not a production aerospace security system.`,
      },
      "Why is this relevant to drones and IoT?": {
        simple: `Drones and IoT systems depend on network communication, sensors, and connected devices.\n\nIf those systems are attacked, AI may be used to detect unusual or malicious behavior.\n\nUnderstanding the AI's reasoning can help engineers investigate and respond to those threats.`,
        technical: `The benchmark spans drone-related and general network intrusion datasets: UAVCAN, UAVIDS-2025, UAV-Cyber, UAV-Attack, ISOT, and CICIDS2017 — a cross-dataset environment for examining detection performance versus explanation stability.`,
      },
      "If the explanation tools disagree, what should we do?": {
        simple: `When agreement (ESS) is low, no single explanation should be trusted on its own. To reduce risk, a team can:\n\n1. Cross-check several explanation methods before acting on one.\n2. Add human review for high-stakes alarms instead of acting automatically.\n3. Report the agreement score (ESS) next to the detection result, so operators know how much to trust the "why."\n4. Be extra cautious with single-method explanations on complex, high-signal systems, where disagreement is more likely.`,
        technical: `Low ESS indicates that feature-attribution rankings are method-dependent, so any single explanation is unreliable for that model/dataset. Risk-reduction options:\n\n• Require agreement across multiple XAI methods before an explanation informs an operational decision.\n• Route low-ESS detections to human-in-the-loop review.\n• Surface ESS alongside F1 in monitoring, treating explanation stability as a first-class signal.\n• Prefer higher-ESS model/data regimes, or additional validation, in high-dimensional deployments where ESS tends to fall.`,
      },
    },
  },
  "Professor / Researcher": {
    icon: "🎓",
    subtitle: "What exactly does this project contribute to XAI and intrusion-detection research?",
    questions: {
      "What does ESS add to normal AI evaluation?": {
        simple: `F1 tells us how well the AI detects attacks.\n\nESS asks a different question: do different explanation methods agree about which features are important?\n\nSo ESS adds an explanation-stability dimension to normal model evaluation.`,
        technical: `ESS (Explainability Stability Score) is the mean pairwise top-5 Jaccard overlap across usable XAI methods for a model and dataset.\n\nIt complements predictive metrics such as F1 by quantifying agreement among explanation methods.`,
      },
      "Can a highly accurate AI still have an explanation problem?": {
        simple: `Yes. High detection accuracy and stable explanations are different properties.\n\nA model can detect attacks correctly while its explanation methods disagree about which features are important.`,
        technical: `Yes. The benchmark explicitly separates predictive performance from explanation stability.\n\nUAV-Attack Random Forest: F1 = 1.000, ESS = 0.000. This illustrates why predictive performance alone cannot characterize XAI behavior.`,
      },
      "What does the dimensionality finding mean?": {
        simple: `In this benchmark, datasets with fewer features generally show more agreement between explanation methods; datasets with more features show less.\n\nThis result needs careful interpretation because ESS uses top-5 feature rankings.`,
        technical: `The relationship is consistent with decreasing top-5 Jaccard overlap as feature-space dimensionality increases.\n\nAn important limitation: part of this behavior may arise intrinsically from top-k selection as the feature space grows. It should not be read as a universal causal law.`,
      },
      "What are the main research contributions?": {
        simple: `This project compares 6 datasets, 4 models, and 4 XAI methods across detection performance and explanation stability.\n\nThe main goal is to show that detecting an attack and explaining the detection are related but different problems.`,
        technical: `Four model families (Random Forest, 1D-CNN, Autoencoder, Isolation Forest) across six IDS datasets and four XAI methods (SHAP, LIME, Permutation Importance, Integrated Gradients).\n\nThe study introduces ESS as an explanation-stability metric and examines its behavior across differing feature dimensionalities.`,
      },
      "F1 or ESS — which is the key for a final decision?": {
        simple: `Neither one alone — they answer different questions.\n\nF1 tells you whether to trust the ALARM (is it really an attack?).\nESS tells you whether to trust the REASON (can you believe the 'why'?).\n\nSo the answer depends on the decision:\n• Acting on an alarm → rely on F1.\n• Trusting or auditing the explanation → rely on ESS.\n• Choosing which model to deploy → you need BOTH.\n\nThe key lesson of this project is that F1 alone is not enough. A model can score a perfect F1 while its explanations completely disagree (UAV-Attack Random Forest: F1 = 1.000, ESS = 0.000). ESS should be reported next to F1, not ignored.`,
        technical: `They are complementary, not interchangeable. F1 quantifies predictive reliability; ESS quantifies explanation reliability.\n\nDecision mapping:\n• Operational detection decision → F1.\n• Trust/audit/debug of the attribution → ESS.\n• Model or method selection → jointly optimize both.\n\nThe benchmark's central argument is that predictive performance is an insufficient selection criterion for explainable security systems. The UAV-Attack Random Forest case (F1 = 1.000, ESS = 0.000) demonstrates that a maximally accurate model can have maximally unstable explanations, so ESS must be treated as a first-class metric alongside F1.`,
      },
    },
  },
  "Everyday User": {
    icon: "👤",
    subtitle: "Why should an ordinary person care about AI security and autonomous systems?",
    questions: {
      "What does this have to do with me?": {
        simple: `You may not build AI systems, but autonomous and connected systems affect everyday life.\n\nDrones, connected devices, transportation, and infrastructure increasingly depend on software and AI.\n\nMaking those systems easier to understand and verify can support safety and trust.`,
        technical: `The immediate contribution is methodological rather than a consumer product.\n\nThis research studies how reliably XAI methods explain security models applied to network and drone-related datasets — the broader relevance being AI-assisted security in autonomous connected systems.`,
      },
      "Why should I care if AI can explain itself?": {
        simple: `Imagine an AI says: "This drone or device may be under attack."\n\nA person may reasonably ask: "Why?"\n\nAn explanation can help people investigate the decision instead of blindly trusting the AI.`,
        technical: `Explainability supports investigation, debugging, auditing, and human oversight.\n\nBut this research also shows why an explanation should not automatically be trusted simply because an XAI tool produced one — different XAI methods can disagree.`,
      },
      "Does this make drones safer?": {
        simple: `Not by itself. This project does not deploy a security system on a real aircraft.\n\nInstead, it studies how well AI models detect attacks and how consistent their explanations are — which can help identify issues engineers should consider when designing future systems.`,
        technical: `The benchmark is an evaluation study, not a production safety certification or deployed aerospace system.\n\nIts contribution is evaluating predictive performance and explanation stability across multiple IDS datasets and model types.`,
      },
      "Why does drone cybersecurity matter?": {
        simple: `A connected drone depends on communication and sensor data.\n\nIf an attacker interferes with that information, the system may make the wrong decision.\n\nCybersecurity helps protect systems that people increasingly depend on.`,
        technical: `Drone datasets here include traffic associated with attacks such as Flooding, Blackhole, Sybil, and Wormhole.\n\nUAVIDS-2025, for example, contains five categories: Normal Traffic, Blackhole, Flooding, Sybil, and Wormhole attacks.`,
      },
    },
  },
};

// ---------------- theme ----------------
const C = {
  bg: "#0e1420", panel: "#161f30", soft: "#1d2840", line: "#2a3a4a",
  ink: "#e8eef7", sub: "#8fa3bf", cyan: "#35d0d6", accent: "#5b8def",
  amber: "#e8c14b", green: "#3fd68a", red: "#e8615e", violet: "#9b8cf0",
};
const mono = "'JetBrains Mono','SF Mono',ui-monospace,monospace";
const sans = "'Inter',system-ui,-apple-system,sans-serif";
const essColor = (v) => (v < 0.2 ? C.red : v < 0.4 ? C.amber : C.green);
const essWord = (v) => (v < 0.2 ? "tools disagree" : v < 0.4 ? "some agreement" : "tools agree");
const T = (mode, obj) => obj[mode];

export default function App() {
  const [section, setSection] = useState("overview");
  const [mode, setMode] = useState("simple"); // simple | technical

  // jump to a section, optionally scrolling to an element id inside it
  const goTo = (target, anchorId) => {
    setSection(target);
    if (anchorId) {
      // wait for the section to render, then scroll to the anchor
      setTimeout(() => {
        const el = document.getElementById(anchorId);
        if (el) el.scrollIntoView({ behavior: "smooth", block: "center" });
      }, 60);
    } else {
      window.scrollTo({ top: 0, behavior: "smooth" });
    }
  };

  const nav = [
    { key: "overview", label: "Overview" },
    { key: "qa", label: "Q&A" },
    { key: "models", label: "Models" },
    { key: "xai", label: "XAI Tools" },
    { key: "eval", label: "Evaluation" },
    { key: "data", label: "Explore Data" },
    { key: "finding", label: "The Finding" },
  ];
  return (
    <div style={{ fontFamily: sans, background: C.bg, color: C.ink, minHeight: "100vh" }}>
      {/* header */}
      <div style={{ background: C.panel, borderBottom: `1px solid ${C.soft}`, position: "sticky", top: 0, zIndex: 10 }}>
        <div style={{ maxWidth: 1000, margin: "0 auto", padding: "16px 24px 0" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: 12 }}>
            <div>
              <div style={{ fontFamily: mono, fontSize: 11, color: C.cyan, letterSpacing: 2, marginBottom: 4 }}>XAI_STABILITY_BENCHMARK</div>
              <h1 style={{ margin: 0, fontSize: 22, fontWeight: 800 }}>Explainable AI for Autonomous Cyber Security</h1>
            </div>
            {/* global toggle */}
            <div style={{ display: "flex", background: C.bg, borderRadius: 10, padding: 4, border: `1px solid ${C.soft}` }}>
              {[["simple", "Simple"], ["technical", "Technical"]].map(([m, l]) => (
                <button key={m} onClick={() => setMode(m)}
                  style={{ border: "none", cursor: "pointer", fontFamily: sans, fontSize: 13, fontWeight: 600, padding: "7px 14px", borderRadius: 7,
                    background: mode === m ? C.cyan : "transparent", color: mode === m ? C.bg : C.sub }}>
                  {l}
                </button>
              ))}
            </div>
          </div>
          <div style={{ display: "flex", gap: 2, marginTop: 12, overflowX: "auto" }}>
            {nav.map((n) => (
              <button key={n.key} onClick={() => setSection(n.key)}
                style={{ border: "none", cursor: "pointer", fontFamily: sans, fontSize: 13.5, whiteSpace: "nowrap",
                  fontWeight: section === n.key ? 700 : 500, padding: "11px 15px", background: "transparent",
                  color: section === n.key ? C.cyan : C.sub, borderBottom: `2px solid ${section === n.key ? C.cyan : "transparent"}` }}>
                {n.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div style={{ maxWidth: 1000, margin: "0 auto", padding: "26px 24px 60px" }}>
        {section === "overview" && <Overview mode={mode} go={setSection} />}
        {section === "qa" && <QASection mode={mode} />}
        {section === "models" && <ModelsSection mode={mode} />}
        {section === "xai" && <XAISection mode={mode} />}
        {section === "eval" && <EvalSection mode={mode} />}
        {section === "data" && <DataSection mode={mode} />}
        {section === "finding" && <FindingSection mode={mode} go={setSection} goTo={goTo} />}
      </div>

      <div style={{ background: C.panel, borderTop: `1px solid ${C.soft}` }}>
        <div style={{ maxWidth: 1000, margin: "0 auto", padding: "16px 24px", fontFamily: mono, fontSize: 11, color: C.sub, display: "flex", justifyContent: "space-between", flexWrap: "wrap", gap: 8 }}>
          <span>6 datasets · 4 models · 4 XAI methods · ESS stability metric</span>
          <span>Yanhong (Linda) Miao</span>
        </div>
      </div>
    </div>
  );
}

function Card({ children, style }) {
  return <div style={{ background: C.panel, border: `1px solid ${C.soft}`, borderRadius: 14, padding: 20, ...style }}>{children}</div>;
}

// =============== OVERVIEW ===============
function Overview({ mode, go }) {
  const stats = [
    ["6", "datasets", "data"],
    ["4", "AI models", "models"],
    ["4", "explanation tools", "xai"],
    ["3", "evaluation types", "eval"],
  ];
  return (
    <div>
      <Card style={{ marginBottom: 16, borderLeft: `3px solid ${C.cyan}` }}>
        <div style={{ fontFamily: mono, fontSize: 12, color: C.cyan, marginBottom: 10 }}>// the question</div>
        <p style={{ fontSize: 18, lineHeight: 1.7, margin: 0 }}>
          {T(mode, {
            simple: "When an AI flags a drone's traffic as an attack, we also want to know WHY. Different tools can answer that — but do they agree? If two tools give different reasons for the same alarm, which do you trust? This project measures that.",
            technical: "Intrusion-detection models are increasingly paired with post-hoc XAI methods. This benchmark asks whether those methods produce consistent feature-attribution rankings for the same model and input — a stability property (ESS) not captured by accuracy metrics alone — across six datasets of varying dimensionality.",
          })}
        </p>
      </Card>
      <div style={{ fontFamily: mono, fontSize: 11, color: C.sub, letterSpacing: 1, marginBottom: 8 }}>THE SCOPE — click any card to jump</div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(130px,1fr))", gap: 12, marginBottom: 16 }}>
        {stats.map(([n, l, target]) => (
          <button key={l} onClick={() => go(target)}
            style={{ cursor: "pointer", textAlign: "center", padding: 18, borderRadius: 14,
              background: C.panel, border: `1px solid ${C.soft}`, transition: "all .2s" }}
            onMouseEnter={(e) => { e.currentTarget.style.borderColor = C.cyan; }}
            onMouseLeave={(e) => { e.currentTarget.style.borderColor = C.soft; }}>
            <div style={{ fontFamily: mono, fontSize: 30, fontWeight: 800, color: C.cyan }}>{n}</div>
            <div style={{ fontSize: 12.5, color: C.sub, marginTop: 4 }}>{l}</div>
            <div style={{ fontSize: 10, color: C.cyan, marginTop: 6, fontFamily: mono }}>view →</div>
          </button>
        ))}
      </div>
      <Card>
        <div style={{ fontWeight: 700, fontSize: 15, marginBottom: 4 }}>Two things we measure for every model</div>
        <div style={{ fontSize: 12.5, color: C.sub, marginBottom: 12 }}>
          {T(mode, { simple: "These are the two ideas the whole project rests on.", technical: "These two axes — detection and explanation stability — are referenced throughout." })}
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
          <div id="def-f1" style={{ background: C.bg, borderRadius: 10, padding: 14, borderLeft: `3px solid ${C.accent}` }}>
            <div style={{ color: C.accent, fontWeight: 700, fontSize: 14 }}>Detection — F1</div>
            <div style={{ color: C.sub, fontSize: 13, marginTop: 5, lineHeight: 1.55 }}>
              {T(mode, {
                simple: "How well a model CATCHES attacks. Higher F1 = it misses fewer attacks. This is about being right, not about explaining.",
                technical: "Attack-focused F1 on the detection task — the harmonic mean of precision and recall for the attack class. Measures predictive performance only.",
              })}
            </div>
          </div>
          <div id="def-ess" style={{ background: C.bg, borderRadius: 10, padding: 14, borderLeft: `3px solid ${C.green}` }}>
            <div style={{ color: C.green, fontWeight: 700, fontSize: 14 }}>Agreement — ESS</div>
            <div style={{ color: C.sub, fontSize: 13, marginTop: 5, lineHeight: 1.55 }}>
              {T(mode, {
                simple: "How much the explanation tools AGREE on WHY. High ESS = the tools point to the same signals, so you can trust the explanation. Low ESS = they disagree.",
                technical: "Explainability Stability Score — mean pairwise top-5 Jaccard overlap across usable XAI methods. Measures explanation consistency, entirely separate from detection.",
              })}
            </div>
          </div>
        </div>
        <div style={{ marginTop: 12, padding: "10px 12px", background: C.soft, borderRadius: 9, fontSize: 12.5, color: C.sub, lineHeight: 1.5 }}>
          {T(mode, {
            simple: "Key idea: a model can be great at catching attacks (high F1) but its explanations can still disagree (low ESS). Detecting and explaining are different jobs.",
            technical: "Central point: high F1 does not imply high ESS. Predictive performance and explanation stability are distinct properties — the motivation for this benchmark.",
          })}
        </div>
        <div style={{ marginTop: 10, padding: "12px 14px", background: C.bg, borderRadius: 9, borderLeft: `3px solid ${C.cyan}` }}>
          <div style={{ fontFamily: mono, fontSize: 10.5, color: C.cyan, marginBottom: 5 }}>WHICH ONE DECIDES?</div>
          <div style={{ fontSize: 13, color: C.ink, lineHeight: 1.6 }}>
            {T(mode, {
              simple: "Use F1 to decide whether to TRUST THE ALARM (is it really an attack?). Use ESS to decide whether to TRUST THE REASON (can you believe the 'why'?). To choose which model to deploy, you need BOTH — the main lesson of this project is that F1 alone is not enough.",
              technical: "F1 governs trust in the detection; ESS governs trust in the explanation. Model selection requires both — the benchmark's core argument is that ESS should be reported alongside F1 rather than F1 being treated as sufficient.",
            })}
          </div>
        </div>
        <button onClick={() => go("data")}
          style={{ marginTop: 16, border: "none", cursor: "pointer", background: C.cyan, color: C.bg, fontFamily: sans, fontWeight: 700, fontSize: 14, padding: "11px 20px", borderRadius: 9 }}>
          Explore the data →
        </button>
      </Card>
    </div>
  );
}

// =============== Q&A ===============
function QASection({ mode }) {
  const [aud, setAud] = useState("Aerospace / Defense");
  const [qIdx, setQIdx] = useState(0);
  const audData = QA[aud];
  const qKeys = Object.keys(audData.questions);
  const qKey = qKeys[qIdx];
  const answer = audData.questions[qKey][mode];
  return (
    <div>
      <div style={{ fontFamily: mono, fontSize: 11, color: C.sub, letterSpacing: 1, marginBottom: 10 }}>WHO ARE YOU?</div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(190px,1fr))", gap: 10, marginBottom: 24 }}>
        {Object.entries(QA).map(([key, d]) => (
          <button key={key} onClick={() => { setAud(key); setQIdx(0); }}
            style={{ cursor: "pointer", textAlign: "left", padding: 15, borderRadius: 13, border: `1px solid ${aud === key ? C.cyan : C.soft}`, background: aud === key ? C.soft : C.panel }}>
            <div style={{ fontSize: 21, marginBottom: 5 }}>{d.icon}</div>
            <div style={{ fontWeight: 700, fontSize: 14, color: aud === key ? C.cyan : C.ink }}>{key}</div>
            <div style={{ fontSize: 11.5, color: C.sub, marginTop: 4, lineHeight: 1.4 }}>{d.subtitle}</div>
          </button>
        ))}
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "minmax(0,290px) minmax(0,1fr)", gap: 16, alignItems: "start" }}>
        <div>
          <div style={{ fontFamily: mono, fontSize: 11, color: C.sub, letterSpacing: 1, marginBottom: 10 }}>QUESTIONS</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {qKeys.map((q, i) => (
              <button key={q} onClick={() => setQIdx(i)}
                style={{ cursor: "pointer", textAlign: "left", padding: "12px 14px", borderRadius: 10, border: `1px solid ${qIdx === i ? C.cyan : C.soft}`,
                  background: qIdx === i ? C.soft : C.panel, color: qIdx === i ? C.ink : C.sub, fontSize: 13, fontWeight: qIdx === i ? 600 : 500, lineHeight: 1.45 }}>
                {q}
              </button>
            ))}
          </div>
        </div>
        <Card>
          <div style={{ fontSize: 12, color: C.cyan, fontWeight: 700, marginBottom: 6 }}>{audData.icon} {aud}</div>
          <h2 style={{ margin: "0 0 16px", fontSize: 19, lineHeight: 1.35 }}>{qKey}</h2>
          <div style={{ whiteSpace: "pre-wrap", fontSize: 15, lineHeight: 1.75, color: C.ink }}>{answer}</div>
        </Card>
      </div>
    </div>
  );
}

// =============== MODELS ===============
function ModelsSection({ mode }) {
  return (
    <div>
      <p style={{ color: C.sub, fontSize: 14, margin: "0 0 16px" }}>
        Four detectors — two learn from labeled attacks (supervised), two learn only what "normal" looks like (unsupervised).
      </p>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(280px,1fr))", gap: 14 }}>
        {MODELS.map((m) => (
          <Card key={m.key}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
              <div style={{ fontWeight: 800, fontSize: 16 }}>{m.name}</div>
              <span style={{ fontFamily: mono, fontSize: 10.5, color: m.type === "supervised" ? C.accent : C.violet, border: `1px solid ${m.type === "supervised" ? C.accent : C.violet}`, borderRadius: 6, padding: "2px 8px" }}>{m.type}</span>
            </div>
            <p style={{ fontSize: 13.5, color: C.sub, lineHeight: 1.6, margin: "0 0 12px" }}>{T(mode, m)}</p>
            <div style={{ background: C.bg, borderRadius: 9, padding: "10px 12px", borderLeft: `3px solid ${C.cyan}` }}>
              <div style={{ fontFamily: mono, fontSize: 10, color: C.cyan, marginBottom: 4 }}>WHY IN THIS PROJECT</div>
              <div style={{ fontSize: 12.5, color: C.sub, lineHeight: 1.55 }}>{m.why}</div>
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}

// =============== XAI ===============
function XAISection({ mode }) {
  return (
    <div>
      <p style={{ color: C.sub, fontSize: 14, margin: "0 0 16px" }}>
        Four ways to explain WHY a model flagged something. ESS measures how much these agree.
      </p>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(280px,1fr))", gap: 14 }}>
        {XAI.map((x) => (
          <Card key={x.name}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
              <div style={{ fontWeight: 800, fontSize: 16 }}>{x.name}</div>
              <span style={{ fontFamily: mono, fontSize: 10.5, color: C.cyan }}>{x.applies}</span>
            </div>
            <p style={{ fontSize: 13.5, color: C.sub, lineHeight: 1.6, margin: "0 0 12px" }}>{T(mode, x)}</p>
            <div style={{ background: C.bg, borderRadius: 9, padding: "10px 12px", borderLeft: `3px solid ${C.cyan}` }}>
              <div style={{ fontFamily: mono, fontSize: 10, color: C.cyan, marginBottom: 4 }}>WHY IN THIS PROJECT</div>
              <div style={{ fontSize: 12.5, color: C.sub, lineHeight: 1.55 }}>{x.why}</div>
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}

// =============== EVALUATION ===============
function EvalSection({ mode }) {
  const colorOf = { accent: C.accent, amber: C.amber, green: C.green };
  return (
    <div>
      <Card style={{ marginBottom: 16, borderLeft: `3px solid ${C.cyan}` }}>
        <div style={{ fontFamily: mono, fontSize: 12, color: C.cyan, marginBottom: 10 }}>// three ways to evaluate</div>
        <p style={{ fontSize: 16, lineHeight: 1.7, margin: 0 }}>
          {T(mode, {
            simple: "Most benchmarks only ask 'is the AI accurate?' This project evaluates in three ways — how well it detects (ML), how practical it is to run (SE), and how much its explanations agree (ESS). Together they give a fuller, more honest picture.",
            technical: "Evaluation combines three layers: standard ML performance metrics, software-engineering (SE) metrics for deployability, and the ESS stability metric introduced here. A model can score well on one layer and poorly on another — which is precisely the point.",
          })}
        </p>
      </Card>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(280px,1fr))", gap: 14 }}>
        {EVAL.map((e) => (
          <Card key={e.name}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
              <div style={{ fontWeight: 800, fontSize: 16 }}>{e.name}</div>
              <span style={{ fontFamily: mono, fontSize: 10.5, color: colorOf[e.color], border: `1px solid ${colorOf[e.color]}`, borderRadius: 6, padding: "2px 8px" }}>{e.tag}</span>
            </div>
            <p style={{ fontSize: 13.5, color: C.sub, lineHeight: 1.6, margin: "0 0 12px" }}>{T(mode, e)}</p>
            <div style={{ background: C.bg, borderRadius: 9, padding: "10px 12px", borderLeft: `3px solid ${colorOf[e.color]}` }}>
              <div style={{ fontFamily: mono, fontSize: 10, color: colorOf[e.color], marginBottom: 4 }}>WHY IN THIS PROJECT</div>
              <div style={{ fontSize: 12.5, color: C.sub, lineHeight: 1.55 }}>{e.why}</div>
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}

// =============== DATA ===============
function DataSection({ mode }) {
  const [sel, setSel] = useState("uavcan");
  const [showInfo, setShowInfo] = useState(true); // show the dataset description panel
  const d = DATASETS.find((x) => x.key === sel);
  const a = avgEss(d);
  return (
    <div>
      <p style={{ color: C.sub, fontSize: 14, margin: "0 0 14px" }}>
        {T(mode, {
          simple: "Pick a dataset (ordered by number of signals). Tap ⓘ to learn what each dataset is for. Blue = catches attacks. Green/red = do the explanation tools agree?",
          technical: "Datasets ordered by feature count. Tap ⓘ for each dataset's role. Detection = attack-focused F1. Agreement = ESS (mean pairwise top-5 Jaccard).",
        })}
      </p>

      {/* dataset chips with info toggle */}
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 16 }}>
        {byFeatures.map((x) => (
          <div key={x.key} style={{ display: "flex", alignItems: "center", border: `1px solid ${sel === x.key ? C.cyan : C.soft}`, borderRadius: 20, overflow: "hidden", background: sel === x.key ? C.cyan : C.panel }}>
            <button onClick={() => { setSel(x.key); setShowInfo(true); }}
              style={{ cursor: "pointer", fontFamily: mono, fontSize: 12, fontWeight: 600, padding: "8px 6px 8px 12px", border: "none", background: "transparent", color: sel === x.key ? C.bg : C.sub }}>
              {x.name} · {x.features}f
            </button>
            <button onClick={() => { setSel(x.key); setShowInfo(true); }} title="What is this dataset?"
              style={{ cursor: "pointer", border: "none", background: "transparent", padding: "8px 10px 8px 4px",
                color: sel === x.key ? C.bg : C.cyan, fontSize: 13, fontWeight: 700 }}>
              ⓘ
            </button>
          </div>
        ))}
      </div>

      {/* dataset description panel */}
      {showInfo && (
        <Card style={{ marginBottom: 16, borderLeft: `3px solid ${C.cyan}` }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 10 }}>
            <div>
              <div style={{ fontSize: 16, fontWeight: 800, marginBottom: 4 }}>{d.name} <span style={{ fontFamily: mono, fontSize: 11, color: C.sub, fontWeight: 400 }}>· {d.features} features · {d.classes} classes</span></div>
              <div style={{ fontSize: 13.5, color: C.ink, lineHeight: 1.6, marginBottom: 10 }}>{d.role}</div>
              <div style={{ background: C.bg, borderRadius: 9, padding: "10px 12px" }}>
                <div style={{ fontFamily: mono, fontSize: 10, color: C.cyan, marginBottom: 4 }}>WHY THIS DATASET</div>
                <div style={{ fontSize: 12.5, color: C.sub, lineHeight: 1.55 }}>{d.why}</div>
              </div>
            </div>
            <button onClick={() => setShowInfo(false)} title="Hide"
              style={{ cursor: "pointer", border: "none", background: "transparent", color: C.sub, fontSize: 18, lineHeight: 1 }}>×</button>
          </div>
        </Card>
      )}

      <div style={{ display: "grid", gridTemplateColumns: "minmax(0,1fr) 220px", gap: 16 }}>
        <Card>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 16 }}>
            <div style={{ fontSize: 19, fontWeight: 700 }}>{d.name}</div>
            <div style={{ fontFamily: mono, fontSize: 11.5, color: C.sub }}>{d.domain}</div>
          </div>
          {MODELS.map((m) => (
            <div key={m.key} style={{ marginBottom: 15 }}>
              <div style={{ fontSize: 13, marginBottom: 6 }}><b>{m.name}</b> <span style={{ color: C.sub, fontFamily: mono, fontSize: 11 }}>{m.type}</span></div>
              <MiniBar label={T(mode, { simple: "Detection", technical: "F1" })} val={d.f1[m.key]} color={C.accent} />
              <MiniBar label={T(mode, { simple: "Agreement", technical: "ESS" })} val={d.ess[m.key]} color={essColor(d.ess[m.key])} tag={essWord(d.ess[m.key])} />
            </div>
          ))}
        </Card>
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <Card style={{ padding: 16 }}>
            <div style={{ fontFamily: mono, fontSize: 10.5, color: C.sub, textTransform: "uppercase", letterSpacing: 1, marginBottom: 10 }}>at a glance</div>
            <Row k={T(mode, { simple: "Signals (features)", technical: "Features" })} v={d.features} />
            <Row k={T(mode, { simple: "Attack types", technical: "Classes" })} v={d.classes} />
            <Row k={T(mode, { simple: "Avg. agreement", technical: "Avg. ESS" })} v={a.toFixed(3)} color={essColor(a)} />
          </Card>
          {/* legend: what numbers + colors mean */}
          <Card style={{ padding: 16 }}>
            <div style={{ fontFamily: mono, fontSize: 10.5, color: C.sub, textTransform: "uppercase", letterSpacing: 1, marginBottom: 10 }}>how to read this</div>
            <LegendRow color={C.accent} label={T(mode, { simple: "Blue = Detection (F1)", technical: "F1 — attack detection" })} />
            <LegendRow color={C.green} label={T(mode, { simple: "Green = tools agree", technical: "ESS ≥ 0.40" })} />
            <LegendRow color={C.amber} label={T(mode, { simple: "Amber = some agreement", technical: "0.20 ≤ ESS < 0.40" })} />
            <LegendRow color={C.red} label={T(mode, { simple: "Red = tools disagree", technical: "ESS < 0.20" })} />
          </Card>
          {d.key === "uav_cyber" && (
            <div style={{ background: C.soft, borderRadius: 14, padding: 15, borderLeft: `3px solid ${C.red}` }}>
              <div style={{ fontSize: 12.5, color: C.sub, lineHeight: 1.5 }}>
                The Autoencoder here catches almost no attacks (F1 0.023). A real result, shown honestly — the model genuinely fails on this dataset.
              </div>
            </div>
          )}
        </div>
      </div>

      {/* group summary: why these 6 together */}
      <Card style={{ marginTop: 16, borderLeft: `3px solid ${C.cyan}` }}>
        <div style={{ fontFamily: mono, fontSize: 11, color: C.cyan, marginBottom: 8 }}>// why these six datasets together</div>
        <p style={{ fontSize: 14, color: C.ink, lineHeight: 1.7, margin: 0 }}>
          {T(mode, {
            simple: "These six were chosen to span a wide range of complexity — from just 10 signals (UAVCAN) up to 80 (UAV-Attack, CICIDS). Four are drone-specific; two are broad network datasets that ground the study in familiar benchmarks. Spreading across that range is exactly what lets the project see how explanation agreement changes as data gets more complex — the central finding.",
            technical: "The six datasets span feature dimensionalities from 10 to 80, mixing drone-specific IDS data (UAVCAN, UAVIDS-2025, UAV-Cyber, UAV-Attack) with established general benchmarks (ISOT, CICIDS2017). This deliberate spread across dimensionality is what enables the cross-dataset ESS analysis: without varied feature counts, the dimensionality–stability relationship could not be observed. UAVIDS-2025 additionally serves as an out-of-sample check on that relationship.",
          })}
        </p>
      </Card>
    </div>
  );
}
function LegendRow({ color, label }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "5px 0", fontSize: 12.5, color: C.sub }}>
      <span style={{ width: 12, height: 12, borderRadius: 3, background: color, flexShrink: 0 }} />
      {label}
    </div>
  );
}
function MiniBar({ label, val, color, tag }) {
  return (
    <div style={{ marginBottom: 6 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11.5, color: C.sub, marginBottom: 3 }}>
        <span>{label}{tag && <span style={{ color, fontWeight: 700 }}> · {tag}</span>}</span>
        <span style={{ fontFamily: mono, color: C.ink }}>{val.toFixed(3)}</span>
      </div>
      <div style={{ height: 8, background: C.bg, borderRadius: 5, overflow: "hidden" }}>
        <div style={{ width: `${Math.max(2, val * 100)}%`, height: "100%", background: color, transition: "width .35s" }} />
      </div>
    </div>
  );
}
function Row({ k, v, color }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", padding: "6px 0", borderBottom: `1px solid ${C.soft}`, fontSize: 12.5 }}>
      <span style={{ color: C.sub }}>{k}</span>
      <b style={{ color: color || C.ink, fontFamily: mono }}>{v}</b>
    </div>
  );
}

// =============== FINDING ===============
function FindingSection({ mode, go, goTo }) {
  const pts = byFeatures.map((d) => ({ name: d.name, f: d.features, e: avgEss(d) }));
  const W = 640, H = 300, pl = 52, pb = 44, pt = 20, pr = 20, xmax = 88, ymax = 0.55;
  const px = (f) => pl + (f / xmax) * (W - pl - pr);
  const py = (e) => H - pb - (e / ymax) * (H - pt - pb);
  const unit = mode === "simple" ? "signals" : "features";
  return (
    <div>
      <Card style={{ marginBottom: 16, borderLeft: `3px solid ${C.cyan}` }}>
        <div style={{ fontFamily: mono, fontSize: 12, color: C.cyan, marginBottom: 10 }}>// the finding</div>
        <p style={{ fontSize: 16.5, lineHeight: 1.7, margin: 0 }}>
          {T(mode, {
            simple: `The fewer ${unit} a dataset has, the more the explanation tools agree. With ~10 ${unit}, tools mostly agree. With 80, they scatter. So how much you can trust an explanation depends partly on how complex the data is.`,
            technical: "Average ESS declines as feature count grows: low-dimensional datasets (10–37 features) reach higher agreement, while 80-feature datasets fall lowest. UAVIDS-2025 (18 features) landed where the trend predicted. Note: part of this may stem from top-5 selection over larger feature spaces.",
          })}
        </p>
      </Card>

      {/* what this chart shows — ESS, not F1 */}
      <div style={{ background: C.soft, borderRadius: 12, padding: "12px 16px", marginBottom: 14, display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
        <span style={{ fontSize: 13, color: C.ink, lineHeight: 1.5 }}>
          {T(mode, {
            simple: "This chart is about AGREEMENT (ESS) — how much the explanation tools agree on why. It is NOT about detection (F1).",
            technical: "This chart plots average ESS (explanation agreement), not F1 (detection). The two are measured separately.",
          })}
        </span>
        <button onClick={() => goTo("overview", "def-f1")}
          style={{ cursor: "pointer", border: `1px solid ${C.cyan}`, background: "transparent", color: C.cyan, fontFamily: mono, fontSize: 11, fontWeight: 600, padding: "6px 10px", borderRadius: 7, whiteSpace: "nowrap" }}>
          what are F1 &amp; ESS? →
        </button>
      </div>

      <Card>
        {/* axis meaning legend */}
        <div style={{ display: "flex", justifyContent: "space-between", flexWrap: "wrap", gap: 8, marginBottom: 6, fontFamily: mono, fontSize: 11, color: C.sub }}>
          <span>↑ up = explanation tools <span style={{ color: C.green }}>agree</span> (high ESS)</span>
          <span>↓ down = they <span style={{ color: C.red }}>disagree</span> (low ESS)</span>
        </div>
        <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto" }}>
          {[0, 0.2, 0.4].map((t) => (
            <g key={t}>
              <line x1={pl} y1={py(t)} x2={W - pr} y2={py(t)} stroke={C.soft} strokeDasharray="3 4" />
              <text x={pl - 8} y={py(t) + 4} fontSize="10" fill={C.sub} textAnchor="end" fontFamily={mono}>{t.toFixed(1)}</text>
            </g>
          ))}
          <text x={16} y={py(0.27)} fontSize="10" fill={C.sub} fontFamily={mono} transform={`rotate(-90, 16, ${py(0.27)})`} textAnchor="middle">agreement (ESS) →</text>
          <text x={pl + 4} y={py(0.47)} fontSize="10" fill={C.green} fontFamily={mono}>agree</text>
          <text x={pl + 4} y={py(0.06)} fontSize="10" fill={C.red} fontFamily={mono}>disagree</text>
          <line x1={px(10)} y1={py(0.45)} x2={px(80)} y2={py(0.12)} stroke={C.cyan} strokeWidth="2" strokeDasharray="6 5" opacity="0.45" />
          {pts.map((p) => (
            <g key={p.name}>
              <circle cx={px(p.f)} cy={py(p.e)} r="7" fill={essColor(p.e)} stroke={C.bg} strokeWidth="2" />
              <text x={px(p.f)} y={py(p.e) - 13} fontSize="9.5" fill={C.ink} textAnchor="middle" fontWeight="700">{p.name}</text>
              <text x={px(p.f)} y={H - pb + 15} fontSize="9" fill={C.sub} textAnchor="middle" fontFamily={mono}>{p.f}</text>
            </g>
          ))}
          <text x={W / 2} y={H - 4} fontSize="11" fill={C.sub} textAnchor="middle">number of {unit} (more {unit} →) </text>
        </svg>
      </Card>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(210px,1fr))", gap: 12, marginTop: 16 }}>
        {[
          ["Why it happens", T(mode, { simple: `Picking the 'top 5' from only 10 ${unit} forces big overlap — agreement is almost automatic. From 80, each tool can pick a different 5.`, technical: "Top-5 selection over a small feature space forces high Jaccard overlap; over a large space, overlap collapses." })],
          ["Why it matters", T(mode, { simple: "For complex, high-signal security data, no single explanation tool is fully trustworthy on its own — cross-checking helps.", technical: "High-dimensional IDS settings yield unstable single-method explanations; report multi-method agreement alongside detection metrics." })],
          ["Interpret with care", T(mode, { simple: "Part of this pattern may come from the top-5 method itself, so it's a benchmark observation, not a universal law.", technical: "The relationship is partly intrinsic to top-k selection as dimensionality grows; not claimed as a universal causal law." })],
        ].map(([h, b]) => (
          <Card key={h}>
            <div style={{ fontWeight: 700, fontSize: 14, marginBottom: 6, color: C.cyan }}>{h}</div>
            <p style={{ fontSize: 13, color: C.sub, lineHeight: 1.55, margin: 0 }}>{b}</p>
          </Card>
        ))}
      </div>

      {/* what a company should do when tools disagree */}
      <Card style={{ marginTop: 16, borderLeft: `3px solid ${C.amber}` }}>
        <div style={{ fontFamily: mono, fontSize: 11, color: C.amber, marginBottom: 8 }}>// when the tools disagree — reducing risk</div>
        <p style={{ fontSize: 14, color: C.ink, lineHeight: 1.65, margin: "0 0 12px" }}>
          {T(mode, {
            simple: "If agreement is low, a security team shouldn't act on one explanation alone. They can cross-check several tools, send high-stakes alarms to a human, and report the agreement score next to the detection so operators know how much to trust the 'why'.",
            technical: "Low ESS implies method-dependent attributions. Practical mitigations: require cross-method agreement before an explanation informs a decision, route low-ESS detections to human review, and surface ESS alongside F1 as a first-class monitoring signal.",
          })}
        </p>
        <button onClick={() => go("qa")}
          style={{ cursor: "pointer", border: `1px solid ${C.amber}`, background: "transparent", color: C.amber, fontFamily: mono, fontSize: 11, fontWeight: 600, padding: "7px 12px", borderRadius: 7 }}>
          more for engineers → Q&amp;A
        </button>
      </Card>
    </div>
  );
}
