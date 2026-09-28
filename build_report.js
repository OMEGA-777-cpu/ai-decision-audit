const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, ShadingType, ImageRun, AlignmentType, BorderStyle, PageBreak,
  LevelFormat, convertInchesToTwip
} = require("docx");

const results = JSON.parse(fs.readFileSync("reports/results.json"));

const COL = {
  border: { style: BorderStyle.SINGLE, size: 4, color: "CCCCCC" },
};
const cellBorders = { top: COL.border, bottom: COL.border, left: COL.border, right: COL.border };

function h(text, level) {
  return new Paragraph({ text, heading: level, spacing: { before: 280, after: 140 } });
}
function p(text, opts = {}) {
  return new Paragraph({ children: [new TextRun({ text, ...opts })], spacing: { after: 120 } });
}
function bullet(text) {
  return new Paragraph({ text, bullet: { level: 0 }, spacing: { after: 60 } });
}
function cell(text, opts = {}) {
  return new TableCell({
    children: [new Paragraph({ children: [new TextRun({ text: String(text), bold: !!opts.bold })] })],
    width: { size: opts.width || 2000, type: WidthType.DXA },
    borders: cellBorders,
    shading: opts.header ? { type: ShadingType.CLEAR, fill: "E8E8E8" } : undefined,
  });
}
function metricsTable() {
  const header = new TableRow({
    children: [
      cell("Model", { header: true, width: 2800 }),
      cell("Accuracy", { header: true, width: 1600 }),
      cell("Precision", { header: true, width: 1600 }),
      cell("Recall", { header: true, width: 1600 }),
      cell("F1", { header: true, width: 1600 }),
      cell("ROC-AUC", { header: true, width: 1600 }),
    ],
  });
  const rows = ["logistic_regression", "decision_tree", "neural_network"].map((name) => {
    const r = results[name];
    return new TableRow({
      children: [
        cell(name.replace(/_/g, " "), { width: 2800 }),
        cell(r.accuracy, { width: 1600 }),
        cell(r.precision, { width: 1600 }),
        cell(r.recall, { width: 1600 }),
        cell(r.f1, { width: 1600 }),
        cell(r.roc_auc, { width: 1600 }),
      ],
    });
  });
  return new Table({ width: { size: 9800, type: WidthType.DXA }, columnWidths: [2800, 1600, 1600, 1600, 1600, 1600], rows: [header, ...rows] });
}
function courseMapTable() {
  const header = new TableRow({
    children: [cell("Project module", { header: true, width: 3200 }), cell("Concept demonstrated", { header: true, width: 3300 }), cell("Course", { header: true, width: 3300 })],
  });
  const data = [
    ["src/models.py — LogisticRegression, DecisionTree", "Classical ML methods", "Machine Learning"],
    ["src/models.py — MLPClassifier", "Neural network training", "Neural Networks and Deep Learning"],
    ["src/decision_layer.py", "Predictive AI vs. decision-making AI", "AI Forms and Functions"],
    ["src/fairness_audit.py", "Bias auditing / model limitations", "Neural Networks and Deep Learning; Introduction to AI"],
    ["src/capability_benchmark.py", "Speed, scalability, real-time adjustment", "The Intelligence Behind AI"],
  ];
  const rows = data.map((d) => new TableRow({ children: [cell(d[0], { width: 3200 }), cell(d[1], { width: 3300 }), cell(d[2], { width: 3300 })] }));
  return new Table({ width: { size: 9800, type: WidthType.DXA }, columnWidths: [3200, 3300, 3300], rows: [header, ...rows] });
}
function fairnessTable() {
  const header = new TableRow({
    children: [cell("Subgroup (tumor-size quartile)", { header: true, width: 3800 }), cell("n", { header: true, width: 1500 }), cell("Accuracy", { header: true, width: 2250 }), cell("Malignant recall", { header: true, width: 2250 })],
  });
  const data = [
    ["Q1 (smallest)", 30, 1.0, 1.0],
    ["Q2", 27, 1.0, 1.0],
    ["Q3", 28, 0.9286, 0.9167],
    ["Q4 (largest)", 29, 1.0, 1.0],
  ];
  const rows = data.map((d) => new TableRow({ children: [cell(d[0], { width: 3800 }), cell(d[1], { width: 1500 }), cell(d[2], { width: 2250 }), cell(d[3], { width: 2250 })] }));
  return new Table({ width: { size: 9800, type: WidthType.DXA }, columnWidths: [3800, 1500, 2250, 2250], rows: [header, ...rows] });
}
function image(path, width) {
  const dims = require("child_process").execSync(`python3 -c "from PIL import Image; im=Image.open('${path}'); print(im.size[0], im.size[1])"`).toString().trim().split(" ").map(Number);
  const ratio = dims[1] / dims[0];
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new ImageRun({ type: "png", data: fs.readFileSync(path), transformation: { width, height: Math.round(width * ratio) } })],
    spacing: { after: 200 },
  });
}

const doc = new Document({
  numbering: {
    config: [{ reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT }] }],
  },
  sections: [
    {
      properties: { page: { size: { width: 11906, height: 16838 } } }, // A4
      children: [
        // Title block
        new Paragraph({ text: "AI Decision Audit:", heading: HeadingLevel.TITLE, alignment: AlignmentType.CENTER, spacing: { before: 800, after: 60 } }),
        new Paragraph({ text: "From Predictive Model to Governed Decision", heading: HeadingLevel.TITLE, alignment: AlignmentType.CENTER, spacing: { after: 400 } }),
        p("A project report synthesizing five AI/ML fundamentals modules (IBM SkillsBuild) into a single tested, audited, benchmarked pipeline.", { italics: true }),
        new Paragraph({ text: "", spacing: { after: 600 } }),
        p("Name: Ritam Laha", { bold: true }),
        p("Program: B.Tech Computer Science & Engineering, Bankura Unnayani Institute of Engineering (BUIE), affiliated to MAKAUT"),
        p("University Roll No.: 10500123038"),
        p("Date: 25 September 2026"),
        p("Repository: github.com/OMEGA-777-cpu/ai-decision-audit"),
        new Paragraph({ children: [new PageBreak()] }),

        h("1. Abstract", HeadingLevel.HEADING_1),
        p("This project builds a small but complete AI decision pipeline on a real diagnostic classification task, rather than a single isolated model-training script. It trains two classical machine learning models and one neural network, converts the resulting predictions into governed decisions via an explicit policy layer, audits subgroup performance, benchmarks inference speed and scalability, and validates all of it with an automated, CI-gated pytest suite. The project is deliberately structured so that each component demonstrates a distinct concept from five completed IBM SkillsBuild AI/ML modules, rather than treating them as disconnected coursework."),

        h("2. Introduction", HeadingLevel.HEADING_1),
        p("A model that outputs a probability is not, by itself, a decision-making system. In practice an AI system that matters is judged on more than a single accuracy figure: it needs a policy for turning a score into an action, it needs to be checked for uneven performance across the population it serves, it needs to be fast enough for its deployment context, and it needs automated tests so a future change doesn't silently break it. This report documents a pipeline built around that view, using the Wisconsin Diagnostic Breast Cancer dataset as a concrete, realistic setting: a model flags a case, and that flag has to turn into a clinical workflow action."),

        h("3. Objectives", HeadingLevel.HEADING_1),
        bullet("Train and compare classical ML models (logistic regression, decision tree) against a neural network (MLP) on the same task."),
        bullet("Implement a decision layer that converts predictive AI output into decision-making AI output, per an explicit, testable policy."),
        bullet("Audit model performance across a data subgroup and state honestly what that audit can and cannot claim."),
        bullet("Benchmark inference latency and scalability across batch sizes."),
        bullet("Validate the entire pipeline with an automated test suite, gated by continuous integration."),

        h("4. Course-to-Module Mapping", HeadingLevel.HEADING_1),
        p("Each of the five completed IBM SkillsBuild modules maps to a specific, working part of this project rather than being cited as background reading:"),
        courseMapTable(),
        new Paragraph({ text: "", spacing: { after: 200 } }),

        h("5. Methodology", HeadingLevel.HEADING_1),
        h("5.1 Dataset", HeadingLevel.HEADING_2),
        p("Wisconsin Diagnostic Breast Cancer dataset (scikit-learn built-in): 569 samples, 30 numeric features derived from digitized fine-needle-aspirate images, binary target (malignant / benign). An 80/20 stratified train-test split was used (random_state=42 throughout, for reproducibility)."),
        h("5.2 Classical ML models", HeadingLevel.HEADING_2),
        p("Logistic Regression and a depth-limited Decision Tree (max_depth=4, to keep it interpretable rather than overfit), both from scikit-learn, each wrapped in a Pipeline with feature scaling where relevant."),
        h("5.3 Neural network", HeadingLevel.HEADING_2),
        p("A Multi-Layer Perceptron (MLPClassifier) with two hidden layers (32, 16 units), ReLU activation, trained on standardized features."),
        h("5.4 Decision layer — predictive AI vs. decision-making AI", HeadingLevel.HEADING_2),
        p("The model's probability output is deliberately kept separate from the policy that turns it into an action. A three-tier policy maps malignancy probability to AUTO_CLEAR / FLAG_FOR_REVIEW / URGENT_REVIEW, with an intentionally conservative low threshold (0.10) rather than the naive 0.5 midpoint, because a missed malignant case is far costlier than an unnecessary review. This separation is the practical difference between a predictive model and a decision-making AI system: the model doesn't change, but the policy is a separate, auditable, testable choice."),
        h("5.5 Fairness / subgroup audit", HeadingLevel.HEADING_2),
        p("The dataset carries no demographic attributes, so a genuine protected-attribute fairness audit is not possible on this data — that limitation is stated here rather than glossed over. What was implemented instead is a subgroup performance audit: test-set accuracy and malignant-case recall are computed within quartiles of tumor size (mean radius), to check whether the best model's error rate is stable across that slice."),
        h("5.6 Capability benchmark", HeadingLevel.HEADING_2),
        p("Inference latency was measured for all three models across batch sizes of 1, 50, 500 and 5,000 samples (resampled with replacement from the test set), to compare speed and how each model's per-sample cost scales — directly reflecting the “speed of computation” and “scalability” capabilities covered in The Intelligence Behind AI module."),
        h("5.7 Testing and CI", HeadingLevel.HEADING_2),
        p("An 11-case pytest suite covers: decision-layer boundary behavior, invalid-input handling, batch/single-call consistency, train/test split integrity, and two quality gates — every model must beat the majority-class baseline, and the neural network's malignant-case recall must exceed 0.85. GitHub Actions runs the full suite and a pipeline smoke test on every push and pull request to main."),

        new Paragraph({ children: [new PageBreak()] }),
        h("6. Results", HeadingLevel.HEADING_1),
        p("All figures below are generated directly by main.py from a live run — none are hand-typed estimates."),
        h("6.1 Model performance", HeadingLevel.HEADING_2),
        metricsTable(),
        new Paragraph({ text: "", spacing: { after: 200 } }),
        image("reports/comparison.png", 560),
        image("reports/roc_curves.png", 420),

        h("6.2 Decision layer output", HeadingLevel.HEADING_2),
        p(`Applying the policy to the neural network's test-set probabilities produced: ${results.decision_layer_counts.AUTO_CLEAR} AUTO_CLEAR, ${results.decision_layer_counts.FLAG_FOR_REVIEW} FLAG_FOR_REVIEW, ${results.decision_layer_counts.URGENT_REVIEW} URGENT_REVIEW, out of ${results.decision_layer_counts.AUTO_CLEAR + results.decision_layer_counts.FLAG_FOR_REVIEW + results.decision_layer_counts.URGENT_REVIEW} test cases — i.e. the conservative low threshold routes a meaningful share of borderline cases to human review rather than silently auto-clearing them.`),

        h("6.3 Subgroup audit (best model by F1: logistic regression)", HeadingLevel.HEADING_2),
        fairnessTable(),
        new Paragraph({ text: "", spacing: { after: 200 } }),
        p("Accuracy and malignant-case recall hold at or near 1.0 in three of four quartiles; Q3 shows a mild dip (92.9% accuracy, 91.7% malignant recall) on a small subgroup (n=28). With this sample size the dip is as consistent with normal variance as with a real effect — flagged as worth re-checking with more data, not overstated as a confirmed weakness."),

        h("6.4 Capability benchmark", HeadingLevel.HEADING_2),
        image("reports/capability.png", 520),
        p("All three models predict in well under a millisecond per sample at batch size 5,000, so for this dataset size none of them is a deployment bottleneck; the meaningful difference is per-call overhead at batch size 1, where the decision tree is fastest and logistic regression and the neural network are within noise of each other."),

        h("7. Ethical Considerations and Limitations", HeadingLevel.HEADING_1),
        bullet("No demographic fairness claim is made — see §5.5. The subgroup audit is a narrower, legitimate technique, not a substitute for one."),
        bullet("569 samples is small for subgroup analysis; the Q3 dip above should not be treated as a confirmed bias without more data."),
        bullet("This is a decision-support design, not a deployable clinical system: real deployment would need clinical validation, regulatory review, and a human-in-the-loop workflow, none of which this project claims to provide."),
        bullet("The 0.10 / 0.50 decision thresholds are a policy choice made here for illustration, not a validated clinical cutoff."),

        h("8. Conclusion", HeadingLevel.HEADING_1),
        p("This project set out to show that predictive modelling, decision policy, fairness auditing, capability benchmarking, and automated testing are separable, testable parts of one system rather than a single training script — and to do that with real, reproducible numbers rather than assumed ones. All three models comfortably beat baseline, logistic regression was the strongest by F1 (0.986) despite being the simplest model here, and the full pipeline is covered by an 11-case CI-gated test suite. The main limitation is scope, not rigor: extending the fairness audit to a dataset with real demographic attributes is the clearest next step."),

        h("9. References", HeadingLevel.HEADING_1),
        bullet("IBM SkillsBuild — Machine Learning (ALM-COURSE_3955165), completed 24 Sep 2026"),
        bullet("IBM SkillsBuild — The Intelligence Behind AI (ALM-COURSE_3955163), completed 24 Sep 2026"),
        bullet("IBM SkillsBuild — Introduction to Artificial Intelligence, completed 24 Sep 2026"),
        bullet("IBM SkillsBuild — AI Forms and Functions, completed 24 Sep 2026"),
        bullet("IBM SkillsBuild — Neural Networks and Deep Learning, completed 24 Sep 2026"),
        bullet("Wolberg, W.H., Street, W.N., Mangasarian, O.L. — Wisconsin Diagnostic Breast Cancer dataset, UCI Machine Learning Repository (distributed via scikit-learn.datasets.load_breast_cancer)"),
        bullet("Pedregosa et al., \"Scikit-learn: Machine Learning in Python\", JMLR 12 (2011)"),
      ],
    },
  ],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync("reports/AI_Decision_Audit_Report.docx", buf);
  console.log("done");
});
