// Rebuild every published table from the run artefacts, in JavaScript.
//
// reports/benchmark.md and reports/results.md are the tables a reader actually
// looks at. Both are written by Python (src/edd/benchmark.py and
// src/edd/report.py) out of the same process that produced the numbers, so a
// formatting or ordering mistake there would be invisible: the markdown and the
// script agree by construction because one made the other.
//
// This reads reports/bench_*.json, reports/compare_*.json and
// reports/threshold_check.json and rebuilds every row of both documents from
// scratch, then requires the published files to match line for line. It also
// rebuilds the two sentences that carry derived numbers, since a sentence goes
// stale exactly as easily as a table and nothing regenerates it.
//
// Run: node verify/tables.mjs [root]

import { readFileSync } from "node:fs";
import { join } from "node:path";

const root = process.argv[2] ?? ".";
const read = (p) => JSON.parse(readFileSync(join(root, p), "utf8"));
const text = (p) => readFileSync(join(root, p), "utf8");

const CATEGORIES = [
  "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
  "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood",
  "zipper",
];

// Python's format specs, spelled out. f is {:.Nf}, s is {:+.Nf} and p is {:.N%}.
//
// toFixed and Python disagree on one case and it is not hypothetical: an exact
// tie. Four of the timings are whole halves, and 58.5 formats as 59 in
// JavaScript and 58 in Python, which rounds a tie to the even neighbour. So
// ties are handled here rather than papered over with a tolerance.
// A tie has to be found in the decimal expansion of the double itself, not by
// scaling it: 0.91875 is not exactly representable and multiplying it by 10000
// lands on 9187.5, which looks like a tie and is not one.
function f(x, n) {
  const exact = Math.abs(x).toFixed(n + 17);
  const dot = exact.indexOf(".");
  if (/^50*$/.test(exact.slice(dot + 1 + n))) {
    let k = BigInt(exact.slice(0, dot) + exact.slice(dot + 1, dot + 1 + n));
    if (k % 2n === 1n) k += 1n;
    const d = k.toString().padStart(n + 1, "0");
    const out = n === 0 ? d : d.slice(0, d.length - n) + "." + d.slice(d.length - n);
    return (x < 0 ? "-" : "") + out;
  }
  return x.toFixed(n);
}
const s = (x, n) => (x >= 0 ? "+" : "") + f(x, n);
const p = (x, n) => f(x * 100, n) + "%";
const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;

const bench = CATEGORIES.map((c) => read(`reports/bench_${c}.json`));
const col = (k) => bench.map((r) => r[k]);

// ---------------------------------------------------------------- benchmark.md
function benchmarkLines() {
  const L = [
    "| category | image AUROC | paper | gap | AP | acc @F1 | majority acc |",
    "|" + "---|".repeat(7),
  ];
  for (const r of bench) {
    L.push(
      `| ${r.category} | **${f(r.image_auroc, 4)}** | ${f(r.paper_image_auroc, 3)} ` +
      `| ${s(r.image_auroc - r.paper_image_auroc, 4)} ` +
      `| ${f(r.average_precision, 4)} | ${f(r.accuracy_at_best_f1, 4)} ` +
      `| ${f(r.majority_class_accuracy, 4)} |`);
  }
  const mi = mean(col("image_auroc")), mp = mean(col("paper_image_auroc"));
  L.push(`| **mean** | **${f(mi, 4)}** | ${f(mp, 3)} | ${s(mi - mp, 4)} | | | |`);

  L.push("| category | pixel AUROC | ctrl | AUPRO | ctrl | peak-in-mask | ctrl " +
         "| top-1% prec | ctrl | defect px |");
  L.push("|" + "---|".repeat(10));
  for (const r of bench) {
    L.push(
      `| ${r.category} | ${f(r.pixel_auroc, 4)} | ${f(r.control_pixel_auroc, 3)} ` +
      `| **${f(r.aupro, 4)}** | ${f(r.control_aupro, 3)} ` +
      `| **${f(r.peak_in_mask, 4)}** | ${f(r.control_peak_in_mask, 3)} ` +
      `| ${f(r.top1pct_precision, 4)} | ${f(r.control_top1pct_precision, 3)} ` +
      `| ${f(r.defect_pixel_fraction, 4)} |`);
  }
  L.push(
    `| **mean** | ${f(mean(col("pixel_auroc")), 4)} | | ` +
    `**${f(mean(col("aupro")), 4)}** | | ` +
    `**${f(mean(col("peak_in_mask")), 4)}** | | ` +
    `${f(mean(col("top1pct_precision")), 4)} | | |`);
  return L;
}

function benchmarkProse() {
  const worst = bench.reduce((a, b) => (b.peak_in_mask < a.peak_in_mask ? b : a));
  const best = bench.reduce((a, b) => (b.peak_in_mask > a.peak_in_mask ? b : a));
  return [
    `**Localisation is not uniform.** Peak-in-mask ranges from ` +
    `${p(worst.peak_in_mask, 0)} (\`${worst.category}\`, defects cover ` +
    `${p(worst.defect_pixel_fraction, 2)} of the image) to ${p(best.peak_in_mask, 0)} ` +
    `(\`${best.category}\`). Pixel AUROC hides this: \`${worst.category}\` still scores ` +
    `${f(worst.pixel_auroc, 4)} there, because the metric is dominated by easy ` +
    `background. A single headline number for 'explainability' would be misleading.`,
  ];
}

// ------------------------------------------------------------------ results.md
function resultsLines() {
  const L = [
    "| category | image AUROC | paper | Δ | pixel AUROC | paper | Δ | AUPRO | " +
    "peak-in-mask | sec |",
    "|" + "---|".repeat(10),
  ];
  for (const r of bench) {
    L.push(
      `| ${r.category} | ${f(r.image_auroc, 4)} | ${f(r.paper_image_auroc, 3)} ` +
      `| ${s(r.image_auroc - r.paper_image_auroc, 3)} ` +
      `| ${f(r.pixel_auroc, 4)} | ${f(r.paper_pixel_auroc, 3)} ` +
      `| ${s(r.pixel_auroc - r.paper_pixel_auroc, 3)} ` +
      `| ${f(r.aupro, 4)} | ${f(r.peak_in_mask, 4)} | ${f(r.seconds, 0)} |`);
  }
  const dImage = mean(bench.map((r) => r.image_auroc - r.paper_image_auroc));
  const dPixel = mean(bench.map((r) => r.pixel_auroc - r.paper_pixel_auroc));
  L.push(
    `| **mean (${bench.length})** | **${f(mean(col("image_auroc")), 4)}** | ` +
    `${f(mean(col("paper_image_auroc")), 3)} | ${s(dImage, 3)} ` +
    `| **${f(mean(col("pixel_auroc")), 4)}** | ${f(mean(col("paper_pixel_auroc")), 3)} ` +
    `| ${s(dPixel, 3)} | ${f(mean(col("aupro")), 4)} | ${f(mean(col("peak_in_mask")), 4)} ` +
    `| ${f(col("seconds").reduce((a, b) => a + b, 0), 0)} |`);

  L.push("| category | pixel AUROC | (random) | AUPRO | (random) | peak-in-mask | " +
         "(random) | top-1% prec | defect px |");
  L.push("|" + "---|".repeat(9));
  for (const r of bench) {
    L.push(
      `| ${r.category} | ${f(r.pixel_auroc, 4)} | ${f(r.control_pixel_auroc, 4)} ` +
      `| ${f(r.aupro, 4)} | ${f(r.control_aupro, 4)} ` +
      `| **${f(r.peak_in_mask, 4)}** | ${f(r.control_peak_in_mask, 4)} ` +
      `| ${f(r.top1pct_precision, 4)} | ${f(r.defect_pixel_fraction, 4)} |`);
  }

  const cmp = ["bottle", "pill", "screw"].map((c) => read(`reports/compare_${c}.json`));
  L.push("| category | held-out | classifier AUROC | PatchCore AUROC | Grad-CAM " +
         "peak-in-mask | PatchCore peak-in-mask | Grad-CAM pixel AUROC | PatchCore pixel " +
         "AUROC |");
  L.push("|" + "---|".repeat(8));
  for (const d of cmp) {
    const cl = d.classifier, pc = d.patchcore;
    L.push(
      `| ${d.category} | ${d.n_held_out} | ${f(cl.image_auroc, 4)} | ${f(pc.image_auroc, 4)} ` +
      `| **${f(cl.gradcam_peak_in_mask, 4)}** | **${f(pc.map_peak_in_mask, 4)}** ` +
      `| ${f(cl.gradcam_pixel_auroc, 4)} | ${f(pc.map_pixel_auroc, 4)} |`);
  }

  const tc = read("reports/threshold_check.json");
  L.push("| category | threshold | target FPR | realised FPR | recall |");
  L.push("|" + "---|".repeat(5));
  for (const r of tc) {
    L.push(`| ${r.category} | ${f(r.threshold, 3)} | ${p(r.target_fpr, 0)} ` +
           `| ${p(r.realised_fpr_on_test, 1)} | ${p(r.recall_on_test, 1)} |`);
  }
  return L;
}

function resultsProse() {
  const worst = [...bench].sort((a, b) => a.image_auroc - b.image_auroc).slice(0, 3);
  const tc = read("reports/threshold_check.json");
  const over = tc.filter((r) => r.realised_fpr_on_test > r.target_fpr)
                 .sort((a, b) => b.realised_fpr_on_test - a.realised_fpr_on_test);
  return [
    "Weakest categories by image AUROC: " +
    worst.map((d) => `\`${d.category}\` (${f(d.image_auroc, 3)})`).join(", ") + ".",

    `${tc.length - over.length} of the ${tc.length} categories land at or below the ` +
    "1% false-alarm target. The ones that do not are " +
    over.map((r) =>
      `\`${r.category}\` at ${p(r.realised_fpr_on_test, 1)}, ` +
      `${Math.round(r.realised_fpr_on_test * r.n_normal)} of ${r.n_normal} ` +
      "normal test images").join(", ") +
    ". What that buys is paid for in recall, which averages " +
    `${p(mean(tc.map((r) => r.recall_on_test)), 1)} here. Both numbers are ` +
    "measured on the test split; neither was used to pick the threshold.",
  ];
}

// ----------------------------------------------------------------------- check
let failures = 0;

function compareTables(doc, want) {
  const got = text(doc).split("\n").filter((l) => l.startsWith("|"));
  const n = Math.max(got.length, want.length);
  let bad = 0;
  for (let i = 0; i < n; i++) {
    if (got[i] !== want[i]) {
      if (bad < 5) {
        console.log(`  ${doc} line ${i + 1} of the tables differs`);
        console.log(`    published: ${got[i] ?? "(missing)"}`);
        console.log(`    rebuilt:   ${want[i] ?? "(missing)"}`);
      }
      bad++;
    }
  }
  failures += bad;
  console.log(`  ${doc}: ${want.length} table lines rebuilt, ${bad} disagree`);
}

function requireProse(doc, sentences) {
  const body = text(doc);
  let bad = 0;
  for (const line of sentences) {
    if (!body.includes(line)) {
      console.log(`  ${doc} does not contain the rebuilt sentence:`);
      console.log(`    ${line}`);
      bad++;
    }
  }
  failures += bad;
  console.log(`  ${doc}: ${sentences.length} derived sentences rebuilt, ${bad} disagree`);
}

compareTables("reports/benchmark.md", benchmarkLines());
requireProse("reports/benchmark.md", benchmarkProse());
compareTables("reports/results.md", resultsLines());
requireProse("reports/results.md", resultsProse());

if (failures > 0) {
  console.log(`\n${failures} published lines do not match the run artefacts`);
  process.exit(1);
}
console.log("\nJavaScript rebuilt both published tables from reports/*.json, character for character");
