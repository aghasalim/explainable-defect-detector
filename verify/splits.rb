# Is the classifier comparison actually run on a held-out half?
#
# reports/results.md claims the supervised classifier and PatchCore were scored
# on "the same held-out half" of the test split. That claim is the whole basis
# for the section: if the two halves overlapped, the classifier would be scored
# partly on images it trained on and its AUROC would mean nothing. Nothing in
# the repository checked it. The split is committed in reports/split_*.json and
# the metrics in reports/compare_*.json, and no script ever compares them.
#
# So this rebuilds the comparison from the two rawer files:
#
#   * the fit and held-out lists are disjoint, have no repeats, and together
#     cover the whole test split of that category as the dataset index counts it
#   * the counts published in compare_*.json are the lengths of those lists, and
#     the anomalous count is what the held-out paths say it is
#   * the published PatchCore AUROC and average precision for the held-out half
#     are recomputed from the per image scores in patchcore-224crop_*.json,
#     restricted to exactly those paths
#
# The last one is the real check. It ties three files written by three different
# scripts to one number, and it can only agree if the split used to report the
# metric is the split that was committed.
#
# Run: ruby verify/splits.rb [root]

require "json"

root = ARGV[0] || "."
CATEGORIES = %w[bottle pill screw].freeze
TOL = 1e-12

def load(root, rel)
  JSON.parse(File.read(File.join(root, rel)))
end

# Mann-Whitney U with mid ranks for ties, which is what AUROC is.
def auroc(points)
  pos = points.count { |_, l| l == 1 }
  neg = points.length - pos
  return nil if pos.zero? || neg.zero?

  asc = points.sort_by { |s, _| s }
  rank_sum = 0.0
  i = 0
  while i < asc.length
    j = i
    j += 1 while j + 1 < asc.length && asc[j + 1][0] == asc[i][0]
    mid = (i + j) / 2.0 + 1.0
    (i..j).each { |k| rank_sum += mid if asc[k][1] == 1 }
    i = j + 1
  end
  (rank_sum - pos * (pos + 1) / 2.0) / (pos.to_f * neg)
end

# Average precision as scikit-learn computes it: a step sum over the
# precision-recall curve, with the curve cut at the lowest threshold that still
# reaches full recall.
def average_precision(points)
  asc = points.sort_by { |s, _| s }
  n = asc.length
  npos = asc.count { |_, l| l == 1 }
  return nil if npos.zero? || npos == n

  prec = []
  rec = []
  i = 0
  while i < n
    j = i
    j += 1 while j + 1 < n && asc[j + 1][0] == asc[i][0]
    tp = asc[i..].count { |_, l| l == 1 }
    prec << tp.to_f / (n - i)
    rec << tp.to_f / npos
    i = j + 1
  end
  start = 0
  start += 1 while start + 1 < rec.length && rec[start + 1] == 1.0
  prec = prec[start..] + [1.0]
  rec = rec[start..] + [0.0]
  (0...(rec.length - 1)).sum { |k| -(rec[k + 1] - rec[k]) * prec[k] }
end

index = load(root, "data/_mvtec_index.json")["samples"]
test_per_category = Hash.new(0)
index.each do |s|
  test_per_category[s["category"]["label"]] += 1 if s["split"] == "test"
end

failures = []
def check(failures, cond, msg)
  failures << msg unless cond
  cond
end

CATEGORIES.each do |cat|
  split = load(root, "reports/split_#{cat}.json")
  cmp = load(root, "reports/compare_#{cat}.json")
  run = load(root, "reports/patchcore-224crop_#{cat}.json")

  fit = split["fit"]
  held = split["held_out"]
  scored = run["scores"].map { |s| [s["path"], [s["score"], s["label"]]] }.to_h

  check(failures, fit.uniq.length == fit.length, "#{cat}: the fit list repeats an image")
  check(failures, held.uniq.length == held.length, "#{cat}: the held-out list repeats an image")
  overlap = fit & held
  check(failures, overlap.empty?,
        "#{cat}: #{overlap.length} images are in both halves, so the comparison is contaminated")
  check(failures, fit.length + held.length == test_per_category[cat],
        "#{cat}: #{fit.length} + #{held.length} images split, the index has " \
        "#{test_per_category[cat]} test images")
  check(failures, scored.length == test_per_category[cat],
        "#{cat}: patchcore-224crop scored #{scored.length} images, the index has " \
        "#{test_per_category[cat]}")
  missing = (fit + held).reject { |p| scored.key?(p) }
  check(failures, missing.empty?,
        "#{cat}: #{missing.length} split images were never scored, first #{missing.first}")

  check(failures, cmp["n_fit"] == fit.length,
        "#{cat}: compare says n_fit #{cmp['n_fit']}, the split holds #{fit.length}")
  check(failures, cmp["n_held_out"] == held.length,
        "#{cat}: compare says n_held_out #{cmp['n_held_out']}, the split holds #{held.length}")
  anomalous = held.count { |p| !p.include?("/test/good/") }
  check(failures, cmp["n_anomalous_held_out"] == anomalous,
        "#{cat}: compare says #{cmp['n_anomalous_held_out']} anomalous held out, the paths " \
        "say #{anomalous}")

  points = held.map { |p| scored[p] }.compact
  got_auroc = auroc(points)
  got_ap = average_precision(points)
  want_auroc = cmp["patchcore"]["image_auroc"]
  want_ap = cmp["patchcore"]["average_precision"]
  d_auroc = (got_auroc - want_auroc).abs
  d_ap = (got_ap - want_ap).abs
  ok = d_auroc <= TOL && d_ap <= TOL
  check(failures, ok,
        "#{cat}: held-out AUROC #{got_auroc} vs published #{want_auroc}, AP #{got_ap} vs #{want_ap}")

  printf("  %-7s %3d fit + %3d held out = %3d test images, %3d anomalous   " \
         "auroc %.15f |d| %.1e   ap |d| %.1e  %s\n",
         cat, fit.length, held.length, test_per_category[cat], anomalous,
         got_auroc, d_auroc, d_ap, ok ? "ok" : "FAIL")
end

if failures.any?
  puts "\n#{failures.length} failed:"
  failures.each { |f| puts "  #{f}" }
  exit 1
end
puts "\nRuby: the two halves are disjoint and cover the test split, and the published " \
     "held-out\nPatchCore metrics recompute from the committed per image scores"
