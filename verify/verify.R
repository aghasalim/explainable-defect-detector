# The statistical half: the claims in the README that are not arithmetic.
#
# Three of them are inferences rather than measurements, and each was made once,
# in the script that produced the number.
#
#   1. "needs 299 normal calibration images for a 95%-confidence 1% bound".
#      That 299 is written into all 15 model cards. It is a consequence of
#      1 - 0.99^n >= 0.95 and nothing in the repo ever solved that equation
#      again.
#   2. "the measured peak-in-mask rate clears its control by a wide margin in
#      every category". Clears it by how much, and is the gap larger than the
#      sampling error of a proportion measured on 30 to 141 images?
#   3. The threshold is sold as a 1% false-alarm rate. On the test split it is
#      not 1% anywhere. Which of the departures are larger than chance?
#
# Base R only, no packages, no JSON library: the report files are machine
# written with one key per line, so a regex reads them.

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) > 0) args[1] else "."

read_flat <- function(txt) {
  m <- regmatches(txt, gregexpr('"[A-Za-z0-9_]+"[[:space:]]*:[[:space:]]*[^,\n}]+', txt))[[1]]
  keys <- sub('^"([A-Za-z0-9_]+)".*$', "\\1", m)
  vals <- sub('^"[A-Za-z0-9_]+"[[:space:]]*:[[:space:]]*', "", m)
  vals <- trimws(gsub('"', "", vals))
  out <- as.list(vals)
  names(out) <- keys
  out
}

read_object <- function(path) read_flat(paste(readLines(file.path(root, path), warn = FALSE), collapse = "\n"))

read_array <- function(path) {
  txt <- paste(readLines(file.path(root, path), warn = FALSE), collapse = "\n")
  chunks <- strsplit(txt, "\\},")[[1]]
  lapply(chunks, read_flat)
}

n <- function(rec, key) as.numeric(rec[[key]])

cats <- c("bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
          "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor",
          "wood", "zipper")

fails <- character(0)
require_that <- function(cond, msg) if (!isTRUE(cond)) fails <<- c(fails, msg)

# ---------------------------------------------------------------- 1. the 299
# Smallest n with P(the largest of n normal scores sits above the 99th
# percentile of the normal distribution) >= 0.95. Solved by search, not by the
# closed form, so a sign error in the closed form would show up.
coverage <- 1 - 0.99^(1:2000)
required <- min(which(coverage >= 0.95))
cat(sprintf("  distribution-free bound: smallest n with 1-0.99^n >= 0.95 is %d (coverage %.6f, %d gives %.6f)\n",
            required, coverage[required], required - 1, coverage[required - 1]))
require_that(coverage[required - 1] < 0.95, "the search returned a value that is not minimal")

models <- lapply(cats, function(c) read_object(file.path("models", paste0(c, ".json"))))
names(models) <- cats
for (c in cats) {
  m <- models[[c]]
  require_that(n(m, "n_required_for_guarantee") == required,
               sprintf("models/%s.json claims %s calibration images are needed, the bound needs %d",
                       c, m$n_required_for_guarantee, required))
}
cat(sprintf("  all %d model cards agree that the guarantee needs %d images\n", length(cats), required))

# Per category, the confidence actually achieved by the calibration set used.
met <- 0
for (c in cats) {
  m <- models[[c]]
  nc <- n(m, "n_calib")
  tgt <- n(m, "target_fpr")
  conf <- 1 - (1 - tgt)^nc
  claimed <- identical(tolower(m$guarantee_met), "true")
  require_that(claimed == (conf >= 0.95),
               sprintf("models/%s.json says guarantee_met=%s but %d images give confidence %.4f",
                       c, m$guarantee_met, nc, conf))
  if (claimed) met <- met + 1
}
cat(sprintf("  %d of %d categories reach 95%% confidence at a 1%% false-alarm rate\n", met, length(cats)))

# ------------------------------------------------- 2. localisation vs control
# peak-in-mask is a proportion over the anomalous test images of the category,
# so it has an exact binomial interval. Clopper-Pearson, two sided 95%.
worst_margin <- Inf
worst_cat <- ""
wins <- 0
for (c in cats) {
  b <- read_object(file.path("reports", paste0("bench_", c, ".json")))
  na <- n(b, "n_anomalous")
  p <- n(b, "peak_in_mask")
  ctrl <- n(b, "control_peak_in_mask")
  k <- p * na
  require_that(abs(k - round(k)) < 1e-9,
               sprintf("bench_%s.json peak_in_mask %.10f is not a whole number of %d images", c, p, na))
  k <- round(k)
  lo <- if (k == 0) 0 else qbeta(0.025, k, na - k + 1)
  hi <- if (k == na) 1 else qbeta(0.975, k + 1, na - k)
  require_that(lo > ctrl,
               sprintf("%s: peak-in-mask %.4f has 95%% lower bound %.4f, control is %.4f", c, p, lo, ctrl))
  if (lo > ctrl) wins <- wins + 1
  if (lo - ctrl < worst_margin) { worst_margin <- lo - ctrl; worst_cat <- c }
  cat(sprintf("  %-11s peak-in-mask %.4f  95%% CI [%.4f, %.4f] on %3d images   control %.4f\n",
              c, p, lo, hi, na, ctrl))
}
cat(sprintf("  %d of %d categories clear their control, tightest margin %.4f on %s\n",
            wins, length(cats), worst_margin, worst_cat))
# All 15 in the same direction: a sign test on the paired comparison.
sign_p <- binom.test(wins, length(cats), 0.5, alternative = "greater")$p.value
cat(sprintf("  sign test over the 15 paired categories: p = %.3g\n", sign_p))
require_that(sign_p < 0.001, "the model does not beat the control consistently across categories")

# -------------------------------------------------- 3. the realised false-alarm rate
tc <- read_array("reports/threshold_check.json")
above <- character(0)
for (r in tc) {
  cat_r <- r$category
  nn <- n(r, "n_normal")
  fpr <- n(r, "realised_fpr_on_test")
  tgt <- n(r, "target_fpr")
  k <- fpr * nn
  require_that(abs(k - round(k)) < 1e-9,
               sprintf("threshold_check %s realised FPR %.10f is not a whole number of %d images", cat_r, fpr, nn))
  k <- round(k)
  # one sided exact test, is the realised rate above the target it claims
  p <- pbinom(k - 1, nn, tgt, lower.tail = FALSE)
  if (k > 0)
    cat(sprintf("  %-11s %d of %2d normal test images above threshold, exact p vs %.0f%% target = %.4g\n",
                cat_r, k, nn, 100 * tgt, p))
  if (p < 0.05) above <- c(above, cat_r)
}
cat(sprintf("  %d of %d categories exceed the 1%% target by more than chance: %s\n",
            length(above), length(tc), if (length(above)) paste(above, collapse = ", ") else "none"))
require_that(length(above) == 1 && above[1] == "carpet",
             sprintf("expected carpet alone to exceed its target, got: %s", paste(above, collapse = ", ")))

# also check recall is a whole number of the anomalous images
for (r in tc) {
  k <- n(r, "recall_on_test") * n(r, "n_anomalous")
  require_that(abs(k - round(k)) < 1e-9,
               sprintf("threshold_check %s recall %.10f is not a whole number of images", r$category, n(r, "recall_on_test")))
}

if (length(fails)) {
  cat("\n", length(fails), " failed:\n", sep = "")
  for (f in fails) cat("  ", f, "\n", sep = "")
  quit(status = 1)
}
cat("\nR: every inference in the README reproduces\n")
