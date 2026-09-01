// The two things Python could not afford to do, and so never did.
//
// 1. AUROC by brute force. The published image AUROC comes from
//    sklearn.roc_auc_score, which sorts and works on ranks. verify/auroc.c does
//    the same thing in C, so a shared misunderstanding of tie handling would
//    survive both. This instead compares every positive against every negative
//    directly, which is the definition rather than the fast form of it, and is
//    O(P*N) per file.
//
// 2. The 299. models/*.json claim that 299 normal calibration images are what a
//    95%-confidence 1% false-alarm bound needs. verify/verify.R solves
//    1 - 0.99^n >= 0.95 for n, which is the same algebra the exporter used, so
//    a wrong formula would pass both. This checks the formula against the thing
//    it claims to describe: draw n normal scores, take the largest as the
//    threshold, and count how often the tail left above it is really under 1%.
//    If 1 - 0.99^n were not the coverage of that procedure, the simulated rate
//    would not match it.
//
//    What the simulation cannot do is show that 298 is too few. The analytic
//    coverage at 298 is 0.949963, which is 3.7e-5 short of 0.95, and separating
//    that from 0.95 by sampling would take of the order of 1e9 trials. The
//    minimality of 299 is left to the exact arithmetic in verify/verify.R.
//
// No crates. The generator is a xorshift64*, seeded fixed so a run is
// reproducible.

use std::env;
use std::fs;
use std::process::exit;

const TRIALS: u64 = 10_000_000;
const TARGET_FPR: f64 = 0.01;
const CONFIDENCE: f64 = 0.95;
const TOL: f64 = 1e-12;

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    // uniform on [0, 1), 53 bits
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }
}

// Value text just after "<key>":, matched with quotes and colon so that
// "score" never matches inside "scores".
fn value_of<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    let pat = format!("\"{}\":", key);
    text.find(&pat).map(|i| &text[i + pat.len()..])
}

fn number_of(text: &str, key: &str) -> Option<f64> {
    let rest = value_of(text, key)?;
    let end = rest
        .find(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-' || c == '+' || c == 'e' || c == 'E' || c == ' '))
        .unwrap_or(rest.len());
    rest[..end].trim().parse::<f64>().ok()
}

fn scores_of(text: &str) -> Vec<(i32, f64)> {
    let mut out = Vec::new();
    let mut rest = match value_of(text, "scores") {
        Some(r) => r,
        None => return out,
    };
    loop {
        let l = match rest.find("\"label\":") {
            Some(i) => i,
            None => break,
        };
        let after_l = &rest[l + "\"label\":".len()..];
        let s = match after_l.find("\"score\":") {
            Some(i) => i,
            None => break,
        };
        let label = match number_of(&rest[l..], "label") {
            Some(v) => v as i32,
            None => break,
        };
        let score = match number_of(&after_l[s..], "score") {
            Some(v) => v,
            None => break,
        };
        out.push((label, score));
        rest = &after_l[s + 1..];
    }
    out
}

// Every positive against every negative. Ties count a half, which is what the
// rank form of AUROC means by an averaged rank.
fn auroc_pairwise(points: &[(i32, f64)]) -> Option<f64> {
    let pos: Vec<f64> = points.iter().filter(|p| p.0 == 1).map(|p| p.1).collect();
    let neg: Vec<f64> = points.iter().filter(|p| p.0 == 0).map(|p| p.1).collect();
    if pos.is_empty() || neg.is_empty() {
        return None;
    }
    let mut wins = 0.0f64;
    for &a in &pos {
        for &b in &neg {
            if a > b {
                wins += 1.0;
            } else if a == b {
                wins += 0.5;
            }
        }
    }
    Some(wins / (pos.len() as f64 * neg.len() as f64))
}

// Fraction of runs in which the largest of n draws leaves less than the target
// above it. The draws are uniform, so "the tail above the threshold" is
// 1 - max directly, and the result does not depend on the score distribution.
fn coverage(n: usize, trials: u64, seed: u64) -> f64 {
    let mut rng = Rng::new(seed);
    let mut covered = 0u64;
    for _ in 0..trials {
        let mut hit = false;
        for _ in 0..n {
            if rng.next_f64() > 1.0 - TARGET_FPR {
                hit = true;
                break;
            }
        }
        if hit {
            covered += 1;
        }
    }
    covered as f64 / trials as f64
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: tolerance <root> <file.json> [...]");
        exit(2);
    }
    let root = &args[1];
    let mut failures = 0usize;
    let mut worst = 0.0f64;

    for path in &args[2..] {
        let text = match fs::read_to_string(path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("cannot read {}: {}", path, e);
                failures += 1;
                continue;
            }
        };
        let points = scores_of(&text);
        let published = number_of(&text, "image_auroc");
        let (got, want) = match (auroc_pairwise(&points), published) {
            (Some(g), Some(w)) => (g, w),
            _ => {
                eprintln!("{}: no usable scores or no published image_auroc", path);
                failures += 1;
                continue;
            }
        };
        let d = (got - want).abs();
        if d > worst {
            worst = d;
        }
        let bad = d > TOL;
        if bad {
            failures += 1;
        }
        println!(
            "  {:<44} {:5} pairs  auroc {:.15}  |d| {:.1e}  {}",
            path,
            points.iter().filter(|p| p.0 == 1).count() * points.iter().filter(|p| p.0 == 0).count(),
            got,
            d,
            if bad { "FAIL" } else { "ok" }
        );
    }
    println!(
        "  exhaustive pairwise AUROC on {} files, worst disagreement {:.1e}",
        args.len() - 2,
        worst
    );

    // ---- the sample size behind the guarantee
    let claimed: usize = {
        let text = fs::read_to_string(format!("{}/models/screw.json", root)).unwrap_or_default();
        number_of(&text, "n_required_for_guarantee").unwrap_or(0.0) as usize
    };
    if claimed == 0 {
        eprintln!("cannot read n_required_for_guarantee from models/screw.json");
        exit(2);
    }

    println!("  {} trials per sample size, uniform draws, xorshift64* with a fixed seed", TRIALS);
    let mut worst_z = 0.0f64;
    for (i, &n) in [100usize, claimed - 1, claimed, 600].iter().enumerate() {
        let sim = coverage(n, TRIALS, 0x9E37_79B9_7F4A_7C15 ^ (i as u64 + 1).wrapping_mul(0x1234_5678_9ABC_DEF1));
        let analytic = 1.0 - (1.0 - TARGET_FPR).powi(n as i32);
        let se = (analytic * (1.0 - analytic) / TRIALS as f64).sqrt();
        let z = (sim - analytic).abs() / se;
        if z > worst_z {
            worst_z = z;
        }
        println!(
            "  n = {:3}: simulated {:.6}  1-0.99^n = {:.6}  z = {:.2}  {}",
            n,
            sim,
            analytic,
            z,
            if z <= 4.0 { "ok" } else { "FAIL" }
        );
        if z > 4.0 {
            failures += 1;
        }
        if n == claimed {
            let margin = (sim - CONFIDENCE) / se;
            println!(
                "           and {:.6} >= {:.2} by {:.1} standard errors  {}",
                sim,
                CONFIDENCE,
                margin,
                if sim >= CONFIDENCE { "ok" } else { "FAIL" }
            );
            if sim < CONFIDENCE {
                failures += 1;
            }
        }
    }
    println!(
        "  the closed form the exporter used matches the simulated procedure, worst z = {:.2}",
        worst_z
    );

    if failures > 0 {
        println!("\n{} files disagree with their published AUROC", failures);
        exit(1);
    }
    println!("\nRust agrees with every published image AUROC to better than {:.0e}", TOL);
}
