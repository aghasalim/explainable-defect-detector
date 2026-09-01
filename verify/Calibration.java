// The calibration story in notes/METHODS.md, recomputed from the run artefacts.
//
// Section 5 of the notes reports three calibration rules, a per category table
// of what the shipped rule does on the test split, and the sentence that says
// what it cost in recall. Every one of those numbers was typed into the notes
// by hand from a run that is now months old. The run itself survives in
// reports/calibration_compare.json, which holds the false-alarm rate and recall
// of both rules for all 15 categories, so the summary rows are an aggregation
// nobody has ever repeated.
//
// This repeats it, and then does the thing the notes cannot do for themselves:
// the shipped threshold appears in three files written by three different
// scripts, reports/calibration_compare.json (tol_thr), models/<cat>.json
// (threshold) and reports/threshold_check.json (threshold), and the realised
// rate appears in two of them. They are required to be the same number.
//
// The first row of the summary table, the 10% holdout attempt, has no artefact
// left in the repository and is not checked here. Saying so is better than
// pretending the check covers it.
//
// Run: java verify/Calibration.java [root]

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class Calibration {

    // Half of the last digit these tables print, so a published 3.4% may be
    // anything the run rounds to 3.4%, and nothing else.
    static final double DISPLAY_TOL = 0.05;
    static final double EXACT_TOL = 1e-12;

    static final List<String> FAILURES = new ArrayList<>();

    static void require(boolean cond, String message) {
        if (!cond) FAILURES.add(message);
    }

    static String read(String root, String rel) throws Exception {
        return Files.readString(Path.of(root, rel));
    }

    /** The report files are machine written with one key per line, so the value
     *  after "key": is enough of a parser. */
    static Double number(String chunk, String key) {
        Matcher m = Pattern.compile("\"" + key + "\"\\s*:\\s*(-?[0-9.eE+]+)").matcher(chunk);
        return m.find() ? Double.parseDouble(m.group(1)) : null;
    }

    static String string(String chunk, String key) {
        Matcher m = Pattern.compile("\"" + key + "\"\\s*:\\s*\"([^\"]*)\"").matcher(chunk);
        return m.find() ? m.group(1) : null;
    }

    /** Split a top level JSON array into one string per object. */
    static List<String> objects(String text) {
        List<String> out = new ArrayList<>();
        int depth = 0, start = -1;
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            if (c == '{') {
                if (depth == 0) start = i;
                depth++;
            } else if (c == '}') {
                depth--;
                if (depth == 0) out.add(text.substring(start, i + 1));
            }
        }
        return out;
    }

    static double mean(List<Double> xs) {
        double s = 0;
        for (double x : xs) s += x;
        return s / xs.size();
    }

    public static void main(String[] args) throws Exception {
        String root = args.length > 0 ? args[0] : ".";

        List<String> calib = objects(read(root, "reports/calibration_compare.json"));
        List<String> checks = objects(read(root, "reports/threshold_check.json"));
        String methods = read(root, "notes/METHODS.md");
        require(calib.size() == 15, "calibration_compare.json holds " + calib.size() + " rows");
        require(checks.size() == 15, "threshold_check.json holds " + checks.size() + " rows");

        // ---- one threshold, three files -----------------------------------
        int agreed = 0;
        for (String row : calib) {
            String cat = string(row, "cat");
            String model = read(root, "models/" + cat + ".json");
            String check = null;
            for (String c : checks) if (cat.equals(string(c, "category"))) check = c;
            if (check == null) {
                require(false, cat + " has no row in threshold_check.json");
                continue;
            }
            double tolThr = number(row, "tol_thr");
            require(Math.abs(number(model, "threshold") - tolThr) <= EXACT_TOL,
                    cat + ": models/" + cat + ".json threshold " + number(model, "threshold")
                            + " is not the calibrated " + tolThr);
            require(Math.abs(number(check, "threshold") - tolThr) <= EXACT_TOL,
                    cat + ": threshold_check.json threshold " + number(check, "threshold")
                            + " is not the calibrated " + tolThr);
            require(Math.abs(number(check, "realised_fpr_on_test") - number(row, "tol_fpr")) <= EXACT_TOL,
                    cat + ": realised FPR disagrees between threshold_check and calibration_compare");
            require(Math.abs(number(check, "recall_on_test") - number(row, "tol_recall")) <= EXACT_TOL,
                    cat + ": recall disagrees between threshold_check and calibration_compare");
            agreed++;
        }
        System.out.printf("  %d categories: one threshold and one realised rate across "
                + "calibration_compare, models/ and threshold_check%n", agreed);

        // ---- the two summary rows that still have an artefact --------------
        String[][] rules = {
            {"q99", "5-fold cross-calibration \\+ 99th percentile"},
            {"tol", "\\*\\*5-fold \\+ tolerance bound \\(shipped\\)\\*\\*"},
        };
        for (String[] rule : rules) {
            List<Double> fpr = new ArrayList<>(), recall = new ArrayList<>();
            int within = 0;
            for (String row : calib) {
                double f = number(row, rule[0] + "_fpr");
                fpr.add(f);
                recall.add(number(row, rule[0] + "_recall"));
                if (f <= 0.01) within++;
            }
            Pattern p = Pattern.compile("\\| " + rule[1]
                    + " \\| \\*{0,2}(\\d+) / (\\d+)\\*{0,2} \\| \\*{0,2}([0-9.]+)%\\*{0,2}"
                    + " \\| \\*{0,2}([0-9.]+)%\\*{0,2} \\|");
            Matcher m = p.matcher(methods);
            if (!m.find()) {
                require(false, "notes/METHODS.md has no summary row for " + rule[0]);
                continue;
            }
            double pubFpr = Double.parseDouble(m.group(3));
            double pubRecall = Double.parseDouble(m.group(4));
            boolean ok = Integer.parseInt(m.group(1)) == within
                    && Integer.parseInt(m.group(2)) == calib.size()
                    && Math.abs(pubFpr - 100 * mean(fpr)) <= DISPLAY_TOL
                    && Math.abs(pubRecall - 100 * mean(recall)) <= DISPLAY_TOL;
            require(ok, "notes/METHODS.md publishes " + m.group(1) + "/" + m.group(2) + ", "
                    + m.group(3) + "%, " + m.group(4) + "% for " + rule[0] + "; the run gives "
                    + within + "/" + calib.size() + ", "
                    + String.format("%.4f%%, %.4f%%", 100 * mean(fpr), 100 * mean(recall)));
            System.out.printf("  %-4s rule: %2d of %d within the 1%% target, mean FPR %.4f%%, "
                    + "mean recall %.4f%%   published %s/%s, %s%%, %s%%  %s%n",
                    rule[0], within, calib.size(), 100 * mean(fpr), 100 * mean(recall),
                    m.group(1), m.group(2), m.group(3), m.group(4), ok ? "ok" : "FAIL");
        }

        // ---- the per category table ----------------------------------------
        int rows = 0;
        for (String row : calib) {
            String cat = string(row, "cat");
            Matcher m = Pattern.compile("\\| " + cat + " \\| (\\d+) \\| \\*{0,2}([0-9.]+)%\\*{0,2}"
                    + " \\| \\*{0,2}([0-9.]+)%\\*{0,2} \\|").matcher(methods);
            if (!m.find()) {
                require(false, "notes/METHODS.md has no calibration row for " + cat);
                continue;
            }
            rows++;
            require(Integer.parseInt(m.group(1)) == (int) (double) number(row, "n"),
                    cat + ": notes say " + m.group(1) + " calibration images, the run used "
                            + number(row, "n"));
            require(Math.abs(Double.parseDouble(m.group(2)) - 100 * number(row, "tol_fpr")) <= DISPLAY_TOL,
                    cat + ": notes say FPR " + m.group(2) + "%, the run gives "
                            + String.format("%.4f%%", 100 * number(row, "tol_fpr")));
            require(Math.abs(Double.parseDouble(m.group(3)) - 100 * number(row, "tol_recall")) <= DISPLAY_TOL,
                    cat + ": notes say recall " + m.group(3) + "%, the run gives "
                            + String.format("%.4f%%", 100 * number(row, "tol_recall")));
        }
        System.out.printf("  %d per category rows in notes/METHODS.md match "
                + "reports/calibration_compare.json%n", rows);

        // ---- what it cost ---------------------------------------------------
        Matcher cost = Pattern.compile("Mean recall dropped from ([0-9.]+)% to ([0-9.]+)%")
                .matcher(methods);
        if (cost.find()) {
            List<Double> q99 = new ArrayList<>(), tol = new ArrayList<>();
            for (String row : calib) {
                q99.add(number(row, "q99_recall"));
                tol.add(number(row, "tol_recall"));
            }
            boolean ok = Math.abs(Double.parseDouble(cost.group(1)) - 100 * mean(q99)) <= DISPLAY_TOL
                    && Math.abs(Double.parseDouble(cost.group(2)) - 100 * mean(tol)) <= DISPLAY_TOL;
            require(ok, "notes/METHODS.md says recall dropped from " + cost.group(1) + "% to "
                    + cost.group(2) + "%, the run gives "
                    + String.format("%.4f%% to %.4f%%", 100 * mean(q99), 100 * mean(tol)));
            System.out.printf("  the cost sentence: %.4f%% to %.4f%%, published %s%% to %s%%  %s%n",
                    100 * mean(q99), 100 * mean(tol), cost.group(1), cost.group(2),
                    ok ? "ok" : "FAIL");
        } else {
            require(false, "notes/METHODS.md no longer states what the tolerance bound cost");
        }

        if (!FAILURES.isEmpty()) {
            System.out.println("\n" + FAILURES.size() + " failed:");
            for (String f : FAILURES) System.out.println("  " + f);
            System.exit(1);
        }
        System.out.println("\nJava reproduces the calibration tables in notes/METHODS.md "
                + "from reports/calibration_compare.json");
    }
}
