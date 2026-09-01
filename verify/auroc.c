/* Recompute the image level detection metrics from the per image scores, in C.
 *
 * Every run file under reports/ that carries a "scores" list also carries a
 * "metrics" block that was produced from those same scores by scikit-learn,
 * inside the script that wrote the file. Nothing has ever recomputed the block
 * from the list. This does, with no library at all: ranks for AUROC, a
 * precision-recall sweep for average precision and best F1.
 *
 * The one convention borrowed from scikit-learn rather than reinvented is that
 * the precision-recall curve stops at the lowest threshold that still reaches
 * full recall. That affects which threshold is reported at the best F1, not
 * AUROC or average precision.
 *
 * Usage: auroc <file.json> [file.json ...]
 * Exits non-zero on the first file that disagrees past the tolerance.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TOL 1e-12

typedef struct { double score; int label; } Point;

static char *slurp(const char *path, size_t *len)
{
    FILE *f = fopen(path, "rb");
    if (!f)
        return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n < 0) { fclose(f); return NULL; }
    char *buf = malloc((size_t)n + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t got = fread(buf, 1, (size_t)n, f);
    fclose(f);
    buf[got] = '\0';
    *len = got;
    return buf;
}

/* Value text just after "<key>": , or NULL. Keys are matched with their quotes
 * and colon, so "score" never matches inside "scores". */
static const char *value_of(const char *from, const char *key)
{
    char pat[128];
    snprintf(pat, sizeof pat, "\"%s\":", key);
    const char *p = strstr(from, pat);
    return p ? p + strlen(pat) : NULL;
}

static int number_of(const char *from, const char *key, double *out)
{
    const char *v = value_of(from, key);
    if (!v)
        return 0;
    char *end;
    double d = strtod(v, &end);
    if (end == v)
        return 0;
    *out = d;
    return 1;
}

static int cmp_point(const void *a, const void *b)
{
    const double x = ((const Point *)a)->score, y = ((const Point *)b)->score;
    return x < y ? -1 : x > y ? 1 : 0;
}

/* Mann-Whitney U with mid-ranks for ties. */
static double auroc_of(const Point *asc, int n, int npos, int nneg)
{
    double rank_sum = 0.0;
    int i = 0;
    while (i < n) {
        int j = i;
        while (j + 1 < n && asc[j + 1].score == asc[i].score)
            j++;
        const double mid = (i + j) / 2.0 + 1.0;    /* ranks are 1 based */
        for (int k = i; k <= j; k++)
            if (asc[k].label)
                rank_sum += mid;
        i = j + 1;
    }
    return (rank_sum - (double)npos * (npos + 1) / 2.0) / ((double)npos * nneg);
}

static int check(const char *path)
{
    size_t len = 0;
    char *buf = slurp(path, &len);
    if (!buf || len == 0) { fprintf(stderr, "cannot read %s\n", path); free(buf); return 2; }

    const char *arr = value_of(buf, "scores");
    if (!arr) { fprintf(stderr, "%s has no scores list\n", path); free(buf); return 2; }

    int cap = 256, n = 0;
    Point *pts = malloc((size_t)cap * sizeof *pts);
    const char *p = arr;
    for (;;) {
        const char *l = strstr(p, "\"label\":");
        if (!l)
            break;
        const char *s = strstr(l, "\"score\":");
        if (!s)
            break;
        if (n == cap) {
            cap *= 2;
            pts = realloc(pts, (size_t)cap * sizeof *pts);
        }
        pts[n].label = (int)strtol(l + strlen("\"label\":"), NULL, 10);
        pts[n].score = strtod(s + strlen("\"score\":"), NULL);
        if (pts[n].label != 0 && pts[n].label != 1) {
            fprintf(stderr, "%s: label %d is not 0 or 1\n", path, pts[n].label);
            free(pts); free(buf); return 2;
        }
        n++;
        p = s + 1;
    }
    if (n < 2) { fprintf(stderr, "%s: %d scores\n", path, n); free(pts); free(buf); return 2; }

    int npos = 0;
    for (int i = 0; i < n; i++)
        npos += pts[i].label;
    const int nneg = n - npos;
    if (npos == 0 || nneg == 0) {
        fprintf(stderr, "%s: one class only\n", path);
        free(pts); free(buf); return 2;
    }

    qsort(pts, (size_t)n, sizeof *pts, cmp_point);

    const double auroc = auroc_of(pts, n, npos, nneg);

    /* Distinct thresholds ascending, with the counts at or above each. */
    double *thr = malloc((size_t)n * sizeof *thr);
    double *prec = malloc((size_t)(n + 1) * sizeof *prec);
    double *rec = malloc((size_t)(n + 1) * sizeof *rec);
    int m = 0, i = 0;
    while (i < n) {
        int j = i;
        while (j + 1 < n && pts[j + 1].score == pts[i].score)
            j++;
        int tp = 0;
        for (int k = i; k < n; k++)
            tp += pts[k].label;
        thr[m] = pts[i].score;
        prec[m] = (double)tp / (double)(n - i);
        rec[m] = (double)tp / (double)npos;
        m++;
        i = j + 1;
    }
    /* Drop the thresholds below the lowest one that still reaches full recall,
     * then close the curve at (precision 1, recall 0). */
    int start = 0;
    while (start + 1 < m && rec[start + 1] == 1.0)
        start++;
    const int mm = m - start;
    memmove(thr, thr + start, (size_t)mm * sizeof *thr);
    memmove(prec, prec + start, (size_t)mm * sizeof *prec);
    memmove(rec, rec + start, (size_t)mm * sizeof *rec);
    prec[mm] = 1.0;
    rec[mm] = 0.0;

    double ap = 0.0;
    for (int k = 0; k < mm; k++)
        ap -= (rec[k + 1] - rec[k]) * prec[k];

    int best = 0;
    double best_f1 = -1.0;
    for (int k = 0; k <= mm; k++) {
        const double d = prec[k] + rec[k] > 1e-12 ? prec[k] + rec[k] : 1e-12;
        const double f1 = 2.0 * prec[k] * rec[k] / d;
        if (f1 > best_f1) { best_f1 = f1; best = k; }
    }
    const double t = thr[best < mm ? best : mm - 1];

    int correct = 0;
    for (int k = 0; k < n; k++)
        correct += (pts[k].score >= t) == (pts[k].label == 1);
    const double acc = (double)correct / (double)n;
    const double pos_rate = (double)npos / (double)n;
    const double majority = pos_rate > 1.0 - pos_rate ? pos_rate : 1.0 - pos_rate;

    struct { const char *key; double got; } want[] = {
        { "image_auroc",             auroc },
        { "average_precision",       ap },
        { "best_f1",                 best_f1 },
        { "precision_at_best_f1",    prec[best] },
        { "recall_at_best_f1",       rec[best] },
        { "threshold",               t },
        { "accuracy_at_best_f1",     acc },
        { "majority_class_accuracy", majority },
        { "n_normal",                (double)nneg },
        { "n_anomalous",             (double)npos },
    };

    int bad = 0, checked = 0;
    double worst = 0.0;
    const char *worst_key = "";
    for (size_t k = 0; k < sizeof want / sizeof want[0]; k++) {
        double published;
        if (!number_of(buf, want[k].key, &published))
            continue;                     /* not every file publishes every field */
        const double d = fabs(published - want[k].got);
        if (d > worst) { worst = d; worst_key = want[k].key; }
        if (d > TOL) {
            printf("    %-24s published %.17g  recomputed %.17g  |d| %.2e  FAIL\n",
                   want[k].key, published, want[k].got, d);
            bad++;
        }
        checked++;
    }
    if (checked < 8) {
        fprintf(stderr, "%s: only %d published fields to compare\n", path, checked);
        bad++;
    }
    printf("  %-44s %4d scores  %2d fields  worst |d| %.1e (%s)  %s\n",
           path, n, checked, worst, *worst_key ? worst_key : "-", bad ? "FAIL" : "ok");

    free(thr); free(prec); free(rec); free(pts); free(buf);
    return bad ? 1 : 0;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s <file.json> [...]\n", argv[0]);
        return 2;
    }
    int failures = 0;
    for (int i = 1; i < argc; i++)
        failures += check(argv[i]) != 0;
    printf("\nC recomputed %d of %d score files with no disagreement above %.0e\n",
           argc - 1 - failures, argc - 1, TOL);
    return failures ? 1 : 0;
}
