// Structural validation of every committed result file, plus the one
// recomputation that only the raw dataset index can settle.
//
// The per-category numbers in reports/ were written by the scripts that
// produced them, and the dataset index in data/_mvtec_index.json was written
// by the download step months earlier. Nothing had ever compared the two. If a
// category were scored on the wrong split, or a run file were regenerated for
// some categories and not others, the only trace would be a count that no
// longer matches the index. So this walks the index, counts the test and train
// images per category itself, and requires every published count to agree.
//
// The rest is the boring half: valid JSON, no NaN or Inf, rates inside [0, 1],
// list lengths that match their own headers, every referenced file present.
//
// Usage: go run . -root /path/to/repo
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

var categories = []string{
	"bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
	"metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood",
	"zipper",
}

// Field names whose value is a rate, a proportion or an area under a curve.
// Anything matching has to sit inside [0, 1].
var unitInterval = []string{
	"auroc", "aupro", "precision", "recall", "_f1", "fpr", "peak_in_mask",
	"accuracy", "fraction", "frac", "confidence",
}

type checker struct {
	root   string
	fails  []string
	checks int
}

func (c *checker) ok(cond bool, format string, a ...any) bool {
	c.checks++
	if !cond {
		c.fails = append(c.fails, fmt.Sprintf(format, a...))
	}
	return cond
}

func (c *checker) load(rel string) map[string]any {
	var v map[string]any
	c.loadInto(rel, &v)
	return v
}

func (c *checker) loadInto(rel string, out any) {
	b, err := os.ReadFile(filepath.Join(c.root, rel))
	if !c.ok(err == nil, "%s: %v", rel, err) {
		return
	}
	err = json.Unmarshal(b, out)
	c.ok(err == nil, "%s: not valid JSON: %v", rel, err)
}

// walk every number in a decoded document, rejecting NaN, Inf and rates that
// have escaped [0, 1].
func (c *checker) numbers(rel, path string, v any) {
	switch t := v.(type) {
	case map[string]any:
		keys := make([]string, 0, len(t))
		for k := range t {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			c.numbers(rel, path+"."+k, t[k])
		}
	case []any:
		for i, e := range t {
			c.numbers(rel, fmt.Sprintf("%s[%d]", path, i), e)
		}
	case float64:
		if math.IsNaN(t) || math.IsInf(t, 0) {
			c.ok(false, "%s: %s is %v", rel, path, t)
			return
		}
		leaf := path[strings.LastIndex(path, ".")+1:]
		for _, u := range unitInterval {
			if strings.Contains(leaf, u) {
				// 1e-9 of slack: a few of these are sums of floats that land a
				// couple of ulps past 1.0, which is arithmetic, not a bad value.
				c.ok(t >= -1e-9 && t <= 1+1e-9, "%s: %s = %v is outside [0,1]", rel, path, t)
				break
			}
		}
	}
}

// Some reports are a top level array. Only the ones that are an object can
// carry a "scores" list, so peek before decoding into the object shape.
func isObject(c *checker, rel string) bool {
	b, err := os.ReadFile(filepath.Join(c.root, rel))
	if err != nil {
		return false
	}
	return len(strings.TrimSpace(string(b))) > 0 && strings.TrimSpace(string(b))[0] == '{'
}

func num(m map[string]any, k string) (float64, bool) {
	v, ok := m[k].(float64)
	return v, ok
}

func main() {
	root := flag.String("root", ".", "repository root")
	flag.Parse()
	c := &checker{root: *root}

	// ---- every committed JSON parses, and carries no NaN, Inf or stray rate.
	var jsonFiles []string
	for _, pat := range []string{"reports/*.json", "models/*.json", "assets/*.json", "data/*.json"} {
		hits, _ := filepath.Glob(filepath.Join(*root, pat))
		for _, h := range hits {
			rel, _ := filepath.Rel(*root, h)
			jsonFiles = append(jsonFiles, rel)
		}
	}
	sort.Strings(jsonFiles)
	c.ok(len(jsonFiles) >= 40, "expected at least 40 committed JSON files, found %d", len(jsonFiles))
	for _, rel := range jsonFiles {
		var v any
		c.loadInto(rel, &v)
		c.numbers(rel, strings.TrimSuffix(filepath.Base(rel), ".json"), v)
	}
	fmt.Printf("  %3d committed JSON files parse, no NaN or Inf, every rate inside [0,1]\n", len(jsonFiles))

	// ---- the recomputation: counts straight off the dataset index.
	var index struct {
		Samples []struct {
			Filepath string `json:"filepath"`
			Category struct {
				Label string `json:"label"`
			} `json:"category"`
			Defect struct {
				Label string `json:"label"`
			} `json:"defect"`
			Split string `json:"split"`
		} `json:"samples"`
	}
	c.loadInto("data/_mvtec_index.json", &index)
	type counts struct{ train, test, testAnom int }
	byCat := map[string]*counts{}
	seenPath := map[string]bool{}
	for _, s := range index.Samples {
		c.ok(!seenPath[s.Filepath], "index: duplicate filepath %s", s.Filepath)
		seenPath[s.Filepath] = true
		if byCat[s.Category.Label] == nil {
			byCat[s.Category.Label] = &counts{}
		}
		k := byCat[s.Category.Label]
		switch s.Split {
		case "train":
			k.train++
			c.ok(s.Defect.Label == "good", "index: %s is a train image labelled %q", s.Filepath, s.Defect.Label)
		case "test":
			k.test++
			if s.Defect.Label != "good" {
				k.testAnom++
			}
		default:
			c.ok(false, "index: %s has split %q", s.Filepath, s.Split)
		}
	}
	c.ok(len(byCat) == 15, "index holds %d categories, expected 15", len(byCat))
	fmt.Printf("  %4d index samples over %d categories, no duplicate path, train split is normal only\n",
		len(index.Samples), len(byCat))

	calib := map[string]float64{}
	var cc []map[string]any
	c.loadInto("reports/calibration_compare.json", &cc)
	for _, r := range cc {
		if cat, ok := r["cat"].(string); ok {
			calib[cat], _ = num(r, "n")
		}
	}

	mismatch := 0
	var totalPt int64
	for _, cat := range categories {
		k := byCat[cat]
		if !c.ok(k != nil, "index has no category %s", cat) {
			continue
		}
		b := c.load("reports/bench_" + cat + ".json")
		nTest, _ := num(b, "n_test")
		nAnom, _ := num(b, "n_anomalous")
		if !c.ok(int(nTest) == k.test, "bench_%s.json n_test %v, index says %d", cat, nTest, k.test) {
			mismatch++
		}
		if !c.ok(int(nAnom) == k.testAnom, "bench_%s.json n_anomalous %v, index says %d", cat, nAnom, k.testAnom) {
			mismatch++
		}
		c.ok(b["category"] == cat, "bench_%s.json says category %v", cat, b["category"])

		m := c.load("models/" + cat + ".json")
		nTrain, _ := num(m, "n_train_total")
		if !c.ok(int(nTrain) == k.train, "models/%s.json n_train_total %v, index says %d", cat, nTrain, k.train) {
			mismatch++
		}
		nCalib, _ := num(m, "n_calib")
		c.ok(nCalib == nTrain, "models/%s.json n_calib %v != n_train_total %v", cat, nCalib, nTrain)
		c.ok(calib[cat] == nTrain, "calibration_compare %s n %v != index train count %d", cat, calib[cat], k.train)

		// the bank is a fixed fraction of one patch grid per training image
		frac, _ := num(m, "coreset_frac")
		grid, _ := m["grid"].([]any)
		if c.ok(len(grid) == 2, "models/%s.json grid is %v", cat, m["grid"]) {
			g0, _ := grid[0].(float64)
			g1, _ := grid[1].(float64)
			want := math.Floor(nTrain * g0 * g1 * frac)
			got, _ := num(m, "bank_size")
			c.ok(got == want, "models/%s.json bank_size %v, %v images x %vx%v patches x %v is %v",
				cat, got, nTrain, g0, g1, frac, want)
		}

		// the threshold is the largest calibration score, and the guarantee is
		// met exactly when there were enough calibration images for it
		thr, _ := num(m, "threshold")
		cmax, _ := num(m, "calib_score_max")
		cmin, _ := num(m, "calib_score_min")
		c.ok(thr == cmax, "models/%s.json threshold %v != calib_score_max %v", cat, thr, cmax)
		c.ok(cmin < cmax, "models/%s.json calib_score_min %v is not below max %v", cat, cmin, cmax)
		req, _ := num(m, "n_required_for_guarantee")
		met, _ := m["guarantee_met"].(bool)
		c.ok(met == (nCalib >= req), "models/%s.json guarantee_met %v with n_calib %v and required %v",
			cat, met, nCalib, req)

		// the exported weights are the size the metadata claims
		fi, err := os.Stat(filepath.Join(*root, "models", cat+".pt"))
		if c.ok(err == nil, "models/%s.pt: %v", cat, err) {
			totalPt += fi.Size()
			mb, _ := num(m, "artefact_mb")
			c.ok(math.Abs(float64(fi.Size())/1e6-mb) < 0.01,
				"models/%s.pt is %d bytes, artefact_mb says %v", cat, fi.Size(), mb)
		}
	}
	c.ok(mismatch == 0, "%d published counts disagree with the dataset index", mismatch)
	fmt.Printf("  %3d test and train counts in reports/ and models/ recomputed from the index, all agree\n",
		len(categories)*4)
	fmt.Printf("  %d exported banks, %.1f MB of .pt files, every artefact_mb within 0.01 MB of its file\n",
		len(categories), float64(totalPt)/1e6)

	// ---- score lists agree with their own headers.
	scoreFiles, _ := filepath.Glob(filepath.Join(*root, "reports", "*.json"))
	sort.Strings(scoreFiles)
	withScores := 0
	for _, f := range scoreFiles {
		rel, _ := filepath.Rel(*root, f)
		if !isObject(c, rel) {
			continue // threshold_check.json and friends are top level arrays
		}
		var d struct {
			Metrics map[string]any `json:"metrics"`
			Scores  []struct {
				Path  string  `json:"path"`
				Label int     `json:"label"`
				Score float64 `json:"score"`
			} `json:"scores"`
		}
		c.loadInto(rel, &d)
		if len(d.Scores) == 0 {
			continue
		}
		withScores++
		pos, seen := 0, map[string]bool{}
		for _, s := range d.Scores {
			c.ok(s.Label == 0 || s.Label == 1, "%s: label %d", rel, s.Label)
			c.ok(!seen[s.Path], "%s: duplicate image %s", rel, s.Path)
			seen[s.Path] = true
			c.ok(s.Score > 0, "%s: %s has score %v", rel, s.Path, s.Score)
			pos += s.Label
		}
		nn, _ := num(d.Metrics, "n_normal")
		na, _ := num(d.Metrics, "n_anomalous")
		c.ok(int(na) == pos, "%s: n_anomalous %v, list holds %d", rel, na, pos)
		c.ok(int(nn)+int(na) == len(d.Scores), "%s: %v + %v != %d rows", rel, nn, na, len(d.Scores))
	}
	c.ok(withScores == 12, "expected 12 files with a scores list, found %d", withScores)
	fmt.Printf("  %3d score lists: labels binary, no repeated image, lengths match their own header\n", withScores)

	// ---- the demo manifest points at files that exist and flags consistently.
	var samples []struct {
		Category  string  `json:"category"`
		Threshold float64 `json:"threshold"`
		NDefects  int     `json:"n_defects"`
		NMissed   int     `json:"n_missed"`
		Recall    float64 `json:"recall_at_threshold"`
		Samples   []struct {
			Name    string  `json:"name"`
			File    string  `json:"file"`
			Score   float64 `json:"score"`
			Flagged bool    `json:"flagged"`
			Defect  string  `json:"defect"`
		} `json:"samples"`
	}
	c.loadInto("assets/samples.json", &samples)
	c.ok(len(samples) == 15, "assets/samples.json has %d categories", len(samples))
	nFiles := 0
	for _, s := range samples {
		c.ok(s.NMissed == int(math.Round(float64(s.NDefects)*(1-s.Recall))),
			"%s: %d defects at recall %v does not give %d missed", s.Category, s.NDefects, s.Recall, s.NMissed)
		for _, im := range s.Samples {
			nFiles++
			_, err := os.Stat(filepath.Join(*root, "assets", "samples", im.File))
			c.ok(err == nil, "assets/samples/%s: %v", im.File, err)
			c.ok(strings.HasPrefix(im.File, s.Category+"__"), "%s is filed under %s", im.File, s.Category)
			// the score is rounded to 3 decimals in the manifest, so allow the
			// rounding when it lands on the threshold
			if math.Abs(im.Score-s.Threshold) > 5e-4 {
				c.ok(im.Flagged == (im.Score >= s.Threshold),
					"%s: score %v against threshold %v but flagged=%v", im.File, im.Score, s.Threshold, im.Flagged)
			}
			if strings.Contains(im.Name, "MISSED") {
				c.ok(!im.Flagged && im.Defect != "good", "%s is named MISSED but flagged=%v defect=%q",
					im.File, im.Flagged, im.Defect)
			}
			if im.Defect == "good" {
				c.ok(!im.Flagged, "%s is a good image and was flagged", im.File)
			}
		}
	}
	fmt.Printf("  %3d demo images referenced by assets/samples.json, all present and consistently flagged\n", nFiles)

	// ---- every image a markdown file points at is committed.
	link := regexp.MustCompile(`!\[[^\]]*\]\(([^)]+)\)`)
	docs := []string{"README.md", "notes/METHODS.md", "reports/results.md", "reports/benchmark.md", "spaces/README.md"}
	nLinks := 0
	for _, doc := range docs {
		b, err := os.ReadFile(filepath.Join(*root, doc))
		if err != nil {
			continue
		}
		for _, mm := range link.FindAllStringSubmatch(string(b), -1) {
			target := mm[1]
			if strings.HasPrefix(target, "http") {
				continue
			}
			nLinks++
			p := filepath.Join(filepath.Dir(filepath.Join(*root, doc)), target)
			_, err := os.Stat(p)
			c.ok(err == nil, "%s points at %s which is missing", doc, target)
		}
	}
	fmt.Printf("  %3d images linked from the markdown, all committed\n", nLinks)

	// ---- README says how big the exported models are.
	readme, err := os.ReadFile(filepath.Join(*root, "README.md"))
	if c.ok(err == nil, "README.md: %v", err) {
		re := regexp.MustCompile(`committed under ` + "`models/`" + ` \((\d+) MB total\)`)
		mm := re.FindSubmatch(readme)
		if c.ok(mm != nil, "README.md does not state the size of models/") {
			claim, _ := strconv.Atoi(string(mm[1]))
			c.ok(claim == int(math.Round(float64(totalPt)/1e6)),
				"README.md claims %d MB of models, the files are %.2f MB", claim, float64(totalPt)/1e6)
		}
	}

	fmt.Printf("\nGo ran %d checks over %d JSON files\n", c.checks, len(jsonFiles))
	if len(c.fails) > 0 {
		fmt.Printf("%d failed:\n", len(c.fails))
		for i, f := range c.fails {
			if i == 20 {
				fmt.Printf("  ... and %d more\n", len(c.fails)-20)
				break
			}
			fmt.Println("  " + f)
		}
		os.Exit(1)
	}
	fmt.Println("all passed")
}
