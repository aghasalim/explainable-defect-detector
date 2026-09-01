-- Rebuild the numbers README.md and notes/METHODS.md state in prose, in SQL.
--
-- The headline of this repository is a mean over 15 per category run files, and
-- the claims around it are minima, maxima and single cells picked out of the
-- same 15 files. Every one of them was computed once, by numpy, inside the
-- script that also wrote the run files, and then typed into the prose by hand.
-- Nothing has ever read the run files back and checked that the prose still
-- describes them. That is exactly how the README came to claim a control of
-- 0.01 for a category whose control is 0.00.
--
-- So this loads reports/bench_*.json and the four screw ablation runs, rebuilds
-- each published line, and looks it up in the document that publishes it. The
-- documents are flattened to single spaces first, because the sentences wrap
-- across lines and a line break is not a disagreement.
--
-- reports/benchmark.md and reports/results.md are rebuilt by verify/tables.mjs
-- instead, which is why they are not repeated here.
--
-- Run from the repository root:
--   sqlite3 -init verify/aggregate.sql :memory: ""

.mode list
.headers off

CREATE TEMP TABLE cats(c TEXT);
INSERT INTO cats(c) VALUES
    ('bottle'), ('cable'), ('capsule'), ('carpet'), ('grid'), ('hazelnut'),
    ('leather'), ('metal_nut'), ('pill'), ('screw'), ('tile'), ('toothbrush'),
    ('transistor'), ('wood'), ('zipper');

-- readfile() returns a blob, so cast before handing it to the JSON functions.
CREATE TEMP TABLE bench AS
SELECT c AS category,
       json_extract(b, '$.image_auroc')          AS image_auroc,
       json_extract(b, '$.pixel_auroc')          AS pixel_auroc,
       json_extract(b, '$.paper_image_auroc')    AS paper_image_auroc,
       json_extract(b, '$.paper_pixel_auroc')    AS paper_pixel_auroc,
       json_extract(b, '$.aupro')                AS aupro,
       json_extract(b, '$.peak_in_mask')         AS peak_in_mask,
       json_extract(b, '$.control_peak_in_mask') AS control_peak_in_mask
FROM (SELECT c, CAST(readfile('reports/bench_' || c || '.json') AS TEXT) AS b
      FROM cats);

CREATE TEMP TABLE ablation AS
SELECT run, json_extract(b, '$.metrics.image_auroc') AS image_auroc
FROM (SELECT run, CAST(readfile('reports/' || run || '_screw.json') AS TEXT) AS b
      FROM (SELECT 'patchcore' AS run UNION ALL SELECT 'patchcore-random'
            UNION ALL SELECT 'patchcore-0.1' UNION ALL SELECT 'patchcore-224crop'));

CREATE TEMP TABLE docs(name TEXT, flat TEXT);
INSERT INTO docs
SELECT 'README.md',
       replace(replace(CAST(readfile('README.md') AS TEXT), char(13), ' '), char(10), ' ');
INSERT INTO docs
SELECT 'notes/METHODS.md',
       replace(replace(CAST(readfile('notes/METHODS.md') AS TEXT), char(13), ' '), char(10), ' ');

CREATE TEMP TABLE want(doc TEXT, label TEXT, line TEXT);

-- A missing file makes json_extract return NULL rather than failing, so refuse
-- to average over holes.
INSERT INTO want SELECT 'README.md', 'files read',
       CASE WHEN count(*) = 15 AND count(image_auroc) = 15 AND count(peak_in_mask) = 15
            THEN 'Explainable Visual Defect Detector'
            ELSE 'only ' || count(image_auroc) || ' of 15 bench files are readable' END
FROM bench;

-- ---- the headline -------------------------------------------------------
INSERT INTO want SELECT 'README.md', 'mean image AUROC',
       'Mean image AUROC **' || printf('%.4f', avg(image_auroc))
       || '** over all 15 MVTec AD categories.' FROM bench;
INSERT INTO want SELECT 'README.md', 'paper mean',
       'The PatchCore paper reports ' || printf('%.3f', avg(paper_image_auroc)) || '.'
FROM bench;

-- ---- the weakest localisation, and its control --------------------------
INSERT INTO want SELECT 'README.md', 'worst localisation',
       'worst case `' || category || '` at ' || printf('%.2f', peak_in_mask)
       || ' against ' || printf('%.2f', control_peak_in_mask) || '.'
FROM bench ORDER BY peak_in_mask LIMIT 1;
INSERT INTO want SELECT 'notes/METHODS.md', 'worst localisation',
       'worst case `' || category || '` at ' || printf('%.2f', peak_in_mask)
       || ' peak-in-mask against a control of ' || printf('%.2f', control_peak_in_mask) || '.'
FROM bench ORDER BY peak_in_mask LIMIT 1;

-- ---- detection and localisation are different problems ------------------
INSERT INTO want SELECT 'README.md', 'toothbrush split',
       '`toothbrush` scores a perfect ' || printf('%.4f', image_auroc)
       || ' image AUROC, but its heatmap points at the actual defect only '
       || CAST(round(100 * peak_in_mask) AS INT) || '% of the time.'
FROM bench WHERE category = 'toothbrush';
INSERT INTO want SELECT 'notes/METHODS.md', 'screw split',
       '`screw` detects at ' || printf('%.3f', image_auroc) || ' and locates at '
       || printf('%.3f', peak_in_mask) || '.'
FROM bench WHERE category = 'screw';

-- ---- the section 3 table, which nothing else rebuilds -------------------
INSERT INTO want SELECT 'notes/METHODS.md', 'row ' || category,
       '| ' || category || ' | ' || printf('%.4f', image_auroc) || ' | '
       || printf('%.3f', paper_image_auroc) || ' | ' || printf('%.4f', pixel_auroc) || ' | '
       || printf('%.3f', paper_pixel_auroc) || ' | ' || printf('%.4f', aupro) || ' | '
       || printf('%.4f', peak_in_mask) || ' |'
FROM bench ORDER BY category;
INSERT INTO want SELECT 'notes/METHODS.md', 'mean row',
       '| **mean** | **' || printf('%.4f', avg(image_auroc)) || '** | '
       || printf('%.3f', avg(paper_image_auroc)) || ' | **' || printf('%.4f', avg(pixel_auroc))
       || '** | ' || printf('%.3f', avg(paper_pixel_auroc)) || ' | **'
       || printf('%.4f', avg(aupro)) || '** | **' || printf('%.4f', avg(peak_in_mask)) || '** |'
FROM bench;

-- ---- the ablations, read back out of the runs they describe -------------
INSERT INTO want SELECT 'notes/METHODS.md', 'coreset ablation',
       'random sampling scores '
       || printf('%.4f', (SELECT image_auroc FROM ablation WHERE run = 'patchcore-random'))
       || ' and greedy k-center scores '
       || printf('%.4f', (SELECT image_auroc FROM ablation WHERE run = 'patchcore')) || '.';
INSERT INTO want SELECT 'notes/METHODS.md', 'bank size ablation',
       'from 1% to 10% took `screw` from '
       || printf('%.4f', (SELECT image_auroc FROM ablation WHERE run = 'patchcore'))
       || ' to '
       || printf('%.4f', (SELECT image_auroc FROM ablation WHERE run = 'patchcore-0.1'))
       || ' and cost 8x the compute.';
INSERT INTO want SELECT 'notes/METHODS.md', 'crop ablation',
       'centre crop got '
       || printf('%.4f', (SELECT image_auroc FROM ablation WHERE run = 'patchcore-224crop'))
       || ' at 1%';

-- A failing line is printed in full, since the disagreement is usually in the
-- part a truncated line would hide.
SELECT CASE WHEN instr((SELECT flat FROM docs WHERE name = want.doc), line) > 0
            THEN 'ok   ' || printf('%-18s', doc) || printf('%-20s', label)
                 || substr(line, 1, 60)
            ELSE 'FAIL ' || printf('%-18s', doc) || printf('%-20s', label) || line END
FROM want;

SELECT 'rebuilt ' || count(*) || ' published lines from 15 bench files and 4 screw runs'
       || ' (sqlite ' || sqlite_version() || ', '
       || (SELECT group_concat(name || ' ' || length(flat) || ' chars', ', ') FROM docs)
       || ')' FROM want;
