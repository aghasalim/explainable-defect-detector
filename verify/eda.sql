-- Rebuild reports/eda_bottle.md from the 292 per image rows behind it.
--
-- The EDA report is the one place in the repository where a published summary
-- has a genuinely rawer file sitting next to it: reports/eda_bottle.json holds
-- one record per image, with its split, its defect class, its mean grey level,
-- its sha1 and the fraction of its pixels the ground truth mask marks. Every
-- table and every sentence with a number in reports/eda_bottle.md is an
-- aggregation of those rows, computed once, in numpy, inside src/edd/eda.py.
--
-- This recomputes all of them in SQL: the class balance, the quantiles of the
-- defect area with numpy's own linear interpolation, the per class medians, the
-- duplicate check and the grey level means and standard deviations. Each result
-- is assembled into the exact line the markdown should contain and looked up in
-- the published file, so a disagreement in any digit shows up as a line that is
-- not there.
--
-- Run from the repository root:
--   sqlite3 -init verify/eda.sql :memory: ""

.mode list
.headers off

CREATE TEMP TABLE img AS
SELECT json_extract(value, '$.split')       AS split,
       json_extract(value, '$.defect')      AS defect,
       json_extract(value, '$.label')       AS label,
       json_extract(value, '$.w')           AS w,
       json_extract(value, '$.h')           AS h,
       json_extract(value, '$.mode')        AS mode,
       json_extract(value, '$.mean')        AS grey,
       json_extract(value, '$.sha')         AS sha,
       json_extract(value, '$.defect_frac') AS frac
FROM json_each(CAST(readfile('reports/eda_bottle.json') AS TEXT));

CREATE TEMP TABLE md AS
SELECT CAST(readfile('reports/eda_bottle.md') AS TEXT) AS doc;

-- Every masked image twice: once in the overall group, once in its own class.
CREATE TEMP TABLE ordered AS
SELECT grp, v,
       row_number() OVER (PARTITION BY grp ORDER BY v) - 1 AS i,
       count(*)     OVER (PARTITION BY grp)                AS c
FROM (SELECT '*' AS grp, frac AS v FROM img WHERE frac IS NOT NULL
      UNION ALL
      SELECT defect, frac FROM img WHERE frac IS NOT NULL);

CREATE TEMP TABLE probs(name TEXT, q REAL);
INSERT INTO probs VALUES ('p25', 0.25), ('median', 0.5), ('p75', 0.75);

-- numpy's default: interpolate linearly between the two order statistics that
-- straddle q*(n-1). Taking the nearest one instead moves p25 by a whole image.
CREATE TEMP TABLE quant AS
SELECT t.grp, t.name, lo.v + (t.p - CAST(t.p AS INT)) * (hi.v - lo.v) AS v
FROM (SELECT g.grp, probs.name, probs.q * (g.c - 1) AS p
      FROM (SELECT grp, max(c) AS c FROM ordered GROUP BY grp) g, probs) t
JOIN ordered lo ON lo.grp = t.grp AND lo.i = CAST(t.p AS INT)
JOIN ordered hi ON hi.grp = t.grp
                AND hi.i = CAST(t.p AS INT) + (t.p > CAST(t.p AS INT));

CREATE TEMP TABLE want(label TEXT, line TEXT);

INSERT INTO want SELECT 'image count', count(*) || ' images.' FROM img;

INSERT INTO want SELECT 'balance ' || split || '/' || defect,
       '| ' || split || ' | ' || defect || ' | ' || count(*) || ' |'
FROM img GROUP BY split, defect ORDER BY split, defect;

INSERT INTO want SELECT 'defects in train',
       '**Defective images in `train`: ' || count(*) || '.**'
FROM img WHERE split = 'train' AND label = 1;

INSERT INTO want SELECT 'test balance',
       '**Test set is ' || good || ' good / ' || bad || ' defective.** A model predicting '
       || 'the majority class scores '
       || printf('%.1f%%', 100.0 * max(good, bad) / (good + bad)) || ' accuracy'
FROM (SELECT sum(label = 0) AS good, sum(label = 1) AS bad FROM img WHERE split = 'test');

INSERT INTO want SELECT 'mask count', count(*) || ' masks found.'
FROM img WHERE frac IS NOT NULL;

INSERT INTO want SELECT 'defect area min',
       '| min | ' || printf('%.4f%%', 100 * min(v)) || ' |' FROM ordered WHERE grp = '*';
INSERT INTO want SELECT 'defect area ' || name,
       '| ' || name || ' | ' || printf('%.4f%%', 100 * v) || ' |' FROM quant WHERE grp = '*';
INSERT INTO want SELECT 'defect area max',
       '| max | ' || printf('%.4f%%', 100 * max(v)) || ' |' FROM ordered WHERE grp = '*';
INSERT INTO want SELECT 'defect area mean',
       '| mean | ' || printf('%.4f%%', 100 * avg(v)) || ' |' FROM ordered WHERE grp = '*';

INSERT INTO want SELECT 'median area ' || o.grp,
       '| ' || o.grp || ' | ' || max(o.c) || ' | ' || printf('%.4f%%', 100 * q.v) || ' |'
FROM ordered o JOIN quant q ON q.grp = o.grp AND q.name = 'median'
WHERE o.grp <> '*' GROUP BY o.grp, q.v ORDER BY o.grp;

INSERT INTO want SELECT 'downsampling sentence',
       'The median defect covers **' || printf('%.2f%%', 100 * med) || '** of the image. '
       || 'Predicting ''no defect'' for every pixel already yields ~'
       || printf('%.2f%%', 100 * (1 - mean)) || ' pixel accuracy, so pixel accuracy is '
       || 'useless - use pixel AUROC / PRO. It also bounds how far we can downsample: at '
       || '224x224 the smallest defect here occupies about '
       || printf('%.1f', smallest * 224 * 224) || ' pixels.'
FROM (SELECT (SELECT v FROM quant WHERE grp = '*' AND name = 'median') AS med,
             avg(v) AS mean, min(v) AS smallest FROM ordered WHERE grp = '*');

-- One row per distinct resolution, so a second resolution would produce a
-- second line and neither would be in the file.
INSERT INTO want SELECT 'resolution', '| size | [(' || w || ', ' || h || ')] |'
FROM img GROUP BY w, h;
INSERT INTO want SELECT 'channels', '| mode | [''' || mode || '''] |'
FROM img GROUP BY mode;

INSERT INTO want SELECT 'duplicate images',
       CASE WHEN count(*) = count(DISTINCT sha)
            THEN 'None. No train/test contamination via identical files.'
            ELSE '**' || (count(*) - count(DISTINCT sha)) || ' duplicated image(s):**' END
FROM img;

-- Population standard deviation, which is what numpy returns by default.
INSERT INTO want SELECT 'exposure confound',
       'Mean grey level - good ' || printf('%.1f', gm) || '+/-' || printf('%.1f', gs)
       || ', defective ' || printf('%.1f', bm) || '+/-' || printf('%.1f', bs)
       || ' (difference ' || printf('%.1f', abs(gm - bm)) || ').'
FROM (SELECT
        (SELECT avg(grey) FROM img WHERE label = 0) AS gm,
        (SELECT sqrt(avg(grey * grey) - avg(grey) * avg(grey)) FROM img WHERE label = 0) AS gs,
        (SELECT avg(grey) FROM img WHERE label = 1) AS bm,
        (SELECT sqrt(avg(grey * grey) - avg(grey) * avg(grey)) FROM img WHERE label = 1) AS bs);

-- A failing line is printed in full, since the disagreement is usually in the
-- part a truncated line would hide.
SELECT CASE WHEN instr((SELECT doc FROM md), line) > 0
            THEN 'ok   ' || printf('%-22s', label) || substr(line, 1, 78)
            ELSE 'FAIL ' || printf('%-22s', label) || line END
FROM want;

SELECT 'rebuilt ' || count(*) || ' lines of reports/eda_bottle.md from '
       || (SELECT count(*) FROM img) || ' per image rows'
       || ' (sqlite ' || sqlite_version() || ')' FROM want;
