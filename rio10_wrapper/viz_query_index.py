# New file, part of the RIO10 wrapper. Does not modify any existing mast3r/dust3r file.
#
# Loads and joins the already-completed oracle run's per-query results
# (results/oracle/{scene}_raw_errors.csv) with the texture/change failure-analysis features
# (analysis/texture_change_features.csv) into one per-query table, and supports filtering/sorting
# so the viz tool's query picker can jump straight to e.g. "high-change failures in scene04"
# instead of scrubbing through thousands of frames one by one. Pure CPU, no GPU, no model.
import csv
import os


def load_raw_errors(output_dir, scene):
    """ {image_name: (transl_err_m, angular_err_deg)}, or {} if this scene hasn't been bulk-run
    into output_dir yet (e.g. retrieval mode on a scene with no matching run). """
    path = os.path.join(output_dir, f'{scene}_raw_errors.csv')
    if not os.path.isfile(path):
        return {}
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            out[row['image_name']] = (float(row['transl_err_m']), float(row['angular_err_deg']))
    return out


def load_texture_change_features(csv_path):
    """ {(scene, query_image_name): (texture_lapvar, change_frac)}. Today's committed file has
    full coverage (34,415 rows, one per oracle query across all 10 scenes), but the underlying
    extraction script still supports per-scene sampling for future runs, so callers must tolerate
    a missing key rather than assume every query has features. """
    if not os.path.isfile(csv_path):
        return {}
    out = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            out[(row['scene'], row['query'])] = (float(row['texture_lapvar']), float(row['change_frac']))
    return out


def build_query_table(scene, all_query_ids, output_dir, texture_change):
    """ One row per query id: image_name, transl_err_m, angular_err_deg, success,
    texture_lapvar (float or None), change_frac (float or None). """
    errors = load_raw_errors(output_dir, scene)
    rows = []
    for qid in all_query_ids:
        t_err, a_err = errors.get(qid, (float('inf'), float('inf')))
        tex, chg = texture_change.get((scene, qid), (None, None))
        rows.append({
            'image_name': qid,
            'transl_err_m': t_err,
            'angular_err_deg': a_err,
            'success': t_err == t_err and t_err != float('inf'),  # nan-safe isfinite
            'texture_lapvar': tex,
            'change_frac': chg,
        })
    return rows


def filter_and_sort(table, success_filter=None, min_transl_err=None, min_change_frac=None,
                    min_texture=None, sort_by='transl_err_m', descending=True, limit=200):
    """
    success_filter: None (no filter), 'success', or 'failed'.
    min_* filters keep only rows with that field >= the threshold (None-valued fields are
    excluded by a non-None min_* filter, since there's nothing to compare).
    sort_by: one of the row keys in build_query_table's output.
    """
    rows = table
    if success_filter == 'success':
        rows = [r for r in rows if r['success']]
    elif success_filter == 'failed':
        rows = [r for r in rows if not r['success']]
    if min_transl_err is not None:
        rows = [r for r in rows if r['transl_err_m'] != float('inf') and r['transl_err_m'] >= min_transl_err]
    if min_change_frac is not None:
        rows = [r for r in rows if r['change_frac'] is not None and r['change_frac'] >= min_change_frac]
    if min_texture is not None:
        rows = [r for r in rows if r['texture_lapvar'] is not None and r['texture_lapvar'] >= min_texture]

    def sort_key(r):
        v = r.get(sort_by)
        if v is None:
            return float('-inf') if descending else float('inf')
        return v

    rows = sorted(rows, key=sort_key, reverse=descending)
    return rows[:limit] if limit else rows
