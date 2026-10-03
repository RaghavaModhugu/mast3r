# New file, part of the RIO10 wrapper. Does not modify any existing mast3r/dust3r file.
#
# Pools every {scene}_raw_errors.csv produced by run_visloc_rio10.py for one experiment (all
# queries across all 10 scenes together, matching how the paper reports one aggregate RIO10
# number rather than an average of per-scene medians) and prints a table in the paper's own
# format, alongside the paper's own reported RIO10 rows (arxiv 2510.00978, Table 3) for a direct
# comparison.
import argparse
import csv
import glob
import os

import numpy as np

# Transcribed directly from the paper (arxiv 2510.00978, Table 3, RIO10 columns), for reference.
PAPER_RIO10_ROWS = [
    ('MASt3R+Kapture (paper, Seen)', None, None, 24.8, 32.6),
    ('MASt3R (paper, Unseen)', 0.17, 5.5, 45.1, 58.2),
    ('FastForward (paper, Unseen)', 0.18, 5.5, 40.6, 59.7),
]


def pool_raw_errors(output_dir, scenes):
    t_errs, r_errs = [], []
    for scene in scenes:
        path = os.path.join(output_dir, f'{scene}_raw_errors.csv')
        if not os.path.isfile(path):
            print(f'warning: missing {path}, skipping this scene in the pooled table')
            continue
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                t_errs.append(float(row['transl_err_m']))
                r_errs.append(float(row['angular_err_deg']))
    return np.asarray(t_errs), np.asarray(r_errs)


def paper_style_row(t_errs, r_errs):
    finite = np.isfinite(t_errs) & np.isfinite(r_errs)
    n = len(t_errs)
    e_t = float(np.median(t_errs)) if finite.any() else float('nan')
    e_r = float(np.median(r_errs)) if finite.any() else float('nan')
    acc10 = 100.0 * np.sum((t_errs <= 0.10) & (r_errs <= 10)) / n if n else 0.0
    acc20 = 100.0 * np.sum((t_errs <= 0.20) & (r_errs <= 20)) / n if n else 0.0
    return n, e_t, e_r, acc10, acc20


def print_table(rows):
    header = f'{"method":44s} {"n":>6s} {"e_t(m)":>8s} {"e_r(deg)":>9s} {"10cm,10°(%)":>12s} {"20cm,20°(%)":>12s}'
    print(header)
    print('-' * len(header))
    for label, n, e_t, e_r, acc10, acc20 in rows:
        print(f'{label:44s} {n:6d} {e_t:8.3f} {e_r:9.3f} {acc10:12.1f} {acc20:12.1f}')
    for name, pt, pr, p10, p20 in PAPER_RIO10_ROWS:
        pt_s = f'{pt:8.3f}' if pt is not None else f'{"N/A":>8s}'
        pr_s = f'{pr:9.3f}' if pr is not None else f'{"N/A":>9s}'
        print(f'{name:44s} {"-":>6s} {pt_s} {pr_s} {p10:12.1f} {p20:12.1f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', required=True, help='dir containing {scene}_raw_errors.csv files')
    parser.add_argument('--scenes', nargs='+', default=[f'scene{i:02d}' for i in range(1, 11)])
    parser.add_argument('--label', default='Ours')
    args = parser.parse_args()

    t_errs, r_errs = pool_raw_errors(args.output_dir, args.scenes)
    n, e_t, e_r, acc10, acc20 = paper_style_row(t_errs, r_errs)
    ok = np.isfinite(t_errs) & np.isfinite(r_errs)
    n_s, e_ts, e_rs, acc10_s, acc20_s = paper_style_row(t_errs[ok], r_errs[ok])
    print_table([
        (f'{args.label}, all queries', n, e_t, e_r, acc10, acc20),
        (f'{args.label}, PnP success only', n_s, e_ts, e_rs, acc10_s, acc20_s),
    ])
