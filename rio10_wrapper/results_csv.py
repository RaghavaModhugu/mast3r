# New file, part of the RIO10 wrapper. Does not modify any existing mast3r/dust3r file.
#
# Shared per-scene CSV reporting helper, used by every run script.
import csv
import os
import numpy as np

HEADER = ['sequence', 'num_frames', 'median_r_err_deg', 'median_t_err_cm',
          'within_5deg_5cm', 'within_10deg_10cm', 'within_20deg_20cm']


def _median_or_na(values):
    m = float(np.median(values)) if len(values) else float('nan')
    return 'N/A' if not np.isfinite(m) else m


def _accuracy_pct(pose_errors_m, angular_errors_deg, thr_deg):
    thr_m = thr_deg / 100.0  # "Xdeg,Xcm" convention: same numeric X for both
    pose_errors_m = np.asarray(pose_errors_m, dtype=np.float64)
    angular_errors_deg = np.asarray(angular_errors_deg, dtype=np.float64)
    if len(pose_errors_m) == 0:
        return 0.0
    correct = (pose_errors_m <= thr_m) & (angular_errors_deg <= thr_deg)
    return 100.0 * correct.sum() / len(pose_errors_m)


def append_scene_row(csv_path, sequence, pose_errors_m, angular_errors_deg):
    """
    pose_errors_m: list of per-query translation errors in meters (inf for failed PnP).
    angular_errors_deg: list of per-query rotation errors in degrees (inf for failed PnP).
    """
    pose_errors_m = [float(v) for v in pose_errors_m]
    angular_errors_deg = [float(v) for v in angular_errors_deg]

    row = {
        'sequence': sequence,
        'num_frames': len(pose_errors_m),
        'median_r_err_deg': _median_or_na(angular_errors_deg),
        'median_t_err_cm': _median_or_na([v * 100.0 for v in pose_errors_m]),
        'within_5deg_5cm': _accuracy_pct(pose_errors_m, angular_errors_deg, 5),
        'within_10deg_10cm': _accuracy_pct(pose_errors_m, angular_errors_deg, 10),
        'within_20deg_20cm': _accuracy_pct(pose_errors_m, angular_errors_deg, 20),
    }

    os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return row
