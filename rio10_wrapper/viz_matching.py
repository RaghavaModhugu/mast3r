# New file, part of the RIO10 wrapper. Does not modify any existing mast3r/dust3r file.
#
# Single-query, per-neighbor match breakdown + PnP + inlier classification, for the interactive
# viz tool. Reuses predicted_matching.coarse_matching_predicted_batch unmodified (by calling it
# once per neighbor instead of batched across queries -- the viz tool only ever handles one query
# at a time, so the batched function's whole purpose, cross-QUERY throughput for the bulk
# pipeline, doesn't apply here; looping avoids coupling that throughput-critical file to UI
# concerns), and dust3r_visloc.localization.run_pnp unmodified.
#
# dust3r_visloc.localization.run_pnp discards every backend's own inlier mask (confirmed by
# reading it in full): cv2.solvePnPRansac's 4th return is discarded via `_`, poselib's second
# return is discarded via `_`, pycolmap's ret['num_inliers'] is read internally but never
# surfaced. classify_inliers() below recovers inlier/outlier status independently by
# reprojection, since there's no way to extract it from run_pnp's return value.
import math
import random

import numpy as np
import cv2

import mast3r.utils.path_to_dust3r  # noqa
from dust3r_visloc.localization import run_pnp

from predicted_matching import coarse_matching_predicted_batch, undistort_query_points


def match_query_against_neighbors(query_view, map_views, model, device, fast_nn_params,
                                   point_conf_thr=1.5, pixel_tol=5, seed_stride=8):
    """ One coarse_matching_predicted_batch call per neighbor (unmodified), so the viz tool gets
    a per-neighbor breakdown that the batched-across-queries function doesn't expose. Returns one
    dict per neighbor: map_view, world_pts, matches_im_query, matches_im_map, matches_confs. """
    results = []
    for map_view in map_views:
        world_pts, m_query, m_map, m_confs = coarse_matching_predicted_batch(
            [(query_view, [map_view])], model, device, fast_nn_params,
            point_conf_thr=point_conf_thr, pixel_tol=pixel_tol, seed_stride=seed_stride)[0]
        results.append({'map_view': map_view, 'world_pts': world_pts, 'matches_im_query': m_query,
                        'matches_im_map': m_map, 'matches_confs': m_confs})
    return results


def concat_neighbor_matches(neighbor_matches):
    """ Concatenates per-neighbor arrays, plus a new parallel neighbor_idx int array recording
    provenance (which neighbor each aggregated correspondence came from) -- bookkeeping that
    doesn't exist in predicted_matching.py's concatenated-across-neighbors contract. """
    world_pts_list, mq_list, mm_list, conf_list, nidx_list = [], [], [], [], []
    for i, nm in enumerate(neighbor_matches):
        n = len(nm['matches_confs'])
        if n == 0:
            continue
        world_pts_list.append(nm['world_pts'])
        mq_list.append(nm['matches_im_query'])
        mm_list.append(nm['matches_im_map'])
        conf_list.append(nm['matches_confs'])
        nidx_list.append(np.full(n, i, dtype=int))
    if not world_pts_list:
        return (np.zeros((0, 3), dtype=np.float32), np.zeros((0, 2)), np.zeros((0, 2)),
                np.zeros((0,)), np.zeros((0,), dtype=int))
    return (np.concatenate(world_pts_list), np.concatenate(mq_list), np.concatenate(mm_list),
            np.concatenate(conf_list), np.concatenate(nidx_list))


def run_pnp_from_matches(world_pts, m_query, m_map, m_confs, query_view, confidence_threshold,
                         pnp_mode, reprojection_error, reprojection_error_diag_ratio,
                         pnp_max_points, rng_seed=0):
    """
    Mirrors run_visloc_rio10.py's per-query control flow exactly (confidence mask -> pnp_max_points
    subsample -> undistort_query_points -> run_pnp(..., distortion=None, ...) -> the NaN/inf-pose
    guard), for a single query. Returns a dict including 'used_match_indices': positions into the
    ORIGINAL (world_pts, m_query, m_map, m_confs) arrays that were actually passed to PnP -- so a
    caller holding the matching neighbor_idx array from concat_neighbor_matches can attribute
    inliers back to specific neighbors.

    Deliberate deviation from run_visloc_rio10.py: that script's >pnp_max_points subsample uses
    the unseeded global `random` module -- fine averaged over thousands of bulk queries, but would
    mean re-clicking "Run" on the same query in this interactive tool silently changes the pose.
    This function seeds random.Random(rng_seed) instead, so interactive runs are reproducible.
    """
    kept_idx = np.arange(0, dtype=int)
    if len(m_confs) > 0:
        mask = m_confs >= confidence_threshold
        world_pts, m_query = world_pts[mask], m_query[mask]
        kept_idx = np.where(mask)[0]

    empty = {'success': False, 'pose': None, 'used_match_indices': np.array([], dtype=int),
            'query_pts2d_undist': np.zeros((0, 2), dtype=np.float32),
            'query_pts3d': np.zeros((0, 3), dtype=np.float32),
            'reprojection_error_used': reprojection_error}
    if len(m_query) == 0:
        return empty

    query_pts2d = m_query.astype(np.float32)
    query_pts3d = world_pts.astype(np.float32)
    sel_idx = np.arange(len(query_pts2d))
    if len(query_pts2d) > pnp_max_points:
        rng = random.Random(rng_seed)
        sel = rng.sample(range(len(query_pts2d)), pnp_max_points)
        sel_idx = np.array(sorted(sel))
        query_pts2d, query_pts3d = query_pts2d[sel_idx], query_pts3d[sel_idx]

    query_pts2d_undist = undistort_query_points(query_pts2d, query_view['intrinsics'],
                                                query_view['distortion']).astype(np.float32)

    W, H = query_view['rgb'].size
    if reprojection_error_diag_ratio is not None:
        reproj_err = reprojection_error_diag_ratio * math.sqrt(W**2 + H**2)
    else:
        reproj_err = reprojection_error

    success, pose = run_pnp(query_pts2d_undist, query_pts3d, query_view['intrinsics'],
                            distortion=None, mode=pnp_mode, reprojectionError=reproj_err,
                            img_size=[W, H])
    if success and not np.all(np.isfinite(pose)):
        success, pose = False, None

    used_match_indices = kept_idx[sel_idx] if len(kept_idx) else sel_idx
    return {
        'success': success, 'pose': pose,
        'used_match_indices': used_match_indices,
        'query_pts2d_undist': query_pts2d_undist,
        'query_pts3d': query_pts3d,
        'reprojection_error_used': reproj_err,
    }


def classify_inliers(pose, query_pts2d_undist, query_pts3d, K, reprojection_error):
    """
    GT-free inlier/outlier recovery for a pose run_pnp already returned (run_pnp discards every
    backend's own inlier mask -- see module docstring). Reprojects query_pts3d through the pose
    and thresholds against the same reprojection_error used for that PnP call. Backend-agnostic:
    all three run_pnp backends (cv2/poselib/pycolmap) return the identical cam_to_world 4x4
    definition, so reprojecting in plain OpenCV pixel convention from the returned pose, against
    the already-undistorted query_pts2d_undist, is valid regardless of which backend produced it
    (even though it doesn't replay each backend's internal colmap-shifted-K bookkeeping).
    """
    if pose is None or len(query_pts3d) == 0:
        return np.zeros((0,), dtype=bool)
    world_to_cam = np.linalg.inv(pose)
    rvec, _ = cv2.Rodrigues(world_to_cam[:3, :3])
    tvec = world_to_cam[:3, 3]
    proj, _ = cv2.projectPoints(query_pts3d, rvec, tvec, K, distCoeffs=None)
    proj = proj.reshape(-1, 2)
    err = np.linalg.norm(proj - query_pts2d_undist, axis=1)
    return err <= reprojection_error
