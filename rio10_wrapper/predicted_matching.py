# New file, part of the RIO10 wrapper. Does not modify any existing mast3r/dust3r file.
#
# Adapts the descriptor-matching mechanics of mast3r/visloc.py's own coarse_matching() (same
# fast_reciprocal_NNs call pattern, same cv2<->colmap pixel-center convention) to build 2D-3D
# correspondences from MASt3R's OWN predicted pointmap instead of any dataset depth/SfM model --
# this is what the paper (arxiv 2510.00978) calls the "MASt3R" (Unseen) baseline: "relies directly
# in its 3D point and matching heads instead of building a SfM model".
#
# Key trick: a MASt3R forward pass on a pair [imgA, imgB] returns pred1['pts3d'] (imgA's pointmap,
# in imgA's OWN local camera frame) and pred2['pts3d_in_other_view'] (imgB's points, already in
# imgA's frame). So we put the MAP image at index 0 of every pair: its pointmap is then directly
# transformable to world coordinates via geotrf(map_cam_to_world, pred1['pts3d']) -- using the
# map's known POSE (standard/required for any relocalization map) but never its depth.
#
# coarse_matching_predicted_batch batches the topk map-view pairs of MULTIPLE queries into a
# single inference() call (configurable via run_visloc_rio10.py's --query_batch_size), since a
# single query's topk=10 pairs leaves most of a 40GB+ GPU idle (empirically ~8GB/42% util for
# topk=10 at 512px) -- batching more queries' pairs together per forward pass trades that idle
# headroom for throughput.
import numpy as np
import torch

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.inference import inference
from dust3r.utils.geometry import geotrf
from mast3r.fast_nn import fast_reciprocal_NNs


def _make_input(img_tensor, idx):
    return dict(img=img_tensor.unsqueeze(0), true_shape=np.int32([img_tensor.shape[1:]]),
                idx=idx, instance=str(idx))


_EMPTY = (np.zeros((0, 3), dtype=np.float32), np.zeros((0, 2)), np.zeros((0, 2)), np.zeros((0,)))


def _finish_match(query_view, map_view, matches_im_map, matches_im_query, pts3d_map_local,
                   desc_conf_map, desc_conf_query):
    world_pts_map = geotrf(map_view['cam_to_world'], pts3d_map_local)
    valid_pts3d = world_pts_map[matches_im_map[:, 1], matches_im_map[:, 0]]
    matches_confs = np.minimum(
        desc_conf_map[matches_im_map[:, 1], matches_im_map[:, 0]],
        desc_conf_query[matches_im_query[:, 1], matches_im_query[:, 0]],
    )

    # rescale matched pixels from the resized-tensor grid back to each image's original pixel
    # coordinates (same cv2<->colmap +-0.5 convention as mast3r/visloc.py:coarse_matching)
    matches_im_query = matches_im_query.astype(np.float64)
    matches_im_map = matches_im_map.astype(np.float64)
    matches_im_query[:, 0] += 0.5
    matches_im_query[:, 1] += 0.5
    matches_im_map[:, 0] += 0.5
    matches_im_map[:, 1] += 0.5
    matches_im_query = geotrf(query_view['to_orig'], matches_im_query, norm=True)
    matches_im_map = geotrf(map_view['to_orig'], matches_im_map, norm=True)
    matches_im_query[:, 0] -= 0.5
    matches_im_query[:, 1] -= 0.5
    matches_im_map[:, 0] -= 0.5
    matches_im_map[:, 1] -= 0.5
    return valid_pts3d, matches_im_query, matches_im_map, matches_confs


@torch.no_grad()
def coarse_matching_predicted_batch(query_map_list, model, device, fast_nn_params,
                                     point_conf_thr=1.5, pixel_tol=5, seed_stride=8):
    """
    query_map_list: list of (query_view, map_views) tuples, one per query in this batch.

    Batches ALL (map_view, query_view) pairs across every query in the list into a SINGLE
    inference() call (cost is bounded by whatever --query_batch_size the caller chose), then
    returns a list of (world_pts, matches_im_query, matches_im_map, matches_confs) tuples, one per
    query in the same order -- same per-query shape contract as the original single-query version.
    """
    pairs, offsets = [], []
    for query_view, map_views in query_map_list:
        query_img = _make_input(query_view['rgb_rescaled'], 1)
        start = len(pairs)
        for map_view in map_views:
            pairs.append((_make_input(map_view['rgb_rescaled'], 0), query_img))
        offsets.append((start, len(pairs)))

    if len(pairs) == 0:
        return [_EMPTY for _ in query_map_list]

    output = inference(pairs, model, device, batch_size=len(pairs), verbose=False)
    pred1, pred2 = output['pred1'], output['pred2']  # pred1=map side (idx0), pred2=query side (idx1)

    results = []
    for (query_view, map_views), (start, end) in zip(query_map_list, offsets):
        all_world_pts, all_m_query, all_m_map, all_confs = [], [], [], []
        for local_i, slot in enumerate(range(start, end)):
            map_view = map_views[local_i]
            pts3d_map_local = pred1['pts3d'][slot].numpy()
            conf_map = pred1['conf'][slot].numpy()
            desc_map = pred1['desc'][slot]
            desc_query = pred2['desc'][slot]
            desc_conf_map = pred1['desc_conf'][slot].numpy()
            desc_conf_query = pred2['desc_conf'][slot].numpy()

            valid_map = conf_map >= point_conf_thr
            if not valid_map.any():
                continue
            H, W = valid_map.shape
            yy, xx = np.mgrid[seed_stride // 2:H:seed_stride, seed_stride // 2:W:seed_stride]
            yy, xx = yy.ravel(), xx.ravel()
            keep = valid_map[yy, xx]
            yM, xM = yy[keep], xx[keep]
            if len(yM) == 0:
                continue

            matches_im_map, matches_im_query = fast_reciprocal_NNs(
                desc_map, desc_query, (xM, yM), pixel_tol=pixel_tol, **fast_nn_params)
            if len(matches_im_map) == 0:
                continue

            valid_pts3d, m_query, m_map, m_confs = _finish_match(
                query_view, map_view, matches_im_map, matches_im_query,
                pts3d_map_local, desc_conf_map, desc_conf_query)
            all_world_pts.append(valid_pts3d)
            all_m_query.append(m_query)
            all_m_map.append(m_map)
            all_confs.append(m_confs)

        if len(all_world_pts) == 0:
            results.append(_EMPTY)
        else:
            results.append((np.concatenate(all_world_pts, axis=0),
                            np.concatenate(all_m_query, axis=0),
                            np.concatenate(all_m_map, axis=0),
                            np.concatenate(all_confs, axis=0)))
    return results


def undistort_query_points(pts2D, intrinsics, distortion):
    """
    dust3r_visloc.localization.run_pnp hardcodes per-backend distortion-vector lengths that don't
    accept RIO10's raw 3-coefficient [k1,k2,k3] radial-only vector (cv2 needs exactly 4/5/8/12/14;
    poselib/pycolmap's hardcoded OPENCV camera model expects exactly 4: k1,k2,p1,p2). Since
    localization.py can't be edited, we pre-undistort the query's matched 2D pixels ourselves with
    RIO10's full calibration here, then call run_pnp(..., distortion=None, ...).
    """
    import cv2
    if len(pts2D) == 0:
        return pts2D
    k1, k2, k3 = distortion
    dist_vec = np.array([k1, k2, 0.0, 0.0, k3], dtype=np.float64)
    pts = np.ascontiguousarray(pts2D, dtype=np.float64).reshape(-1, 1, 2)
    undistorted = cv2.undistortPoints(pts, intrinsics, dist_vec, R=None, P=intrinsics)
    return undistorted.reshape(-1, 2)
