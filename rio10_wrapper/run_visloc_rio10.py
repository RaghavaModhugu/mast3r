# New file, part of the RIO10 wrapper. Does not modify any existing mast3r/dust3r file.
#
# Driver for one scene: loads VislocRIO10, runs predicted_matching + PnP per query (mirrors
# mast3r/visloc.py's __main__ loop structurally, reusing its evaluation/localization helpers
# unmodified), writes a raw per-query error dump, appends one row to the shared per-scene CSV, and
# prints the standard aggregate_stats summary.
import argparse
import math
import os
import random

import numpy as np
import torch
from tqdm import tqdm

from mast3r.model import AsymmetricMASt3R

import mast3r.utils.path_to_dust3r  # noqa
from dust3r_visloc.localization import run_pnp
from dust3r_visloc.evaluation import get_pose_error, aggregate_stats, export_results

from rio10_dataset import VislocRIO10
from predicted_matching import coarse_matching_predicted_batch, undistort_query_points
from results_csv import append_scene_row


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, help='path to rio10 dataset dir')
    parser.add_argument('--scene', required=True, help='e.g. scene01')
    parser.add_argument('--pairs_file', required=True)
    parser.add_argument('--topk', type=int, default=10)

    parser_weights = parser.add_mutually_exclusive_group(required=True)
    parser_weights.add_argument('--weights', type=str, default=None)
    parser_weights.add_argument('--model_name', type=str,
                                choices=['MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric'])

    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--pixel_tol', type=int, default=5)
    parser.add_argument('--point_conf_threshold', type=float, default=1.5,
                        help='filters MASt3R\'s own point-head confidence for the map image (NOT '
                        'desc_conf). Note: the paper\'s tau in {1,5} does not transfer directly -- '
                        'empirically this checkpoint\'s point confidence only ranges ~[1.0, 3.3], '
                        'so tau=5 would filter out everything; tune relative to that range instead.')
    parser.add_argument('--seed_stride', type=int, default=8,
                        help='grid stride for seeding the reciprocal NN search (bounds cost '
                        'regardless of how many pixels pass point_conf_threshold; matches '
                        'mast3r/visloc.py\'s own stride-8 default)')
    parser.add_argument('--confidence_threshold', type=float, default=1.001,
                        help='descriptor-match confidence threshold (matches_conf >= thr)')
    parser.add_argument('--pnp_mode', type=str, default='poselib', choices=['cv2', 'poselib', 'pycolmap'])
    parser_reproj = parser.add_mutually_exclusive_group()
    parser_reproj.add_argument('--reprojection_error', type=float, default=5.0)
    parser_reproj.add_argument('--reprojection_error_diag_ratio', type=float, default=None)
    parser.add_argument('--pnp_max_points', type=int, default=5000, help='matches paper\'s cap')
    parser.add_argument('--query_batch_size', type=int, default=4,
                        help='how many queries\' topk pairs to batch into a single inference() '
                        'call. A single query (topk=10 pairs @ 512px) empirically uses only ~8GB '
                        'and ~42%% GPU util on a 46GB GPU -- raise this to use more of a larger '
                        'GPU\'s headroom and improve throughput. Tune to your GPU\'s memory.')

    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--output_label', type=str, default='')
    parser.add_argument('--results_csv', type=str, default=None,
                        help='shared per-scene CSV to append this scene\'s summary row to')
    parser.add_argument('--max_queries', type=int, default=None,
                        help='debug: only process the first N queries')
    return parser


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    device = args.device

    weights_path = args.weights if args.weights is not None else 'naver/' + args.model_name
    model = AsymmetricMASt3R.from_pretrained(weights_path).to(device)
    fast_nn_params = dict(device=device, dist='dot', block_size=2**13)

    dataset = VislocRIO10(args.root, args.scene, args.pairs_file, topk=args.topk)
    dataset.set_resolution(model)
    if args.max_queries is not None:
        dataset.query_ids = dataset.query_ids[:args.max_queries]

    query_names, poses_pred, pose_errors, angular_errors = [], [], [], []

    pbar = tqdm(total=len(dataset))
    for batch_start in range(0, len(dataset), args.query_batch_size):
        batch_idxs = range(batch_start, min(batch_start + args.query_batch_size, len(dataset)))
        batch_views = [dataset[idx] for idx in batch_idxs]
        query_map_list = [(views[0], views[1:]) for views in batch_views]
        for query_view, _ in query_map_list:
            query_names.append(query_view['image_name'])

        batch_results = coarse_matching_predicted_batch(
            query_map_list, model, device, fast_nn_params,
            point_conf_thr=args.point_conf_threshold, pixel_tol=args.pixel_tol,
            seed_stride=args.seed_stride)

        for (query_view, _), (world_pts, m_query, m_map, m_confs) in zip(query_map_list, batch_results):
            if len(m_confs) > 0:
                mask = m_confs >= args.confidence_threshold
                world_pts, m_query = world_pts[mask], m_query[mask]

            if len(m_query) == 0:
                success, pr_cam_to_world = False, None
            else:
                query_pts2d = m_query.astype(np.float32)
                query_pts3d = world_pts.astype(np.float32)
                if len(query_pts2d) > args.pnp_max_points:
                    sel = random.sample(range(len(query_pts2d)), args.pnp_max_points)
                    query_pts2d, query_pts3d = query_pts2d[sel], query_pts3d[sel]

                query_pts2d = undistort_query_points(query_pts2d, query_view['intrinsics'],
                                                     query_view['distortion'])

                W, H = query_view['rgb'].size
                if args.reprojection_error_diag_ratio is not None:
                    reproj_err = args.reprojection_error_diag_ratio * math.sqrt(W**2 + H**2)
                else:
                    reproj_err = args.reprojection_error

                success, pr_cam_to_world = run_pnp(query_pts2d, query_pts3d, query_view['intrinsics'],
                                                   distortion=None, mode=args.pnp_mode,
                                                   reprojectionError=reproj_err, img_size=[W, H])
                # degenerate PnP can report success with a NaN/inf pose; treat it as a failure
                if success and not np.all(np.isfinite(pr_cam_to_world)):
                    success, pr_cam_to_world = False, None

            if not success:
                abs_transl_error, abs_angular_error = float('inf'), float('inf')
            else:
                abs_transl_error, abs_angular_error = get_pose_error(pr_cam_to_world, query_view['cam_to_world'])
                abs_transl_error, abs_angular_error = float(abs_transl_error), float(abs_angular_error)

            pose_errors.append(abs_transl_error)
            angular_errors.append(abs_angular_error)
            poses_pred.append(pr_cam_to_world)
        pbar.update(len(batch_idxs))
    pbar.close()

    xp_label = f'{args.scene}_tol_{args.pixel_tol}_conf_{args.confidence_threshold}'
    if args.output_label:
        xp_label = args.output_label + '_' + xp_label

    os.makedirs(args.output_dir, exist_ok=True)
    raw_path = os.path.join(args.output_dir, f'{args.scene}_raw_errors.csv')
    with open(raw_path, 'w') as f:
        f.write('image_name,transl_err_m,angular_err_deg\n')
        for name, t_err, a_err in zip(query_names, pose_errors, angular_errors):
            f.write(f'{name},{t_err},{a_err}\n')

    if args.results_csv:
        row = append_scene_row(args.results_csv, args.scene, pose_errors, angular_errors)
        print('per-scene CSV row:', row)

    print(aggregate_stats(f'{args.scene} ({args.pairs_file})', pose_errors, angular_errors))
    export_results(args.output_dir, xp_label, query_names, poses_pred)
