# New file, part of the RIO10 wrapper. Does not modify any existing mast3r/dust3r file.
#
# Builds an "oracle retrieval" pairs file for one RIO10 scene: for every query frame, ranks the
# scene's mapping frames by ground-truth camera-center (translation-only) distance and keeps the
# top-k. This is an upper bound on retrieval quality (no learned retrieval model involved, no
# network, no depth), used to check how much headroom a real retriever is leaving on the table.
#
# Pairs-file format (plain text, one line per query):
#   query_frame_id map_frame_id_1 map_frame_id_2 ... map_frame_id_k
# where every id is the composite "sceneNN/seqNN_0X/frame-NNNNNN" (see rio10_dataset.py).
import argparse
import os
import numpy as np
from scipy.spatial import cKDTree

from rio10_dataset import map_subscan, query_subscan, list_frame_ids, parse_frame_id, subscan_dir


def load_camera_center(root, frame_id):
    scene, subscan, frame_name = parse_frame_id(frame_id)
    d = subscan_dir(root, scene, subscan)
    cam_to_world = np.loadtxt(os.path.join(d, frame_name + '.pose.txt'))
    return cam_to_world[:3, 3]


def build_oracle_pairs(root, scene, topk):
    map_ids = list_frame_ids(root, scene, map_subscan(scene))
    query_ids = list_frame_ids(root, scene, query_subscan(scene))

    map_centers = np.stack([load_camera_center(root, fid) for fid in map_ids])
    query_centers = np.stack([load_camera_center(root, fid) for fid in query_ids])

    tree = cKDTree(map_centers)
    k = min(topk, len(map_ids))
    _, nn_idx = tree.query(query_centers, k=k)
    if k == 1:
        nn_idx = nn_idx[:, None]

    pairs = {}
    for qi, qid in enumerate(query_ids):
        pairs[qid] = [map_ids[j] for j in nn_idx[qi]]
    return query_ids, pairs


def write_pairs_file(out_path, query_ids, pairs):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        for qid in query_ids:
            f.write(' '.join([qid] + pairs[qid]) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, help='path to rio10 dataset dir')
    parser.add_argument('--scene', required=True, help='e.g. scene01')
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--output', required=True, help='output pairs .txt path')
    args = parser.parse_args()

    query_ids, pairs = build_oracle_pairs(args.root, args.scene, args.topk)
    write_pairs_file(args.output, query_ids, pairs)
    print(f'{args.scene}: wrote {len(query_ids)} query pairs (topk={args.topk}) to {args.output}')
