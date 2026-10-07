# New file, part of the RIO10 wrapper. Does not modify any existing mast3r/dust3r file.
#
# Builds an "oracle retrieval" pairs file for one RIO10 scene: for every query frame, ranks the
# scene's mapping frames by a combined ground-truth camera-center distance + viewing-direction
# score and keeps the top-k. This is an upper bound on retrieval quality (no learned retrieval
# model involved, no network, no depth), used to check how much headroom a real retriever leaves
# on the table.
#
# Position alone is NOT enough here: empirically (checked directly against this dataset, 1000
# query/map pairs sampled from scene01), ranking purely by camera-center distance has a MEDIAN
# viewing-direction cosine similarity of -0.14 (facing away, not toward) with zero pairs even
# moderately aligned (>0.5) -- RIO10's scan trajectory does not correlate position with viewing
# direction (an operator can stand in one spot and look at many different walls). Position-only
# oracle pairs produced 0% successful PnP across every query tested -- the "nearest" map frames by
# position routinely have no shared field of view with the query at all. So the score combines
# camera-center distance (meters) with the angle (radians) between viewing directions.
#
# Pairs-file format (plain text, one line per query):
#   query_frame_id map_frame_id_1 map_frame_id_2 ... map_frame_id_k
# where every id is the composite "sceneNN/seqNN_0X/frame-NNNNNN" (see rio10_dataset.py).
import argparse
import os
import numpy as np

from rio10_dataset import map_subscan, query_subscan, list_frame_ids, parse_frame_id, subscan_dir

FORWARD_CAM = np.array([0.0, 0.0, -1.0])  # OpenCV/dust3r camera convention: looks down -Z


def load_center_and_forward(root, frame_id):
    scene, subscan, frame_name = parse_frame_id(frame_id)
    d = subscan_dir(root, scene, subscan)
    cam_to_world = np.loadtxt(os.path.join(d, frame_name + '.pose.txt'))
    center = cam_to_world[:3, 3]
    forward = cam_to_world[:3, :3] @ FORWARD_CAM
    return center, forward


def load_all_centers_forwards(root, frame_ids):
    """ Loads (center, forward) for every frame id -- pure pose arithmetic, cheap even for the
    largest scenes (confirmed: ~10K frames in a fraction of a second). Shared by both the
    scene-level bulk builder (build_oracle_pairs) and the image-level on-demand path
    (rank_single_query), so a scene's map-side arrays can be computed once and reused per image. """
    centers = np.zeros((len(frame_ids), 3))
    forwards = np.zeros((len(frame_ids), 3))
    for i, fid in enumerate(frame_ids):
        centers[i], forwards[i] = load_center_and_forward(root, fid)
    return centers, forwards


def rank_single_query(query_center, query_forward, map_ids, map_centers, map_forwards,
                      topk, angle_weight=1.0):
    """
    Image-level oracle ranking: scores ONE query's (center, forward) against an already-loaded
    set of map centers/forwards (e.g. cached once per scene via load_all_centers_forwards) and
    returns the topk map ids, nearest first. Same distance+angle score as build_oracle_pairs,
    just for a single query instead of the whole scene's queries at once.
    """
    dist = np.linalg.norm(map_centers - query_center[None, :], axis=-1)
    cos_sim = np.clip(map_forwards @ query_forward, -1.0, 1.0)
    angle = np.arccos(cos_sim)
    score = dist + angle_weight * angle
    k = min(topk, len(map_ids))
    part = np.argpartition(score, kth=k - 1)[:k]
    order = part[np.argsort(score[part])]
    return [map_ids[j] for j in order]


def build_oracle_pairs(root, scene, topk, angle_weight=1.0, chunk_size=500):
    """
    angle_weight: meters of position-distance penalty per radian of viewing-direction mismatch.
    1.0 makes a quarter-turn (~1.57 rad) roughly as costly as being 1.57m farther away -- a
    reasonable default given typical indoor camera-center distances here are sub-meter to a few
    meters; tune if oracle pairs still show poor viewing-direction alignment for your scenes.
    """
    map_ids = list_frame_ids(root, scene, map_subscan(scene))
    query_ids = list_frame_ids(root, scene, query_subscan(scene))

    map_centers, map_forwards = load_all_centers_forwards(root, map_ids)
    query_centers, query_forwards = load_all_centers_forwards(root, query_ids)

    k = min(topk, len(map_ids))
    pairs = {}
    for start in range(0, len(query_ids), chunk_size):
        end = min(start + chunk_size, len(query_ids))
        qc, qf = query_centers[start:end], query_forwards[start:end]

        dist = np.linalg.norm(qc[:, None, :] - map_centers[None, :, :], axis=-1)   # [B, n_map]
        cos_sim = np.clip(qf @ map_forwards.T, -1.0, 1.0)                           # [B, n_map]
        angle = np.arccos(cos_sim)                                                  # radians
        score = dist + angle_weight * angle

        part = np.argpartition(score, kth=k - 1, axis=1)[:, :k]
        for bi, qi in enumerate(range(start, end)):
            order = part[bi][np.argsort(score[bi, part[bi]])]
            pairs[query_ids[qi]] = [map_ids[j] for j in order]
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
    parser.add_argument('--angle_weight', type=float, default=1.0,
                        help='meters of distance-penalty per radian of viewing-direction mismatch')
    parser.add_argument('--output', required=True, help='output pairs .txt path')
    args = parser.parse_args()

    query_ids, pairs = build_oracle_pairs(args.root, args.scene, args.topk, args.angle_weight)
    write_pairs_file(args.output, query_ids, pairs)
    print(f'{args.scene}: wrote {len(query_ids)} query pairs (topk={args.topk}) to {args.output}')
