# New file, part of the RIO10 wrapper. Does not modify any existing mast3r/mast3r/retrieval file.
#
# Builds a learned-retrieval pairs file for one RIO10 scene using mast3r's own ASMK-based
# Retriever/extract_local_features, in DISJOINT query-vs-database mode: the IVF index is built
# from the mapping (database) images only, then queried with the query images -- unlike
# Retriever.__call__ which always does self-retrieval. No existing mast3r file is modified; we
# just call the public pieces of Retriever (self.model, self.asmk) directly.
import argparse
import os
import numpy as np
import torch

# mast3r/retrieval/processor.py's Retriever.__init__ calls torch.load(modelname, 'cpu') without
# weights_only=False; PyTorch >=2.6 defaults weights_only=True and refuses to unpickle the
# checkpoint's argparse.Namespace args. We can't edit processor.py, so default weights_only=False
# for this process only (the checkpoint comes directly from the official naver download server).
_torch_load_orig = torch.load


def _torch_load_patched(*a, **kw):
    kw.setdefault('weights_only', False)
    return _torch_load_orig(*a, **kw)


torch.load = _torch_load_patched

from mast3r.model import AsymmetricMASt3R
from mast3r.retrieval.processor import Retriever
from mast3r.retrieval.model import extract_local_features

from rio10_dataset import map_subscan, query_subscan, list_frame_ids, color_path


def build_retrieval_pairs(root, scene, retrieval_model_path, backbone_weights_path, topk, device='cuda'):
    map_ids = list_frame_ids(root, scene, map_subscan(scene))
    query_ids = list_frame_ids(root, scene, query_subscan(scene))
    map_paths = [color_path(root, fid) for fid in map_ids]
    query_paths = [color_path(root, fid) for fid in query_ids]

    # Retriever.__init__ defaults to AsymmetricMASt3R.from_pretrained(ckpt_args.pretrained), but
    # that field is a training-time-only internal checkpoint name ("finalmast3r") baked into the
    # released retrieval checkpoint, not resolvable here (not a local file, not a real HF repo).
    # Passing our own already-loaded backbone (same architecture, loaded from our local base
    # checkpoint) skips that lookup entirely -- Retriever only loads non-backbone weights
    # (prewhiten/projector/postwhiten) from the retrieval checkpoint on top of it.
    backbone = AsymmetricMASt3R.from_pretrained(backbone_weights_path).to(device)
    retriever = Retriever(retrieval_model_path, backbone=backbone, device=device)

    db_feat, db_ids = extract_local_features(retriever.model, map_paths, retriever.imsize,
                                              tocpu=True, device=retriever.device)
    db_feat, db_ids = db_feat.numpy(), db_ids.numpy()
    asmk_dataset = retriever.asmk.build_ivf(db_feat, db_ids)

    q_feat, q_ids = extract_local_features(retriever.model, query_paths, retriever.imsize,
                                           tocpu=True, device=retriever.device)
    q_feat, q_ids = q_feat.numpy(), q_ids.numpy()
    _, ranked_query_ids, ranks, _ = asmk_dataset.query_ivf(q_feat, q_ids)

    k = min(topk, len(map_ids))
    pairs = {}
    for row, qidx in enumerate(ranked_query_ids):
        qid = query_ids[int(qidx)]
        db_indices = np.asarray(ranks[row])[:k]
        pairs[qid] = [map_ids[int(j)] for j in db_indices]

    # preserve the natural query order even if asmk returned rows in a different order
    ordered_query_ids = [qid for qid in query_ids if qid in pairs]
    return ordered_query_ids, pairs


def write_pairs_file(out_path, query_ids, pairs):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        for qid in query_ids:
            f.write(' '.join([qid] + pairs[qid]) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, help='path to rio10 dataset dir')
    parser.add_argument('--scene', required=True, help='e.g. scene01')
    parser.add_argument('--retrieval_model', required=True,
                        help='path to the *_retrieval_trainingfree.pth checkpoint '
                        '(the matching *_codebook.pkl must sit alongside it)')
    parser.add_argument('--backbone_weights', required=True,
                        help='path to the base MASt3R_ViTLarge_..._metric.pth checkpoint')
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output', required=True, help='output pairs .txt path')
    args = parser.parse_args()

    query_ids, pairs = build_retrieval_pairs(args.root, args.scene, args.retrieval_model,
                                             args.backbone_weights, args.topk, device=args.device)
    write_pairs_file(args.output, query_ids, pairs)
    print(f'{args.scene}: wrote {len(query_ids)} query pairs (topk={args.topk}) to {args.output}')
