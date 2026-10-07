# New file, part of the RIO10 wrapper. Does not modify any existing mast3r/mast3r/retrieval file.
#
# Builds a learned-retrieval pairs file for one RIO10 scene using mast3r's own ASMK-based
# Retriever/extract_local_features, in DISJOINT query-vs-database mode: the IVF index is built
# from the mapping (database) images only, then queried with the query images -- unlike
# Retriever.__call__ which always does self-retrieval. No existing mast3r file is modified; we
# just call the public pieces of Retriever (self.model, self.asmk) directly.
import argparse
import gc
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


def _extract_local_features_chunked(model, image_paths, imsize, device, chunk_size=500):
    """
    mast3r.retrieval.model.extract_local_features (a core mast3r file we can't edit) appends one
    descriptor tensor per image to a plain Python list for the ENTIRE call, with nothing freed
    until a single torch.cat at the very end, and builds a fresh 8-worker DataLoader that stays
    alive for that whole call too. For a ~4700-5500 image scene this is enough to OOM-kill a
    long-running process that already holds a model in memory (confirmed directly: the viz app's
    server was killed mid-extraction, both before and after reusing the backbone model -- the
    second time at 94.7% through just the map-image pass alone).

    Calling extract_local_features once per chunk instead bounds each call's internal growing list
    and DataLoader-worker lifetime to chunk_size images: the chunk's workers are torn down and its
    list is freed as soon as that call returns, before the next chunk starts, instead of all of it
    staying resident for the full image set. ids are offset per chunk to remain valid GLOBAL
    indices into image_paths (extract_local_features itself only ever returns chunk-local 0..N-1
    ids, since it enumerates whatever list it's given).
    """
    feats, ids = [], []
    for start in range(0, len(image_paths), chunk_size):
        chunk = image_paths[start:start + chunk_size]
        feat, local_ids = extract_local_features(model, chunk, imsize, tocpu=True, device=device)
        feats.append(feat.numpy())
        ids.append(local_ids.numpy() + start)
        del feat, local_ids
        gc.collect()
        if device != 'cpu' and torch.cuda.is_available():
            torch.cuda.empty_cache()
    return np.concatenate(feats, axis=0), np.concatenate(ids, axis=0)


def build_retrieval_pairs(root, scene, retrieval_model_path, backbone_weights_path, topk,
                          device='cuda', backbone=None):
    """
    backbone: an already-loaded AsymmetricMASt3R instance to reuse instead of loading a fresh one
    from backbone_weights_path. Pass this when calling from a long-running process that already
    holds a model in memory (e.g. the viz app) -- loading a second full model instance alongside
    it is what OOM-killed the viz app's on-demand retrieval-index build in practice (confirmed:
    the server process was killed mid-extraction on this node's memory cap). backbone_weights_path
    is still required and used as a fallback when backbone is None (the normal CLI path below).
    """
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
    if backbone is None:
        backbone = AsymmetricMASt3R.from_pretrained(backbone_weights_path).to(device)
    retriever = Retriever(retrieval_model_path, backbone=backbone, device=device)

    db_feat, db_ids = _extract_local_features_chunked(retriever.model, map_paths, retriever.imsize,
                                                       retriever.device)
    asmk_dataset = retriever.asmk.build_ivf(db_feat, db_ids)
    del db_feat, db_ids
    gc.collect()

    q_feat, q_ids = _extract_local_features_chunked(retriever.model, query_paths, retriever.imsize,
                                                     retriever.device)
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
