# Per-query texture (Laplacian variance of the query image) and change share (query pixels showing
# instances absent from its 10 oracle map views). Run from rio10_wrapper/. Output: CSV of features.
import csv, os, sys, numpy as np
from PIL import Image
from scipy.ndimage import laplace
from multiprocessing import Pool
BASE = '../../ace-g/rio10'
PAIRS_DIR = 'results/oracle/pairs'
def frame_path(fid, kind):
    scene, sub, fr = fid.split('/')
    return f'{BASE}/{scene}/{sub[:5]}/{sub}/{fr}.{kind}'
def mask(fid):
    scene, sub, fr = fid.split('/')
    return np.array(Image.open(f'{BASE}/{scene}/{sub}/{fr}.instances.png'))
def work(args):
    scene, qid, maps = args
    gray = np.asarray(Image.open(frame_path(qid, 'color.jpg')).convert('L'), dtype=np.float32)
    tex = float(laplace(gray).var())
    qm = mask(qid)
    union = set()
    for m in maps:
        union |= set(np.unique(mask(m)).tolist())
    valid = qm > 0
    absent = ~np.isin(qm, list(union - {0})) & valid
    change = float(absent.sum() / max(valid.sum(), 1))
    return scene, qid, tex, change
if __name__ == '__main__':
    n_per_scene = int(sys.argv[1])
    rng = np.random.default_rng(0)
    jobs = []
    for s in range(1, 11):
        scene = f'scene{s:02d}'
        lines = open(f'{PAIRS_DIR}/{scene}_oracle_pairs.txt').read().splitlines()
        sel = rng.choice(len(lines), size=min(n_per_scene, len(lines)), replace=False)
        for i in sel:
            parts = lines[i].split()
            jobs.append((scene, parts[0], parts[1:11]))
    with Pool(16) as p:
        res = p.map(work, jobs, chunksize=8)
    with open(sys.argv[2], 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['scene', 'query', 'texture_lapvar', 'change_frac'])
        for r in res: w.writerow(r)
    print('wrote', len(res))
