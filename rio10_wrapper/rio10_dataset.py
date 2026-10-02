# New file, part of the RIO10 wrapper. Does not modify any existing mast3r/dust3r file.
#
# VislocRIO10: a depth-free BaseVislocDataset for RIO10. Unlike the existing
# VislocSevenScenes/VislocCambridgeLandmarks loaders, no depth or SfM point cloud is ever loaded:
# the "MASt3R" (Unseen) baseline this reproduces gets its map-side 3D points from MASt3R's own
# predicted pointmap at inference time (see predicted_matching.py), anchored only by each map
# image's known pose. See the plan doc for the full rationale.
#
# RIO10 frame numbers collide between the map (_01) and query (_02) subscans within a scene, and
# across scenes, so every identifier used anywhere in this wrapper (pairs files, dict keys,
# image_name) is the composite path "sceneNN/seqNN_0X/frame-NNNNNN" (no extension), never a bare
# frame name.
import os
import re
import numpy as np
import torch
import PIL.Image
import yaml

from dust3r_visloc.datasets.base_dataset import BaseVislocDataset
from dust3r_visloc.datasets.utils import get_resize_function
from dust3r.datasets.utils.transforms import ImgNorm


def scene_num(scene):
    """ 'scene01' -> '01' """
    m = re.match(r'scene(\d+)$', scene)
    assert m, f'unexpected scene name {scene!r}'
    return m.group(1)


def map_subscan(scene):
    n = scene_num(scene)
    return f'seq{n}_01'


def query_subscan(scene):
    n = scene_num(scene)
    return f'seq{n}_02'


def subscan_dir(root, scene, subscan):
    n = scene_num(scene)
    return os.path.join(root, scene, f'seq{n}', subscan)


def make_frame_id(scene, subscan, frame_name):
    """ frame_name is e.g. 'frame-000000' (no extension) """
    return f'{scene}/{subscan}/{frame_name}'


def parse_frame_id(frame_id):
    scene, subscan, frame_name = frame_id.split('/')
    return scene, subscan, frame_name


def color_path(root, frame_id):
    scene, subscan, frame_name = parse_frame_id(frame_id)
    return os.path.join(subscan_dir(root, scene, subscan), frame_name + '.color.jpg')


def list_frame_ids(root, scene, subscan):
    d = subscan_dir(root, scene, subscan)
    frame_names = sorted({
        fn[:len('frame-000000')]
        for fn in os.listdir(d)
        if fn.startswith('frame-') and fn.endswith('.color.jpg')
    })
    return [make_frame_id(scene, subscan, fn) for fn in frame_names]


class VislocRIO10(BaseVislocDataset):
    def __init__(self, root, scene, pairs_file, topk=10):
        super().__init__()
        self.root = root
        self.scene = scene
        self.topk = topk
        self.num_views = topk + 1
        self.maxdim = None
        self.patch_size = None
        self._camera_cache = {}

        self.pairs = {}
        self.query_ids = []
        with open(pairs_file) as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                qid, mids = parts[0], parts[1:]
                self.pairs[qid] = mids
                self.query_ids.append(qid)

    def __len__(self):
        return len(self.query_ids)

    def _load_camera(self, scene, subscan):
        key = (scene, subscan)
        if key not in self._camera_cache:
            yaml_path = os.path.join(subscan_dir(self.root, scene, subscan), 'camera.yaml')
            with open(yaml_path) as f:
                cam = yaml.safe_load(f)['camera_intrinsics']
            fx, fy, cx, cy = cam['model']
            k1, k2, k3 = cam['distortion']
            intrinsics = np.float32([(fx, 0, cx), (0, fy, cy), (0, 0, 1)])
            self._camera_cache[key] = (intrinsics, [float(k1), float(k2), float(k3)])
        return self._camera_cache[key]

    def _load_view(self, frame_id):
        scene, subscan, frame_name = parse_frame_id(frame_id)
        d = subscan_dir(self.root, scene, subscan)
        intrinsics, distortion = self._load_camera(scene, subscan)

        cam_to_world = np.loadtxt(os.path.join(d, frame_name + '.pose.txt')).astype(np.float32)

        rgb_image = PIL.Image.open(os.path.join(d, frame_name + '.color.jpg')).convert('RGB')
        rgb_image.load()
        W, H = rgb_image.size
        resize_func, to_resize, to_orig = get_resize_function(self.maxdim, self.patch_size, H, W)
        rgb_tensor = resize_func(ImgNorm(rgb_image))

        return {
            'intrinsics': intrinsics,
            'distortion': distortion,
            'cam_to_world': cam_to_world,
            'rgb': rgb_image,
            'rgb_rescaled': rgb_tensor,
            'to_orig': to_orig,
            'image_name': frame_id,
        }

    def __getitem__(self, idx):
        assert self.maxdim is not None and self.patch_size is not None
        query_id = self.query_ids[idx]
        map_ids = self.pairs[query_id][:self.topk]

        views = [self._load_view(query_id)]
        views[0]['idx'] = 0
        for i, map_id in enumerate(map_ids):
            view = self._load_view(map_id)
            view['idx'] = i + 1
            views.append(view)
        return views
