# New file, part of the RIO10 wrapper. Does not modify any existing mast3r/dust3r file.
#
# RIO10 ships a full textured 3D reconstruction (mesh.obj) and a semantically-labeled point cloud
# (labels.ply, per-vertex instance IDs) for EACH subscan -- including both the mapping scan
# (prior) and the rescan (current). This module loads them, cross-references the same
# instances.txt-based changed-object logic already used by analysis/texture_change.py (at the
# mesh/vertex level instead of the per-frame-mask level), and exports .glb files (with camera
# frustums overlaid via dust3r.viz, unmodified) for gradio.Model3D display.
#
# IMPORTANT path note: mesh.obj/labels.ply/instances.txt/*.instances.png live FLAT at
# {root}/{scene}/{subscan}/ -- a DIFFERENT, shallower directory than
# rio10_dataset.subscan_dir(), which points at the nested {root}/{scene}/seq{NN}/{subscan}/ used
# for camera.yaml/frame images/poses. Confirmed via readlink -f: these are two distinct real
# directories, not a symlink pair. Do not reuse rio10_dataset.subscan_dir() here.
#
# IMPORTANT trimesh note: both mesh.obj and labels.ply MUST be loaded with process=False.
# trimesh's default vertex-merging drops ~26% of vertices (confirmed on scene01: 67272 -> 49518),
# which desyncs labels.ply's per-vertex objectId column from the mesh. With process=False,
# mesh.obj and labels.ply have identical vertex count and order (coordinates match to ~1e-6,
# confirmed directly) -- this is what makes per-vertex instance-based highlighting possible.
import os

import numpy as np
import trimesh

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.viz import add_scene_cam, CAM_COLORS

from rio10_dataset import map_subscan, query_subscan


def semantic_subscan_dir(root, scene, subscan):
    """ Flat path for mesh/labels/instances files -- NOT rio10_dataset.subscan_dir(). """
    return os.path.join(root, scene, subscan)


def load_subscan_mesh(root, scene, subscan):
    path = os.path.join(semantic_subscan_dir(root, scene, subscan), 'mesh.obj')
    return trimesh.load(path, process=False, force='mesh')


_PLY_DTYPE = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
              ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
              ('objectId', 'u2'), ('globalId', 'u2'),
              ('NYU40', 'u1'), ('Eigen13', 'u1'), ('RIO27', 'u1')]


def load_labels_ply(path):
    """ Hand-rolled ASCII-PLY parser for RIO10's fixed labels.ply schema. trimesh's generic PLY
    loader only exposes vertex_colors (the r,g,b columns), not the custom integer columns
    (objectId etc.) this module needs -- a small parser for this fixed, simple, ASCII schema is
    cheaper and safer than adding a new PLY-with-custom-properties dependency. """
    with open(path) as f:
        n_vertices = None
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f'{path}: end of file before end_header')
            line = line.strip()
            if line.startswith('element vertex'):
                n_vertices = int(line.split()[-1])
            elif line == 'end_header':
                break
        rows = np.loadtxt(f, dtype=_PLY_DTYPE, max_rows=n_vertices)

    vertices = np.stack([rows['x'], rows['y'], rows['z']], axis=-1).astype(np.float64)
    vertex_colors = np.stack([rows['red'], rows['green'], rows['blue']], axis=-1).astype(np.uint8)
    object_id = rows['objectId'].astype(np.int64)
    return {'vertices': vertices, 'vertex_colors': vertex_colors, 'object_id': object_id}


def _unreferenced_vertex_indices(obj_path, n_vertices):
    """
    trimesh's OBJ loader silently drops any vertex not referenced by at least one face, even with
    process=False (confirmed directly on scene02/seq02_01: 3 of 79398 raw vertices are never used
    by an 'f ' line, and trimesh's loaded mesh has exactly 79395 vertices, in the same relative
    order as the raw file with those 3 rows removed -- process=False only disables vertex
    merging/cleanup, not this face-reference filtering, which happens during parsing itself). This
    re-derives the same 0-based indices trimesh dropped, purely from the raw OBJ file's 'f ' lines,
    so load_labels_ply's rows (one per raw vertex, no such filtering) can be realigned to match.
    """
    referenced = np.zeros(n_vertices, dtype=bool)
    with open(obj_path) as f:
        for line in f:
            if not line.startswith('f '):
                continue
            for tok in line.split()[1:]:
                idx = int(tok.split('/')[0])
                idx = idx - 1 if idx > 0 else n_vertices + idx
                referenced[idx] = True
    return np.where(~referenced)[0]


def load_instance_names(instances_txt_path):
    """ {id: class_name} """
    names = {}
    with open(instances_txt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            names[int(parts[0])] = parts[1] if len(parts) > 1 else ''
    return names


def load_instance_id_set(root, scene, subscan):
    names = load_instance_names(os.path.join(semantic_subscan_dir(root, scene, subscan), 'instances.txt'))
    return set(names.keys())


def removed_instance_ids(prior_ids, current_ids):
    """ Instances present in the prior (mapping) scan but gone from the current (rescan) scan. """
    return prior_ids - current_ids


def added_instance_ids(prior_ids, current_ids):
    """ Instances present in the current (rescan) scan but absent from the prior (mapping) scan. """
    return current_ids - prior_ids


def vertex_highlight_colors(base_colors, object_id, highlight_ids, highlight_rgb=(255, 60, 60)):
    """ Pure numpy recolor: copies base_colors (Nx3 or Nx4 uint8) and overwrites the RGB of every
    vertex whose object_id is in highlight_ids. """
    out = base_colors.copy()
    if not highlight_ids:
        return out
    mask = np.isin(object_id, list(highlight_ids))
    out[mask, 0] = highlight_rgb[0]
    out[mask, 1] = highlight_rgb[1]
    out[mask, 2] = highlight_rgb[2]
    return out


def _load_side(root, scene, subscan):
    obj_path = os.path.join(semantic_subscan_dir(root, scene, subscan), 'mesh.obj')
    mesh = load_subscan_mesh(root, scene, subscan)
    labels = load_labels_ply(os.path.join(semantic_subscan_dir(root, scene, subscan), 'labels.ply'))

    if len(labels['object_id']) != len(mesh.vertices):
        orphans = _unreferenced_vertex_indices(obj_path, len(labels['object_id']))
        keep = np.ones(len(labels['object_id']), dtype=bool)
        keep[orphans] = False
        if keep.sum() != len(mesh.vertices):
            raise ValueError(
                f'{scene}/{subscan}: mesh.obj has {len(mesh.vertices)} vertices, labels.ply has '
                f'{len(labels["object_id"])}, and removing the {len(orphans)} face-unreferenced '
                f'vertices found in mesh.obj still leaves {keep.sum()} -- counts should match '
                f'after that removal (see _unreferenced_vertex_indices). Refusing to use a '
                f'possibly-desynced labels array.')
        max_coord_diff = np.abs(labels['vertices'][keep] - np.asarray(mesh.vertices)).max()
        if max_coord_diff > 1e-3:
            raise ValueError(
                f'{scene}/{subscan}: vertex coordinates do not line up after removing '
                f'face-unreferenced rows (max diff {max_coord_diff}) -- refusing to use a '
                f'possibly-desynced labels array.')
        labels['object_id'] = labels['object_id'][keep]
        labels['vertex_colors'] = labels['vertex_colors'][keep]

    names = load_instance_names(os.path.join(semantic_subscan_dir(root, scene, subscan), 'instances.txt'))
    return {
        'mesh': mesh,
        'object_id': labels['object_id'],
        'vertex_colors': labels['vertex_colors'],
        'names': names,
        'ids': set(names.keys()),
    }


def build_scene_meshes(root, scene):
    """ The expensive, per-scene-only loader (~1-2s per subscan, confirmed) -- cache the result
    (e.g. in gradio.State) and reload only on scene change, never per-query. """
    prior = _load_side(root, scene, map_subscan(scene))
    current = _load_side(root, scene, query_subscan(scene))
    return {
        'prior': prior,
        'current': current,
        'removed_ids': removed_instance_ids(prior['ids'], current['ids']),
        'added_ids': added_instance_ids(prior['ids'], current['ids']),
    }


def export_subscan_glb(side_bundle, out_path, highlight_ids=None, cam_poses=(), cam_colors=(),
                       cam_images=None, focal=None, imsize=None, cam_size=0.1):
    """
    side_bundle: one of build_scene_meshes(...)'s 'prior'/'current' dicts.
    highlight_ids: instance ids to recolor (e.g. mesh_bundle['removed_ids'] on the prior mesh), or
      None/empty for the plain textured view.
    cam_poses/cam_colors: parallel sequences of 4x4 cam_to_world and (r,g,b) tuples, drawn via
      dust3r.viz.add_scene_cam (unmodified import -- same frustum drawer mast3r/demo.py uses).
    cam_images: optional parallel sequence of PIL images to texture each frustum with (or None
      per-camera / None for the whole sequence to draw untextured frustums).
    """
    base_mesh = side_bundle['mesh']
    scene = trimesh.Scene()

    if highlight_ids:
        mesh = base_mesh.copy()
        colors = vertex_highlight_colors(side_bundle['vertex_colors'], side_bundle['object_id'], highlight_ids)
        mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=colors)
    else:
        mesh = base_mesh
    scene.add_geometry(mesh)

    if cam_images is None:
        cam_images = [None] * len(cam_poses)
    for pose, color, image in zip(cam_poses, cam_colors, cam_images):
        add_scene_cam(scene, pose, color, image=image, focal=focal, imsize=imsize, screen_width=cam_size)

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    scene.export(out_path)
    return out_path
