# New file, part of the RIO10 wrapper. Does not modify any existing mast3r/dust3r file.
#
# Interactive Gradio debugging tool: pick a scene and a query image (optionally jumping straight
# to a known failure/high-change case from the completed oracle run), choose oracle or learned
# retrieval, and see -- for that one query -- its retrieved neighbors, the MASt3R dense matches
# against each neighbor (color-coded inlier/outlier once PnP has run), the estimated vs.
# ground-truth pose, and the prior vs. current 3D mesh side by side with changed objects
# highlighted. UI wiring only -- all matching/mesh/plotting logic lives in the sibling
# viz_*.py modules; this file does not implement any of it itself.
#
# Launch (same env/PYTHONPATH convention as run_visloc_rio10.py):
#   source ace-g/helper_scripts/setup.sh
#   export PYTHONPATH="$MAST3R_ROOT:$MAST3R_ROOT/dust3r:$MAST3R_ROOT/rio10_wrapper:$PYTHONPATH"
#   python viz_app.py --root <rio10_root> --weights <ckpt.pth> --local_network --server_port 7860
# Then from your machine: ssh -L 7860:<node-hostname>:7860 <cluster-login>, browse localhost:7860.
import argparse
import os
import tempfile

import gradio as gr
import numpy as np

from mast3r.model import AsymmetricMASt3R
from dust3r_visloc.evaluation import get_pose_error

from rio10_dataset import VislocRIO10, map_subscan, query_subscan, list_frame_ids, color_path
import pairs_oracle
import pairs_retrieval
import viz_query_index as qidx
import viz_matching as vmatch
import viz_mesh as vmesh
import viz_match_plot as vplot

SCENES = [f'scene{i:02d}' for i in range(1, 11)]


def get_args_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True, help='path to rio10 dataset dir')
    p.add_argument('--weights', required=True, help='path to the base MASt3R checkpoint')
    p.add_argument('--retrieval_model', default=None,
                   help='path to *_retrieval_trainingfree.pth, needed to build retrieval pairs '
                   'on demand for scenes other than scene01')
    p.add_argument('--device', default='cuda')
    p.add_argument('--output_dir', default='results/oracle',
                   help='bulk-run output dir to enrich the query picker with (raw_errors.csv)')
    p.add_argument('--pairs_dir', default='results',
                   help='root under which precomputed/on-demand pairs files are read/written')
    p.add_argument('--texture_change_csv', default='analysis/texture_change_features.csv')
    p.add_argument('--local_network', action='store_true', default=False)
    p.add_argument('--server_name', default=None)
    p.add_argument('--server_port', type=int, default=None)
    p.add_argument('--share', action='store_true', default=False)
    return p


def oracle_pairs_path(pairs_dir, scene):
    return os.path.join(pairs_dir, 'oracle', 'pairs', f'{scene}_oracle_pairs.txt')


def retrieval_pairs_path(pairs_dir, scene):
    # scene01's existing precomputed file lives at results/pairs/scene01_retrieval_pairs.txt;
    # on-demand builds for other scenes are written to the same convention.
    return os.path.join(pairs_dir, 'pairs', f'{scene}_retrieval_pairs.txt')


def ensure_oracle_pairs(root, pairs_dir, scene, topk):
    path = oracle_pairs_path(pairs_dir, scene)
    if not os.path.isfile(path):
        query_ids, pairs = pairs_oracle.build_oracle_pairs(root, scene, topk)
        pairs_oracle.write_pairs_file(path, query_ids, pairs)
    return path


def build_app(args, model):
    tmpdir = tempfile.mkdtemp(prefix='rio10_viz_')
    fast_nn_params = dict(device=args.device, dist='dot', block_size=2**13)
    texture_change = qidx.load_texture_change_features(args.texture_change_csv)

    def load_scene(scene, pairs_mode, topk, processing_level):
        """
        'scene' level: unchanged original behavior -- a full scene-wide pairs file (oracle:
        built on demand, cheap; retrieval: must already exist or be built via the button) drives
        VislocRIO10 and every query's neighbors are precomputed up front.

        'image' level: VislocRIO10 is built with pairs_file=None (query_ids listed directly, no
        neighbor precomputation at all). Oracle neighbors are computed per-image on demand in
        on_run_matching using map_ids/map_centers/map_forwards cached here once per scene (cheap:
        pure pose arithmetic). Retrieval neighbors need a database index built once per scene
        (unavoidable -- a retrieval index needs a database to search) via the same "Build
        retrieval index" button, but that button now builds ONLY the map-side index here (no
        query-side extraction at all, no pairs file), cached into the returned bundle and reused
        for however many individual images the user picks.
        """
        if processing_level == 'scene':
            if pairs_mode == 'oracle':
                pairs_path = ensure_oracle_pairs(args.root, args.pairs_dir, scene, topk)
            else:
                pairs_path = retrieval_pairs_path(args.pairs_dir, scene)
                if not os.path.isfile(pairs_path):
                    return None, f'No retrieval pairs for {scene} yet -- click "Build retrieval index".'
            dataset = VislocRIO10(args.root, scene, pairs_path, topk=topk)
        else:
            dataset = VislocRIO10(args.root, scene, pairs_file=None, topk=topk)
        dataset.set_resolution(model)

        mesh_bundle = vmesh.build_scene_meshes(args.root, scene)
        table = qidx.build_query_table(scene, dataset.query_ids, args.output_dir, texture_change)
        bundle = {'dataset': dataset, 'mesh_bundle': mesh_bundle, 'table': table,
                 'processing_level': processing_level, 'retrieval_index': None}

        if processing_level == 'image':
            map_ids = list_frame_ids(args.root, scene, map_subscan(scene))
            map_centers, map_forwards = pairs_oracle.load_all_centers_forwards(args.root, map_ids)
            bundle.update(map_ids=map_ids, map_centers=map_centers, map_forwards=map_forwards)

        return bundle, f'{scene}: {len(dataset)} queries loaded ({processing_level}-level).'

    def _retrieval_build_needed(bundle, pairs_mode, processing_level):
        if pairs_mode != 'retrieval':
            return False
        if processing_level == 'scene':
            return bundle is None  # load_scene already returned None iff no pairs file exists yet
        return bundle is not None and bundle.get('retrieval_index') is None

    def on_scene_or_mode_change(scene, pairs_mode, topk, processing_level):
        bundle, msg = load_scene(scene, pairs_mode, topk, processing_level)
        build_visible = _retrieval_build_needed(bundle, pairs_mode, processing_level)
        rows = _table_rows(bundle, None, None, None, None, None, None, 'transl_err_m', True)
        return bundle, msg, gr.update(visible=build_visible), rows, None, None, None

    def on_build_retrieval(scene, topk, processing_level, progress=gr.Progress()):
        if not args.retrieval_model:
            return None, 'No --retrieval_model configured at launch; cannot build on demand.', gr.update(visible=True), []
        if processing_level == 'scene':
            progress(0, desc=f'Building ASMK retrieval index + pairs for {scene} (~1-3 min)...')
            query_ids, pairs = pairs_retrieval.build_retrieval_pairs(
                args.root, scene, args.retrieval_model, args.weights, topk, device=args.device,
                backbone=model)  # reuse the loaded model -- a second full instance OOM-killed this
            pairs_retrieval.write_pairs_file(retrieval_pairs_path(args.pairs_dir, scene), query_ids, pairs)
            bundle, msg = load_scene(scene, 'retrieval', topk, 'scene')
        else:
            progress(0, desc=f'Building ASMK database index for {scene} (map images only, faster)...')
            index = pairs_retrieval.build_database_index(
                args.root, scene, args.retrieval_model, args.weights, device=args.device, backbone=model)
            bundle, msg = load_scene(scene, 'retrieval', topk, 'image')
            bundle['retrieval_index'] = index
            msg += f" Index built from {len(index['map_ids'])} map images."
        rows = _table_rows(bundle, None, None, None, None, None, None, 'transl_err_m', True)
        return bundle, msg, gr.update(visible=False), rows

    def _table_rows(bundle, success_filter, min_t, min_chg, min_tex, max_t_err, max_ang_err,
                    sort_by, descending):
        if bundle is None:
            return []
        success_f = None if success_filter in (None, 'any') else success_filter
        rows = qidx.filter_and_sort(bundle['table'], success_filter=success_f,
                                    min_transl_err=min_t or None, min_change_frac=min_chg or None,
                                    min_texture=min_tex or None, max_transl_err=max_t_err or None,
                                    max_angular_err=max_ang_err or None, sort_by=sort_by,
                                    descending=descending)
        def _r(v, n):
            return None if v is None else round(v, n)
        return [[r['image_name'], r['success'], _r(r['transl_err_m'], 3), _r(r['angular_err_deg'], 2),
                _r(r['texture_lapvar'], 1), _r(r['change_frac'], 3)] for r in rows]

    def on_filter_change(bundle, success_filter, min_t, min_chg, min_tex, max_t_err, max_ang_err,
                         sort_by, descending):
        return _table_rows(bundle, success_filter, min_t, min_chg, min_tex, max_t_err, max_ang_err,
                           sort_by, descending)

    def on_row_select(bundle, evt: gr.SelectData):
        if bundle is None:
            return None, None, ''
        qid = evt.row_value[0]
        view = bundle['dataset']._load_view(qid)
        errs = qidx.load_raw_errors(args.output_dir, bundle['dataset'].scene).get(qid)
        tex, chg = texture_change.get((bundle['dataset'].scene, qid), (None, None))
        stats = f"**{qid}**\n\n"
        if errs:
            stats += f"GT-compare error (debug only): {errs[0]*100:.1f} cm, {errs[1]:.2f} deg\n\n"
        stats += f"texture={tex if tex is None else round(tex,1)}  change_frac={chg if chg is None else round(chg,3)}"
        return qid, np.array(view['rgb']), stats

    def on_run_matching(bundle, qid, pairs_mode, topk, point_conf_thr, pixel_tol, seed_stride):
        if bundle is None or qid is None:
            return None, [], 'Pick a scene and a query first.'
        dataset = bundle['dataset']

        if bundle['processing_level'] == 'scene':
            idx = dataset.query_ids.index(qid)
            views = dataset[idx]
            query_view, map_views = views[0], views[1:]
        else:
            query_view = dataset._load_view(qid)
            if pairs_mode == 'oracle':
                qcenter, qforward = pairs_oracle.load_center_and_forward(args.root, qid)
                neighbor_ids = pairs_oracle.rank_single_query(
                    qcenter, qforward, bundle['map_ids'], bundle['map_centers'], bundle['map_forwards'],
                    int(topk))
            else:
                if bundle.get('retrieval_index') is None:
                    return None, [], 'Retrieval index not built yet -- click "Build retrieval index".'
                neighbor_ids = pairs_retrieval.query_single_image(
                    bundle['retrieval_index'], color_path(args.root, qid), int(topk))
            map_views = [dataset._load_view(mid) for mid in neighbor_ids]

        neighbor_matches = vmatch.match_query_against_neighbors(
            query_view, map_views, model, args.device, fast_nn_params,
            point_conf_thr=point_conf_thr, pixel_tol=pixel_tol, seed_stride=int(seed_stride))
        world_pts, mq, mm, mc, nidx = vmatch.concat_neighbor_matches(neighbor_matches)
        match_state = {
            'query_view': query_view, 'neighbor_matches': neighbor_matches,
            'world_pts': world_pts, 'm_query': mq, 'm_map': mm, 'm_confs': mc, 'neighbor_idx': nidx,
        }
        gallery = [(np.array(nm['map_view']['rgb']),
                   f"#{i} {nm['map_view']['image_name'].split('/')[-1]} ({len(nm['matches_confs'])} matches)")
                  for i, nm in enumerate(neighbor_matches)]
        return match_state, gallery, f'Matched against {len(map_views)} neighbors, {len(mc)} total correspondences.'

    def on_pnp_change(match_state, confidence_threshold, pnp_mode, reprojection_error, pnp_max_points):
        if match_state is None:
            return None, ''
        pnp = vmatch.run_pnp_from_matches(
            match_state['world_pts'], match_state['m_query'], match_state['m_map'], match_state['m_confs'],
            match_state['query_view'], confidence_threshold=confidence_threshold, pnp_mode=pnp_mode,
            reprojection_error=reprojection_error, reprojection_error_diag_ratio=None,
            pnp_max_points=int(pnp_max_points), rng_seed=hash(match_state['query_view']['image_name']) & 0xffff)
        msg = f"PnP solver: {'success' if pnp['success'] else 'FAILED'}"
        if pnp['success']:
            inliers = vmatch.classify_inliers(pnp['pose'], pnp['query_pts2d_undist'], pnp['query_pts3d'],
                                              match_state['query_view']['intrinsics'], pnp['reprojection_error_used'])
            ratio = float(inliers.mean()) if len(inliers) else 0.0
            degenerate = ratio < 0.1
            t_err, a_err = get_pose_error(pnp['pose'], match_state['query_view']['cam_to_world'])
            msg += (f", inlier ratio {ratio:.2f} ({inliers.sum()}/{len(inliers)})"
                   f"{' [DEGENERATE: low inlier ratio]' if degenerate else ''}\n\n"
                   f"GT-compare error (debug only): {float(t_err)*100:.1f} cm, {float(a_err):.2f} deg")
            pnp['inliers'] = inliers
        return pnp, msg

    def _inlier_mask_for_neighbor(match_state, pnp_state, i):
        """
        match_state['neighbor_idx'] groups the concatenated correspondence arrays by neighbor in
        order (neighbor 0's matches first, then neighbor 1's, ...), so neighbor i's matches occupy
        one contiguous slice [start:end) of those arrays -- found via searchsorted. Maps
        pnp_state['used_match_indices']/'inliers' (indices into + values over those SAME
        concatenated arrays) back onto that slice. Matches never passed to PnP (filtered out by
        the confidence threshold, or dropped by the pnp_max_points random subsample) are treated
        as outliers for coloring purposes -- a deliberate simplification, since plot_matches only
        supports a 2-color (inlier/outlier) scheme, not a 3rd "not evaluated" state.
        Returns None if PnP hasn't been run yet (match_state has no pnp_state, or it failed).
        """
        if pnp_state is None or not pnp_state.get('success') or 'inliers' not in pnp_state:
            return None
        n_total = len(match_state['neighbor_idx'])
        global_inlier = np.zeros(n_total, dtype=bool)
        global_inlier[pnp_state['used_match_indices']] = pnp_state['inliers']
        start = int(np.searchsorted(match_state['neighbor_idx'], i, side='left'))
        end = int(np.searchsorted(match_state['neighbor_idx'], i, side='right'))
        return global_inlier[start:end]

    def on_neighbor_select(match_state, pnp_state, evt: gr.SelectData):
        if match_state is None:
            return None
        i = evt.index
        nm = match_state['neighbor_matches'][i]
        inlier_mask_i = _inlier_mask_for_neighbor(match_state, pnp_state, i)
        return vplot.plot_matches(match_state['query_view']['rgb'], nm['map_view']['rgb'],
                                  nm['matches_im_query'], nm['matches_im_map'],
                                  inlier_mask=inlier_mask_i, matches_confs=nm['matches_confs'],
                                  title=f"neighbor #{i}")

    def on_mesh_refresh(bundle, qid, match_state, pnp_state, highlight, show_query_on_prior, cam_size):
        if bundle is None:
            return None, None
        mesh_bundle = bundle['mesh_bundle']
        dataset = bundle['dataset']
        prior_cams, prior_colors, prior_imgs = [], [], []
        current_cams, current_colors, current_imgs = [], [], []
        if qid is not None:
            qview = dataset._load_view(qid)
            current_cams.append(qview['cam_to_world']); current_colors.append((0, 0, 255)); current_imgs.append(None)
            if show_query_on_prior:
                prior_cams.append(qview['cam_to_world']); prior_colors.append((0, 0, 255)); prior_imgs.append(None)
        if match_state is not None:
            for i, nm in enumerate(match_state['neighbor_matches']):
                c = vmesh.CAM_COLORS[i % len(vmesh.CAM_COLORS)]
                prior_cams.append(nm['map_view']['cam_to_world']); prior_colors.append(c); prior_imgs.append(None)
        if pnp_state is not None and pnp_state.get('success'):
            current_cams.append(pnp_state['pose']); current_colors.append((255, 0, 0)); current_imgs.append(None)

        prior_path = os.path.join(tmpdir, 'prior.glb')
        current_path = os.path.join(tmpdir, 'current.glb')
        vmesh.export_subscan_glb(mesh_bundle['prior'], prior_path,
                                 highlight_ids=mesh_bundle['removed_ids'] if highlight else None,
                                 cam_poses=prior_cams, cam_colors=prior_colors, cam_images=prior_imgs,
                                 cam_size=cam_size)
        vmesh.export_subscan_glb(mesh_bundle['current'], current_path,
                                 highlight_ids=mesh_bundle['added_ids'] if highlight else None,
                                 cam_poses=current_cams, cam_colors=current_colors, cam_images=current_imgs,
                                 cam_size=cam_size)
        return prior_path, current_path

    with gr.Blocks(title='RIO10 Relocalization Debugger') as demo:
        scene_state = gr.State(None)
        query_state = gr.State(None)
        match_state = gr.State(None)
        pnp_state = gr.State(None)

        with gr.Row():
            scene_dd = gr.Dropdown(SCENES, value='scene01', label='Scene')
            pairs_mode_dd = gr.Radio(['oracle', 'retrieval'], value='oracle', label='Retrieval mode')
            processing_level_radio = gr.Radio(['scene', 'image'], value='scene', label='Processing level')
            topk_slider = gr.Slider(1, 10, value=10, step=1, label='topk')
            scene_msg = gr.Markdown()
        build_retrieval_btn = gr.Button('Build retrieval index for this scene (~1-3 min)', visible=False)

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown('### Query picker')
                with gr.Row():
                    success_filter_dd = gr.Dropdown(['any', 'success', 'failed'], value='any', label='PnP status')
                    min_t_num = gr.Number(label='min transl err (m)', value=None)
                    min_chg_num = gr.Number(label='min change_frac', value=None)
                    min_tex_num = gr.Number(label='min texture', value=None)
                    sort_by_dd = gr.Dropdown(['transl_err_m', 'angular_err_deg', 'change_frac', 'texture_lapvar'],
                                             value='transl_err_m', label='sort by')
                with gr.Row():
                    max_t_err_num = gr.Number(
                        label='max transl err for "success" (m)', value=None,
                        info='Leave blank for pure solver success/fail. Set this (and/or the angular '
                        'threshold) to also count a solver-"success" pose as a failure once its GT '
                        'error exceeds the bound -- catches degenerate/inaccurate poses.')
                    max_ang_err_num = gr.Number(label='max angular err for "success" (deg)', value=None)
                query_table = gr.Dataframe(
                    headers=['image_name', 'success', 'transl_err_m', 'angular_err_deg', 'texture', 'change_frac'],
                    interactive=False)
            with gr.Column(scale=1):
                query_preview = gr.Image(label='Query image')
                query_stats_md = gr.Markdown()

        with gr.Row():
            point_conf_slider = gr.Slider(1.0, 5.0, value=1.5, step=0.05, label='point_conf_threshold')
            pixel_tol_slider = gr.Slider(0, 20, value=5, step=1, label='pixel_tol')
            seed_stride_slider = gr.Slider(1, 32, value=8, step=1, label='seed_stride')
            run_match_btn = gr.Button('Run matching', variant='primary')
        match_msg = gr.Markdown()

        with gr.Row():
            confidence_slider = gr.Slider(1.0, 1.1, value=1.001, step=0.001, label='confidence_threshold (desc_conf)')
            pnp_mode_dd = gr.Dropdown(['poselib', 'cv2', 'pycolmap'], value='poselib', label='pnp_mode')
            reproj_err_slider = gr.Slider(0.5, 20, value=5.0, step=0.5, label='reprojection_error (px)')
            pnp_max_points_slider = gr.Slider(500, 20000, value=5000, step=500, label='pnp_max_points')
        pnp_msg = gr.Markdown()

        with gr.Row():
            neighbor_gallery = gr.Gallery(label='Retrieved neighbors (rank, inlier-aware once PnP runs)', columns=5)
            match_plot_img = gr.Image(label='Query <-> neighbor matches (green=inlier, red=outlier)')

        with gr.Row():
            highlight_cb = gr.Checkbox(value=True, label='Highlight changed-instance vertices')
            show_query_on_prior_cb = gr.Checkbox(value=True, label='Also show query pose on prior mesh')
            cam_size_slider = gr.Slider(0.05, 1.0, value=0.3, step=0.05, label='Camera frustum size (m)')
        with gr.Row():
            prior_model3d = gr.Model3D(label='Prior (mapping scan)', height=600)
            current_model3d = gr.Model3D(label='Current (rescan / query)', height=600)

        scene_inputs = [scene_dd, pairs_mode_dd, topk_slider, processing_level_radio]
        scene_or_mode_outputs = [scene_state, scene_msg, build_retrieval_btn, query_table,
                                 prior_model3d, current_model3d, query_state]
        for trigger in [scene_dd.change, pairs_mode_dd.change, topk_slider.change,
                       processing_level_radio.change]:
            trigger(on_scene_or_mode_change, inputs=scene_inputs, outputs=scene_or_mode_outputs)

        build_retrieval_btn.click(on_build_retrieval,
                                  inputs=[scene_dd, topk_slider, processing_level_radio],
                                  outputs=[scene_state, scene_msg, build_retrieval_btn, query_table])

        filter_inputs = [scene_state, success_filter_dd, min_t_num, min_chg_num, min_tex_num,
                        max_t_err_num, max_ang_err_num, sort_by_dd]
        for trigger in [success_filter_dd.change, min_t_num.change, min_chg_num.change,
                       min_tex_num.change, max_t_err_num.change, max_ang_err_num.change,
                       sort_by_dd.change]:
            trigger(lambda *a: on_filter_change(*a, True), inputs=filter_inputs, outputs=query_table)

        query_table.select(on_row_select, inputs=[scene_state], outputs=[query_state, query_preview, query_stats_md])

        run_match_btn.click(on_run_matching,
                            inputs=[scene_state, query_state, pairs_mode_dd, topk_slider,
                                   point_conf_slider, pixel_tol_slider, seed_stride_slider],
                            outputs=[match_state, neighbor_gallery, match_msg])

        pnp_inputs = [match_state, confidence_slider, pnp_mode_dd, reproj_err_slider, pnp_max_points_slider]
        for trigger in [confidence_slider.change, pnp_mode_dd.change, reproj_err_slider.change,
                       pnp_max_points_slider.change, match_state.change]:
            trigger(on_pnp_change, inputs=pnp_inputs, outputs=[pnp_state, pnp_msg])

        neighbor_gallery.select(on_neighbor_select, inputs=[match_state, pnp_state], outputs=match_plot_img)

        mesh_inputs = [scene_state, query_state, match_state, pnp_state, highlight_cb,
                      show_query_on_prior_cb, cam_size_slider]
        for trigger in [query_state.change, match_state.change, pnp_state.change,
                       highlight_cb.change, show_query_on_prior_cb.change, cam_size_slider.change]:
            trigger(on_mesh_refresh, inputs=mesh_inputs, outputs=[prior_model3d, current_model3d])

    return demo


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    model = AsymmetricMASt3R.from_pretrained(args.weights).to(args.device)
    demo = build_app(args, model)
    server_name = args.server_name if args.server_name is not None else ('0.0.0.0' if args.local_network else '127.0.0.1')
    demo.launch(share=args.share, server_name=server_name, server_port=args.server_port)
