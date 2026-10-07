# New file, part of the RIO10 wrapper. Does not modify any existing mast3r/dust3r file.
#
# 2D correspondence-line visualization between a query image and one map neighbor, with
# inlier/outlier coloring. Adapts the existing inline pad+hconcat+pl.plot(...) idiom from
# mast3r/visloc.py's debug viz_matches blocks and mast3r/colmap/database.py's
# convert_im_matches_pairs (viz branch), factored into a standalone, reusable function -- neither
# of those is importable as-is (both are embedded in larger functions with other side effects),
# and neither currently colors matches by inlier/outlier status (only by match index via a jet
# colormap), which this adds.
import numpy as np
from PIL import Image


def plot_matches(query_rgb, map_rgb, matches_im_query, matches_im_map, inlier_mask=None,
                 matches_confs=None, max_lines=300, title=None):
    """
    query_rgb, map_rgb: PIL Images (original resolution, as returned by VislocRIO10's view dicts).
    matches_im_query, matches_im_map: Nx2 arrays of matched pixel coordinates in each image.
    inlier_mask: optional bool array of length N. When given, inliers are drawn green and
      outliers red. When None (no PnP run yet), falls back to a jet colormap by match index
      (or by matches_confs rank, if given) -- same convention as the existing inline code.
    Returns a PIL.Image (the two images side by side with match lines drawn across them).
    """
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as pl

    imgs = [np.array(query_rgb), np.array(map_rgb)]
    H0, W0 = imgs[0].shape[:2]
    H1, W1 = imgs[1].shape[:2]
    img0 = np.pad(imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    img1 = np.pad(imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    canvas = np.concatenate((img0, img1), axis=1)

    n = len(matches_im_query)
    if n > max_lines:
        if matches_confs is not None:
            idx = np.argsort(-matches_confs)[:max_lines]
        else:
            idx = np.round(np.linspace(0, n - 1, max_lines)).astype(int)
    else:
        idx = np.arange(n)

    fig = pl.figure(figsize=(canvas.shape[1] / 100, canvas.shape[0] / 100), dpi=100)
    pl.imshow(canvas)
    pl.axis('off')
    if title:
        pl.title(title)

    if inlier_mask is not None:
        for i in idx:
            color = 'lime' if inlier_mask[i] else 'red'
            x0, y0 = matches_im_query[i]
            x1, y1 = matches_im_map[i]
            pl.plot([x0, x1 + W0], [y0, y1], '-+', color=color, linewidth=0.6, markersize=3,
                    scalex=False, scaley=False)
    else:
        cmap = pl.get_cmap('jet')
        for j, i in enumerate(idx):
            x0, y0 = matches_im_query[i]
            x1, y1 = matches_im_map[i]
            color = cmap(j / max(len(idx) - 1, 1))
            pl.plot([x0, x1 + W0], [y0, y1], '-+', color=color, linewidth=0.6, markersize=3,
                    scalex=False, scaley=False)

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    out = Image.fromarray(buf).convert('RGB')
    pl.close(fig)
    return out
