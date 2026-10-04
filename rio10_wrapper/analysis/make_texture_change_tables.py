# Builds the success/texture/change/error tables from texture_change_features.csv and the
# oracle run's per-scene raw_errors.csv files. Usage: python make_texture_change_tables.py <results/oracle dir>
import csv, sys, glob, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
err = {}
for f in glob.glob(os.path.join(sys.argv[1], 'scene*_raw_errors.csv')):
    for r in csv.DictReader(open(f)):
        err[r['image_name']] = (float(r['transl_err_m']), float(r['angular_err_deg']))
rows = list(csv.DictReader(open(os.path.join(HERE, 'texture_change_features.csv'))))
t = np.array([err[r['query']][0] for r in rows]); a = np.array([err[r['query']][1] for r in rows])
ch = np.array([float(r['change_frac']) for r in rows]); tex = np.array([float(r['texture_lapvar']) for r in rows])
ok = np.isfinite(t); inl = ok & (t <= 10)
print(f'# Texture / change analysis (oracle run, all {len(t)} queries, PnP success {100*ok.mean():.1f}%)\n')
print('## Definitions\n')
print('- **Texture** = variance of the Laplacian (scipy.ndimage.laplace) of the grayscale query image (540x960 W x H RGB, converted with PIL "L"). Higher = more high-frequency detail; lower = smoother, less textured content.')
print('- **Change share** = for each query, the fraction of its labelled pixels (instance ID > 0 in the query `*.instances.png`) whose instance ID does not appear in any of its 10 oracle map frames (`*.instances.png` of the mapping subscan). ID 0 (unlabelled) is excluded from both numerator and denominator.')
print('- **PnP success** = poselib returned a finite pose. **Outliers >10 m** = successful PnP poses with translation error above 10 m (degenerate solutions). Error statistics exclude these.\n')
print(f'Median texture (Laplacian variance of grayscale query image), all frames: {np.median(tex):.1f}\n')
print('## Table A: success and error statistics by texture quartile\n')
print('Error statistics use PnP-successful queries with translation error <= 10 m; outliers are counted separately.\n')
print('| Texture quartile | n | Success | Outliers >10 m | Median change | Median t (cm) | IQR t (cm) | Std t (cm) | Median r (deg) | IQR r (deg) | Std r (deg) |')
print('|---|---|---|---|---|---|---|---|---|---|---|')
qs = np.quantile(tex, [0, .25, .5, .75, 1])
for i in range(4):
    g = (tex >= qs[i]) & (tex <= qs[i+1]) if i == 3 else (tex >= qs[i]) & (tex < qs[i+1])
    il = g & inl; out = (g & ok & ~inl).sum()
    tt = t[il]*100; aa = a[il]
    print(f'| Q{i+1} [{qs[i]:.1f}-{qs[i+1]:.1f}] | {g.sum()} | {100*ok[g].mean():.1f}% | {out} | {100*np.median(ch[g]):.2f}% | {np.median(tt):.1f} | {np.subtract(*np.percentile(tt,[75,25])):.1f} | {np.std(tt):.1f} | {np.median(aa):.2f} | {np.subtract(*np.percentile(aa,[75,25])):.2f} | {np.std(aa):.2f} |')
print('\n## Table B: success and error statistics by change share\n')
print('Error statistics use PnP-successful queries with translation error <= 10 m; outliers are counted separately.\n')
print('| Change share | n | Success | Outliers >10 m | Median t (cm) | IQR t (cm) | Std t (cm) | Median r (deg) | IQR r (deg) | Std r (deg) | Median texture |')
print('|---|---|---|---|---|---|---|---|---|---|---|')
for lo, hi, name in [(0, .001, '<0.1%'), (.001, .05, '0.1-5%'), (.05, .2, '5-20%'), (.2, 1.01, '>=20%')]:
    g = (ch >= lo) & (ch < hi); i = g & inl; out = (g & ok & ~inl).sum()
    tt = t[i]*100; aa = a[i]
    print(f'| {name} | {g.sum()} | {100*ok[g].mean():.1f}% | {out} | {np.median(tt):.1f} | {np.subtract(*np.percentile(tt,[75,25])):.1f} | {np.std(tt):.1f} | {np.median(aa):.2f} | {np.subtract(*np.percentile(aa,[75,25])):.2f} | {np.std(aa):.2f} | {np.median(tex[g]):.1f} |')
tt = t[inl]*100; aa = a[inl]
print(f'| All | {len(t)} | {100*ok.mean():.1f}% | {(ok & ~inl).sum()} | {np.median(tt):.1f} | {np.subtract(*np.percentile(tt,[75,25])):.1f} | {np.std(tt):.1f} | {np.median(aa):.2f} | {np.subtract(*np.percentile(aa,[75,25])):.2f} | {np.std(aa):.2f} | {np.median(tex):.1f} |')
print('\n## Table C: error buckets (per-frame translation error)\n')
print('| Translation error | n | Median rot (deg) | Median change | Change >=20% | Median texture |\n|---|---|---|---|---|---|')
for name, lo, hi in [('<=5 cm', 0, .05), ('5-10 cm', .05, .10), ('10-20 cm', .10, .20), ('20-50 cm', .20, .50), ('50 cm-1 m', .50, 1.0), ('>1 m', 1.0, 1e9)]:
    m = ok & (t > lo) & (t <= hi)
    print(f'| {name} | {m.sum()} | {np.median(a[m]):.2f} | {100*np.median(ch[m]):.2f}% | {100*np.mean(ch[m] >= .2):.1f}% | {np.median(tex[m]):.1f} |')
m = ~ok
print(f'| PnP failed | {m.sum()} | - | {100*np.median(ch[m]):.2f}% | {100*np.mean(ch[m] >= .2):.1f}% | {np.median(tex[m]):.1f} |')
