# Texture / change analysis (oracle run, all 34415 queries, PnP success 75.8%)

Median texture (Laplacian variance of grayscale query image), all frames: 20.1

## Table A: success and error statistics by texture quartile

Error statistics use PnP-successful queries with translation error <= 10 m; outliers are counted separately.

| Texture quartile | n | Success | Outliers >10 m | Median change | Median t (cm) | IQR t (cm) | Std t (cm) | Median r (deg) | IQR r (deg) | Std r (deg) |
|---|---|---|---|---|---|---|---|---|---|---|
| Q1 [0.5-11.8] | 8604 | 67.5% | 194 | 1.32% | 16.6 | 19.2 | 65.4 | 3.78 | 5.39 | 30.08 |
| Q2 [11.8-20.1] | 8603 | 76.0% | 81 | 3.04% | 15.4 | 16.8 | 65.2 | 3.09 | 3.53 | 22.33 |
| Q3 [20.1-60.7] | 8604 | 76.5% | 71 | 2.10% | 13.4 | 16.7 | 61.9 | 2.82 | 2.75 | 23.73 |
| Q4 [60.7-8452.2] | 8604 | 83.1% | 22 | 2.22% | 10.0 | 12.2 | 57.8 | 2.04 | 2.20 | 21.64 |

## Table B: success and error statistics by change share

Error statistics use PnP-successful queries with translation error <= 10 m; outliers are counted separately.

| Change share | n | Success | Outliers >10 m | Median t (cm) | IQR t (cm) | Std t (cm) | Median r (deg) | IQR r (deg) | Std r (deg) | Median texture |
|---|---|---|---|---|---|---|---|---|---|---|
| <0.1% | 13173 | 82.8% | 64 | 12.6 | 16.7 | 47.9 | 2.72 | 2.82 | 22.10 | 19.8 |
| 0.1-5% | 7081 | 85.2% | 14 | 12.2 | 13.5 | 44.5 | 2.71 | 2.66 | 19.30 | 21.9 |
| 5-20% | 6662 | 75.0% | 27 | 14.2 | 15.1 | 57.9 | 3.07 | 3.29 | 21.28 | 20.3 |
| >=20% | 7499 | 55.2% | 263 | 18.4 | 28.5 | 106.7 | 3.63 | 7.91 | 37.24 | 18.9 |
| All | 34415 | 75.8% | 368 | 13.6 | 16.5 | 62.6 | 2.87 | 3.27 | 24.52 | 20.1 |

## Table C: error buckets (per-frame translation error)

| Translation error | n | Median rot (deg) | Median change | Change >=20% | Median texture |
|---|---|---|---|---|---|
| <=5 cm | 2913 | 1.46 | 0.27% | 11.3% | 54.2 |
| 5-10 cm | 6193 | 2.10 | 0.50% | 10.0% | 29.4 |
| 10-20 cm | 8236 | 2.82 | 1.67% | 13.9% | 20.0 |
| 20-50 cm | 5997 | 4.68 | 0.66% | 16.4% | 17.1 |
| 50 cm-1 m | 1111 | 21.41 | 4.93% | 28.5% | 18.2 |
| >1 m | 1626 | 114.17 | 13.57% | 45.9% | 16.9 |
| PnP failed | 8336 | - | 11.87% | 40.3% | 16.1 |
