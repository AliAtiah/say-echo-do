# Controlled experiments: summary

## Out-of-sample performance (main world, 5 seeds)

| Strategy | IC | IC t | Sharpe/period |
|---|---:|---:|---:|
| Naive sentiment (follow tone) | -0.119 ± 0.013 | -9.9 | -1.69 |
| Naive contrarian (fade tone) | +0.119 ± 0.013 | +9.9 | +1.69 |
| Price reversal (fade reaction) | +0.162 ± 0.018 | +11.4 | +1.57 |
| Follow institutions (Say) | -0.047 ± 0.006 | -4.5 | -0.82 |
| Follow positioning (Do) | +0.035 ± 0.011 | +3.6 | +0.66 |
| News/echo split only | +0.091 ± 0.018 | +6.7 | +1.14 |
| Theory composite (unit weights) | +0.049 ± 0.010 | +4.6 | +0.83 |
| Theory composite, firm-level absorption | +0.027 ± 0.012 | +2.3 | +0.42 |
| Linear, raw voices | +0.169 ± 0.017 | +12.3 | +1.59 |
| Linear, raw + theory features | +0.171 ± 0.018 | +12.4 | +1.63 |
| Gradient boosting, raw voices | +0.109 ± 0.015 | +8.3 | +1.42 |
| Gradient boosting, raw + theory | +0.118 ± 0.019 | +9.5 | +1.61 |

## Theory tests (test half, firm-clustered t)

| Test | coef | t | predicted | seeds |t|>2 as predicted |
|---|---:|---:|---|---:|
| T1 news | +0.346 | +2.13 | + | 3/5 |
| T1 echo | -1.515 | -7.96 | − | 5/5 |
| T2 Say | Cov<0 | -0.055 | -0.29 | − | 1/5 |
| T2 Say | Cov>=0 | +0.101 | +0.39 | either | 0/5 |
| T2 Do | +0.142 | +0.88 | + | 1/5 |
| T3 absorption | -1.268 | -11.63 | either | 5/5 |
| T3 absorption | informed trading | -1.267 | -11.09 | either | 5/5 |
| T3 absorption | no informed trading | -1.245 | -3.62 | − | 4/5 |
| T3 absorption, firm-level multiplier | -0.723 | -9.01 | either | 5/5 |
| T3 firm-level | informed trading | -0.704 | -8.39 | either | 5/5 |
| T3 firm-level | no informed trading | -1.026 | -4.03 | − | 5/5 |
| T4 signed Levy area | +0.233 | +1.01 | + | 1/5 |

## False-alarm detection (AUC)

- all features: 0.934 ± 0.010
- Say-Do correlation (rolling): 0.895 ± 0.014
- Levy area: 0.627 ± 0.009
- absorption: 0.507 ± 0.006
- echo share: 0.530 ± 0.018
- Say x Do: 0.746 ± 0.012

## Composite ablation (IC when a term is removed)

- drop news: +0.061
- drop echo: +0.002
- drop do: +0.047
- drop say: +0.041
- drop absorption: +0.096
- drop levy: +0.022
- drop none: +0.049

## Say-Do switch: window length

- W=5: accuracy 0.847, t(words | fade group) -3.02
- W=10: accuracy 0.902, t(words | fade group) -3.05
- W=20: accuracy 0.934, t(words | fade group) -3.27
- W=40: accuracy 0.960, t(words | fade group) -3.21

## Robustness (3 seeds each; 'base' is the reference world on the same seeds)

| Variant | price reversal IC | linear raw | linear + theory | GBM raw | GBM + theory | T1 echo t | T1 news t | T4 Levy t | detection AUC | echo share true/est |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| main (5 seeds) | +0.162 | +0.169 | +0.171 | +0.109 | +0.118 | -8.0 | +2.1 | +1.0 | 0.934 | 0.427/0.422 |
| base | +0.184 | +0.187 | +0.190 | +0.126 | +0.124 | -7.5 | +3.6 | +3.6 | 0.941 | 0.430/0.425 |
| fat_tails | +0.134 | +0.137 | +0.141 | +0.078 | +0.080 | -8.1 | +2.4 | +0.5 | 0.924 | 0.428/0.423 |
| no_semantics | +0.184 | +0.187 | +0.189 | +0.126 | +0.126 | -6.0 | +2.6 | +3.7 | 0.940 | 0.430/0.368 |
| nonlinear_crowd | +0.090 | +0.099 | +0.101 | +0.052 | +0.059 | -5.9 | +3.2 | -0.0 | 0.942 | 0.430/0.425 |
| partial_disclosure | +0.184 | +0.183 | +0.191 | +0.129 | +0.130 | -7.5 | +4.1 | +2.9 | 0.857 | 0.430/0.425 |
| powerlaw_kernel | +0.171 | +0.176 | +0.177 | +0.113 | +0.117 | -7.9 | +2.1 | +1.9 | 0.939 | 0.429/0.417 |
| weak_semantics | +0.173 | +0.187 | +0.188 | +0.119 | +0.129 | -7.2 | +1.9 | +2.2 | 0.941 | 0.429/0.373 |

## Events per firm

- E=20: detection AUC 0.934, T1 echo t -5.4, GBM+theory IC +0.077
- E=40: detection AUC 0.940, T1 echo t -6.9, GBM+theory IC +0.084
- E=80: detection AUC 0.941, T1 echo t -9.3, GBM+theory IC +0.135

## Text experiment

Default setting, 3 seeds (mean ± sd):

| Method | analog IC | top-10 neighbours: same consequence | same topic |
|---|---:|---:|---:|
| TF-IDF (generic) | +0.192 ± 0.003 | 0.506 | 0.846 |
| Random projection (generic, 16-d) | +0.039 ± 0.016 | 0.327 | 0.622 |
| Ridge regression (supervised, one target) | -0.019 ± 0.016 | — | — |
| Gradient boosting (supervised, one target) | +0.837 ± 0.008 | — | — |
| Return-aligned embedding (16-d) | +0.892 ± 0.009 | 1.000 | 0.344 |

Neighbour certificate (Theorem 6.3) on 150 training anchors after training:

| seed | outcome noise | bandwidth scale | final loss | certified fraction | violations | analog IC |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0.3 | 3.7721 | 0.4534 | 0 | +1.000 |
| 0 | 0 | 1 | 4.9775 | 0.0000 | 0 | +0.993 |
| 0 | 0.3 | 1 | 4.9655 | 0.0000 | 0 | +0.983 |
| 0 | 1 | 0.3 | 3.8565 | 0.0000 | 0 | +0.931 |
| 0 | 1 | 1 | 4.9670 | 0.0000 | 0 | +0.885 |
| 1 | 1 | 1 | 4.9690 | 0.0000 | 0 | +0.902 |
| 2 | 1 | 1 | 4.9705 | 0.0000 | 0 | +0.891 |
