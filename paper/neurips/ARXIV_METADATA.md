# arXiv submission metadata (NeurIPS-style edition)

Upload `arxiv_upload.zip` (it contains `say_echo_do_neurips.tex` and `neurips_2026.sty`); arXiv compiles it with pdfLaTeX.
Upload only one edition of this paper to arXiv (this one or `paper/arxiv`), not both.

## Title

Say, Echo, Do: Strategic Narratives and Revealed Positioning in Financial Markets

## Authors

Ali Atiah Alzahrani

## Abstract (1565 characters; limit 1,920)

Machine-learning signals built from financial text treat what institutions say, and what the media repeat, as evidence about value. But whoever shapes a narrative may be trading against it. We study markets with three observable voices: institutional statements (Say), media repetition (Echo) and revealed positioning (Do). We ask when words should be followed and when they should be faded. In a linear–quadratic model of an informed institution that speaks and trades before a partly credulous crowd, talking an asset down while buying it is optimal exactly when $\varphi^2<2\lambda k<\varphi$. A distribution-free identity then shows that when the observable Say–Do covariance is negative, words carry negative predictive content and should be faded. For measurement, we derive (i) an exact factorised posterior over which articles are echoes, combining arrival times with embedding similarity; (ii) a return-aligned contrastive objective that attains its bound exactly when squared embedding distances are an increasing affine function of squared outcome distances, with the tightest loss-based certificate of which neighbour rankings survive imperfect training; and (iii) a path-signature statistic for who moved first. In a controlled market with known ground truth, echo sentiment predicts returns with a significantly negative sign in all 29 simulated markets, the rolling Say–Do correlation flags false-alarm events with an AUC of 0.90, and return-aligned embeddings organise headlines by consequence rather than topic. We also report where the tools fail.

## Comments

First draft. 9 pages main text, 31 pages total, 3 figures in the main text. Code: https://github.com/AliAtiah/say-echo-do

## Categories

- Primary: q-fin.TR (Trading and Market Microstructure)
- Cross-lists: cs.LG (Machine Learning), q-fin.MF (Mathematical Finance)

## MSC / JEL (optional fields)

- MSC 2020: 91G15, 91B44, 60G55, 62M20
- JEL: G12, G14, G41, C45, C58
