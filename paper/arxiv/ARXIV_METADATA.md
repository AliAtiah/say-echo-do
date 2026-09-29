# arXiv submission metadata

Paste these into the arXiv submission form. The PDF keeps its longer abstract;
the form field is limited to 1,920 characters (this version: 1908).

## Title

Say, Echo, Do: Strategic Narratives and Revealed Positioning in Financial Markets

## Authors

[Author Name]

## Abstract

Stock prices sometimes fall on bad news by far less than the news seems to warrant, and then recover, while later disclosures show that someone was buying all along. We ask when such "false alarms" are an equilibrium outcome rather than an accident, and what an outside observer can learn from them. We model a market with three voices: institutions that speak, a media that repeats them, and capital whose positions are eventually revealed. When an informed institution trades while addressing a partly credulous crowd, talking the asset down while buying it turns out to be optimal in a narrow but explicit region, $\varphi^2<2\lambda k<\varphi$, where $\varphi$ measures credulity, $\lambda$ price impact and k the cost of misreporting. Outside that region the same institution shades the truth, or exaggerates and sells. An identity that holds for any distribution of fundamentals links the predictive content of institutional statements to their covariance with institutional trades, so the sign of an observable covariance tells us whether to follow words or fade them. Media amplification decides which kind of manipulation appears, and when the crowd learns whom to trust, credibility can cycle and even become chaotic. On the measurement side, we show how to separate news from repetition exactly using both the timing and the meaning of articles, how to align text embeddings with market outcomes with a guarantee that survives imperfect training, and how to tell whether money moved before the narrative did. In a simulated market with known truth, these tools separate echo from news, confirm that fading the echo pays, and, given labelled examples, flag false-alarm institutions with an AUC of 0.93. They also show that absorption is only as reliable as the narrative model it subtracts, that lead–lag is weak, and that the theory adds little to a well-fitted linear forecast. Code is available.

## Comments

37 pages, 21 figures, 5 tables. Includes controlled experiments in a simulated market. Code: https://github.com/AliAtiah/say-echo-do

## Categories

- Primary: q-fin.TR (Trading and Market Microstructure)
- Cross-lists: q-fin.MF (Mathematical Finance), econ.TH (Theoretical Economics)

## MSC / JEL (optional fields)

- MSC 2020: 91G15, 91B44, 60G55, 62M20
- JEL: G12, G14, G41, C45, C58

## Upload

Upload `say_echo_do_arxiv.tex` alone: it is self-contained (bibliography and all
figures are inside the file), and arXiv compiles it with pdfLaTeX.
