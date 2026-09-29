# Build every paper version.  Requires a TeX Live installation with pdflatex and bibtex.
LATEX = pdflatex -interaction=nonstopmode -halt-on-error

.PHONY: main all dark neurips journal icml conference arxiv explainer report clean

# Default target: the main paper.
main: arxiv

all: arxiv dark neurips journal icml conference explainer report

dark:
	cd arxiv-dark && $(LATEX) say_echo_do_arxiv_dark && $(LATEX) say_echo_do_arxiv_dark && $(LATEX) say_echo_do_arxiv_dark

# NeurIPS 2026 style, non-anonymous preprint
neurips:
	cd neurips && $(LATEX) say_echo_do_neurips && $(LATEX) say_echo_do_neurips && $(LATEX) say_echo_do_neurips

journal:
	cp ../figures/fig_declustering.pdf journal/ 2>/dev/null || true
	cd journal && $(LATEX) say_echo_do_journal && bibtex say_echo_do_journal && $(LATEX) say_echo_do_journal && $(LATEX) say_echo_do_journal

icml:
	cd icml && $(LATEX) say_echo_do_icml && bibtex say_echo_do_icml && $(LATEX) say_echo_do_icml && $(LATEX) say_echo_do_icml

conference:
	cd conference && $(LATEX) say_echo_do_conference && $(LATEX) say_echo_do_conference

arxiv:
	cd arxiv && $(LATEX) say_echo_do_arxiv && $(LATEX) say_echo_do_arxiv && $(LATEX) say_echo_do_arxiv

explainer:
	cd explainer && $(LATEX) say_echo_do_explainer && $(LATEX) say_echo_do_explainer

report:
	cd report && $(LATEX) say_echo_do_report && $(LATEX) say_echo_do_report && $(LATEX) say_echo_do_report

clean:
	find . -name '*.aux' -o -name '*.log' -o -name '*.out' -o -name '*.toc' -o -name '*.bbl' -o -name '*.blg' | xargs rm -f
