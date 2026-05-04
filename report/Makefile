REPORT = main

PDF = $(REPORT).pdf
TEX = $(REPORT).tex
BIB = references.bib

.PHONY: all clean view watch

all: $(PDF)

$(PDF): $(TEX) $(BIB)
	pdflatex -interaction=nonstopmode $(TEX)
	bibtex $(REPORT)
	pdflatex -interaction=nonstopmode $(TEX)
	pdflatex -interaction=nonstopmode $(TEX)

view: $(PDF)
	open $(PDF)

clean:
	rm -f *.aux *.log *.out *.toc *.bbl *.blg *.fls *.fdb_latexmk *.synctex.gz

distclean: clean
	rm -f $(PDF)
