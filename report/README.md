# COMP3308 A2 Report

LaTeX source for Part 2 of the assignment.

## Build

Requires a TeX distribution with `pdflatex` and `bibtex` (TeX Live, MacTeX,
MikTeX). From this folder:

```bash
make          # produces main.pdf
make view     # opens main.pdf in macOS Preview / xdg-open
make clean    # removes intermediate aux files
make distclean # removes pdf as well
```

If you don't have `make`, run the same three commands by hand:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## What you must edit before submitting

1. **SIDs.** Replace `<SID1>`, `<SID2>` (and `<SID3>` if 3-person group)
   in the title block and in the Reflection / AI-acknowledgement
   sections.
2. **Weka results.** The Weka cells in Tables 2 and 3 are filled with
   plausible placeholder numbers. After you run Weka on `heart.csv`
   (10-fold cross-validation, default settings, J48 with and without
   pruning, IBk at k=1 and k=5, Bagging and AdaBoostM1 with J48 as
   the base classifier), update the macros in the
   `WEKA RESULTS` block at the top of `main.tex`. Every number in the
   tables and discussion will update automatically.
3. **Reflection.** Each group member writes their own short reflection.
4. **AI acknowledgement.** Each group member writes their own statement
   listing the AI tools they used and how. There is a draft for SID 1
   that you may keep or rewrite.

## What is *not* edited

The `My*` rows of Table 3 and the SD column of Table 4 come from the
formal evaluation we ran with `python3 evaluate.py` on the official
`heart-folds.csv`. Re-run that script if you change the algorithm or
the folds; otherwise leave the numbers as they are.

The two appendix DT diagrams are read directly from
`../diagrams.txt`. Re-run `python3 build_diagrams.py > diagrams.txt`
in the parent folder if you change the DT or DT+ algorithm.
