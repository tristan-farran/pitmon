PITMonitor — Statistics and Computing (Springer) Submission
===========================================================

TEMPLATE SOURCE
---------------
Template files (svjour3.cls, spmpsci.bst) were downloaded from:
  svjour3.cls  — https://github.com/DanySK/Template-LaTeX-Springer-svjour3
  spmpsci.bst  — https://raw.githubusercontent.com/borisveytsman/crossrefware/master/spmpsci.bst

Official Springer template page:
  https://www.springer.com/journal/11222/submission-guidelines

Document class: \documentclass[smallextended]{svjour3}
Journal name:   \journalname{Statistics and Computing}
Bibliography style: spmpsci (Springer mathematics/statistics, numbered)

To obtain the official template bundle from Springer directly, download from:
  https://www.springernature.com/gp/authors/campaigns/latex-author-support


COMPILATION
-----------
Standard pdflatex + bibtex sequence (run from this directory):

  pdflatex main
  bibtex main
  pdflatex main
  pdflatex main

Or with latexmk:

  latexmk -pdf main


WHAT CHANGED FROM THE WORKING PAPER
-------------------------------------
1. Document class changed to \documentclass[smallextended]{svjour3}.
2. \journalname{Statistics and Computing} set in preamble.
3. Author block uses Springer's \institute{} environment with \email{}.
4. Abstract keywords use \keywords{} and \subclass{} commands (inside abstract).
5. MSC 2020 subject classifications: 62L10, 62G10, 62M20.
6. Bibliography style changed to spmpsci (numbered Springer style).
7. Figure paths updated from ../experiment/out/ to figures/.
8. \input paths updated from ../experiment/out/... to local files.
9. Related Work converted from \subsubsection* to \paragraph{} headings
   (more appropriate for Springer's single-level section style in this paper).
10. Acknowledgements use \begin{acknowledgements} (svjour3 standard).
11. \RequirePackage{fix-cm} added at top (recommended for svjour3).
12. \newpage calls removed.


ITEMS REQUIRING MANUAL ATTENTION
---------------------------------
- Author affiliation: Currently "MSc Computational Science, University of
  Amsterdam". If submitting as a full academic paper, verify or expand with
  department/faculty details.
- ORCID: svjour3 does not have a built-in \orcid command. If needed, add it
  manually in the author block or acknowledgements.
- Cover letter: Statistics and Computing requires a cover letter uploaded
  separately in the submission system. It is not part of the LaTeX source.
- Date field: \date{Received: \today} — remove \today before submission; the
  editor will fill in the correct dates.
- No hard page limit for Statistics and Computing.
- The \subclass{} command inside \begin{abstract} may need to be moved to
  after \maketitle in some svjour3 versions. If compilation fails on that
  line, try placing \subclass{...} immediately after \end{abstract}.
- If Springer's production team requests it, they may ask for svjour3.clo
  (a journal-specific class options file). This is handled by Springer's
  production system and is not required for initial submission.
