MLWA EDITABLE SOURCE - FINAL CONFIDENCE-FOCUSED EDITION

Compile the main manuscript:
  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

Compile the supplementary material:
  latexmk -pdf -interaction=nonstopmode -halt-on-error supplementary_material.tex

Verified compiled lengths:
  Main manuscript: 28 pages
  Supplementary material: 30 pages

The source contains editable LaTeX tables and vector PDF artwork. The manuscript
reports the broad 24-condition compact-summary analysis and the eight-condition
strict output-only sensitivity as distinct estimands. Risk@80% is evaluated as a
fixed-coverage ranking metric. Repeated splits are averaged within each
model-dataset condition before crossed model/dataset sensitivity analysis.

The manuscript cites Zenodo DOI 10.5281/zenodo.20732806. Before submission,
confirm that GitHub, Zenodo, the author list, funding details, and Editorial
Manager metadata match this exact manuscript version.
