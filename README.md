---
title: "Representation and Outcomes at the Board of Veterans' Appeals, 2010-2025"
tags: []
type: note
project: BVA Project
status: active
created: 2026-03-07
modified: 2026-03-07
---
# Representation and Outcomes at the Board of Veterans' Appeals, 2010-2025

[[index_bva-project]]

Replication package for a reproducible cohort analysis of 1.19 million Board of Veterans' Appeals (BVA) decisions examining the effect of legal representation on case outcomes.

## Problem

Veterans who appeal disability claim denials to the BVA face a complex adjudicatory process. Whether legal representation affects outcomes, and by how much, has significant policy implications for access to justice in the VA system.

## Data

- 1,189,770 BVA decisions from CY 2010 through CY 2025
- Source: publicly available BVA decision records
- Decision-level dataset with representation type, disposition, issue categories, era markers

## Method

14-step pipeline from raw decision download through publication figures:

1. Download BVA decisions from public source
2. Clean encoding and structure
3. Index and parse structured fields (representation type, disposition, issue categories)
4. Quality control checkpoints
5. Build analysis dataset at decision level
6. Compute rollup statistics by year and representation type
7. Generate summary statistics (Table 1 inputs)
8. Run primary logistic regression models (Tables 2A, 2B)
9. Run multinomial and inverse probability weighting models (Table 3)
10. Build sensitivity and robustness figures
11. Compute effect sizes
12. Generate publication figures (Figures 1-4, S1-S6, S8)
13. Export methods tables and text snippets
14. Generate 300-case audit sample and data snapshot

Statistical approach: inverse probability weighting (IPW) with trimming and balance diagnostics, wild cluster bootstrap inference for small-cluster standard errors, robust variance estimators.

## How to Run

```
# Install dependencies
python 00_install_python_requirements.vbs

# Run full pipeline (steps 01 through 14)
python 99_run_full_pipeline.vbs

# Or run individual steps
python 01_download_bva_decisions.pyw
python 02_clean_bva_decisions.py
# ... through step 14
```

See `RUN_ORDER_code.txt` for the complete execution sequence with descriptions.

## Outputs

- **20 analysis tables** (CSV): descriptives, logistic regression, multinomial, IPW balance, robustness checks
- **11 publication figures** (PNG/SVG): sample flow, coverage trends, outcome comparisons, forest plots, diagnostics
- **300-case audit sample** with backing data for all figures

See `MANIFEST_code.txt`, `MANIFEST_data_tables.txt`, and `MANIFEST_figures.txt` for complete catalogs.

## Key Finding

Attorney representation is associated with approximately 6x odds of a favorable outcome at the BVA, after adjusting for case characteristics and year effects via inverse probability weighting.

## Citation

See `CITATION.cff` for citation information.

## License

MIT. See `LICENSE`.

## OSF Project

https://osf.io/vzusf/
