# 🧬 PhenoSeq v3

**Bilingual clinical phenotype extraction · HPO linking · Orphanet differential diagnosis · DNA conservation analysis**

[![Tests](https://img.shields.io/badge/tests-29%2F30%20passing-brightgreen)](.)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](.)
[![CPU only](https://img.shields.io/badge/hardware-CPU--only-lightgrey)](.)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Languages](https://img.shields.io/badge/languages-EN%20%2B%20RU-teal)](.)
[![HPO](https://img.shields.io/badge/HPO-2024-orange)](https://hpo.jax.org/)

---

## Key results at a glance

| Metric | Value |
|--------|-------|
| Hit@3 overall | **0.830** |
| Hit@3 Russian formal | **1.000** |
| Wu–Palmer similarity @3 | **0.872** |
| Russian NER F₁ | **0.765** |
| English NER F₁ | **0.713** |
| Synonym index rows | **43 596** |
| Test suite | **29 / 30 (96%)** |

---

## Table of contents

1. [Overview](#1-overview)
2. [NGS context & ConservSeq](#2-ngs-context--conservseq)
3. [System architecture](#3-system-architecture)
4. [Russian NER — 4-layer pipeline](#4-russian-ner--4-layer-pipeline)
5. [HPO linking](#5-hpo-linking)
6. [Evaluation results](#6-evaluation-results)
7. [Installation](#7-installation)
8. [Project structure](#8-project-structure)
9. [Limitations](#9-limitations)
10. [Citation](#10-citation)

---

## 1. Overview

PhenoSeq is an end-to-end pipeline that transforms unstructured clinical
narratives in **English** and **Russian** into structured
[HPO](https://hpo.jax.org/) phenotype codes, generates
[Orphanet](https://www.orpha.net/)-based differential diagnoses, exports
[Phenopackets v2](https://phenopacket-schema.readthedocs.io/), and identifies
functionally conserved DNA regions in candidate disease genes.

> Rare disease patients wait an average of **4–6 years** for a correct
> diagnosis. PhenoSeq shortens this path by automating two bottlenecks:
> phenotype extraction from free text and variant prioritisation in sequencing
> data.

**All models run on CPU — no GPU required.**

---

## 2. NGS context & ConservSeq

Next-generation sequencing (NGS) can return **thousands of variants** per
patient. Without prioritisation, clinicians face an impossible manual review
burden. ConservSeq bridges the phenotype-extraction step and genomic variant
interpretation:

```
Clinical text  →  NER + HPO linking  →  Orphanet DDx  →  ConservSeq  →  Variant priority list
  (EN / RU)        HP:0001250 …         ORPHA:2524 …    conserved        check these first
                                                         regions
```

### What NGS sequencing data looks like

| Library prep & sequencing | Alignment & variant calling | Conservation tracks |
|:---:|:---:|:---:|
| ![NGS workflow](https://hbctraining.github.io/Intro-to-rnaseq-hpc-salmon-flipped/img/sequencing_workflow.png) | ![NGS pipeline](https://irepertoire.com/wp-content/uploads/2020/03/NGS-Overview.png) | ![UCSC conservation](https://genome.ucsc.edu/images/ucscHelp.jpg) |
| Raw reads from Illumina flow cell | Alignment to reference genome & variant calling | UCSC phastCons tracks — ConservSeq emulates this statistically |

### ConservSeq scoring formula

A 20 bp sliding window scores each genomic position using three signals:

```
score(pos) = w_GC · [GC(pos) ≥ 0.50]
           + w_H  · [H₃(pos) ≤ 1.90]      # 3-mer Shannon entropy
           + 2.5  · motif_hits(pos)        # TATA, GC-box, CAAT, E-box,
                                           #   Kozak, CpG, PolyA
```

Regions with `norm_score ≥ 0.4` are labelled **conservative**.
Gaps ≤ 5 bp between regions are merged.

> ⚠️ **Limitation:** Validated against UCSC phastCons100way on the PTEN
> promoter (chr10:89,623,195–89,623,395): Spearman ρ = −0.231 (p < 0.001).
> Statistical signals are a lightweight proxy — not a replacement for
> multi-species phylogenetic conservation.

---

## 3. System architecture

### Language detection & routing

```python
if f_cyrillic > 0.70:   pipeline = "russian"
elif f_latin  > 0.80:   pipeline = "english"
else:                   pipeline = "russian"   # mixed → russian
```

### English NER pipeline

- **Model:** `d4data/biomedical-ner-all` (`aggregation_strategy="max"`)
- **Entity types:** `Disease_disorder`, `Sign_symptom`, `Biological_structure`,
  `Diagnostic_procedure`, `Lab_value`
- **Dictionary layer:** regex over 42,355 HPO terms + synonyms,
  longest-match span resolution
- **Modifier blacklist:** 18 terms (`mri`, `ct`, `eeg`, `bilateral`, `mild`,
  `moderate`, `severe`, …)

**Performance** on 30-sentence annotated corpus
(Jaccard ≥ 0.6, length-ratio ≥ 0.5):

| Precision | Recall | F₁ |
|-----------|--------|----|
| 0.620 | 0.838 | **0.713** |

---

## 4. Russian NER — 4-layer pipeline

| # | Method | Model / tool | Confidence |
|---|--------|-------------|------------|
| 1 | HPO dictionary (RU) | 107-term dict · 4 case forms · 28 informal phrases | 0.95 |
| 2 | NER model | `graviada/labse-ner-runne-ru` | model score |
| 3 | Cross-lingual retrieval | `paraphrase-multilingual-MiniLM-L12-v2` · 2 000-term index | ≥ 0.55 |
| 4 | Rule-based | Cyrillic regex · prefixes гипо-/гипер-/дис- · 80 stop-words | 0.30–0.50 |

**Performance** on 8-sentence evaluation set:

| Precision | Recall | F₁ |
|-----------|--------|----|
| 0.765 | 0.765 | **0.765** |

> **Note:** pymorphy2 lemmatisation is gracefully disabled on Python 3.11+
> due to a known incompatibility.

---

## 5. HPO linking

- **Embedding model:** `paraphrase-multilingual-MiniLM-L12-v2` (384-dim)
- **Index:** 43,596 rows — canonical names **and** all synonyms

Three improvements over vanilla cosine similarity:

```python
# 1. Synonym-expanded index (43 596 rows vs ~19 000 canonical terms)

# 2. Hierarchy depth bonus
delta(t) = 0.05 * depth(t) / depth_max

# 3. Optional cross-encoder rerank (top-5 candidates)
#    cross-encoder/ms-marco-MiniLM-L-6-v2
#    ⚠ degrades on this test set (ΔHit@3 = -0.083)
#      → biomedical fine-tuning required for reliable reranking
```

**Wu–Palmer similarity:**

```
WP(a, b) = 2 · depth(LCA(a, b)) / (depth(a) + depth(b))  ∈ [0, 1]
```

---

## 6. Evaluation results

Test set: **24 cases · 47 HPO annotations**
(bootstrap 95% CI, n = 1 000 resamples)

### By language / formality group

| Group | n | Hit@1 | Hit@3 | Hit@5 | MRR | WP@3 |
|-------|---|-------|-------|-------|-----|------|
| en_formal | 19 | 0.368 | 0.789 | 0.842 | 0.575 | 0.842 |
| en_template | 5 | 0.600 | **1.000** | **1.000** | 0.800 | **1.000** |
| ru_formal | 17 | 0.353 | **1.000** | **1.000** | 0.627 | **1.000** |
| ru_informal | 6 | 0.167 | 0.333 | 0.333 | 0.250 | 0.500 |
| **Overall** | **47** | **0.362** | **0.830** | **0.851** | **0.576** | **0.872** |

> 💡 The 47-point gap between Hit@1 (0.362) and Hit@3 (0.830) is a **ranking
> problem**, not a retrieval problem — the correct HPO term is consistently
> retrieved within the top 3 but rarely ranked first.

### By symptom category

| Category | n | Hit@1 | Hit@3 | MRR |
|----------|---|-------|-------|-----|
| Neurological | 29 | 0.345 | 0.828 | 0.560 |
| Ophthalmological | 13 | 0.462 | 0.769 | 0.603 |
| Musculoskeletal | 3 | 0.333 | 1.000 | 0.667 |
| Dysmorphic | 2 | 0.000 | 1.000 | 0.500 |

### Ablation study (English subset, 13 cases / 24 annotations)

| Configuration | Hit@1 | Hit@3 | MRR |
|---------------|-------|-------|-----|
| Vanilla (cosine only, term-name index) | 0.417 | 0.875 | 0.632 |
| +Synonyms (43 596-row index) | 0.417 | 0.875 | 0.635 |
| +Hierarchy (depth bonus α = 0.05) | 0.417 | 0.875 | 0.632 |
| +Cross-encoder (ms-marco-MiniLM-L-6-v2) | 0.417 | 0.792 | 0.614 |

### Sample output

**English input:**
> *"The patient presents with cerebellar hypoplasia, axial hypotonia, ptosis,
> and intermittent seizures. MRI shows pontine hypoplasia."*

| Extracted term | HPO code | Score |
|----------------|----------|-------|
| pontine hypoplasia | HP:0012110 | 0.98 |
| seizures | HP:0001250 | 0.98 |
| ptosis | HP:0000508 | 0.98 |

Orphanet DDx: `ORPHA:2524` Pontocerebellar hypoplasia type 1A (71%) ·
`ORPHA:99803` type 2A (65%)

---

**Russian informal input:**
> *"Ребёнок плохо держит голову, мышцы очень вялые, судороги."*

| Extracted term | HPO code | Score |
|----------------|----------|-------|
| плохо держит голову | HP:0008936 | 0.95 |
| судороги | HP:0001250 | 0.95 |
| мышцы очень вялые | HP:0001324 | 0.81 |

---

## 7. Installation

```bash
# Clone the repository
git clone https://github.com/constantinerosljakov-cyber/PhenoSeq.git
cd PhenoSeq

# Install dependencies (CPU-only, no GPU required)
pip install -r requirements.txt
```

**Key dependencies:**

```
torch==2.12.0
transformers==5.8.1
sentence-transformers==5.5.1
numpy==1.26.4
pandas==2.2.2
scikit-learn==1.8.0
networkx==3.3
```

```bash
# Run the full pipeline
jupyter notebook PhenoSeq_project_NLP.ipynb

# Run the test suite (29/30 tests expected)
python -m pytest hpo_metrics.py -v
```

---

## 8. Project structure

```
PhenoSeq/
├── PhenoSeq_project_NLP.ipynb       # main notebook — full pipeline
├── hpo_metrics.py                   # evaluation metrics module
├── requirements.txt                 # pinned dependencies
├── PhenoSeq_Report_project.tex      # academic paper (LaTeX)
├── lit_project.bib                  # bibliography
├── .gitignore
└── README.md
```

---

## 9. Limitations

| Area | Issue |
|------|-------|
| **ru_informal** | Hit@3 = 0.333 — colloquial Russian is the primary open challenge |
| **NER span splitting** | "cerebellar hypoplasia" → "cerebellar" + "hypoplasia" (4 of 6 FN) |
| **Cross-encoder** | MS MARCO domain mismatch: ΔHit@3 = −0.083; biomedical fine-tuning needed |
| **pymorphy2** | Not functional on Python 3.11+ — lemmatisation gracefully disabled |
| **Negation** | "No seizures" extracted as "seizures" — NegEx integration pending |
| **ConservSeq** | ρ = −0.231 vs phastCons — statistical signals ≠ phylogenetic conservation |
| **Test set size** | 47 annotations — sufficient for a student project, not for clinical validation |

**Future work:**
- Annotated Russian clinical referral letters for the informal language gap
- RuBERT fine-tuned on NEREL-BIO for Russian NER
- PhyloP/GERP scores via Ensembl REST API
- Negation detection (NegEx / spaCy dependency parser)
- Biomedical cross-encoder fine-tuned on HPO term pairs
- Clinical validation on MIMIC-III discharge summaries

---

## 10. Citation

```bibtex
@misc{rosljakov2026phenoseq,
  title   = {PhenoSeq: Bilingual Clinical Phenotype Extraction
             and HPO Linking for Rare Disease Diagnosis},
  author  = {Rosljakov, Constantine},
  year    = {2026},
  url     = {https://github.com/constantinerosljakov-cyber/PhenoSeq}
}
```

---

*Constantine Rosljakov · 2026 · constantinerosljakov@gmail.com*
