# Publication Package: Multimodal Neural Operator

This directory contains the final bilingual paper and technical documentation for the Burgers-equation multimodal neural operator.

## Deliverables

| Language | Paper | Technical document |
| --- | --- | --- |
| English | `multimodal_neural_operator_paper_en.md` | `multimodal_neural_operator_technical_document_en.md` |
| Chinese | `multimodal_neural_operator_paper_zh.md` | `multimodal_neural_operator_technical_document_zh.md` |

Generated PDFs are written to `pdf/`.

The exact metric values, source paths, and evidence status are recorded in
`evidence_summary.json`.

## Evidence Status

**Current main model:** 128-wide, four-layer multimodal operator with 874,209 parameters, eight sensors, and `burgers_augmented_v1`.

**Verified main result:** seed 42, 2,956 training trajectories, 24,000 steps, test relative L2 `0.246087`, PDE RMS `0.001059`. Metrics use `eval_samples=32`.

**Preliminary sensitivity result:** physics weight 0.15 with dropout 0.08 gives test L2 `0.223471` and PDE RMS `0.001062`, but changes two factors and uses one run.

**Legacy status:** the 64-wide, three-layer prototype is retained only as a historical reference. The current paper does not describe it as the active architecture.

## Primary Sources

- Model and training: `../code/demo_burgers_dataset.py`
- Current data: `../datasets/burgers_augmented_v1/`
- Current main run: `../experiments/v_aug3x_xl_d128_l4/`
- Current sensitivity run: `../experiments/v_aug3x_xl_physics15/`

## Build PDFs

From the repository root:

```powershell
python publication\build_pdfs.py
```

The build requires the Python packages already used by `tmp/pdfs/build_all_documents_pdf.py`, including `markdown`, `playwright`, `pymupdf`, `pypdf`, and `reportlab`.

To validate source links, image paths, PDF page counts, metadata, and extracted text:

```powershell
python publication\validate_publication.py
```

## Scope

These documents cover only the multimodal neural operator for the Burgers equation. They do not describe the structural-health-monitoring classifier or the explainer-video materials.

