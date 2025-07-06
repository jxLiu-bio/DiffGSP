## Data Preparation

### Sequencing-Based Spatial Transcriptomics Data

The following data are required to run **DiffGSP**:

1. **Gene expression matrix**: A matrix with dimensions *spots ? genes*.
2. **Spatial coordinate matrix**: A matrix with dimensions *spots ? [x, y]*.
3. **`in_tissue` information** *(optional but recommended)*.

All the above data should be stored in an `AnnData` object:

- Expression data: `adata.X`
- Spatial coordinates: `adata.obsm['spatial']`
- In-tissue information (required for BFGS): `adata.obs['in_tissue']`

In spatial transcriptomics, `in_tissue` typically indicates whether a spot is located within the tissue section.  It is usually represented as binary (`1`/`0`) variable.

---

### Example Datasets

The datasets used in this tutorial can be downloaded from the following sources:

- [Mouse brain Visium dataset](https://www.10xgenomics.com/datasets/mouse-brain-section-coronal-1-standard)
- [Mouse olfactory bulb Slide-seqV2 dataset](https://singlecell.broadinstitute.org/single_cell/study/SCP815/highly-sensitive-spatial-transcriptomics-at-near-cellular-resolution-with-slide-seqv2#study-download)
- [Mouse kidney Stereo-seq dataset](https://db.cngb.org/stomics/datasets/STDS0000240/data)
- [Human colorectal cancer Visium HD dataset](https://www.10xgenomics.com/products/visium-hd-spatial-gene-expression/dataset-human-crc)



