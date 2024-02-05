# FACSPy
Automated flow-, spectral-flow- and mass-cytometry platform 

## Installation
Currently, FACSPy is in beta phase. A pypi distribution will be available once the beta phase is completed.

To install, first clone this repository to your local drive via your terminal:

```shell
>>> git clone https://github.com/TarikExner/FACSPy.git
```

It is recommended to choose conda as your package manager. Conda can be obtained, e.g., by installing the Miniconda distribution, for detailed instructions, please refer to the respective documentation.

With conda installed, open your terminal and create a new environment by executing the following commands.
```shell
>>> conda create -n facspy python=3.10
>>> conda activate facspy
```

Navigate to the folder where you cloned the repository in and run:
```shell
>>> pip install .
```

This installs FACSPy and all dependencies.

To install jupyter, run:
```shell
>>> conda install jupyter
```

Open a notebook by running
```shell
>>> jupyter-notebook
```

To test if everything went successfull open a python console and import FACSPy:
```shell
>>> python
>>> import FACSPy as fp
```

## Getting Started

Code examples are found under "vignettes" and currently include:
- FACSPy dataset explanation
- Metadata objects
- Panel objects
- Cofactor-Table objects

- analysis of a spectral flow cytometry dataset
- analysis of a flow cytometry dataset consisting of mouse lineages

## Features

Currently, the following features are implemented:

### Metadata Annotation

Accompanying Metadata (tabular metadata, the panel information, asinh-transformation cofactors and a FlowJo- or Diva-Workspace) are internally represented to gather pre-existing information!

```python
import FACSPy as fp

metadata = fp.dt.Metadata("../metadata.csv")
panel = fp.dt.Panel("../panel.csv")
workspace = fp.dt.FlowJoWorkspace("../workspace.wsp")
```

### Dataset Creation

The dataset is created using one single function

```python
import FACSPy as fp

dataset = fp.dt.create_dataset(
    metadata = metadata,
    panel = panel,
    workspace = workspace
)
```
<p align="center">
<img src="https://github.com/TarikExner/FACSPy/blob/main/FACSPy/img/FACSPY_graphical_abstract.png" width = 600 alt="FACSPy Schema">
</p>

### Dataset Transformation

The asinh transform requires cofactors, which are calculated automatically:

```python
fp.dt.calculate_cofactors(dataset)
```

Plotting is realized through a dedicated plotting module:

```python
fp.pl.transformation_plot(
    dataset,
    sample_identifier = "2",
    marker = "CD38"
)
```
<p align="center">
<img src="https://github.com/TarikExner/FACSPy/blob/main/FACSPy/img/transformation_plot.png" alt="transformation plot">
</p>

The dataset is then transformed using asinh-transformation, logicle, hyperlog or normal log transformation.

```python
fp.dt.transform(
    dataset,
    transform = "asinh",
    cofactor_table = dataset.uns["cofactors"],
    key_added = "transformed",
    layer = "compensated"
)
```

### Biaxial plotting

```python
fp.pl.biax(
    dataset,
    gate = "CD45+",
    x_channel = "CD3",
    y_channel = "SSC-A",
    color = "density"
)

fp.pl.biax(
    dataset,
    gate = "CD45+",
    x_channel = "CD3",
    y_channel = "SSC-A",
    color = "CD4",
    vmin = 1,
    vmax = 4e4
)

fp.pl.biax(
    dataset,
    gate = "CD45+",
    x_channel = "CD3",
    y_channel = "SSC-A",
    color = "CD8",
    vmin = 1,
    vmax = 4e4

)
```

<p float="left" align="center" width=600>
<img src="https://github.com/TarikExner/FACSPy/blob/main/FACSPy/img/biax_density.png" width = 200 alt="gate frequency plot"/>
<img src="https://github.com/TarikExner/FACSPy/blob/main/FACSPy/img/biax_CD4.png" width = 200 alt="gate frequency plot"/>
<img src="https://github.com/TarikExner/FACSPy/blob/main/FACSPy/img/biax_CD8.png" width = 200 alt="gate frequency plot"/>
</p>


### Cell Count Analysis

```python
fp.pl.cell_counts(
    dataset,
    gate = "live",
    groupby = "diag_main",
    figsize = (2,4)
)

fp.pl.cell_counts(
    dataset,
    gate = "live",
    groupby = "diag_main",
    splitby = "organ",
    stat_test = False,
    figsize = (2,4)
)
```
<p float="left" align="center" width = 400>
<img src="https://github.com/TarikExner/FACSPy/blob/main/FACSPy/img/cell_counts.png" width = 300 alt="gate frequency plot"/>
<img src="https://github.com/TarikExner/FACSPy/blob/main/FACSPy/img/cell_counts_split.png" width = 450 alt="gate frequency plot"/>
</p>


### Gate Frequency Analysis

```python
fp.tl.gate_frequencies(dataset)

fp.pl.gate_frequency(
    dataset,
    gate = "CD45+",
    groupby = "diag_main",
    freq_of = "parent",
    figsize = (2,4)
)

fp.pl.gate_frequency(
    dataset,
    gate = "CD45+",
    groupby = "diag_main",
    splitby = "organ",
    freq_of = "parent",
    figsize = (2,4)
)
```
<p float="left" align="center" width = 400>
<img src="https://github.com/TarikExner/FACSPy/blob/main/FACSPy/img/gate_frequency.png" width = 300 alt="gate frequency plot"/>
<img src="https://github.com/TarikExner/FACSPy/blob/main/FACSPy/img/gate_frequency_split.png" width = 300 alt="gate frequency plot"/>
</p>


### Gating

Gating can be accomplished using a conventional FlowJo-Workspace, unsupervised Gating via Clustering (manually or automated) or supervised Gating using pre-gated example files.

Here, we gate NK cells by looking at the CD16+ and CD56+ clusters manually:
<p float="left" align="center">
<img src="https://github.com/TarikExner/FACSPy/blob/main/FACSPy/img/CD16_umap.png" width = 300 alt="gate frequency plot"/>
<img src="https://github.com/TarikExner/FACSPy/blob/main/FACSPy/img/CD56_umap.png" width = 300 alt="gate frequency plot"/>
</p>
<p float="left" align="center">
<img src="https://github.com/TarikExner/FACSPy/blob/main/FACSPy/img/leiden_umap.png" width = 400 alt="gate frequency plot"/>
</p>

```python
fp.convert_cluster_to_gate(
    dataset,
    obs_column = "CD45+_transformed_leiden",
    positive_cluster = ["10", "17", "2"],
    population_name = "NK_cells",
    parent_name = "CD45+"
)

fp.convert_gate_to_obs(
    dataset,
    "NK_cells"
)
```

<p align="center">
<img src="https://github.com/TarikExner/FACSPy/blob/main/FACSPy/img/NK_cell_umap.png" width = 300 alt="gate frequency plot">
</p>

We can also used the unsupervisedGating class:

```python
gating_strategy = {
    "T_cells": ["CD45+", ["CD3+", "CD45+"]],
    "CD4_T_cells": ["T_cells", ["CD3+", "CD4+", "CD8-", "CD45+"]],
    "CD8_T_cells": ["T_cells", ["CD3+", "CD4-", "CD8+", "CD45+"]]
}

clf = fp.ml.unsupervisedGating(
    dataset,
    gating_strategy = gating_strategy,
    clustering_algorithm = "leiden", 
    layer = "transformed",
    cluster_key = None
)

clf.identify_populations()

fp.convert_gate_to_obs(dataset, "T_cells")
fp.convert_gate_to_obs(dataset, "CD4_T_cells")
fp.convert_gate_to_obs(dataset, "CD8_T_cells")
```
<p align="center">
<img src="https://github.com/TarikExner/FACSPy/blob/main/FACSPy/img/t_cells_umap_relook.png" width = 600 alt="gate frequency plot">
</p>
<p align="center">
<img src="https://github.com/TarikExner/FACSPy/blob/main/FACSPy/img/cd4_t_cells_umap_relook.png" width = 600 alt="gate frequency plot">
</p>
<p align="center">
<img src="https://github.com/TarikExner/FACSPy/blob/main/FACSPy/img/cd8_t_cells_umap_relook.png" width = 600 alt="gate frequency plot">
</p>

### Flow Cytometry Metrics

Common Flow Cytometry metrics like MFI and FOP (frequency of parent) can be plotted:

```python
fp.pl.mfi(
    dataset,
    groupby = "organ",
    marker = "PD-1_(CD279)",
    colorby = "diag_main",
    figsize = (2,4)
)

fp.pl.fop(
    dataset,
    groupby = "organ",
    marker = "PD-1_(CD279)",
    colorby = "diag_main",
    figsize = (2,4)
)
```

<p float="left" align="center" width = 400>
<img src="https://github.com/TarikExner/FACSPy/blob/main/FACSPy/img/mfi_plot.png" width = 200 alt="gate frequency plot"/>
<img src="https://github.com/TarikExner/FACSPy/blob/main/FACSPy/img/fop_plot.png" width = 200 alt="gate frequency plot"/>
</p>



### Dimensionality Reduction

FACSPy offers samplewise dimensionality reductions as well as single cell dimensionality reductions!

```python
fp.tl.mfi(dataset)

fp.pl.pca_samplewise(
    dataset,
    groupby = "organ"
)

fp.pl.mds_samplewise(
    dataset,
    groupby = "organ"
)

fp.pl.umap_samplewise(
    dataset,
    groupby = "organ"
)

fp.pl.tsne_samplewise(
    dataset,
    groupby = "organ"
)

```

<p float="left" align="center" width = 800>
<img src="https://github.com/TarikExner/FACSPy/blob/main/FACSPy/img/pca_samplewise.png" width = 200 alt="gate frequency plot"/>
<img src="https://github.com/TarikExner/FACSPy/blob/main/FACSPy/img/mds_samplewise.png" width = 200 alt="gate frequency plot"/>
<img src="https://github.com/TarikExner/FACSPy/blob/main/FACSPy/img/tsne_samplewise.png" width = 200 alt="gate frequency plot"/>
<img src="https://github.com/TarikExner/FACSPy/blob/main/FACSPy/img/umap_samplewise.png" width = 200 alt="gate frequency plot"/>
</p>


```python
fp.tl.umap(dataset)
fp.tl.tsne(dataset)
fp.tl.diffmap(dataset)
fp.tl.pca(dataset)

fp.pl.pca(
    dataset,
    color = "organ"
)

fp.pl.umap(
    dataset,
    color = "organ"
)

fp.pl.tsne(
    dataset,
    color = "organ"
)

fp.pl.diffmap(
    dataset,
    color = "organ"
)

```

<p float="left" align="center" width = 800>
<img src="https://github.com/TarikExner/FACSPy/blob/main/FACSPy/img/pca.png" width = 200 alt="gate frequency plot"/>
<img src="https://github.com/TarikExner/FACSPy/blob/main/FACSPy/img/diffmap.png" width = 200 alt="gate frequency plot"/>
<img src="https://github.com/TarikExner/FACSPy/blob/main/FACSPy/img/umap.png" width = 200 alt="gate frequency plot"/>
<img src="https://github.com/TarikExner/FACSPy/blob/main/FACSPy/img/tsne.png" width = 200 alt="gate frequency plot"/>
</p>


### Clustering

Currently, PARC, PhenoGraph, Leiden and FlowSOM are implemented:

```python
fp.tl.parc(dataset)
fp.tl.phenograph(dataset)
fp.tl.leiden(dataset)
fp.tl.flowsom(dataset)

fp.pl.umap(
    dataset,
    color = "CD45+_transformed_leiden",
    legend_loc = "on data"
)

fp.pl.umap(
    dataset,
    color = "CD45+_transformed_parc"
    legend_loc = "on data"
)

fp.pl.umap(
    dataset,
    color = "CD45+_transformed_flowsom"
    legend_loc = "on data"
)

fp.pl.umap(
    dataset,
    color = "CD45+_transformed_phenograph"
    legend_loc = "on data"
)

```

<p float="left" align="center" width = 800>
<img src="https://github.com/TarikExner/FACSPy/blob/main/FACSPy/img/leiden.png" width = 150 alt="gate frequency plot"/>
<img src="https://github.com/TarikExner/FACSPy/blob/main/FACSPy/img/parc.png" width = 150 alt="gate frequency plot"/>
<img src="https://github.com/TarikExner/FACSPy/blob/main/FACSPy/img/flowsom.png" width = 150 alt="gate frequency plot"/>
<img src="https://github.com/TarikExner/FACSPy/blob/main/FACSPy/img/phenograph.png" width = 150 alt="gate frequency plot"/>
</p>


### Heatmap visualization

FACSPy implements heatmap visualizations for expression data as well as correlation plots for marker and samples

```python

fp.pl.expression_heatmap(
    dataset,
    metadata_annotation = ["organ", "diag_main", "diag_fine"],
    plot_annotate = "HLA_DR",
    figsize = (5,8)
)

```

<p align="center">
<img src="https://github.com/TarikExner/FACSPy/blob/main/FACSPy/img/expression_heatmap.png" width = 500 alt="gate frequency plot"/>
</p>

```python

fp.pl.marker_correlation(
    dataset,
    y_label_fontsize = 8
    figsize = (6,6)
)

```

<p align="center">
<img src="https://github.com/TarikExner/FACSPy/blob/main/FACSPy/img/marker_correlation.png" width = 500 alt="gate frequency plot"/>
</p>

```python
fp.pl.sample_correlation(
    dataset,
    metadata_annotation = ["organ", "diag_main", "diag_fine"],
    metaclusters = 2,
    corr_method = "spearman",
    label_metaclusters_in_dataset = True,
    label_metaclusters_key = "sample_corr_metaclusters"
)

```

<p align="center">
<img src="https://github.com/TarikExner/FACSPy/blob/main/FACSPy/img/sample_correlation.png" width = 500 alt="gate frequency plot"/>
</p>

```python
fp.pl.sample_distance(
    dataset,
    metadata_annotation = ["organ", "diag_main", "diag_fine"],
    metaclusters = 2,
    corr_method = "spearman",
    label_metaclusters_in_dataset = True,
    label_metaclusters_key = "sample_dist_metaclusters"
)

```

<p align="center">
<img src="https://github.com/TarikExner/FACSPy/blob/main/FACSPy/img/sample_distance.png" width = 500 alt="gate frequency plot"/>
</p>

### Differential Expression Testing

Differential expression testing is visualized as a fold change plot:

```python
fp.pl.fold_change(
    dataset,
    layer = "compensated",
    groupby  = "organ",
    group1 = "PB",
    group2 = "organ2",
    figsize = (2,6)
)
```

<p align="center">
<img src="https://github.com/TarikExner/FACSPy/blob/main/FACSPy/img/diffexp.png" width = 500 alt="gate frequency plot"/>
</p>


## Future Feature Implementation

FACSPy will continue to be developed!

If you have any feature requests, please open an issue on GitHub. To ensure efficient treament of feature requests, please make sure to check if an issue regarding you request hasn't already been created.

For the near future, the following features are meant to be implemented:
- data integration
- minimal spanning trees
- t.b.a.


## Contributing

Your contributions are welcome!

Please submit an issue or pull request via Github! Pull requests with updated documentation and accompanying unit tests are preferred but not obligate!
