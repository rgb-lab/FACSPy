# FACSPy
Automated flow-, spectral-flow- and mass-cytometry platform 

## Installation
Currently, FACSPy is in beta phase. A pypi distribution will be available once the beta phase is completed.

To install, first clone this repository to your local drive via your terminal:

```shell
>>> git clone https://github.com/TarikExner/FACSPy.git
```

It is recommended to choose anaconda as your package manager. Install Anaconda and open the Anaconda terminal.

Create a new environment by executing
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
<img src="https://github.com/TarikExner/FACSPy/blob/main/FACSPy/img/FACSPY_graphical_abstract.png" alt="FACSPy Schema">

### Dataset Transformation

The asinh transform requires cofactors, which are calculated automatically:

```python
fp.dt.calculate_cofactors(dataset)
```

Plotting is realized through dedicated plotting functionality:

```python
fp.pl.transformation_plot(
    dataset,
    sample_identifier = "2",
    marker = "CD38"
)
```
<img src="https://github.com/TarikExner/FACSPy/blob/main/FACSPy/img/transformation_plot.png" alt="transformation plot">

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


### Gating

### Gate Frequency Analysis

```python
fp.pl.gate_frequency(
    dataset,
    gate = "CD45+",
    groupby = "diag_main",
    colorby = "organ",
    freq_of = "parent",
    figsize = (4,4),
    save = "../FACSPy/img/gate_frequency.png"
)
```

<img src="https://github.com/TarikExner/FACSPy/blob/main/FACSPy/img/gate_frequency.png" width = 250 alt="gate frequency plot">

### Flow Cytometry Metrics

### Cell Count Analysis

```python
fp.pl.cell_counts(
    dataset,
    gate = "live",
    groupby = "diag_main",
    colorby = "organ",
    figsize = (4,4)
)
```

<img src="https://github.com/TarikExner/FACSPy/blob/main/FACSPy/img/cell_counts.png" alt="gate frequency plot">


### Dimensionality Reduction

### Clustering

### Heatmap visualization

### Differential Expression Testing

## Demo Code

Code examples are found under "vignettes" and currently include:
    - spectral flow cytometry dataset

## Future Feature Implementation

FACSPy will continue to be developed!

If you have any feature request, please submit them via GitHub.

For the near future, the following features are meant to be implemented:
- data integration
- minimal spanning trees
- t.b.a.


## Contributing

Your contributions are welcome!

Please submit an issue or pull request via Github! Pull requests with updated documentation and accompanying unit tests are preferred but not obligate!


