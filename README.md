# FACSPy
Automated flow-, spectral-flow- and mass-cytometry platform 

## Installation
Currently, FACSPy is in beta phase.
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

Accompanying Metadata (tabular metadata, the panel information, asinh-transformation cofactors and a FlowJo/Diva-Workspace) are internally represented to

### Dataset Creation

### Dataset Transformation

### Gating

### Gate Frequency Analysis

### Flow Cytometry Metrics

### Cell Count Analysis

### Dimensionality Reduction

### Clustering

### Heatmap visualization

### Differential Expression Testing

## Demo Code

Code examples are found under "vignettes" and currently include:
    - spectral flow cytometry dataset

## Future Feature Implementation

FACSPy will continue to be developed!

If you have any feature request, submit them via GitHub.

For the near future, the following features are meant to be implemented:
- data integration
- minimal spanning trees
- t.b.a.


## Contributing

Your contributions are welcome!

Please submit an issue or pull request via Github! Pull requests with updated documentation and accompanying unit tests are preferred but not obligate!


