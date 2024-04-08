# Installation

It is recommended to choose conda as your package manager. Conda can be obtained, e.g., by installing the Miniconda distribution. For detailed instructions, please refer to the respective documentation.

With conda installed, open your terminal and create a new environment by executing the following commands::

    conda create -n facspy python=3.10
    conda activate facspy

## PyPI

Currently, FACSPy is in beta-phase. There will be a PyPI release once the beta phase is finished.

    pip install FACSPy


## Development Version

In order to get the latest version, install from [GitHub](https://github.com/TarikExner/FACSPy) using
    
    pip install git+https://github.com/TarikExner/FACSPy@main

Alternatively, clone the repository to your local hard drive via

    git clone https://github.com/TarikExner/FACSPy.git && cd FACSPy
    git checkout --track origin/main
    pip install .

Note that while FACSPy is in beta phase, you need to have access to the private repo.

## Jupyter notebooks

Jupyter notebooks are highly recommended due to their extensive visualization capabilities. Install jupyter via

    conda install jupyter

and run the notebook by entering `jupyter-notebook` into the terminal.

## Installing with an R environment

AnnData objects can be readily converted to SingleCellExperiments in R via the rpy2 and anndata2ri interface.
Combining R and python can be difficult sometimes, the steps described here have been tested on Windows (r-base=4.1.3)
and linux (r-base=4.2.1).

In order to setup an environment covering both R and python use the following commands in conda:

    conda create -n facspy_r 
    conda activate facspy_r
    conda install python=3.10 jupyter r-base

In order to install FACSPy with the dependencies for the data analysis in R and python, either install from PyPI:

    pip install FACSPy[r_env]

or locally:
    
    git clone https://github.com/TarikExner/FACSPy.git && cd FACSPy
    git checkout --track origin/main
    cd ..
    pip install FACSPy
    pip install anndata2ri rpy2

This is the minimal setup in order to convert an AnnData object to a SingleCellExperiment object. You can then save
the SingleCellExperiment and use your own R distribution for further analysis.

If you want to install R-packages in the same environment, please follow the instructions below.

### Example installation for CATALYST and Spectre

Make sure you have [RTools](https://cran.r-project.org/bin/windows/Rtools/) installed when using Windows.

[CATALYST](https://github.com/HelenaLC/CATALYST) and [Spectre](https://github.com/ImmuneDynamics/Spectre) are R libraries for the analysis of cytometry data.
In order to install CATALYST and Spectre, use the following additional commands:

    conda install -c r rtools
    conda install -c conda-forge r-curl r-scales r-igraph r-cowplot r-ggridges r-ggbeeswarm r-viridis r-devtools r-emmeans r-recipes r-biocmanager r-matrixstats r-ggplot2 r-rainbow r-rcpphnsw r-rcpparmadillo r-sf r-tiff r-cairo r-terra

From within an R session:

    > install.packages(c("RCurl", "curl", "XML")) # don't use compilation
    > BiocManager::install("CATALYST")
    > library(CATALYST)
    > devtools::install_github("immunedynamics/spectre")
    > library(Spectre)

### R package installation
You can install R packages different ways, here we cover the four most relevant scenarios.

1. Open an R session by typing `R` into the terminal. Install packages via:

    ```shell
    > install.packages("ggplot2")
    ```

2. Open an R session by typing `R` into the terminal and install packages via BioConductor:

    ```shell
    > install.packages("BiocManager")
    > BiocManager::install("SingleCellExperiment")
    ```

3. Open an R session by typing `R` into the terminal and install from GitHub via:

    ```shell
    > install.packages("devtools")
    > devtools::install_github("saeyslab/CytoNorm")
    ```

4. Install via conda:

    ```shell
    >>> conda install -c conda-forge r-hdf5r
    ```

Please check the availability of your desired package on the respective websites. If there are conda versions available, this is
often preferred due to build complications when systems libraries and conda libraries for building are not the same.

    
## Issues

If there are any issues, bugs or problems, please do not hesitate to submit an [issue](https://github.com/TarikExner/FACSPy/issues).

## Contributing

Your contributions are welcome!

Please submit an [issue](https://github.com/TarikExner/FACSPy/issues) or pull request via Github. Pull requests with updated documentation and accompanying unit tests are preferred but not obligate!

