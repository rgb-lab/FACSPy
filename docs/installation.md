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
    
## Issues

If there are any issues, bugs or problems, please do not hesitate to submit an [issue](https://github.com/TarikExner/FACSPy/issues).

## Contributing

Your contributions are welcome!

Please submit an [issue](https://github.com/TarikExner/FACSPy/issues) or pull request via Github. Pull requests with updated documentation and accompanying unit tests are preferred but not obligate!

