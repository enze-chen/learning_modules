# MSE learning modules
This folder contains a collection of computational MSE learning modules that I have created, largely with Jupyter notebooks. The specific notebooks are:

* Thermodynamics

    * `Thermo_solution_models`: This module teaches you about Gibbs free energy curves, the common tangent construction, and the corresponding phase diagram for ideal and regular solutions.
    
* Characterization

    * `XRD_indexing`: This module teaches you how to index powder XRD spectra using scientific computing tools in Python. It is targeted towards *complete beginners* and a good notebook to start with if you're new to Python.
    
    * `XRD_trends`: This module teaches you about how changing the experimental parameters will affect the powder XRD spectra. You can adjust some widgets at the end to change the behavior.


## Usage
Most of the Jupyter notebooks have an interactive component that requires dynamic rendering. There are several ways to do this, and I give two below:

### 1: Google Colaboratory
This method is nice because it doesn't require you to have Git or Python on your computer, and you can save a copy of each notebook on your Google account. I recommend this option.

1. Click through the GitHub repo until you've found the notebook you want to render.   
1. In a different tab, go to https://colab.research.google.com and click `File > Open notebook > GitHub`.   
1. Copy and paste the notebook's URL into the blank space and you should be able to run the notebook.

### 2: Cloning the repo
If you're familiar with Git and have Python installed on your computer, this is another option.   

1. Clone the repo.   
1. Install the libraries in `requirements.txt` so that you can run all the notebooks. You can do this with:
    ```bash
    pip install -r requirements.txt 
    ```   
1. Load Jupyter (`jupyter notebook`) and run the notebooks.   


## Contributing
If you have any questions about any of these modules or have an idea for a new module, please let me know! Email, GitHub issue/pull request, anything works.