# MSE learning modules
This folder contains a collection of computational MSE learning modules that I have created, largely with [Jupyter notebooks](https://jupyter.org). All the notebooks have interactive elements, whether it's adjusting sliders or writing your own code. I've tried to keep the coding parts accessible to beginners. The specific notebooks are:

### Computational

* [Cellular automata for metallurgy](Cellular_automaton_models.ipynb): This modules teaches you how to use cellular automata to model two important processes in metallurgy: recrystallization and spinodal decomposition. It is highly interactive, and at the end you will create an animation using Matplotlib! I also provide several links to external resources.

* [Monte Carlo simulations of the Ising model](Monte_Carlo_Ising_model.ipynb): This module teaches you how to use Monte Carlo to simulate the 2D ferromagnetic Ising model on a square lattice. There is a lot of scaffolding, but you must fill in all of the code. A great introduction to [NumPy](https://numpy.org/) and [Matplotlib](https://matplotlib.org/) for someone who is interested in computational materials science in Python.

* [Machine learning for the Ising model](Machine_learning_Ising_model.ipynb): This module teaches you how to use machine learning algorithms to predict the phase transition in the 2D Ising model on a square lattice. It is recommended that you understand the MC simulation notebook and basic Python before jumping into this one.


### Thermodynamics

* [Intro to regular solutions](Regular_solution_plot.ipynb): This module serves as an introduction to scientific computing for *complete beginners* using [NumPy](https://numpy.org/) and [Matplotlib](https://matplotlib.org/). Your task is to generate the Gibbs free energy curves for the regular solution model.

* [Thermodynamics of solution models](Thermo_solution_models.ipynb): This module teaches you about Gibbs free energy curves, the common tangent construction, and the corresponding phase diagrams for ideal and regular solutions. There are interactive widgets and places for you to fill in the code.
    
* [Eutectic system](Eutectic_solution.ipynb): This module is very similar to the previous module but is customized to plot the Gibbs free energy curves and the eutectic phase diagram. There is an interactive widget that allows you to change the temperature and observe the effects on the Gibbs free energy curve.


### Characterization

* [Indexing powder XRD spectra](XRD_indexing.ipynb): This module teaches you how to index powder XRD spectra using scientific computing tools in Python, specifically [pandas](https://pandas.pydata.org/) and [NumPy](https://numpy.org/). It is targeted towards *complete beginners* and a good notebook to start with if you're new to Python.

* [Plotting powder XRD spectra](XRD_plotting.ipynb): This module teaches you how to represent the physical phenomena in XRD in code using [NumPy](https://numpy.org/) and [Matplotlib](https://matplotlib.org/). It is targeted towards students with some Python programming experience and hopefully reinforces the connection between the concepts we learn and the spectra we see.

* [Trends in powder XRD spectra](XRD_trends.ipynb): This module teaches you about how changing the experimental parameters will affect the powder XRD spectra. The widgets at the end correspond to different parameters that will change the peak profile. *No programming experience necessary*.



## Usage
Most of the Jupyter notebooks have an interactive component that requires dynamic rendering. There are several ways to do this, and I give two below:

### 1: Google Colaboratory
This method is nice because it doesn't require you to have Git or Python on your computer, and you can save a copy of each notebook on your Google account. I recommend this option.

1. Click on the specific notebook you want to render.   
1. In a different tab, go to https://colab.research.google.com and click `File > Open notebook > GitHub`.   
1. Copy and paste the notebook's URL into the blank space and you should be able to run the notebook.

### 2: Cloning the repository
If you're familiar with Git and have Python installed on your computer, this is another option.   

1. Clone the entire repository.   
1. Install the libraries in [requirements.txt](../requirements.txt) so that you can run all the notebooks. You can do this with:
    ```bash
    pip install -r requirements.txt 
    ```   
1. Load Jupyter (`jupyter notebook`) and run the notebooks.   


## Contributing
If you have any questions about any of these modules or have an idea for a new module, please let me know! Email, GitHub issue/pull request, anything works.
