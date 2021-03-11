# Python learning modules
This folder contains a collection of [Jupyter notebooks](https://jupyter.org) that introduce *complete beginners* to the Python programming language.
Each notebook is quite long and has many interactive exercises for students to fill in their own code.
I understand that there are many such tutorials online, many of which are probably better than these, so the main reason I created this repo was just to consolidate the resources I used for mentoring other students.
Moreover, the focus is on modules, functions, and algorithms most commonly encountered in computational materials science, so you'll find the exercises most helpful in this regard.
The notebooks are numbered in what I hope is a logical sequence:

00. [Installing Python](00_Installing_Python.md): This is 
01. [Introduction to Python](01_Introduction_to_Python.ipynb):


## Usage
These Jupyter notebooks have sections for you to write your own code. 
To enable interactivity, I recommend one of two ways below:

### 1: Google Colaboratory
This method is nice because it doesn't require you to have Git or Python on your computer, and you can save a copy of each notebook on your Google account. 
I recommend this option.

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
If you have any questions about any of these modules or have an idea for a new module, please let me know! 
Email, GitHub issue/pull request, anything works.


## References
Many of these notebook segments were inspired by, if not taken directly from, [The Materials Project Workshop](https://github.com/materialsproject/workshop/), [Python Practice at UC Berkeley](http://python.berkeley.edu/), and [Berkeley Physics Tutorials](https://github.com/berkeley-physics).