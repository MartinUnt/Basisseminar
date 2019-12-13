# Readme: How to install all required packages for the successfull execution of the jupyter notebooks

## Create Conda environment via anaconda console. First step open an anaconda prompt -> windows search: anaconda prompt

    conda create --name env

    conda activate env

## Install all required packages in this environment via (first navigate to the directory where requirements.txt is located):

    pip install -r requirements.txt

## Make conda environment available as kernel in jupyter notebook

    conda install nb_conda

## To prevent errors with tqdm notebook install

    conda install -c conda-forge ipywidgets

## Start the Jupyter Notebook via

    jupyter notebook

## Select the kernel in the jupyter notebook 

Toolbar -> Kernel -> Change Kernels -> NAME_OF_ENVIRONMENT

## If tensorflow-gpu is the desired version for tensorflow, remove tensorflow from the requirements.txt and install the following conda package instead:

    conda install tensorflow-gpu