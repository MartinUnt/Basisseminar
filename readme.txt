- Create Conda environment via anaconda console:

conda create --name env

conda activate env

-Install all required packages in this environment via:

pip install -r requirements.txt

- Make conda environment available as kernel in jupyter notebook

conda install nb_conda

- For the installation of tensorflow-gpu, remove tensorflow from the requirements and install the following conda package instead:

conda install tensorflow-gpu

- For errors with tqdm notebook install

conda install -c conda-forge ipywidgets