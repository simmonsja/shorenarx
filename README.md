# Code to accompany "Data-driven shoreline modelling at timescales of days to years"
Joshua Simmons 2024

This repository contains the code to reproduce the figures in the publication "Data-driven shoreline modelling at timescales of days to years" by Simmons and Splinter (2024). The code is written in Python and uses the pytorch library to train a neural network to predict the shoreline position at timescales of days to years.

# Notebooks

- `01_Publication_Optuna_Analysis.ipynb`: Analysis of the Optuna hyperparameter search to find appropriate settings for the predictive model.
- `02_Model_Training_Example.ipynb`: Example of how to train the model using the optimal hyperparameters found in the previous notebook.
- `03_Publication_Model_Performance.ipynb`: Analysis of the model performance and various tests. This notebook is the main notebook to reproduce the figures in the publication.

## Scripts

- `01a_Optuna_Run_Script.py`: Script to run the Optuna hyperparameter search given a config file.
- `02a_CV_Run_Script.py`: Script to train the model given a config file.


# Config files
The config files are stored in `config/` as json files converted to dict upon loading. The config files are used to set the hyperparameters for the model and various decisions around the train/test split, input variables, and output variables. Typical settings include:

- `site`: The site for which the configuration is set.
- `saveClass`: The name for the model when saved.
- `saveVersion`: The version of the saved model.
- `xSample`: The sampling rate for the input variables (e.g., "3H", "1D").
- `ySample`: The sampling rate for the output variables (e.g., "1D", "3D").
- `inputVars`: The list of input variables used in the model (e.g., `["Hsig", "Tp", "Wdir"]`).
- `addVars`: The variable containing the shoreline $t-1$ timeseries as the "add" variable - e.g., each batch prediction starts predicting from $y_{t-1}$.
- `yStd`: The variable containing the standard deviation of the shoreline timeseries.
- `yVars`: The output variables of the model.
- `trainCombs`: The train test splits for CV training. A value of 1 indicates the test set and a value of 2 indicates the train set so e.g., [1,2,2,2,2] would indicate a 20% test set at the beginning of the timeseries.
- `peakBool`: A boolean flag indicating whether to include peak values in the model.
- `histBool`: A boolean flag indicating whether to include $y_{t-1}$ as an input variable in the model. If true the model undertakes both *training and testing* using an autoregressive approach where the previous *modelled* output is the input for the next prediction in the timeseries.
- `ar1`: A boolean flag indicating whether to use an autoregressive approach during *testing only* which allows for a model trained with the true $y_{t-1}$ values but then recursively predicting into the future.
- `numLayers`: The number of layers in the neural network.
- `sizeFac`: The size factor for the neural network - gives the size of the first hidden layer and then each subsequent layer is half the size of the layer preceding (with a min size of 12).
- `dropoutRate`: The dropout rate for the neural network.
- `learningRate`: The learning rate for the neural network.
- `learningDecay`: The learning decay rate for the neural network.
- `scalerType`: The type of scaler used to normalize the input variables (e.g., "standard" or "minmax")
- `batchSize`: The batch size for training the neural network.
- `epochs`: The number of epochs for training the neural network.


# Data specification

