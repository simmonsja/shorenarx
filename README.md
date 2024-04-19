# Code to accompany "Data-driven shoreline modelling at timescales of days to years"
Joshua Simmons and Kristen Splinter 2024

This repository contains the code to reproduce the figures in the publication "Data-driven shoreline modelling at timescales of days to years" by Simmons and Splinter (2024). The code is written in Python and uses the pytorch library to train a neural network to predict the shoreline position at timescales of days to years.

# Notebooks

- `01a_Publication_Optuna_Analysis.ipynb`: Narrabeen analysis of the Optuna hyperparameter search to find appropriate settings for the predictive model.
- `01b_Publication_Optuna_Analysis.ipynb`: Tairua analysis of the Optuna hyperparameter search to find appropriate settings for the predictive model.
- `02_Model_Training_Example.ipynb`: Example of how to train the model using the optimal hyperparameters found in the previous notebook.
- `03_Publication_Model_Performance.ipynb`: Analysis of the model performance and various tests. This notebook is the main notebook to reproduce the figures in the publication.
- `04_Publication_Data_Requirements.ipynb`: Specific analysis of data requirements aspects of the model. This produces additional figures used in the publication.
- `99_Location_figure.ipynb`: Produces the location figure used in the publication.

All functions used in the notebooks for analysis and plotting are stored in `shoreNARX`. This include the neural network models themselves, and the classes required to prepared data and train the models.

## Scripts

- `01_Optuna_Run_Script.py`: Script to run the Optuna hyperparameter search given a config file.
- `02a_CV_Run_Script.py`: Script to train the model given a config file.

# Config files
The config files are stored in `config/` as json files converted to dict upon loading. The config files are used to set the hyperparameters for the model and various decisions around the train/test split, input variables, and output variables. Typical settings include:

- `site`: The site for which the configuration is set. ("narra" or "tairua")
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
- `satperc`: The percentage of training target variable data to keep (e.g., 20% keeps every 5th shoreline position).
- `satnoise`: The standard deviation of the noise added to the training target variable data. In the units of the target (m for shoreline position).
- `numLayers`: The number of layers in the neural network.
- `sizeFac`: The size factor for the neural network - gives the size of the first hidden layer and then each subsequent layer is half the size of the layer preceding (with a min size of 12).
- `dropoutRate`: The dropout rate for the neural network.
- `learningRate`: The learning rate for the neural network.
- `learningDecay`: The learning decay rate for the neural network.
- `scalerType`: The type of scaler used to normalize the input variables (e.g., "standard" or "minmax")
- `batchSize`: The batch size for training the neural network.
- `epochs`: The number of epochs for training the neural network.

# Data specification
The data are stored in `input_data/processed/` as csv files. The 0th column stores the index as a datetime (e.g., '2019-01-01 12:00:00') and the subsequent columns store the covariates and target variables (see `config/` for the variable names used). Each row corresponds to one timestep as specified by the `ySample` in the config file and may have missing values where observations are not available. If `xSample` < `ySample` e.g., for the `xSample` = 3H, `ySample` = 3D then the repeated variables are stored as `varname_0`, `varname_1`, etc for, in this case, the 24 3-hourly inputs across the 3 days.

Must contain the variables as specified in `inputVars`, `addVars`, `yVars`, and `yStd` in the config file.

# Environment
The dockerfile (`docker/shoreNARX.dockerfile`) can be used to setup a docker environment for running the code. It lists all the required dependencies for the code to run (see [here](https://docs.docker.com/reference/cli/docker/image/build/)).
