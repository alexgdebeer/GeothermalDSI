# Data Space Inversion for Efficient Predictions and Uncertainty Quantification for Geothermal Models

![3D Reservoir Model](3d_model.png)

This repository contains code written as part of the paper "Data Space Inversion for Efficient Predictions and Uncertainty Quantification for Geothermal Models", by Alex de Beer, Andrew Power, Daniel Wong, Ken Dekkers, Michael Gravatt, John P. O'Sullivan, Michael J. O'Sullivan, Oliver J. Maclaren, and Ruanui Nicholson.

## Simplified Two-Dimensional Reservoir Model

Code to run the experiments for the simplified two-dimensional reservoir model is contained in the `Model2D` folder. The model is written in Julia 1.9. After installing Julia and cloning this repository, open this folder and run
```
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```
to install the project dependencies. You will then be able to run any of the scripts at the top level of the folder.

## Three-Dimensional Reservoir Model

Code to run the experiments for the three-dimensional reservoir model is contained in the `Model3D` folder. The scripts in this folder require Python >= 3.8 to run. After installing Python and cloning this repository, open this folder and run
```
pip install requirements.txt
```
to install the project dependencies. You will also need to install the Waiwera geothermal simulator; for further information, consult the [Waiwera website](https://waiwera.github.io). 
 - To generate the ensemble of reservoir models used to estimate the covariance matrices required as part of the DSI algorithm, run `generate_ensemble.py`. The resulting simulation input files can then be run [locally using Docker](https://waiwera.readthedocs.io/en/latest/run.html) (though this is likely to take a while), or on a high-performance computing cluster which contains a Waiwera build. We have included the jobfile we used to run the simulations on the Maui cluster operated by NeSI (New Zealand eScience Infrastucture).
 - To gather the statistics of the quantities of interest, run `process_output.py`. This will ignore the results associated with failed simulations. 
 - To run the DSI algorithm to approximate the posterior predictive distribution of the quantities of interest, run `run_dsi.py` or `run_dsi_trans.py` (which will run the DSI algorithm with a transformation applied to the pressure predictions).


## Issues

If you have any questions about the code or the paper, please open an [issue](https://github.com/alexgdebeer/GeothermalDSI/issues).