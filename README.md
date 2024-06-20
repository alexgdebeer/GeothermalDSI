# Data Space Inversion for Geothermal

![3D Reservoir Model](3d_model.png)

This repository contains code written as part of the paper "Data Space Inversion for Efficient Predictions and Uncertainty Quantification for Geothermal Models", by Alex de Beer, Andrew Power, Daniel Wong, Ken Dekkers, Michael Gravatt, John P. O'Sullivan, Michael J. O'Sullivan, Oliver J. Maclaren, and Ruanui Nicholson.

## Simplified Two-Dimensional Reservoir

Code to run the experiments for the simplified two-dimensional reservoir model is contained the `Model2D` folder. The model is written in Julia 1.9. After installing Julia and cloning this repository, open this folder and run
```
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```
to install the project dependencies. You will then be able to run any of the scripts at the top level of the folder.

## Three-Dimensional Reservoir

Code to run the experiments for the three-dimensional reservoir model is contained the `Model3D` folder. The scripts in this folder require Python >= 3.8 to run. After installing Python and cloning this repository, open this folder and run
```
pip install requirements.txt
```
to install the project dependencies. You will then be able to run any of the scripts at the top level of the folder.

## Issues

If you have any questions about the code or the paper, please open an [issue](https://github.com/alexgdebeer/GeothermalDSI/issues).