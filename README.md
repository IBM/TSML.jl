# TSML (Time Series Machine Learning)

Julia 1.0: [![Build Status](https://travis-ci.org/ppalmes/TSML.jl.svg?branch=master)](https://travis-ci.org/ppalmes/TSML.jl)


TSML (Time Series Machine Learning) is package for Time Series data processing, classification, and prediction. It combines ML libraries from Python's ScikitLearn, R's Caret, and Julia using a common API and allows seamless ensembling and integration of heterogenous ML libraries to create complex models for robust time-series prediction.

The package assumes a two-column input composed of Dates and Values. The first part of the workflow aggregates values based on the specified date/time interval which minimizes occurence of missing values and noise. The aggregated data is then left-joined to the complete sequence of dates in a specified date/time interval. Remaining missing values are replaced by k nearest neighbors where k is the symmetric distance from the location of missing value. This approach can be called several times until there are no more missing values.

The next part extracts the date features and convert the values into matrix form parameterized by the _size_ and _stride_ of the sliding window representing the dimension of the input for ML training and prediction.

The final part combines the date features and the matrix of values as input to the ML with the output representing the values of the time periods to be predicted ahead of time.

TSML uses a pipeline which iteratively calls the __fit__ and __transform__ families of functions relying on multiple dispatch to select the correct algorithm from the steps outlined above.

Machine learning functions in TSML are wrappers to the corresponding Scikit-learn, Caret, and native Julia ML libraries. There are more than hundred classifiers and regression functions available using a common API. 

Below is an example of the pipeline workflow: (todo) 
