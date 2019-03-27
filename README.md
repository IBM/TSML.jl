# TSML (Time Series Machine Learning)

Julia 1.0: [![Build Status](https://travis-ci.com/IBM/TSML.jl.svg?branch=master)](https://travis-ci.com/IBM/TSML.jl)


TSML (Time Series Machine Learning) is package for Time Series data processing and prediction. It combines ML libraries from Python's ScikitLearn, R's Caret, and Julia using a common API and allows seamless ensembling and integration of heterogenous ML libraries to create complex models for robust time-series prediction.

The package assumes a two-column input composed of Date and Values columns. The first part of the workflow aggregates values based on the specified date/time interval which minimizes occurence of missing values and noise. The aggregated data is then left joined to the complete sequence of dates in a specified date/time interval. Remaining missing values are replaced by k nearest neighbors where k is the symmetric distance from the location of missing value. This approach can be called several times until there are no more missing values.

The next part extract the date features and convert the values into matrix form parameterized by the size and stride of the sliding window representing the dimension of the input for ML.

The final part combines the date features and the matrix of values as input to the ML with the output representing the value of the time period to be predicted ahead of time.

TSML uses a pipeline which iteratively calls fit! and transform! relying on multiple dispatch to do the corresponding algorithm of the steps outlined above.

Machine learning functions in TSML are wrappers to the corresponding Scikit-learn, Caret, and native Julia ML libraries. There are more than hundred classifiers and regression functions available using a common API. 

Below is an example of the pipeline workflow: (todo) 
