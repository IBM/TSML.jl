# Roadmap

### Release v1.0.0
- Transformer - abstract class with fit! and transform! interfaces to be overloaded
- TSLearner : Transformer - learners for classification/prediction with fit! function for training and transform! for prediction
- Baseline : TSLearner - base learner (returns the mode and usually the worst case)
- CaretLearner : TSLearner - API wrapper to expose caret regression/classification libs
- SKLearner : TSLearner - API wrapper to expose scikitlearn regression/classification libs
- Identity : Transformer - identity learner (returns mirror image)
- Imputer : Transformer - removes missing values
- Pipeline : Transformer - iteratively calls fit! and transform! to the set of transformers in the workflow
- DateValizer : Transformer - replace missings with medians grouped by datetime period
- DateValgator : Transformer - Aggregate values grouped by datetime period

### Release v1.0.1
- Matrifier : Transformer - transform vector of values into matrix by sliding windows
- Dateifier : Transformer - get the date boundaries in the sliding windows to correspond with matrifier output
- DateValNNer : Transformer - nearest neighbor replacement of missing data
- CSVDateValReader : Transformer - CSV reader
- CSVDateValWriter : Transformer - CSV writer

### Release v1.0.2
- RandomForest : TSLearner - RF regression/classification wrapper
- PrunedTree : TSLearner - decision tree regression/classification wrapper
- Adaboost : TSLearner - Adaboost regression/classification wrapper
- DataReader : Transformer - hdf5/feather/jld/csv multiformat reader
- DataWriter : Transformer - hdf5/feather/jld/csv multiforma writer
- Dockerization - dockerized notebook tutorial and dockerized TSML

## Future Work
- StatFilter : Transformer - scalar stats for data quality characterization
- MonotonicFilter : Transformer - convert monotonic data using finite difference operator
- Module for automatic classification of TS data type
- API wrapper for KITT interaction
- API wrapper for E2D interaction
- Webserver module to receive/process/output data
- Higher-level API for parameter optimization during prediction/classification
