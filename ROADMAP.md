# Roadmap
__Authored by__: _Paulito P. Palmes_ ([ppalmes-ibm](https://github.com/ppalmes))

### Release v1.0.0 (Base data structures and ML wrappers)
- Transformer - abstract class with __fit__ and __transform__ interfaces to be overloaded
- TSLearner <: Transformer - learners for classification/prediction with __fit__ function for training and __transform__ for prediction
- Baseline <: TSLearner - returns the __mode__ for classification and usually provides the worst case result
- CaretLearner <: TSLearner - API wrapper to expose caret regression/classification libs
- SKLearner <: TSLearner - API wrapper to expose scikitlearn regression/classification libs
- Identity <: Transformer - identity learner (returns mirror image)
- Imputer <: Transformer - removes missing values
- Pipeline <: Transformer - iteratively calls fit! and transform! to the set of transformers in the workflow
- DateValizer <: Transformer - replace missings with medians grouped by datetime period
- DateValgator <: Transformer - Aggregate values grouped by datetime period

### Release v1.0.1 (Matrify TS for ML workflow)
- Matrifier <: Transformer - transform vector of values into matrix by sliding windows
- Dateifier <: Transformer - get the date boundaries in the sliding windows to correspond with matrifier output
- DateValNNer <: Transformer - nearest neighbor replacement of missing data
- CSVDateValReader <: Transformer - CSV reader
- CSVDateValWriter <: Transformer - CSV writer

### Release v1.0.6 (Ensemble wrappers, multiformat data readers/writers)
- RandomForest <: TSLearner - RF regression/classification wrapper
- PrunedTree <: TSLearner - decision tree regression/classification wrapper
- Adaboost <: TSLearner - Adaboost regression/classification wrapper
- DataReader <: Transformer - hdf5/feather/jld/csv multiformat reader
- DataWriter <: Transformer - hdf5/feather/jld/csv multiformat writer
- Dockerization - dockerized notebook tutorial and dockerized TSML

## Release v1.0.7
- Statifier <: Transformer - scalar stats for data quality characterization
- MonotonicFilter <: Transformer - convert monotonic data using finite difference operator
- TSClassifier <: Transformer - automatic classification of TS data type
- High-level wrapper for CLI automation and interfacing with other programs in the docker/shell
- Dockerized branch for Kubernetes deployment

## Future Work (Higher-level reasoning and integration APIs)
- XGBoost wrapper
- API wrapper for KITT interaction
- API wrapper for E2D interaction
- Webserver module to receive/process/output data
- Higher-level API for parameter optimization during prediction/classification
