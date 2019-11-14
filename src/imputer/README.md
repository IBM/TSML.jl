# Impute
[![stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://invenia.github.io/Impute.jl/stable/)
[![latest](https://img.shields.io/badge/docs-latest-blue.svg)](https://invenia.github.io/Impute.jl/latest/)
[![Build Status](https://travis-ci.org/invenia/Impute.jl.svg?branch=master)](https://travis-ci.org/invenia/Impute.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/github/invenia/Impute.jl?svg=true)](https://ci.appveyor.com/project/invenia/Impute-jl)
[![codecov](https://codecov.io/gh/invenia/Impute.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/invenia/Impute.jl)

Impute.jl provides various methods for handling missing data in Vectors, Matrices and [Tables](https://github.com/JuliaData/Tables.jl).

## Installation
```julia
julia> using Pkg; Pkg.add("Impute")
```

## Quickstart
Let's start by loading our dependencies:
```julia
julia> using DataFrames, RDatasets, Impute
```

We'll also want some test data containing missings to work with:

```julia
julia> df = dataset("boot", "neuro")
469×6 DataFrames.DataFrame
│ Row │ V1       │ V2       │ V3      │ V4       │ V5       │ V6       │
│     │ Float64⍰ │ Float64⍰ │ Float64 │ Float64⍰ │ Float64⍰ │ Float64⍰ │
├─────┼──────────┼──────────┼─────────┼──────────┼──────────┼──────────┤
│ 1   │ missing  │ -203.7   │ -84.1   │ 18.5     │ missing  │ missing  │
│ 2   │ missing  │ -203.0   │ -97.8   │ 25.8     │ 134.7    │ missing  │
│ 3   │ missing  │ -249.0   │ -92.1   │ 27.8     │ 177.1    │ missing  │
│ 4   │ missing  │ -231.5   │ -97.5   │ 27.0     │ 150.3    │ missing  │
│ 5   │ missing  │ missing  │ -130.1  │ 25.8     │ 160.0    │ missing  │
│ 6   │ missing  │ -223.1   │ -70.7   │ 62.1     │ 197.5    │ missing  │
│ 7   │ missing  │ -164.8   │ -12.2   │ 76.8     │ 202.8    │ missing  │
⋮
│ 462 │ missing  │ -207.3   │ -88.3   │ 9.6      │ 104.1    │ 218.0    │
│ 463 │ -242.6   │ -142.0   │ -21.8   │ 69.8     │ 148.7    │ missing  │
│ 464 │ -235.9   │ -128.8   │ -33.1   │ 68.8     │ 177.1    │ missing  │
│ 465 │ missing  │ -140.8   │ -38.7   │ 58.1     │ 186.3    │ missing  │
│ 466 │ missing  │ -149.5   │ -40.3   │ 62.8     │ 139.7    │ 242.5    │
│ 467 │ -247.6   │ -157.8   │ -53.3   │ 28.3     │ 122.9    │ 227.6    │
│ 468 │ missing  │ -154.9   │ -50.8   │ 28.1     │ 119.9    │ 201.1    │
│ 469 │ missing  │ -180.7   │ -70.9   │ 33.7     │ 114.8    │ 222.5    │
```

Our first instinct might be to drop all observations, but this leaves us too few rows to work with:

```julia
julia> Impute.drop(df)
4×6 DataFrames.DataFrame
│ Row │ V1      │ V2      │ V3      │ V4      │ V5      │ V6      │
│     │ Float64 │ Float64 │ Float64 │ Float64 │ Float64 │ Float64 │
├─────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┤
│ 1   │ -247.0  │ -132.2  │ -18.8   │ 28.2    │ 81.4    │ 237.9   │
│ 2   │ -234.0  │ -140.8  │ -56.5   │ 28.0    │ 114.3   │ 222.9   │
│ 3   │ -215.8  │ -114.8  │ -18.4   │ 65.3    │ 171.6   │ 249.7   │
│ 4   │ -247.6  │ -157.8  │ -53.3   │ 28.3    │ 122.9   │ 227.6   │
```

We could try imputing the values with linear interpolation, but that still leaves missing
data at the head and tail of our dataset:

```julia
julia> Impute.interp(df)
469×6 DataFrames.DataFrame
│ Row │ V1       │ V2       │ V3      │ V4       │ V5       │ V6       │
│     │ Float64⍰ │ Float64⍰ │ Float64 │ Float64⍰ │ Float64⍰ │ Float64⍰ │
├─────┼──────────┼──────────┼─────────┼──────────┼──────────┼──────────┤
│ 1   │ missing  │ -203.7   │ -84.1   │ 18.5     │ missing  │ missing  │
│ 2   │ missing  │ -203.0   │ -97.8   │ 25.8     │ 134.7    │ missing  │
│ 3   │ missing  │ -249.0   │ -92.1   │ 27.8     │ 177.1    │ missing  │
│ 4   │ missing  │ -231.5   │ -97.5   │ 27.0     │ 150.3    │ missing  │
│ 5   │ missing  │ -227.3   │ -130.1  │ 25.8     │ 160.0    │ missing  │
│ 6   │ missing  │ -223.1   │ -70.7   │ 62.1     │ 197.5    │ missing  │
│ 7   │ missing  │ -164.8   │ -12.2   │ 76.8     │ 202.8    │ missing  │
⋮
│ 462 │ -241.025 │ -207.3   │ -88.3   │ 9.6      │ 104.1    │ 218.0    │
│ 463 │ -242.6   │ -142.0   │ -21.8   │ 69.8     │ 148.7    │ 224.125  │
│ 464 │ -235.9   │ -128.8   │ -33.1   │ 68.8     │ 177.1    │ 230.25   │
│ 465 │ -239.8   │ -140.8   │ -38.7   │ 58.1     │ 186.3    │ 236.375  │
│ 466 │ -243.7   │ -149.5   │ -40.3   │ 62.8     │ 139.7    │ 242.5    │
│ 467 │ -247.6   │ -157.8   │ -53.3   │ 28.3     │ 122.9    │ 227.6    │
│ 468 │ missing  │ -154.9   │ -50.8   │ 28.1     │ 119.9    │ 201.1    │
│ 469 │ missing  │ -180.7   │ -70.9   │ 33.7     │ 114.8    │ 222.5    │
```

Finally, we can chain multiple simple methods together to give a complete dataset:

```julia
julia> Impute.interp(df) |> Impute.locf() |> Impute.nocb()
469×6 DataFrames.DataFrame
│ Row │ V1       │ V2       │ V3      │ V4       │ V5       │ V6       │
│     │ Float64⍰ │ Float64⍰ │ Float64 │ Float64⍰ │ Float64⍰ │ Float64⍰ │
├─────┼──────────┼──────────┼─────────┼──────────┼──────────┼──────────┤
│ 1   │ -233.6   │ -203.7   │ -84.1   │ 18.5     │ 134.7    │ 222.7    │
│ 2   │ -233.6   │ -203.0   │ -97.8   │ 25.8     │ 134.7    │ 222.7    │
│ 3   │ -233.6   │ -249.0   │ -92.1   │ 27.8     │ 177.1    │ 222.7    │
│ 4   │ -233.6   │ -231.5   │ -97.5   │ 27.0     │ 150.3    │ 222.7    │
│ 5   │ -233.6   │ -227.3   │ -130.1  │ 25.8     │ 160.0    │ 222.7    │
│ 6   │ -233.6   │ -223.1   │ -70.7   │ 62.1     │ 197.5    │ 222.7    │
│ 7   │ -233.6   │ -164.8   │ -12.2   │ 76.8     │ 202.8    │ 222.7    │
⋮
│ 462 │ -241.025 │ -207.3   │ -88.3   │ 9.6      │ 104.1    │ 218.0    │
│ 463 │ -242.6   │ -142.0   │ -21.8   │ 69.8     │ 148.7    │ 224.125  │
│ 464 │ -235.9   │ -128.8   │ -33.1   │ 68.8     │ 177.1    │ 230.25   │
│ 465 │ -239.8   │ -140.8   │ -38.7   │ 58.1     │ 186.3    │ 236.375  │
│ 466 │ -243.7   │ -149.5   │ -40.3   │ 62.8     │ 139.7    │ 242.5    │
│ 467 │ -247.6   │ -157.8   │ -53.3   │ 28.3     │ 122.9    │ 227.6    │
│ 468 │ -247.6   │ -154.9   │ -50.8   │ 28.1     │ 119.9    │ 201.1    │
│ 469 │ -247.6   │ -180.7   │ -70.9   │ 33.7     │ 114.8    │ 222.5    │
```

**Warning:**

- Your approach should depend on the properties of you data (e.g., [MCAR, MAR, MNAR](https://en.wikipedia.org/wiki/Missing_data#Types_of_missing_data)).
- In-place calls aren't guaranteed to mutate the original data, but it will try avoid copying if possible.
  In the future, it may be possible to detect whether in-place operations are permitted on an array or table using traits:
    - https://github.com/JuliaData/Tables.jl/issues/116
    - https://github.com/JuliaDiffEq/ArrayInterface.jl/issues/22
