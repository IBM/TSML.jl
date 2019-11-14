using Impute
using Tables
using Test
using AxisArrays
using DataFrames
using Dates
using RDatasets
using Statistics
using StatsBase
using Random

import Impute:
    Drop,
    DropObs,
    DropVars,
    Interpolate,
    Fill,
    LOCF,
    NOCB,
    SRS,
    Context,
    WeightedContext,
    ImputeError


@testset "Impute" begin
    # Defining our missing datasets
    a = allowmissing(1.0:1.0:20.0)
    a[[2, 3, 7]] .= missing
    mask = map(!ismissing, a)
    ctx = Context(; limit=0.2)

    # We call collect to not have a wrapper type that references the same data.
    m = collect(reshape(a, 5, 4))

    aa = AxisArray(
        deepcopy(m),
        Axis{:time}(DateTime(2017, 6, 5, 5):Hour(1):DateTime(2017, 6, 5, 9)),
        Axis{:id}(1:4)
    )

    table = DataFrame(
        :sin => allowmissing(sin.(1.0:1.0:20.0)),
        :cos => allowmissing(sin.(1.0:1.0:20.0)),
    )

    table.sin[[2, 3, 7, 12, 19]] .= missing

    @testset "Equality" begin
        @testset "$T" for T in (DropObs, DropVars, Interpolate, Fill, LOCF, NOCB, SRS)
            @test T() == T()
        end
    end

    @testset "Drop" begin
        @testset "DropObs" begin
            @testset "Vector" begin
                result = impute(a, DropObs(; context=ctx))
                expected = deleteat!(deepcopy(a), [2, 3, 7])

                @test result == expected
                @test result == Impute.dropobs(a; context=ctx)

                a2 = deepcopy(a)
                Impute.dropobs!(a2; context=ctx)
                @test a2 == expected
            end

            @testset "Matrix" begin
                # Because we're removing 2 of our 5 rows we need to change the limit.
                ctx = Context(; limit=0.4)
                result = impute(m, DropObs(; context=ctx))
                expected = m[[1, 4, 5], :]

                @test isequal(result, expected)
                @test isequal(result, Impute.dropobs(m; context=ctx))
                @test isequal(collect(result'), Impute.dropobs(collect(m'); dims=2, context=ctx))

                m_ = Impute.dropobs!(m; context=ctx)
                # The mutating test is broken because we need to making a copy of
                # the original matrix
                @test_broken isequal(m, expected)
                @test isequal(m_, expected)
            end

            @testset "Tables" begin
                ctx = Context(; limit=0.4)
                @testset "DataFrame" begin
                    df = deepcopy(table)
                    result = impute(df, DropObs(; context=ctx))
                    expected = dropmissing(df)

                    @test isequal(result, expected)
                    @test isequal(result, Impute.dropobs(df; context=ctx))

                    df_ = Impute.dropobs!(df; context=ctx)
                    # The mutating test is broken because we need to making a copy of
                    # the original table
                    @test_broken isequal(df, expected)
                    @test isequal(df_, expected)
                end

                @testset "Column Table" begin
                    coltab = Tables.columntable(table)

                    result = impute(coltab, DropObs(; context=ctx))
                    expected = Tables.columntable(dropmissing(table))

                    @test isequal(result, expected)
                    @test isequal(result, Impute.dropobs(coltab; context=ctx))

                    coltab_ = Impute.dropobs!(coltab; context=ctx)
                    # The mutating test is broken because we need to making a copy of
                    # the original table
                    @test_broken isequal(coltab, expected)
                    @test isequal(coltab_, expected)
                end

                @testset "Row Table" begin
                    rowtab = Tables.rowtable(table)
                    result = impute(rowtab, DropObs(; context=ctx))
                    expected = Tables.rowtable(dropmissing(table))

                    @test isequal(result, expected)
                    @test isequal(result, Impute.dropobs(rowtab; context=ctx))

                    rowtab_ = Impute.dropobs!(rowtab; context=ctx)
                    # The mutating test is broken because we need to making a copy of
                    # the original table
                    # @test_broken isequal(rowtab, expected)
                    @test isequal(rowtab_, expected)
                end
            end

            @testset "AxisArray" begin
                # Because we're removing 2 of our 5 rows we need to change the limit.
                ctx = Context(; limit=0.4)
                result = impute(aa, DropObs(; context=ctx))
                expected = aa[[1, 4, 5], :]

                @test isequal(result, expected)
                @test isequal(result, Impute.dropobs(aa; context=ctx))

                aa_ = Impute.dropobs!(aa; context=ctx)
                # The mutating test is broken because we need to making a copy of
                # the original matrix
                @test_broken isequal(aa, expected)
                @test isequal(aa_, expected)
            end
        end

        @testset "DropVars" begin
            @testset "Vector" begin
                @test_throws MethodError Impute.dropvars(a)
            end

            @testset "Matrix" begin
                ctx = Context(; limit=0.5)
                result = impute(m, DropVars(; context=ctx))
                expected = copy(m)[:, 3:4]

                @test isequal(result, expected)
                @test isequal(result, Impute.dropvars(m; context=ctx))
                @test isequal(collect(result'), Impute.dropvars(collect(m'); dims=2, context=ctx))

                m_ = Impute.dropvars!(m; context=ctx)
                # The mutating test is broken because we need to making a copy of
                # the original matrix
                @test_broken isequal(m, expected)
                @test isequal(m_, expected)
            end

            @testset "Tables" begin
                @testset "DataFrame" begin
                    df = deepcopy(table)
                    result = impute(df, DropVars(; context=ctx))
                    expected = select(df, :cos)

                    @test isequal(result, expected)
                    @test isequal(result, Impute.dropvars(df; context=ctx))

                    Impute.dropvars!(df; context=ctx)
                    # The mutating test is broken because we need to making a copy of
                    # the original table
                    @test_broken isequal(df, expected)
                end

                @testset "Column Table" begin
                    coltab = Tables.columntable(table)

                    result = impute(coltab, DropVars(; context=ctx))
                    expected = Tables.columntable(Tables.select(coltab, :cos))

                    @test isequal(result, expected)
                    @test isequal(result, Impute.dropvars(coltab; context=ctx))

                    Impute.dropvars!(coltab; context=ctx)
                    # The mutating test is broken because we need to making a copy of
                    # the original table
                    @test_broken isequal(coltab, expected)
                end

                @testset "Row Table" begin
                    rowtab = Tables.rowtable(table)
                    result = impute(rowtab, DropVars(; context=ctx))
                    expected = Tables.rowtable(Tables.select(rowtab, :cos))

                    @test isequal(result, expected)
                    @test isequal(result, Impute.dropvars(rowtab; context=ctx))

                    Impute.dropvars!(rowtab; context=ctx)
                    # The mutating test is broken because we need to making a copy of
                    # the original table
                    @test_broken isequal(rowtab, expected)
                end
            end
            @testset "AxisArray" begin
                ctx = Context(; limit=0.5)
                result = impute(aa, DropVars(; context=ctx))
                expected = copy(aa)[:, 3:4]

                @test isequal(result, expected)
                @test isequal(result, Impute.dropvars(aa; context=ctx))

                aa_ = Impute.dropvars!(aa; context=ctx)
                # The mutating test is broken because we need to making a copy of
                # the original matrix
                @test_broken isequal(aa, expected)
                @test isequal(aa_, expected)
            end
        end
    end

    @testset "Interpolate" begin
        result = impute(a, Interpolate(; context=ctx))
        @test result == collect(1.0:1.0:20)
        @test result == interp(a; context=ctx)

        # Test in-place method
        a2 = copy(a)
        Impute.interp!(a2; context=ctx)
        @test a2 == result

        # Test interpolation between identical points
        b = ones(Union{Float64, Missing}, 20)
        b[[2, 3, 7]] .= missing
        @test interp(b; context=ctx) == ones(Union{Float64, Missing}, 20)

        # Test interpolation at endpoints
        b = ones(Union{Float64, Missing}, 20)
        b[[1, 3, 20]] .= missing
        result = interp(b; context=ctx)
        @test ismissing(result[1])
        @test ismissing(result[20])
    end

    @testset "Fill" begin
        @testset "Value" begin
            fill_val = -1.0
            result = impute(a, Fill(; value=fill_val, context=ctx))
            expected = copy(a)
            expected[[2, 3, 7]] .= fill_val

            @test result == expected
            @test result == Impute.fill(a; value=fill_val, context=ctx)
        end

        @testset "Mean" begin
            result = impute(a, Fill(; value=mean, context=ctx))
            expected = copy(a)
            expected[[2, 3, 7]] .= mean(a[mask])

            @test result == expected
            @test result == Impute.fill(a; value=mean, context=ctx)

            a2 = copy(a)
            Impute.fill!(a2; context=ctx)
            @test a2 == result
        end

        @testset "Matrix" begin
            ctx = Context(; limit=1.0)
            expected = Matrix(Impute.dropobs(dataset("boot", "neuro"); context=ctx))
            data = Matrix(dataset("boot", "neuro"))

            result = impute(data, Fill(; value=0.0, context=ctx))
            @test size(result) == size(data)
            @test result == Impute.fill(data; value=0.0, context=ctx)

            data2 = copy(data)
            Impute.fill!(data2; value=0.0, context=ctx)
            @test data2 == result
        end
    end

    @testset "LOCF" begin
        result = impute(a, LOCF(; context=ctx))
        expected = copy(a)
        expected[2] = 1.0
        expected[3] = 1.0
        expected[7] = 6.0

        @test result == expected
        @test result == Impute.locf(a; context=ctx)

        a2 = copy(a)
        Impute.locf!(a2; context=ctx)
        @test a2 == result
    end

    @testset "NOCB" begin
        result = impute(a, NOCB(; context=ctx))
        expected = copy(a)
        expected[2] = 4.0
        expected[3] = 4.0
        expected[7] = 8.0

        @test result == expected
        @test result == Impute.nocb(a; context=ctx)

        a2 = copy(a)
        Impute.nocb!(a2; context=ctx)
        @test a2 == result
    end

    @testset "SRS" begin
        result = impute(a, SRS(; rng=MersenneTwister(137), context=ctx))
        expected = copy(a)
        expected[2] = 9.0
        expected[3] = 16.0
        expected[7] = 17.0

        @test result == expected

        @test result == Impute.srs(a; rng=MersenneTwister(137), context=ctx)

        a2 = copy(a)

        Impute.srs!(a2; rng=MersenneTwister(137), context=ctx)
        @test a2 == result
    end

    @testset "Not enough data" begin
        ctx = Context(; limit=0.1)
        @test_throws ImputeError impute(a, DropObs(; context=ctx))
        @test_throws ImputeError Impute.dropobs(a; context=ctx)
    end

    @testset "Chain" begin
        orig = dataset("boot", "neuro")
        ctx = Context(; limit=1.0)

        @testset "DataFrame" begin
            result = Impute.interp(orig; context=ctx) |> Impute.locf!() |> Impute.nocb!()

            @test size(result) == size(orig)
            # Confirm that we don't have any more missing values
            @test all(!ismissing, Matrix(result))

            # We can also use the Chain type with explicit Imputor types
            result2 = impute(
                orig,
                Impute.Chain(
                    Impute.Interpolate(; context=ctx),
                    Impute.LOCF(),
                    Impute.NOCB()
                ),
            )

            # Test creating a Chain via Imputor composition
            imp = Impute.Interpolate(; context=ctx) ∘ Impute.LOCF() ∘ Impute.NOCB()
            result3 = impute(orig, imp)
            @test result == result2
            @test result == result3

            @testset "GroupedDataFrame" begin
                hod = repeat(1:24, 12 * 10)
                obj = repeat(1:12, 24 * 10)
                n = length(hod)

                df = DataFrame(
                    :hod => hod,
                    :obj => obj,
                    :val => allowmissing(
                        [sin(x) * cos(y) for (x, y) in zip(hod, obj)]
                    ),
                )

                df.val[rand(1:n, 20)] .= missing
                gdf1 = groupby(deepcopy(df), [:hod, :obj])
                gdf2 = groupby(df, [:hod, :obj])

                f1 = Impute.interp(; context=ctx) ∘ Impute.locf!() ∘ Impute.nocb!()
                f2 = Impute.interp!(; context=ctx) ∘ Impute.locf!() ∘ Impute.nocb!()

                result = mapreduce(f1, vcat, gdf1)
                @test df != result
                @test size(result) == (24 * 12 * 10, 3)
                @test all(!ismissing, Tables.matrix(result))

                # Test that we can also mutate the dataframe directly
                map(f2, gdf2)
                @test result == sort(df, (:hod, :obj))
            end
        end

        @testset "Column Table" begin
            result = Tables.columntable(orig) |>
                Impute.interp!(; context=ctx) |>
                Impute.locf!() |>
                Impute.nocb!() |>
                Tables.matrix

            @test size(result) == size(orig)
            # Confirm that we don't have any more missing values
            @test all(!ismissing, result)
        end

        @testset "Row Table" begin
            result = Tables.rowtable(orig) |>
                Impute.interp!(; context=ctx) |>
                Impute.locf!() |>
                Impute.nocb!() |>
                Tables.matrix

            @test size(result) == size(orig)
            # Confirm that we don't have any more missing values
            @test all(!ismissing, result)
        end

        @testset "Matrix" begin
            data = Matrix(orig)
            result = Impute.interp(data; context=ctx) |> Impute.locf!() |> Impute.nocb!()

            @test size(result) == size(data)
            # Confirm that we don't have any more missing values
            @test all(!ismissing, result)
        end

        @testset "AxisArray" begin
            data = AxisArray(
                Matrix(orig),
                Axis{:row}(1:size(orig, 1)),
                Axis{:V}(names(orig)),
            )
            result = Impute.interp(data; context=ctx) |> Impute.locf!() |> Impute.nocb!()

            @test size(result) == size(data)
            # Confirm that we don't have any more missing values
            @test all(!ismissing, result)
        end
    end

    @testset "Alternate missing functions" begin
        ctx1 = Context(; limit=1.0)
        ctx2 = Context(; limit=1.0, is_missing=isnan)
        data1 = dataset("boot", "neuro")                    # Missing values with `missing`
        data2 = Impute.fill(data1; value=NaN, context=ctx1)  # Missing values with `NaN`

        @test Impute.dropobs(data1; context=ctx1) == dropmissing(data1)

        result1 = Impute.interp(data1; context=ctx1) |> Impute.dropobs()
        result2 = Impute.interp(data2; context=ctx2) |> Impute.dropobs(; context=ctx2)

        @test result1 == result2
    end

    @testset "Contexts" begin
        @testset "Base" begin
            ctx = Context(; limit=0.1)
            @test_throws ImputeError Impute.dropobs(a; context=ctx)
            @test_throws ImputeError impute(a, DropObs(; context=ctx))
        end

        @testset "Weighted" begin
            # If we use an exponentially weighted context then we won't pass the limit
            # because missing earlier observations is less important than later ones.
            ctx = WeightedContext(eweights(20, 0.3); limit=0.1)
            @test isa(ctx, WeightedContext)
            result = impute(a, DropObs(; context=ctx))
            expected = copy(a)
            deleteat!(expected, [2, 3, 7])
            @test result == expected

            # If we reverse the weights such that earlier observations are more important
            # then our previous limit of 0.2 won't be enough to succeed.
            ctx = WeightedContext(reverse!(eweights(20, 0.3)); limit=0.2)
            @test_throws ImputeError impute(a, DropObs(; context=ctx))
        end
    end

    @testset "Utils" begin
        M = [1.0 2.0 3.0 4.0 5.0; 1.1 2.2 3.3 4.4 5.5]

        @testset "obswise" begin
            @test map(sum, Impute.obswise(M; dims=2)) == [2.1, 4.2, 6.3, 8.4, 10.5]
            @test map(sum, Impute.obswise(M; dims=1)) == [15, 16.5]
        end

        @testset "varwise" begin
            @test map(sum, Impute.varwise(M; dims=2)) == [15, 16.5]
            @test map(sum, Impute.varwise(M; dims=1)) == [2.1, 4.2, 6.3, 8.4, 10.5]
        end

        @testset "filterobs" begin
            @test Impute.filterobs(x -> sum(x) > 5.0, M; dims=2) == M[:, 3:5]
            @test Impute.filterobs(x -> sum(x) > 15.0, M; dims=1) == M[[false, true], :]
        end

        @testset "filtervars" begin
            @test Impute.filtervars(x -> sum(x) > 15.0, M; dims=2) == M[[false, true], :]
            @test Impute.filtervars(x -> sum(x) > 5.0, M; dims=1) == M[:, 3:5]
        end
    end

    include("deprecated.jl")
end
