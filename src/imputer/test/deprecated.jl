@testset "deprecated" begin
    a = allowmissing(1.0:1.0:20.0)
    a[[2, 3, 7]] .= missing
    mask = map(!ismissing, a)

    @testset "Drop" begin
        result = impute(a, :drop; limit=0.2)
        expected = copy(a)
        deleteat!(expected, [2, 3, 7])

        @test result == expected

        # Mutating method
        a2 = copy(a)
        Impute.drop!(a2; limit=0.2)
        @test a2 == expected
    end

    @testset "Interpolate" begin
        result = impute(a, :interp; limit=0.2)
        @test result == collect(1.0:1.0:20)
        @test result == interp(a)

        # Test in-place method
        a2 = copy(a)
        Impute.interp!(a2; limit=0.2)
        @test a2 == result

        # Test interpolation between identical points
        b = ones(Union{Float64, Missing}, 20)
        b[[2, 3, 7]] .= missing
        @test interp(b) == ones(Union{Float64, Missing}, 20)

        # Test interpolation at endpoints
        b = ones(Union{Float64, Missing}, 20)
        b[[1, 3, 20]] .= missing
        result = interp(b)
        @test ismissing(result[1])
        @test ismissing(result[20])
    end

    @testset "Fill" begin
        @testset "Value" begin
            fill_val = -1.0
            result = impute(a, :fill, fill_val; limit=0.2)
            expected = copy(a)
            expected[[2, 3, 7]] .= fill_val

            @test result == expected
        end

        @testset "Mean" begin
            result = impute(a, :fill; limit=0.2)
            expected = copy(a)
            expected[[2, 3, 7]] .= mean(a[mask])

            @test result == expected

            a2 = copy(a)
            Impute.fill!(a2; limit=0.2)
            @test a2 == result
        end
    end

    @testset "LOCF" begin
        result = impute(a, :locf; limit=0.2)
        expected = copy(a)
        expected[2] = 1.0
        expected[3] = 1.0
        expected[7] = 6.0

        @test result == expected
        a2 = copy(a)
        impute!(a2, :locf; limit=0.2)
        @test a2 == result
    end

    @testset "NOCB" begin
        result = impute(a, :nocb; limit=0.2)
        expected = copy(a)
        expected[2] = 4.0
        expected[3] = 4.0
        expected[7] = 8.0

        @test result == expected
        a2 = copy(a)
        Impute.nocb!(a2; limit=0.2)
        @test a2 == result
    end

    @testset "DataFrame" begin
        data = dataset("boot", "neuro")
        df = impute(data, :interp; limit=1.0)
    end

    @testset "Matrix" begin
        data = Matrix(dataset("boot", "neuro"))

        @testset "Drop" begin
            result = Iterators.drop(data)
            @test size(result, 1) == 4
        end

        @testset "Fill" begin
            result = impute(data, :fill, 0.0; limit=1.0)
            @test size(result) == size(data)
        end
    end

    @testset "Not enough data" begin
        @test_throws ImputeError impute(a, :drop)
    end

    @testset "Chain" begin
        orig = dataset("boot", "neuro")

        @testset "DataFrame" begin
            result = chain(
                orig,
                Impute.Interpolate(),
                Impute.LOCF(),
                Impute.NOCB();
                limit=1.0
            )

            @test size(result) == size(orig)
            # Confirm that we don't have any more missing values
            @test all(!ismissing, Matrix(result))
        end

        @testset "Column Table" begin
            data = Tables.columntable(orig)
            result = chain(
                data,
                Impute.Interpolate(),
                Impute.LOCF(),
                Impute.NOCB();
                limit=1.0
            ) |> Tables.matrix

            @test size(result) == size(orig)
            # Confirm that we don't have any more missing values
            @test all(!ismissing, result)
        end

        @testset "Matrix" begin
            data = Matrix(orig)
            result = chain(
                data,
                Impute.Interpolate(),
                Impute.LOCF(),
                Impute.NOCB();
                limit=1.0
            )

            @test size(result) == size(data)
            # Confirm that we don't have any more missing values
            @test all(!ismissing, result)
        end
    end

    @testset "Alternate missing functions" begin
        data1 = dataset("boot", "neuro")                # Missing values with `missing`
        data2 = impute(data1, :fill, NaN; limit=1.0)     # Missing values with `NaN`

        @test impute(data1, :drop; limit=1.0) == dropmissing(data1)

        result1 = chain(data1, Impute.Interpolate(), Impute.Drop(); limit=1.0)
        result2 = chain(data2, isnan, Impute.Interpolate(), Impute.Drop(); limit=1.0)
        @test result1 == result2

        @test Impute.drop(data1; limit=1.0) == impute(data2, isnan, :drop; limit=1.0)
    end
end
