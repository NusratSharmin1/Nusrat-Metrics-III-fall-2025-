using Test, JLD, CSV, DataFrames, Random, LinearAlgebra, Statistics, Distributions

include("Nusrat ps1.jl")

@testset "q1 function" begin
    A, B, C, D = q1()
    @test size(A) == (10, 7)
    @test size(B) == (10, 7)
    @test size(C, 1) == 5
    @test size(D) == (10, 7)
    @test isfile("matrixpractice.jld")
    @test isfile("firstmatrix.jld")
    @test isfile("Cmatrix.csv")
    @test isfile("Dmatrix.dat")
end

@testset "matrixops function" begin
    A, B, C, D = q1()
    out1, out2, out3 = matrixops(A, B)
    @test size(out1) == size(A)
    @test size(out2, 1) == size(A, 2)
    @test isa(out3, Number)
    @test_throws ErrorException matrixops(A, C) # Should throw error for size mismatch
end

@testset "q2 function" begin
    A, B, C, D = q1()
    @test q2(A, B, C) === nothing
end

@testset "q3 function" begin
    @test q3() === nothing
    @test isfile("nlsw88_cleaned.csv")
end

@testset "q4 function" begin
    q4_result = q4()
    @test isa(q4_result, Tuple)
end

println("All tests completed.")

