using Random, LinearAlgebra, Distributions, Statistics, Optim, DataFrames, CSV, HTTP, GLM, FreqTables, ForwardDiff

cd(@__DIR__)

# Read in the function
include("Ps2_Nusrat.jl")

# Execute the function
run_all()
