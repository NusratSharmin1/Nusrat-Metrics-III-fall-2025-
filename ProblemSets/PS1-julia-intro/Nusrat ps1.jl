
using test,JLD, Random, LinearAlgebra, Statistics, CSV, DataFrames, FreqTables, Distributions
# set the seed
using Random
Random.seed!(1234)
function q1()
#--------------------------------------------------------------------------------------------------------
#question 1, part (a)
#--------------------------------------------------------------------------------------------------------
# Draw uniform random numbers
A = -5 .+ 15*rand(10,7)
A = rand(Uniform(-5, 10), 10, 7)

#Draw normal random numbers
B = -2 .+ 15*rand(10, 7)
B = rand(Normal( -2, 15), 10, 7)

# Indexing 
C = [A[1:5, 1:5] B[1:5, 6:7]]
D = ifelse.(A .<= 0, A, 0)
#------------------------------------------------------------------------------------------------------
#1b
#-----------------------------------------------------------------------------------------------------
#number of elements
size(A)
size(A, 1) * size(A, 2)
size(A[:])
#--------------------------------------------------------------------------------------------------------
#1c
#-------------------------------------------------------------------------------------------
length(D)
length(unique(D))
#------------------------------------------------------------------------------------------
#1d
#-----------------------------------------
E = reshape(B, 70, 1)
E = reshape(B, (70, 1))
E = reshape(B, length(B), 1)
E = reshape(B, size(B, 1) * size(B,2), 1)
E =B[:]

#------------------------------------------------
#1E
#-------------------------------
F = cat(A, B; dims=3)
#-----------------------------------------------
#1F
#-------------------------
F =permutedims(F, (3, 1, 2))
#---------------
#1g
#-----------------
#Create a matrix g which is equal to B
G = kron(B, C)
# G = kron(C, F) # this doesnot work
#-------------------
#1h
#-----------------
#save the matrices A, B, C, D, E, F and G as a .jld file named matrixpractice.
save("matrixpractice.jld" ,"A", A, "B", B, "C", C, "D", D,"E", E, "F", F,"G", G)
#---------------------------
#1j
#--------------------------
DataFrame(C,:auto)

CSV.write("Cmatrix.csv")
#--------------
#1k
#-----------------
df_D = DataFrame(D, :auto)
CSV.write("Dmatrix.dat", df_D, delim='\t')
CSV.write("Cmatrix.csv")
DataFrame(C,:auto)

return A, B, C, D
end



#
#q2 A

function q2(A, B, C, D)
    A, B, C, D = q1()
function q2(A, B)
AB = zeros(size(A))
for r in 1:eachindex(A[:, 1])
    for c in 1:eachindex(A[1, :])
        AB[r, c] = A[r,c] *B[r, c]

end

#Part B
Cprime = []
for r in 1:eachindex(C[:, 1])
    for c in 1:eachindex(C[1,:])


    end
    Cprime2 =C[(C .>= -5) .& (C .<=5)]
    #compare the two vectors
Cprime == Cprime2
if Cprime != Cprime2
    @show Cprime
    @show Cprime2

    
    print(Cprime .== Cprime2)
    error("Cprime and Cprime2 are not the same")


#part C
X = zeroes(15_169, 6,5)

#ordering the 2nd dimension:
#intercept
#dummy variable
#continuous variable (normal)
#normal
#binomial ("discrete" normal)
#another binomial
for i in axes(X,1)
X[i, 1, :] .=1.0
X[i, 5, :] .= rand(binomial(20, 0.6))
X[i, 6, :] .=rand(binomial(20, 0.5))

for t in axes(X,3)
    X[i, 2, t] = rand() <= .75 * (6- t) /5
X[i, 3, t] .= rand(normal(15+ t -1, 5*(t-1)))
X[i, 6, t] .=rand(normal(pai*(6 - t), 1/exp))
end

#part D
#comprehesions practice
X = [t for t in 1:10]
β = zeros(K,T)
β = [1:1.25:3]







#2e
Y = zeros(N,T)
Y = [X[:, :, t] * β[:, t] .+ rand(normal(0, 0.36),N) for t in 1:T]

end


function q3()
    #q3
    df = DataFrame(CSV.File("nlsw88.csv"))
    @show df[1:5, :]
    @show typeof(df[:, :grade])

    #partb
    #part C
    #percentage never married

    @show freqtable(df[:, :race])
    #part D
    vars = names(df)
    @summarystats = describe(df)
    @show summarystats
    #part E
    show freqtable(df[:, industry], df[:, :occupation])

    #part F
    df_sub = df[:, [industry, :occupation, :wage]]
    grouped = groupby(df_sub, [:industry, :occupation])
    mean_wage = combine(grouped, :wage => mean =>)
    @show mean_wage






    return nothing
end


#4b and c
"""
matrixops(A, B)
performs the following operations on two matrices A and B:
(1) element-wise product of A and B
(2) compute the matrix product  of 
"""


function matrixops(A: Array, B)
    #part(e) : check dimesion compatibility
    if size(A) != size(B)
        error("Matrices A and B must have the same dimensionselements")
    end
    #(1) element-wise product of A and B
    out1 = A .* B
    #(2) ,atrix product of A' and B
    out2 = A' * B
    #(3) sum of all elements of sum of A and B
    out3 = sum(A + B)
    return out1, out2, out3
end


function q4()
    #parta

    #load("matrixpractice.jld" ,"A", "B", "C", "D","E", "F","G")
    #load "matrixpractice.jld" A B C D E F G
    @load "matrixpractice.jld"
    #D
    matrixops(A, B)

    #part F
    try
        
        matrixops(C, D)
     catch e
        println("trying matrixops(C, D):")
     end
    #part G#read in processed CSV File
    nlsw88 = DataFrame(CSV.File("nlsw88_cleaned.csv"))
    ttlexp = convert(Array, nlsw88[:, :ttlexp])
    wage = convert(Array, nlsw88[:, :wage])
    matrixops(ttl_exp, wage)


    end



#call the function from q1
A, B, C = q1()
#call the function from q2
q2(A, B, C)
#call the function from q3
q3()
#call the function from q4
q4()






