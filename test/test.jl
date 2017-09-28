using ChebyExp, MatrixDepot

# random 10x10
n = 1000
R = Matrix{Float64}(n,2)
A = Matrix{Float64}(10,10)
for i in 1:n
    rand!(A)
    R[i,1] = norm(expm(A)*expm(-A)-I,2)/norm(expm(A),2)
    R[i,2] = norm(chebyexp(A)*chebyexp(-A)-I,2)/norm(chebyexp(A),2)
end
a,b = mean(R,1), std(R,1)
@printf("""
Random 10x10 matrices:
expm error:     %1.2G ± %1.2G
chebyexp error: %1.2G ± %1.2G
""",a[1],b[1],a[2],b[2])


# Chebyshev and Forsythe
R = Matrix{Float64}(2,2)
for (i,A) in enumerate([matrixdepot("chebspec",10), matrixdepot("forsythe",10)])
    R[i,1] = norm(expm(A)*expm(-A)-I,2)/norm(expm(A),2)
    R[i,2] = norm(chebyexp(A)*chebyexp(-A)-I,2)/norm(chebyexp(A),2)
end
a,b = mean(R,1), std(R,1)
@printf("""
Chebyshev and Forsythe 10x10 matrices:
expm error:     %1.2G ± %1.2G
chebyexp error: %1.2G ± %1.2G
""",a[1],b[1],a[2],b[2])

# Ill-conditioned
R = Matrix{Float64}(length(matrixdepot("ill-cond")),2)
for (i,nameA) in enumerate(matrixdepot("ill-cond"))
    A = matrixdepot(nameA,Float64,16)
    norm(full(A))>5 && (A /= norm(full(A))/4)
    R[i,1] = norm(expm(full(A))*expm(-full(A))-I,2)/norm(expm(full(A)),2)
    R[i,2] = norm(chebyexp(A)*chebyexp(-A)-I,2)/norm(chebyexp(A),2)
end
a,b = mean(R,1), std(R,1)
@printf("""
Ill-conditioned 16x16 matrices:
expm error:     %1.2G ± %1.2G
chebyexp error: %1.2G ± %1.2G
""",a[1],b[1],a[2],b[2])
