module ChebyExp
using Base.LinAlg: BLAS.gemm!, BlasFloat, checksquare, tilebufsize, Abuf, Bbuf, Cbuf, rcswap!
import Base.A_mul_B!

export chebyexp

include("exp.jl")


# Utility functions
A_mul_B!(α::T, A::StridedVecOrMat{T}, B::StridedVecOrMat{T}, β::T, C::StridedMatrix{T}) where {T<:BlasFloat} = gemm!('N','N',α,A,B,β,C)

function A_mul_B!(α::Number, A::AbstractVecOrMat{T}, B::AbstractVecOrMat{S}, β::Number, C::AbstractVecOrMat{R}) where {T,S,R}
    mA, nA = size(A)
    mB, nB = size(B)
    if mB != nA
        throw(DimensionMismatch("matrix A has dimensions ($mA,$nA), matrix B has dimensions ($mB,$nB)"))
    end
    if size(C,1) != mA || size(C,2) != nB
        throw(DimensionMismatch("result C has dimensions $(size(C)), needs ($mA,$nB)"))
    end

    @inbounds begin
        if isbits(R) && isbits(T) && isbits(S)
            tile_size = floor(Int, sqrt(tilebufsize / max(sizeof(R), sizeof(S), sizeof(T))))
            sz = (tile_size, tile_size)
            Atile = unsafe_wrap(Array, convert(Ptr{T}, pointer(Abuf)), sz)
            Btile = unsafe_wrap(Array, convert(Ptr{S}, pointer(Bbuf)), sz)

            z1 = zero(A[1, 1]*B[1, 1] + A[1, 1]*B[1, 1])
            z = convert(promote_type(typeof(z1), R), z1)

            if mA < tile_size && nA < tile_size && nB < tile_size
                Base.copy_transpose!(Atile, 1:nA, 1:mA, A, 1:mA, 1:nA)
                copy!(Btile, 1:mB, 1:nB, B, 1:mB, 1:nB)
                for j = 1:nB
                    boff = (j-1)*tile_size
                    for i = 1:mA
                        aoff = (i-1)*tile_size
                        s = z
                        for k = 1:nA
                            s += Atile[aoff+k] * Btile[boff+k]
                        end
                        C[i,j] = α*s + β*C[i,j]
                    end
                end
            else
                Ctile = unsafe_wrap(Array, convert(Ptr{R}, pointer(Cbuf)), sz)
                for jb = 1:tile_size:nB
                    jlim = min(jb+tile_size-1,nB)
                    jlen = jlim-jb+1
                    for ib = 1:tile_size:mA
                        ilim = min(ib+tile_size-1,mA)
                        ilen = ilim-ib+1
                        fill!(Ctile, z)
                        for kb = 1:tile_size:nA
                            klim = min(kb+tile_size-1,mB)
                            klen = klim-kb+1
                            Base.copy_transpose!(Atile, 1:klen, 1:ilen, A, ib:ilim, kb:klim)
                            copy!(Btile, 1:klen, 1:jlen, B, kb:klim, jb:jlim)
                            for j = 1:jlen
                                bcoff = (j-1)*tile_size
                                for i = 1:ilen
                                    aoff = (i-1)*tile_size
                                    s = z
                                    for k = 1:klen
                                        s += Atile[aoff+k] * Btile[bcoff+k]
                                    end
                                    Ctile[bcoff+i] += s
                                end
                            end
                        end
                        for j = 1:jlen, i = 1:ilen
                            coff = (j-1)*tile_size+i
                            Ctile[coff] = α*Ctile[coff] + β*C[ib+i-1,jb+j-1]
                        end
                        copy!(C, ib:ilim, jb:jlim, Ctile, 1:ilen, 1:jlen)
                    end
                end
            end
        else # Multiplication for non-plain-data uses the naive algorithm
            for i = 1:mA, j = 1:nB
                z2 = zero(A[i, 1]*B[1, j] + A[i, 1]*B[1, j])
                Ctmp = convert(promote_type(R, typeof(z2)), z2)
                for k = 1:nA
                    Ctmp += A[i, k]*B[k, j]
                end
                C[i,j] = α*Ctmp + β*C[i,j]
            end
        end
    end # @inbounds
    return C
end

end
