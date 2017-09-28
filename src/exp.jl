"""
    chebyexp(A)

Compute the exponential of `A`. The exponential is calculated with a Chebyshev series expansion. For \$|A| ≤ 1\$, we have that

```math
\\exp(A) = J₀(i) + 2 ∑_{k=1}^∞ iᵏ Jₖ(-i) Tₖ(A)
```

where \$Jₖ\$ is the Bessel function of the first kind, and \$Tₖ\$ is a Chebyshev polynomial. If \$|A| > 1\$, scaling and squaring is used.
"""
chebyexp(A::AbstractMatrix{T}) where T<:Number = chebyexp!(Matrix{float(T)}(size(A)),A)

function chebyexp!(R::AbstractMatrix{<:Number},A::AbstractMatrix{<:Number})
    # Run alg
    normA = norm(A,1)
    if normA <= 1
        _chebyexp!(R,A)
    else
        p = ceil(Int,log2(normA))
        _chebyexp!(R,A/2^p)
        Q = similar(R)
        for i = 1:p
            copy!(Q,R)
            A_mul_B!(R,Q,Q)
        end
    end
    return R
end

function _chebyexp!(R::AbstractMatrix{T},A::AbstractMatrix{<:Number}) where T
    # only valid for |A|<1
    n = checksquare(A)
    if T <: BigFloat
        order, coeff0, coeff = 32, cheby_coeff0_big, cheby_coeff_big
    else
        order, coeff0, coeff = 16, cheby_coeff0, cheby_coeff
    end
    # Initialize required matrices
    fill!(R,zero(T))
    Tₖ₋₂ = similar(R); copy!(Tₖ₋₂,A)
    Tₖ₋₁ = eye(T,n,n); A_mul_B!(T(2),A,Tₖ₋₂,T(-1),Tₖ₋₁)
    # Compute expansion
    for i=1:n; R[i,i] = coeff0; end # R₀ = J₀(i)*I, the 0th order
    R .+= coeff[1].*Tₖ₋₂ .+ coeff[2].*Tₖ₋₁ # 1st and 2nd order
    for k in 3:order # higher orders
        A_mul_B!(T(2),A,Tₖ₋₁,T(-1),Tₖ₋₂) # result Tₖ is stored in Tₖ₋₂
        R .+= coeff[k].*Tₖ₋₂ # coeff[k].*Tₖ
        Tₖ₋₂, Tₖ₋₁ = Tₖ₋₁, Tₖ₋₂ # Tₖ₋₁->Tₖ₋₂ and Tₖ->Tₖ₋₁ for next order
    end
    return R
end

# Bessel function values precomputed in Mathematica

# J₀(i)
const cheby_coeff0_big = big"1.2660658777520083355982446252147175376076703113550"
const cheby_coeff0 = Float64(cheby_coeff0_big)

# 2 * iᵏ * Jₖ(-i)
const cheby_coeff_big = BigFloat[
    big"1.1303182079849700544153920552197266146577992432422"
    big"0.27149533953407656236570513998998184589974213622556"
    big"0.044336849848663804952571495259799231058830698339959"
    big"0.0054742404420937326502761684311864595467579461858053"
    big"0.00054292631191394375036214781030755468476712885351629"
    big"0.000044977322954295146654690328110912699086657650642406"
    big"3.1984364624019905058638729766022957272370458074163E-6"
    big"1.9921248066727957259610643848055890533900933857774E-7"
    big"1.1036771725517344326169960913353241812896390172496E-8"
    big"5.5058960796737472504714204020055270687431547280674E-10"
    big"2.4979566169849825227120109342187675410080716361415E-11"
    big"1.0391522306785700504996346724238478525397128556091E-12"
    big"3.9912633564144015128877204015326949127607826796533E-14"
    big"1.4237580108256571488273680253471752219093588992517E-15"
    big"4.7409261025614961710899305606042914145777617483995E-17"
    big"1.4801800572082975003888571658877975360303747318760E-18"
    big"4.3499194949441698455876297633392992805626063964777E-20"
    big"1.2074289272797528890630463524357806390885570735305E-21"
    big"3.1753567370594449606628945704889798438009317678357E-23"
    big"7.9336719716380401114641564996829844420300175294498E-25"
    big"1.8879484042289160772319706157860669889247560558152E-26"
    big"4.2886738765925870898799133815030885460420950260008E-28"
    big"9.3189852817775768480872792470802866623424437483364E-30"
    big"1.9406469749017397597649278461566813645709017660684E-31"
    big"3.8798022492260012156255855282161124021152712081106E-33"
    big"7.4585028873915195213508204862516351326616201311347E-35"
    big"1.3807477824110645231588753652621331312287399205215E-36"
    big"2.4648623717710962928935138361162240264245603184147E-38"
    big"4.2485421925059913850761703704767643098614220928398E-40"
    big"7.0790011762128954933502123970072670493550456762707E-42"
    big"1.1414867782540890660429322724040802483946870774259E-43"
    big"1.7831510375432838840323081019695093079857962301911E-45"
]
const cheby_coeff = Vector{Float64}(cheby_coeff_big)
