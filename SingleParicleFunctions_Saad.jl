### in order to generate and save samples, highlight all and <shift + enter>, then in REPL run sampleGenerator as needed

sum(LOAD_PATH.==pwd())==0 && push!(LOAD_PATH, pwd()) ###Push the current direcory into the path
using LinearAlgebra
using ProgressTimer
using Polynomials
using SpecialPolynomials
using Plots
using NPZ
Plots.default(show = true)


###QHO/LLL wavefunction using Jason's notation
function Psi2D(x,nl;a=1.0::Real)
    n,l = nl ### nl is a pair of integers, following Jason's notation. 
    z=x/a
    indexn = vec(fill(0,n+1,1)) ###making an array of length n filled with 0's
    indexn[end] = 1 ###making last elmt = 1 to serve as input for Laguerre, to make a Laguerre Polynomial (n,l)
    psi = (z)^l * convert(Polynomial,Laguerre{l}(indexn))(abs(z)^2) * exp(-abs(z)^2/2) * sqrt(factorial(n)/(factorial(n+l)*pi*a^2))
    return psi
end



function RunMetropolis2D(Samples,nList;dt=10,sl=1.0,
                         therm=Int(ceil(0.1*Samples))::Integer,
                         x0_in=nothing)
    Ne=size(nList)[1]

    if 2!=size(nList)[2] || 2!=length(size(nList))
        error("nList size=$(size(nList))has wrong format")
    end

    if therm < 0 || Samples < 1
        error("Nonses values for thermalization=$therm or Samples=$Samples")
    end
     
    ###Run the metropolis step
    ###Generate an initial guess
    if x0_in == nothing
        x0 = randn(Ne) + im .*randn(Ne)  ###Random gaussian number
    elseif length(x0_in) != Ne
        error("Starting seed has wrong numer of particles in it")
    else ###use the external data but make sure co make it comples
        x0 = (1.0+0.0*im) .* x0_in
    end
    ###Evaluate the wfn
    psi0 = SlaterDeterminantPsi(x0,nList)

    ###And the corresponding probablility
    prob0= abs(psi0)^2
    accepted=0
    rejected=0
    XList=fill(0.0*im,Samples,Ne)
    PsiList=fill(0.0*im,Samples)
    TIMESTRUCT=TimingInit()
    for Sno in 1:(Samples+therm)
        therm >0 && Sno==1 &&  println("Begin Thermalizing!")
        for _ in 1:dt ###Run extra steps to make the data more independend
            ### Now generate new coordiantes
            Elem=rand(1:Ne)
            x = copy(x0)
            x[Elem] = x[Elem] .+ (randn() + im .*randn()).*sl
            ## and new wfn
            psi = SlaterDeterminantPsi(x,nList)
            ###And the corresponding probablility
            prob= abs(psi)^2
            ##Then we compare
            #println("x=$x , x0=$x0")
            #println("psi=$psi , psi0=$psi0")
            #println("prob=$prob , prob0=$prob0")
            if prob > prob0*rand() ## accept
                x0 = x
                psi0 = psi
                prob0 = prob
                accepted+=1
                #println("Accepted")
            else
                rejected+=1
                #println("Rejected")
            end
        end
        Sno==therm && println("Thermalization completed!")
        if Sno>therm
            ###Store the sample
            XList[Sno-therm,:]=x0
            ###Store the wfn
            PsiList[Sno-therm]=psi0
        end
        TIMESTRUCT=TimingProgress(TIMESTRUCT,Sno,Samples+therm)
    end
    AccepRatio=accepted/(0.0+accepted+rejected)
    println("$accepted of $(accepted+rejected), Acceptance Ratio $(100*AccepRatio)%")
    return XList,PsiList
end


function SlaterDeterminantPsi(XList,NList)
    if length(XList) != size(NList)[1] || size(NList)[2] != 2 ##sanity check
        error("Lists lengths $(length(XList)) vs $(size(NList)[1]) do not match!)")
    end
    Dim=length(XList)
    SlatDet=fill(0.0im,Dim,Dim)
    for pindx in 1:Dim
        for windx in 1:Dim
            SlatDet[pindx,windx]=Psi2D(XList[pindx],NList[windx,:])
        end
    end
    return det(SlatDet) / sqrt(factorial(Dim))
end

#function to generate samples and save to npy file
function sampleGenerator(Samples::Integer, Ne, nList, file_name = "temp_samples.npy")
    if size(nList)[1] != Ne
        error("List length is longer than number of particles!")
    end
    ###dt is how many MC samples are generated before one is saved
    xList,PsiList=RunMetropolis2D(Samples,nList,dt=Ne*3,sl=1.0)
    npzwrite(file_name,xList)
end