sum(LOAD_PATH.==pwd())==0 && push!(LOAD_PATH, pwd()) ###Push the current direcoty into the path
using LinearAlgebra
using ProgressTimer


function HermitePolynomial(x,n::Integer)
    if n==0
        return 1
    elseif n==1
        return 2 .*x
    elseif n==2
        return 4 .*x.^2  .- 2
    elseif n==3
        return 8 .*x.^3  - 12 .* x
    elseif n==4
        return 16 .*x.^4  - 48 .* x.^2 .+ 12
    else
        error("Hermite polynomial not implemende yet")
    end
end



function PsiHarmonic1D(x,n::Integer;a=1.0::Real)
    xa=x./a
    psi=HermitePolynomial(xa,n) .* exp.(-.5*xa.^2) / ( sqrt(2^n * factorial(n)) * (pi*a^2)^0.25)
    return psi
end


function PsiHarmonic2D(x,n;a=1.0::Real)
    n1,n2=n
    psi=PsiHarmonic1D(real.(x),n1;a=a)*PsiHarmonic1D(imag.(x),n2;a=a)
    return psi
end


function PsiGaussian2D(x,w;a=1.0::Real)
    psi=PsiGaussian(real.(x),real.(w);a=a)*PsiGaussian(imag.(x),imag.(w);a=a)
    return psi
end



function PsiGaussian(x,x0::Real;a=1.0::Real)
    return PsiHarmonic1D(x .- x0,0;a=a)
end



function RunMetropolis1D(Samples;dt=10,sl=1.0,n=0,
                         therm=Int(ceil(0.1*Samples))::Integer)
    ###Run the metropolis step
    ###Generate an initial guess
    x0 = randn()  ###Random gaussian number
    ###Evaluate the wfn
    psi0 = PsiHarmonic1D(x0,n)
    ###And the corresponding probablility
    prob0= abs(psi0)^2
    accepted=0
    rejected=0
    XList=fill(0.0,Samples)
    PsiList=fill(0.0*im,Samples)
    for Sno in 1:(Samples+therm)
        for _ in 1:dt ###Run extra steps to make the data more independend
            ### Now generate new coordiantes
            x = x0 + randn()*sl
            ## and new wfn
            psi = PsiHarmonic1D(x,n)
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
        if Sno>therm
            ###Store the sample
            XList[Sno-therm]=x0
            ###Store the wfn
            PsiList[Sno-therm]=psi0
        end
    end
    AccepRatio=accepted/(0.0+accepted+rejected)
    println("$accepted of $(accepted+rejected), Acceptance Ratio $(100*AccepRatio)%")
    return XList,PsiList
end


function RunMetropolis2D(Samples,nList;dt=10,sl=1.0,
                         therm=Int(ceil(0.1*Samples))::Integer,
                         Gaussian=false,x0_in=nothing)
    Ne=size(nList)[1]
    if Gaussian
        println("nList=$nList")
        if 1!=length(size(nList))
            error("nList size=$(size(nList))has wrong format")
        end
    else
        if 2!=size(nList)[2] || 2!=length(size(nList))
            error("nList size=$(size(nList))has wrong format")
        end
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
    if Gaussian
        psi0 = SlaterDeterminantsGaussian2D(x0,nList)
    else
        psi0 = SlaterDeterminantsHarmOcc2D(x0,nList)
    end
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
            if Gaussian
                psi = SlaterDeterminantsGaussian2D(x,nList)
            else
                psi = SlaterDeterminantsHarmOcc2D(x,nList)
            end
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



function SlaterDeterminantsHarmOcc1D(XList,NList)
    if length(XList) != length(NList) ##sanity check
        error("Lists lengths $(length(XList)) vs $(length(NList)) do not match!)")
    end
    Dim=length(XList)
    SlatDet=fill(0.0im,Dim,Dim)
    for pindx in 1:Dim
        for windx in 1:Dim
            SlatDet[pindx,windx]=PsiHarmonic1D(XList[pindx],NList[windx])
        end
    end
    return det(SlatDet) / sqrt(factorial(Dim))
end



function SlaterDeterminantsHarmOcc2D(XList,NList)
    if length(XList) != size(NList)[1] || size(NList)[2] != 2 ##sanity check
        error("Lists lengths $(length(XList)) vs $(size(NList)[1]) do not match!)")
    end
    Dim=length(XList)
    SlatDet=fill(0.0im,Dim,Dim)
    for pindx in 1:Dim
        for windx in 1:Dim
            SlatDet[pindx,windx]=PsiHarmonic2D(XList[pindx],NList[windx,:])
        end
    end
    return det(SlatDet) / sqrt(factorial(Dim))
end


function SlaterDeterminantsGaussian2D(XList,NList)
    if length(XList) != length(NList) || length(size(NList)) != 1 ##sanity check
        error("Lists lengths $(length(XList)) vs $(size(NList)) do not match!)")
    end
    Dim=length(XList)
    SlatDet=fill(0.0im,Dim,Dim)
    for pindx in 1:Dim
        for windx in 1:Dim
            SlatDet[pindx,windx]=PsiGaussian2D(XList[pindx],NList[windx])
        end
    end
    return det(SlatDet) / sqrt(factorial(Dim))
end

