using Distributions
using Statistics
using StatsBase
using Random
using LinearAlgebra
using SparseArrays
using PoissonRandom
using DelimitedFiles

include("Donne.jl")
include("rates.jl")
include("update_rates.jl")
include("exponential_growth.jl")
include("tauleapswitch2.jl")
include("ssa_switch4.jl")

function DownSampling(;Data,BETA_vec)
    Counts_downsampling=Array{Int64,2}(undef, size(Data))

    for i in 1:size(Data)[1]
        for j in 1:size(Data)[2]
            d=Binomial(Int(Data[i,j]),BETA_vec[j])
            Counts_downsampling[i,j]= rand(d, 1)[1]
        end
    end

    return(Counts_downsampling)
end

function model_construction(k1,k2,k3,k4,k5,k6,gene_num)
    model=[]
    inds=[]
    acts=[]
    for i = 1:gene_num
            push!(model,(name = "transcription$i", rate = [k1], reactants = [:NULL], products = [Symbol("G$(i)_mRNA")], coeff_rea = [1], coeff_pro = [1]))
            push!(inds,length(model))
            push!(model,(name = "decay$i", rate = [k2], reactants = [Symbol("G$(i)_mRNA")], products = [:NULL], coeff_rea = [1], coeff_pro = [1]))
            push!(model,(name = "translation$i", rate = [k3], reactants = [Symbol("G$(i)_mRNA")], products = [Symbol("G$(i)_mRNA"),Symbol("G$(i)")], coeff_rea = [1], coeff_pro = [1,1]))
            push!(model,(name = "decay$i", rate = [k4], reactants = [Symbol("G$(i)")], products = [:NULL], coeff_rea = [1], coeff_pro = [1]))
            for j = 1:gene_num
                push!(model,(name = "activation$(i*(j-1))", rate = [k1,k5,k6], reactants = [Symbol("G$(i)")], products = [Symbol("G$(j)_mRNA"),Symbol("G$(i)")], coeff_rea = [1], coeff_pro = [1,1]))
                push!(inds,length(model))
                push!(acts,length(model))
            end
    end
    return model,inds,acts
end
 
function sampler(k1::Float64,k2::Float64,k3::Float64,k4::Float64,k5::Float64,k6::Float64,gr::Float64,beta::Float64,gene_num::Int64,cell_num::Int64,activations::Vector{Float64},final_time::Float64)
    modelv,inds,acts = model_construction(k1,k2,k3,k4,k5,k6,gene_num)
    model = Tuple(modelv[i] for i in 1:length(modelv))
    count=0
    for i in acts
        count+=1
        model[i].rate[1] = activations[count]
    end
    gene_list  = Symbol.(:G, (1:gene_num))
    protein_list  = gene_list
    mRNA_list = Symbol.(protein_list, :_mRNA)
    initiale_population = [:NULL 0;protein_list ceil.(Int,((1.5*k1*k3)/((k2+gr)*(k4+gr)))*ones(Int, gene_num));mRNA_list ceil.(Int,(1.5*k1/(k2+gr))*ones(Int, gene_num))]
    data = Donne(model, initiale_population, final_time, 0.1, cell_num, gr, 0.03)
    trans_index = Int.(inds)
    t, V, X = exponential_growth(data, trans_index, 0.01, tauleapswitch2, 0.8)
    V_f = V[end,:]
    mRNA_index = findall(x -> x in mRNA_list, data.species)
    mRNA = X[end,:,mRNA_index]

    mRNAd = DownSampling(;Data=mRNA,BETA_vec=beta .* ones(cell_num))
    mRNAdd = mRNAd ./ V_f
    return mRNAdd
end
