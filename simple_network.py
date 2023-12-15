#python3 -m pip install julia need to run this first
import julia
jl = julia.Julia(compiled_modules=False)
from julia import Main
from julia import Main
from julia import Pkg
Pkg.add(['Catalyst','DifferentialEquations',"IfElse"])
Main.include('catalyst_4_gene_sampler.jl')
import time
import numpy as np
import random
from datetime import datetime

#    param[0]  G2 -> G1
#    param[1]  G3 -> G1
#    param[2]  G4 -> G1
#    param[3]  G1 -> G2
#    param[4]  G3 -> G2
#    param[5]  G4 -> G2
#    param[6]  G1 -> G3
#    param[7]  G2 -> G3
#    param[8]  G4 -> G3
#    param[9]  G1 -> G4 
#    param[10] G2 -> G4
#    param[11] G3 -> G4

def simulate(param,cell_num=1000,sampler="sde"):

    if len(param) > 12:
        k11 = 10**param[0]
        k12 = 10**param[1]
        k13 = 10**param[2]
        k14 = 10**param[3]
        lparam = param[4:].copy()
    else:
        k11=0.36787944117207516
        k12=0.36787944117207516
        k13=0.36787944117207516
        k14=0.36787944117207516
        lparam = param.copy()

    k2=0.04987442692380366
    k3=4.3376521035785345
    k4=0.36777724535966944
    k5=10.0
    k6=61.47011276535837

    growthr = 0.03
    #gene_num = 4 # FIXED

    if sampler == "ssa":
        data = Main.ssa_sampler(k11,k12,k13,k14,k2,k3,k4,k5,k6,growthr,cell_num,lparam)
    elif sampler == "sde":
        data = Main.sde_sampler(k11,k12,k13,k14,k2,k3,k4,k5,k6,growthr,cell_num,lparam)

    return data

def bootstrap(data,sam=10000):
    index = np.arange(len(data))
    for i in range(sam):
        ind = np.random.choice(index, replace=True,size=len(data))
        samples = data[ind,:]
        m = np.mean(samples,axis=0)
        cov = np.cov(np.transpose(samples))
        if i == 0:
            M = m
            COV = cov
        else:
            M = np.vstack((M,m))
            COV = np.dstack((COV,cov))
        
    m1 = np.mean(M,axis=0)
    var1 = np.std(M,axis=0)
    c = np.mean(COV,axis=2)
    vc = np.std(COV,axis=2)
    
    return np.vstack((m1,var1,c,vc,data))

def cov_distance(boot,data2,orthogonal=False):
    cov1 = boot[2:6,:]
    varc = boot[6:10,:]
    cov2 = np.cov(np.transpose(data2))
    m1 = boot[0,:]
    m2 = np.mean(data2,axis=0)
    var1 = boot[1,:]
    std2 = np.diagonal(cov2)
    dist1 = np.sum((m1-m2)**2/var1)
    dist2 = np.sum((cov1-cov2)**2/varc)
    if not orthogonal:
        dist = np.sqrt((dist1+dist2)/len(boot[10:,:]))
    else:
        dist = np.sqrt((dist1*dist2)/len(boot[10:,:]))

    return dist

def observe(param,disttype,cell_num=1000):
    data = simulate(param,cell_num=cell_num,sampler="sde")
    if disttype == "Energy":
        return data
    else:
        print("Bootstrapping data")
        boot = bootstrap(data)
        return boot
