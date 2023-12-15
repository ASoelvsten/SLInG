import numpy as np
import matplotlib.pyplot as plt
from glob import glob

#Med = [ 3.22212945e-04, 1.24284146e-04, 5.26897287e+01, 2.50569453e+01, -2.63988937e+01, 2.24782525e-05, -2.51893354e-05,  2.17955548e+01,  5.79472953e+01,  4.62862656e+00]
#Std = [2.15670630e-04, 5.05395160e-03, 1.38382491e+00, 7.71465059e-01, 7.49401430e-01, 2.12837752e-05, 1.74845366e-05, 8.20082363e-01, 1.17209209e+00, 1.08342101e+00]
#Best = [ 1.31407798e-03,  1.31612971e-04,  5.53278499e+01,  2.63256416e+01, -2.79238831e+01, 2.24782525e-05, -2.34001149e-04,  1.82480215e+01,  5.92803069e+01,  4.94890434e+00]

names = sorted(glob("./DR0*/Diabetes_RMSD_00004_step_*txt"))
print(names)

index = [2,3,6,9]
names = [names[x] for x in index]

for i in range(len(names)):
    name = names[i][:-4]

    print(name)

    lab = r"$\delta_\mathrm{abc}=$"+name[name.find("dabc_")+5:name.find("_scale")]

    data = np.genfromtxt(name+".txt")
    burnin = 2500

    print(data.shape)
    theta_t = data[:,10:20]
    EPS = data[:,:10]
    dist = data[:,20]

    Med = np.median(theta_t[burnin:,:],axis=0)
    Std = np.std(theta_t[burnin:,:],axis=0)
    Best = theta_t[dist.argmin(),:]
    p025 = np.percentile(theta_t[burnin:,:],2.5,axis=0)
    p975 = np.percentile(theta_t[burnin:,:],97.5,axis=0)

    xerr = np.vstack((Med-p025,p975-Med))

    y = np.arange(theta_t.shape[1])+1-0.1+0.1*i

    plt.errorbar(np.asarray(Med)*10,y,xerr=10*xerr,fmt="^",capsize=2,alpha=0.5,label=lab)
    plt.axvline(x=0,linestyle="--",color="grey")
#    plt.plot(np.asarray(Best)*10.,y,"x")

plt.legend()
plt.gca().invert_yaxis()
plt.xlabel(r"$\mathrm{Standardized \,\,coefficients}$",fontsize=13)
plt.ylabel(r"$\mathrm{Variable \,\, Number}$",fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.savefig(name+"_Param_Sum.jpeg")

plt.show()

