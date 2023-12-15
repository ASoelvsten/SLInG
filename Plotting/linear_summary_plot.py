import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns

files = sorted(glob("./LRUN*/Sum_*.txt"))
CRES = []
med = []
xmed = []
mea = []
s  = []

for i, ff in enumerate(files):
    res = np.genfromtxt(files[i])
    truth = res[0,:-1]
    truei = np.nonzero(truth)
    n = len(truei[0])
    best = res[1:,:-1]
    best[best<1] = 0
    dabc = np.round(res[1:,-1],5)
    sfiles = glob(files[i][:9]+"LinearD_RMSD_00004_*")
    dabc_real = []
    for f in sfiles:
        dabc_real.extend([np.round(float(f[f.find("dabc_")+5:f.find("_scale")]),5)])

    dabc_real = sorted(dabc_real)

    for j, d in enumerate(dabc):
        if d in dabc_real:
            row = best[j,:]
            n_row = len(np.nonzero(row)[0]) # indices of nonzero elements
            row_right = row[truei]
            
            # If some are ignored, which?
            check = truth.copy()
            check[np.nonzero(row)[0]]=0
            if sum(check)>0 and n>n_row and n >3:
                xmed.extend([np.ones(len(check[np.nonzero(check)[0]]))*d])
                med.extend([check[np.nonzero(check)[0]]/np.median(truth[truei])])
                mea.extend([check[np.nonzero(check)[0]]/np.mean(truth[truei])])
                s.extend([np.ones(len(check[np.nonzero(check)[0]]))*n])

            n_right = len(np.nonzero(row_right)[0])
            counts = [n,n_row,n_right,d]
            if CRES == []:
                CRES = counts.copy()
            else:
                CRES = np.vstack((CRES,counts))

med = np.concatenate(med)
xmed = np.concatenate(xmed)
mea = np.concatenate(mea)
s = np.concatenate(s)
plt.figure()
print(min(xmed),max(xmed))
plt.scatter(xmed,med,c=s,alpha=0.3)
plt.colorbar()
plt.show()

fsize = 14
n = CRES[:,0]
plt.figure(figsize=(14,8))
for nn in range(int(min(n)),int(max(n))+1):
    mask = n == nn
    n_row = CRES[:,1][mask]
    n_right = CRES[:,2][mask]
    dabc= CRES[:,3][mask]
    data = {"n": n_row,
            "nr": n_right,
            "dabc": dabc}

    plt.subplot(2,3,nn)
    sns.lineplot( x = "dabc",
             y = "n",
             color = "b",
             label = r"$\mathrm{Parameters\,\,used}$",
             data = data);

    sns.lineplot( x = "dabc",
             y = "nr",
             color = "orange",
             linestyle="--",
             label = r"$\mathrm{Parameters\,\,correctly\,\,identified}$",
             data = data);

    plt.title(r"$n="+str(nn)+"$")
    plt.xlabel(r"$\delta_\mathrm{abc}$",fontsize=fsize)
    plt.ylabel(r"$\mathrm{Predicted\,\,n}$",fontsize=fsize)
    plt.axhline(y=nn,color="k",linestyle=":")
    plt.xticks(size=fsize)
    plt.yticks(size=fsize)

    if nn == 1:
        plt.legend(bbox_to_anchor=(0., 1.3), ncol =2,  loc=2, borderaxespad=0.,fontsize=fsize)
    else:
        plt.legend([],[], frameon=False)

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.3)
plt.savefig("LinearD_overview.jpeg",bbox_inches='tight')
plt.show()
