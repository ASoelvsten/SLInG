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
tmea = []
tmed = []
ts = []

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
            row = best[j,:] # For this particular d_abc (i.e. we will average over different d_abc in the end)
            n_row = len(np.nonzero(row)[0]) # indices of nonzero elements
            row_right = row[truei]
            
            # If some are ignored, which?
            check = truth.copy()
            check[np.nonzero(row)[0]]=0 # We set those we got right to zero, i.e. we want to figure out the values of those we missed out on
            check2 = row.copy()
            check2[truth == 0]=0 # We only look tat those that we actually get right, i.e. we only keep the entries in the fit that also correspond to the truth
            if sum(check)>0: # Did you get any wrong?
                xmed.extend([np.ones(len(check[np.nonzero(check)[0]]))*d])
                med.extend([check[np.nonzero(check)[0]]/np.median(truth[truei])]) # median of array excluding true values
                mea.extend([check[np.nonzero(check)[0]]/np.mean(truth[truei])]) # corresponding mean
                s.extend([np.ones(len(check[np.nonzero(check)[0]]))*n])
            if sum(check2)>0: # Did you get any right?
                tmed.extend([truth[np.nonzero(check2)[0]]/np.median(truth[truei])])
                tmea.extend([truth[np.nonzero(check2)[0]]/np.mean(truth[truei])])
                ts.extend([np.ones(len(truth[np.nonzero(check2)[0]]))*n])

            n_right = len(np.nonzero(row_right)[0])
            counts = [n,n_row,n_right,d]
            if CRES == []:
                CRES = counts.copy()
            else:
                CRES = np.vstack((CRES,counts))

med = np.concatenate(med)
xmed = np.concatenate(xmed)
mea = np.concatenate(mea)
tmed = np.concatenate(tmed)
tmea = np.concatenate(tmea)
s = np.concatenate(s)
ts = np.concatenate(ts)

plt.figure(figsize=(14,8))
bins = [0,0.25,0.50,0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.50]
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.title(r"$n="+str(i+1)+"$")
    plt.hist(mea[s==i+1],bins,linestyle="--",histtype=u'step',density=True,label="Wrongly discarded")
    plt.hist(tmea[ts==i+1],bins,histtype=u'step',density=True,label="Correctly identified")
    if i == 0:
        plt.legend(bbox_to_anchor=(0., 1.25), ncol =2,  loc=2, borderaxespad=0.,fontsize=14)
    plt.xlabel(r"$\theta_k/\langle \theta \rangle$",fontsize=14)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.ylabel(r"$\mathrm{Normalised\,\, counts}$",fontsize=14)

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.3)
plt.savefig("LinearD_discarded.jpeg",bbox_inches='tight')
plt.show()
