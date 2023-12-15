import numpy as np
from glob import glob
import Adaptation as ada

def read_file(fil):
#    print(fil)

    olno = 0
    blno = 0
    dabc = 0
    EF = False

    with open(fil) as f:
        lines = f.readlines()
        for lno, line in enumerate(lines):
            words = line.split()
            if words != []:
                if words[0] == "Observations:":
                    olno = lno
                elif words[0] == "BEST":
                    blno = lno
                elif words[0] == "Initial" and words[1] == "distance":
                    dabc = float(words[-1])

        if olno != 0 and blno !=0:

            obs  = []
            words = lines[olno].split()
            for w in words[2:]:
                obs.extend([float(w)])

            best = []
            for i in range(blno,blno+7):
                words = lines[i].split()
                if EF:
                    break
                for w in words:
                    if w!="BEST" and w!="FIT:":
                        if "[" in w and w!="[":
                            best.extend([float(w[1:])])
                        elif "]" in w:
                            best.extend([float(w[:-1])])
                            EF = True
                            break
                        elif w!="[":
                            best.extend([float(w)])

            for i, b in enumerate(best[:13]):
                if abs(b) <= 1e-1 and i not in [5,6,7,8]:
                    best[i] = 0
                elif abs(b) <=1.e-1 and i in [5,6,7,8]:
                    best[i] = 0

            S, P = ada.PreSen(best)

            return S, P, best, obs, dabc

        else:
#            print("NOT THERE")       
            return np.nan, np.nan, [],[], dabc

def same_or_not(best,obs):
    k = best[:13].copy()
    print(best)
    for i, b in enumerate(k):
        if b > 0:
            k[i] = 1
        elif b < 0:
            k[i] = -1

    print(k)
    diff = np.asarray(k) - 10*np.asarray(obs[:13])
#    print(diff)

files = sorted(glob("./gibbs.o*"))

counter = 0
success = 0
ds = []

for fil in files:
    S,P,best,obs,dabc = read_file(fil)
    if best != []:
        counter +=1
#    print(S,P)
#    same_or_not(best,obs)
    if S > 1 and P >10:
        print(fil)
        print("Success",S,P)
        same_or_not(best,obs)
        ds.extend([dabc])
        success += 1

print(ds)
print(counter,success)
