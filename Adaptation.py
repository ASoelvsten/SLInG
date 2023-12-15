import numpy as np
from scipy.integrate import odeint
import time
import warnings
warnings.filterwarnings("ignore")

def pn_term(k,K,B,A):
    u = A*k*(np.heaviside(k,0)-np.heaviside(k,0)*B+np.heaviside(-k,0)*B)/(np.heaviside(k,0)-np.heaviside(k,0)*B+np.heaviside(-k,0)*B+K)
    if np.isnan(u):
        u = 0
    return u

def callABC(t_vec,x0,IEF,param):
    
    I, EA, EB, EC = IEF
    
    k = param[:13]
    K = param[13:]

    def ODE(y, t):
        A, B, C = y
        dAdt = pn_term(k[0],K[0],A,A)+pn_term(k[1],K[1],A,B)+pn_term(k[2],K[2],A,C)+pn_term(k[3],K[3],A,EA)+pn_term(k[4],K[4],A,I)
        dBdt = pn_term(k[5],K[5],B,A)+pn_term(k[6],K[6],B,B)+pn_term(k[7],K[7],B,C)+pn_term(k[8],K[8],B,EB)
        dCdt = pn_term(k[9],K[9],C,A)+pn_term(k[10],K[10],C,B)+pn_term(k[11],K[11],C,C)+pn_term(k[12],K[12],C,EC)
        return [dAdt,dBdt,dCdt]

    y = odeint(ODE, y0=x0, t=t_vec)

    return y

def PreSen(param,nolink=[]):

    if nolink != []:
        for i in nolink:
            param[i] = 0.0

    I1 = 0.5
    I2 = 0.6

    EA = 1.0
    EB = 1.0
    EC = 1.0

    threshold = 1e-3

    # WHEN IT IS IMPORTANTANT TO COMPUTE THE MODELS IN THE FIRST PLACE?

    Aps = param[:5]
    Bps = param[5:9]
    Cps = param[9:13]

    Apsp = np.sum(np.array(Aps) >= threshold, axis=0)
    Apsn = np.sum(np.array(Aps) <= -threshold, axis=0)
    Bpsp = np.sum(np.array(Bps) >= 0.1*threshold, axis=0)
    Bpsn = np.sum(np.array(Bps) <= -0.1*threshold, axis=0)
    Cpsp = np.sum(np.array(Cps) >= threshold, axis=0)
    Cpsn = np.sum(np.array(Cps) <= -threshold, axis=0)

    if Apsp >= 1 and Apsn >= 1 and ((Bpsp >= 1 and Bpsn >= 1) or (Bpsp==0 and Bpsn==0)) and Cpsp >= 1 and Cpsn >= 1:

        t_vec = np.linspace(0,10,10)
        y1 = callABC(t_vec,[0.00,0.00,0.00],[I1,EA,EB,EC],param)
        i = 1
        dy0dt = abs(y1[-1,2]-y1[-2,2])
        dy2dt = 1.
        dy1dt = 1.
        mr = 10000
        while (dy0dt > 2e-6 or dy2dt > 2e-6 or dy1dt > 2e-6) and i < mr: # dy1dt > 1e-4 and i < 100:
            y1 = callABC(t_vec,y1[-1],[I1,EA,EB,EC],param)
            dy2dt = abs(y1[-1,2]-y1[-2,2])+abs(y1[-1,2]-y1[-10,2])
            dy1dt = abs(y1[-1,1]-y1[-2,1])+abs(y1[-1,1]-y1[-10,1])
            dy0dt = abs(y1[-1,0]-y1[-2,0])+abs(y1[-1,0]-y1[-10,0])

            i += 1
        
        if i >= mr:
            S = np.nan
            P = np.nan

        else:

            t_vec = np.linspace(0,i*10,i*10)
    
            y2 = callABC(t_vec,y1[-1],[I2,EA,EB,EC],param)
            dy2dt = abs(y2[-1,2]-y2[-2,2])+abs(y2[-1,2]-y2[-10,2])
            dy1dt = abs(y2[-1,1]-y2[-2,1])+abs(y2[-1,1]-y2[-10,1])
            dy0dt = abs(y2[-1,0]-y2[-2,0])+abs(y2[-1,0]-y2[-10,0])
            if (dy0dt > 2e-6 or dy2dt > 2e-6 or dy1dt > 2e-6):
                S = np.nan
                P = np.nan
            else:
                O2 = y2[-1,2]
                O1 = y2[0,2]
                O_peak = np.max(abs(y2[:,2] - O1)) + O1
    
                S = abs(O_peak-O1)/O1/(abs(I2-I1)/I1)
                P = 1.0/(abs(O2-O1)/O1/(abs(I2-I1)/I1))
        
    else:
        S = np.nan
        P = np.nan

    return S, P

def measure(S,P,n=2,ks=1,kp=10):
    # NAN implies that O1, O2 and O_peak are zero, i.e. the system has no output
    if np.isnan(S) or S == np.inf:
        m = np.inf
    elif np.isnan(P) or P == np.inf:
        m = np.inf
    else:
        m = 1-S**n/(S**n+ks**n)*P**n/(P**n+kp**n)
    return m
