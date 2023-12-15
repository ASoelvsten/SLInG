import numpy as np
from scipy.integrate import odeint # Solves systems of ordinary differential equations
import matplotlib.pyplot as plt

#============================================================

def linHD(theta,x_obs):

    y_sim = theta[0]*100 + theta[1]*x_obs + theta[2] *100*np.sin(theta[3]/5.*x_obs) + theta[4]*x_obs**2 

    return y_sim

#============================================================

def lin2HD(theta,x_obs):

    y_sim = theta[0] + theta[1]*x_obs + theta[2]*x_obs**2 + theta[3]*np.sin(theta[4]*2*np.pi*x_obs) + theta[5]*np.tan(theta[6]*2*np.pi*x_obs) + theta[7]*np.exp(theta[8]*x_obs)

    return y_sim

#============================================================

def linsin(theta,x_obs):
    y_sim = theta[0]*x_obs[:,0]+theta[1]*100*np.sin(x_obs[:,1]/5)
    return y_sim

#============================================================

def simple_ode(y, t, a, b, c, d, e, f, g, h, i):

    dydt = a + b*t + c*t**2 + d *np.sin(2*np.pi*e*t) + f *np.tan(2*np.pi*g*t) + h*np.exp(i*t)

    return dydt

#============================================================

def run_simple_ode(theta, y0 = 100, t=np.linspace(0, 100, 1000)):

    a = theta[0]
    b = theta[1]
    c = theta[2]
    d = theta[3]
    e = theta[4]
    f = theta[5]
    g = theta[6]
    h = theta[7]
    i = theta[8]

    y_sim = odeint(simple_ode, y0, t, args=(a, b, c, d, e , f, g, h, i), rtol=1e-9 , atol=1e-12)

    return y_sim

#============================================================

def odes(param, t, a, b, c, s1, s2, mu2, mu3, g1, g2, g3, p1, p2, r2):
    x, y, z = param

    fx = r2*(1-b*x)
    dx = a*x*y/(g2+x)
    py = c*x+p1*y*z/(g1+z)
    pz = p2*x*y/(g3+x)
    ay = mu2*y
    az = mu3*z
    phiy = s1
    phiz = s2

    dxdt = x*fx - dx
    dydt = phiy + py - ay
    dzdt = phiz + pz - az

    return [dxdt, dydt, dzdt]

#============================================================

def kirschner(theta, t = np.linspace(0, 1000, 10000) ,param0 = [1e-9, 1e-9, 1e-9], scale = 1e9):
        
    mu2=theta[0]/1.e1
    p1=0.617
    g1=0.2
    g2=1.e-4
    r2=1.0
    b=1.0
    a=theta[1]
    c=theta[2]/10.
    mu3=55.556
    p2=27.778
    g3=1.e-6
    s1=0.0
    s2=0.0

    y_sim = odeint(odes, param0, t, args=(a, b, c, s1, s2, mu2, mu3, g1, g2, g3, p1, p2, r2),rtol=1e-12 , atol=1e-20)

    return scale*y_sim

#============================================================

def LotkaVolterra(param, t, a, b, c, d):

    x, y = param

    dxdt = a*x-b*y*x  # Prey
    dydt = c*x*y-d*y  # Predator

    return [dxdt, dydt]

#============================================================

def solve_LV(theta, t=np.linspace(0,100,1000), param0 = [10,10]):

    a = theta[0]
    b = theta[1]
    c = theta[2]
    d = theta[3]

    y_sim = odeint(LotkaVolterra, param0, t, args=(a, b, c, d),rtol=1e-12 , atol=1e-20)

    return y_sim

#============================================================

def efron(theta,x_obs):

    beta = theta*10.
    y_sim = np.dot(x_obs,np.transpose(beta))

    return y_sim

#============================================================

def SIR1_eq(param, t, a, b):

    s, i , r = param

    dsdt = -a*i*s
    didt = a*i*s - b *i
    drdt = b*i

    return [dsdt, didt, drdt]

#============================================================

def SIR1(theta, t=np.linspace(0,100,1000), param0 = [1,1.27e-6,0]):

    a = theta[0]
    b = theta[1]

    y_sim = odeint(SIR1_eq, param0, t, args=(a, b),rtol=1e-12 , atol=1e-20)

    return y_sim

#============================================================

def malaria_eq(param, t, P, L, dimm, din, dtreat, p1, p2, A, r0, phi, eta):

    S, I1, I2, R, W = param

    R0 = r0+A*np.cos(2.*np.pi*(t-phi))
    lam = R0*(1./L+1./dtreat)*(I1+I2)/P

    dSdt = P/L - (lam+1/L)*S+R/dimm
    t1 = eta*p1/dtreat+(1-eta*p1)/din
    t2 = eta*p2/dtreat+(1-eta*p2)/din
    dI1dt = lam*S-(t1+1./L)*I1
    dI2dt = lam*R-(t2+1./L)*I2
    dRdt = t1*I1 + t2*I2 - (lam+1./dimm+1./L)*R
    dWdt = lam*S*eta*p1+lam*R*eta*p2

    return [dSdt, dI1dt, dI2dt, dRdt, dWdt]

#============================================================

def malaria(theta,t,param0=[]):

    P = 29203486
    P0 = 25.65e6
    if param0 == []:
        param0 = [P0/2, P0/2, 0, 0,0]
    else:
        param0[4]=0
    L = 66.67
    dimm = 0.93
    din = theta[0]
    dtreat = 2.*7./365.25
    p1 = 0.87
    p2 = 0.08
    A = 0.87
    r0 = 1.23
    phi = 3./12.
    eta = theta[1]/50

    y_sim = odeint(malaria_eq, param0, t, args=(P, L, dimm, din, dtreat, p1, p2, A, r0, phi, eta),rtol=1e-12 , atol=1e-20)

    return y_sim
