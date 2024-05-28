import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib as mpl

mpl.rcParams['figure.figsize'] = [12.0, 8.0]
SMALL_SIZE = 18
MEDIUM_SIZE = 18
BIGGER_SIZE = 24
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


T = 100.
kB = 8.617330350*10**(-5)
kbt = kB*T
pmbias = 1.1

alpha = 0.
beta = -2.7
V = [-1.0]#[-1.0,-0.6,-0.2]

a = [-0.2,-0.1,0.0]
b = [-0.1,-0.05,0.0]
delta = [-0.4]#[-0.4, -0.2, 0.0]

eta = 1.0*10**(-6)
epsilon = np.linspace(-4.,4.,1001)

c0 = cm.YlOrRd(np.linspace(0.3, 0.8, 3))
c1 = cm.GnBu(np.linspace(0.3, 0.8, 3))
c2 = cm.YlGn(np.linspace(0.3, 0.8, 3))

cc = cm.Reds(np.linspace(0.1, 1.1, 3))
cc1 = cm.Blues(np.linspace(0.8, 0.3, 3))
cc = ['r','r','r']
linetype = ['-','--',':']

def transmission(H,Gamma_l, Gamma_r,epsilon,eta):
    T = np.zeros(epsilon.shape)
    Sigma = 1J*(Gamma_l+Gamma_r)
    for i in range(len(epsilon)):
        G_ret = la.inv((epsilon[i]+1J*eta)*np.identity(Gamma_l.shape[0]) - H - 0.5*Sigma)
        G_adv = np.conj(G_ret)
        T[i] = np.trace(np.dot(Gamma_l,np.dot(G_ret,np.dot(Gamma_r,G_adv))))
    return T

def fermiDirac(e,kbt):
    e2 = np.clip(e / kbt, -50., 50.)
    return 1./(np.exp(e2)+1.)

def iv(transmission,energies,pmbias,kbt):
    dE = energies[-1]-energies[-2]
    bias = np.linspace(-pmbias,pmbias,len(energies))
    current = np.zeros(len(bias))
    for i in range(len(bias)):
        mu_left = 0.5*bias[i]
        mu_right = -0.5*bias[i]
        fermiLeft = fermiDirac(energies-mu_left,kbt)
        fermiRight = fermiDirac(energies-mu_right,kbt)
        current[i] = np.sum(transmission*(fermiLeft-fermiRight)) * dE
    
    #bias = np.concatenate((-bias[::-1],bias[1:]), axis=0)
    #current = np.concatenate((-current[::-1],current[1:]),axis=0)
    return current,bias

def dIdV(transmission,energies,pmbias,kbt):
    current,bias = iv(transmission,energies,pmbias,kbt)
    didv = np.diff(current)/np.diff(bias)
    db = bias[1]-bias[0]
    return didv, bias[1:]-db*0.5


##-------------------------------------------------------------------------------------------##
alpha = [0.,0.5,1.0]
for j in range(len(alpha)):
    for i in range(len(V)):
        H = alpha[i] * np.eye(4)
        H[0,1] = beta
        H[1,0] = beta
        H[2,3] = beta
        H[3,2] = beta
        Gamma_l = np.zeros((4,4))
        Gamma_l[0,0] = V[0]**2
        Gamma_l[3,3] = V[0]**2
        Gamma_l[0,3] = delta[0]
        Gamma_l[3,0] = delta[0]
        Gamma_l[0,3] = delta[0]
        Gamma_l[3,0] = delta[0]
        
        Gamma_r = Gamma_l
        
        H[1,2] = a[j]
        H[2,1] = a[j]
        H[0,3] = a[j]
        H[3,0] = a[j]
        H[0,2] = b[j]
        H[2,0] = b[j]
        H[3,1] = b[j]
        H[1,3] = b[j]
        
        T = transmission(H,Gamma_l,Gamma_r,epsilon,eta)
        didv,bias2 = dIdV(T,epsilon,pmbias,kbt)
        if j==0:
            if i==0:
                plt.semilogy(bias2,didv,linetype[i],linewidth=3,c=cc[i], alpha=1.0,label='C2: [a,b] = [-0.2 , -0,1]')
            else:
                plt.semilogy(bias2,didv,linetype[i],linewidth=3,c=cc[i], alpha=1.0)
        if j==1:
            if i==0:
                plt.semilogy(bias2,didv,'-',c=cc1[i], alpha=1.0,label='C2: [a,b] = [-0.1 , -0,05]')
            else:
                plt.semilogy(bias2,didv,'-',c=cc1[i], alpha=1.0)

#if i==0:
#plt.legend()
#plt.title('[V,$\delta$] = ['+str(V[i])+','+str(delta[i])+']')
plt.xlabel('Voltage [V]',fontsize=28)
plt.ylabel('dI/dV [S]',fontsize=28)
plt.ylim(10**-3,1)
#plt.savefig('didv_weak.png',dpi=300)
plt.show()
plt.clf()

for j in range(len(alpha)):
    for i in range(len(V)):
        H = alpha[i] * np.eye(4)
        H[0,1] = beta
        H[1,0] = beta
        H[2,3] = beta
        H[3,2] = beta
        Gamma_l = np.zeros((4,4))
        Gamma_l[0,0] = V[0]**2
        Gamma_l[3,3] = V[0]**2
        Gamma_l[0,3] = delta[0]
        Gamma_l[3,0] = delta[0]
        Gamma_l[0,3] = delta[0]
        Gamma_l[3,0] = delta[0]
        
        Gamma_r = Gamma_l
        
        H[1,2] = a[j]
        H[2,1] = a[j]
        H[0,3] = a[j]
        H[3,0] = a[j]
        H[0,2] = b[j]
        H[2,0] = b[j]
        H[3,1] = b[j]
        H[1,3] = b[j]
        
        T = transmission(H,Gamma_l,Gamma_r,epsilon,eta)
        didv,bias2 = dIdV(T,epsilon,pmbias,kbt)
        if j==0:
            if i==0:
                plt.semilogy(epsilon,T,linetype[i],linewidth=3,c=cc[i], alpha=1.0, label= alpha[i])
            else:
                plt.semilogy(epsilon,T,linetype[i],linewidth=3, c=cc[i], alpha=1.0, label=alpha[i])
        if j==1:
            if i==0:
                plt.semilogy(epsilon,T,'-',c=cc1[i], alpha=1.0, label='C2: [a,b] = [-0.1 , -0,05]')
            else:
                plt.semilogy(epsilon,T,'-',c=cc1[i], alpha=1.0)

#if i==0:
plt.legend()
#plt.title('[V,$\delta$] = ['+str(V[i])+','+str(delta[i])+']')
plt.xlabel('E - E$_f$ [eV]',fontsize=28)
plt.ylabel('Transmission',fontsize=28)
#plt.savefig('T_weak.png',dpi=300)
plt.show()
plt.clf()





