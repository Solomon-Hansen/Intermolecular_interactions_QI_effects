import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib

# ------ Load data --------
data_a = np.loadtxt('aqmt_5x5_c4M_up.txt',delimiter=' ')
data_b = np.loadtxt('aqdt_5x5_c4M_up.txt',delimiter=' ')
#data_c = np.loadtxt('aq_dt_c4_M_up.txt',delimiter=' ')
data_d = np.loadtxt('aq_mt_singleM_T_a_sum.txt',delimiter=' ')
data_e = np.loadtxt('aq_dt_singleM_T_a_sum.txt',delimiter=' ')
#data_f = np.loadtxt('ac_dt_singleM_T_a_sum.txt',delimiter=' ')


transmission_a = data_a[:,0]
transmission_b = data_b[:,0]
#transmission_c = data_c[:,0]
transmission_d = data_d[:,0]
transmission_e = data_e[:,0]
#transmission_f = data_f[:,0]
energies = data_a[:,1]
energies2 = data_b[:,1]
print (len(energies), np.argmin(transmission_a), energies[np.argmin(transmission_a)])


#--------------------- Set parameters ------------------------------#
T = 100.
kB = 8.617330350*10**(-5)
kbt = kB*T
pmbias = 0.5

#------------------------Functions-------------------------#

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
    
    return current,bias

def dIdV(transmission,energies,pmbias,kbt):
    current,bias = iv(transmission,energies,pmbias,kbt)
    didv = np.diff(current)/np.diff(bias)
    db = bias[1]-bias[0]
    return didv, bias[1:]-db*0.5



# -------------- Call the functions ---------------
didv_a,bias2_a = dIdV(transmission_a,energies,pmbias,kbt)
didv_b,bias2_b = dIdV(transmission_b,energies,pmbias,kbt)
#didv_c,bias2_c = dIdV(transmission_c,energies,pmbias,kbt)
didv_d,bias2_d = dIdV(transmission_d,energies,pmbias,kbt)
didv_e,bias2_e = dIdV(transmission_e,energies,pmbias,kbt)
#didv_f,bias2_f = dIdV(transmission_f,energies,pmbias,kbt)


# ------------------ Plot figure -------------------
fig = plt.figure(figsize=(10,8))
ax1 = fig.add_subplot()
ax1.set_xlabel('Bias [V]',fontsize=24)
ax1.set_ylabel('dI/dV [S]',fontsize=24)
ax1.set_ylim(10**(-8),10) #10**(0))
ax1.tick_params(axis='both', labelsize= 20)
#ax1.semilogy(bias2_c, didv_c, '-',c='k', linewidth=3, label='AC-DT (M)')
ax1.semilogy(bias2_b, didv_b, '-',c='k', linewidth=3, label='AQ-DT (C4M)')
ax1.semilogy(bias2_e, didv_e, '-',c='orange', linewidth=3, label='AQ-DT (SM)')
ax1.semilogy(bias2_a, didv_a, '--', c='k', linewidth=3, label='AQ-MT (C4M)')
ax1.semilogy(bias2_d, didv_d, '--' ,c='orange', linewidth=3, label='AQ-MT (SM)')
#ax1.semilogy(bias2_f, didv_f, '-',c='k', linewidth=3, label='AC-DT (SM)')
fig.legend(ncol=2,loc=(0.21,0.75),fontsize=20)

plt.savefig('didv_SM_C4M.png', dpi=250)
plt.show()


# --- Fermi dirac distribution --- 
# plt.plot(energies,fermiDirac(energies,kbt),'--r')
# plt.show()


#----------- plot current ------
# current,bias = iv(transmission_a,energies,pmbias,kbt)
# current1,bias1 = iv(transmission_b,energies,pmbias,kbt)
# current2,bias2 = iv(transmission_c,energies,pmbias,kbt)

# #plt.plot(bias, current)
# plt.plot(bias1,current1)
# #plt.plot(bias2,current2)
# plt.show()

