import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns




def lin_transmission(epsilon, huckel, coupling, delta):  #Huckel transmission code 
    gamma_left = np.zeros(huckel.shape, order='F')
    gamma_left[0,0] = coupling**2
    gamma_left[-1,-1] = coupling**2
    gamma_left[0,-1] = delta
    gamma_left[-1,0] = delta

    gamma_right = np.zeros(huckel.shape, order='F')
    gamma_right[1,1] = coupling**2
    gamma_right[2,2] = coupling**2
    gamma_right[1,2] = delta
    gamma_right[2,1] = delta

    temp_epsilon = np.zeros(huckel.shape)
    np.fill_diagonal(temp_epsilon, epsilon) 

    g_ret = np.linalg.inv(
        temp_epsilon - huckel + (1j/2)*gamma_left + (1j/2)*gamma_right
    )

    g_adv = np.conjugate(g_ret)

    result = np.trace(
        np.matmul(np.matmul(np.matmul(gamma_left, g_ret), gamma_right), g_adv)
    )
    return result

def cross_transmission(epsilon, huckel, coupling,delta):  #Huckel transmission code 
    gamma_left = np.zeros(huckel.shape, order='F')
    gamma_left[0,0] = coupling**2
    gamma_left[-1,-1] = coupling**2
    gamma_left[0,-1] = delta
    gamma_left[-1,0] = delta

    gamma_right = np.zeros(huckel.shape, order='F')
    gamma_right[0,0] = coupling**2
    gamma_right[-1,-1] = coupling**2
    gamma_right[0,-1] = delta
    gamma_right[-1,0] = delta

    temp_epsilon = np.zeros(huckel.shape)
    np.fill_diagonal(temp_epsilon, epsilon) 

    g_ret = np.linalg.inv(
        temp_epsilon - huckel + (1j/2)*gamma_left + (1j/2)*gamma_right
    )

    g_adv = np.conjugate(g_ret)

    result = np.trace(
        np.matmul(np.matmul(np.matmul(gamma_left, g_ret), gamma_right), g_adv)
    )

    return result

# ----- Changeable values ---------------
#color_list = ['orange','b','r']
colors = plt.cm.inferno(np.linspace(0, 0.8, 3))
linestyle = ['-.','--','-']
beta = -2.7
alpha = 0.0
grid = np.linspace(-4,4,10000)

a = [-0.2,-0.1,0.0] #
b = [-0.1,-0.05,0.0] #

v = -1.0
delta =  -0.4

# v = [-1.0,-0.2]
# delta = [-0.4, -0.2, 0.0]

test = ['DM: [a,b] = [-0.2 , -0,1]','M: [a,b] = [-0.1 , -0,05]','SM: [a,b] = [0.0 , 0,0]']


# -------- Run transmission code with Hamiltonian and coupling as input -------- 
for i in range(len(a)):
    # ------ Hamiltonian -------------
    Ham = np.array(
    [[alpha,beta,b[i],a[i]],
    [beta,alpha,a[i],b[i]],
    [b[i],a[i],alpha,beta],
    [a[i],b[i],beta,alpha]])


    trans = [cross_transmission(en, Ham, v,delta) for en in grid] 
    #print (trans)

#-------- plots -------------

    plt.semilogy(grid,trans,label=test[i],color = colors[i],linewidth=2,linestyle=linestyle[i])
    #plt.plot(grid,trans,label=test[i],color = colors[i],linewidth=2,linestyle=linestyle[i])

plt.ylabel('Transmission',fontsize=15)
plt.xlabel('$E-E_{f} [Ev]$',fontsize=14) 
plt.xlim(-4,4)
plt.legend()
#plt.savefig('cross_strong_log',dpi=250)
plt.show()    

