#Runge-Kutta scheme for solving the stochastic SIR model with gaussian noise
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import time
start_time  = time.time()
#intialize vectors
N = 1000
n = 1000
a = 0
b = 0.1
h = (b-a)/N                     #the integration step size
#stochastic vectors
s = np.zeros(N, dtype='float')
j = np.zeros(N, dtype='float')
r = np.zeros(N, dtype='float')
#determinitic vectors
s_det =  np.zeros(N, dtype='float')
j_det =  np.zeros(N, dtype='float')
r_det =  np.zeros(N, dtype='float')
xi1 = np.random.normal(0, 800, N)
xi2 = np.random.normal(0, 100, N)
#t = np.zeros(N, dtype='float')
alpha = 40
beta = 10

#differential equations for the deterministic SIR
def normal_eq(s0, j0):
    sdot = -alpha*s0*j0
    jdot = alpha*s0*j0 - beta*j0
    return [sdot, jdot]

#define a function of differential equations 
def equations(s0, j0, xi10, xi20):
    sdot = -alpha*s0*j0 + (1/np.sqrt(n))*np.sqrt(alpha*s0*j0)*xi10
    jdot = alpha*s0*j0 -beta*j0 - (1/np.sqrt(n))*np.sqrt(alpha*s0*j0)*xi10 + (1/np.sqrt(n))*np.sqrt(beta*j0)*xi20
    return [sdot, jdot]
#define Ringe-Kutta coefficients for the determinitic SIR
def rK1(s0, j0):
    ks1 = h*normal_eq(s0, j0)[0]
    kj1 =  h*normal_eq(s0, j0)[1]
    return [ks1, kj1]
def rK2(s0, j0):
    ks2 = h*normal_eq(s0 + rK1(s0, j0)[0]/2, j0)[0]
    kj2 = h*normal_eq(s0, j0 + rK1(s0, j0)[1]/2)[1]
    return [ks2, kj2]

def rK3(s0, j0):
    ks3 = h*normal_eq(s0 +rK2(s0, j0)[0]/2, j0)[0]
    kj3 = h*normal_eq(s0, j0 + rK2(s0, j0)[1]/2)[1]
    return [ks3, kj3]

def rK4(s0, j0):
    ks4 = h*normal_eq(s0  + rK3(s0, j0)[0], j0)[0]
    kj4 = h*normal_eq(s0, j0 + rK3(s0, j0)[1])[1]
    return [ks4, kj4]


#define functions of Runge-Kutta coefficients for then stochastic SIR
def RK1(s0, j0, xi10, xi20):
    ks1 = h*equations(s0, j0, xi10, xi20)[0]
    kj1 =  h*equations(s0, j0, xi10, xi20)[1]
    return [ks1, kj1]
def RK2(s0, j0, xi10, xi20):
    ks2 = h*equations(s0 + RK1(s0, j0, xi10, xi20)[0]/2, j0, xi10, xi20)[0]
    kj2 = h*equations(s0, j0 + RK1(s0, j0, xi10, xi20)[1]/2, xi10, xi20)[1]
    return [ks2, kj2]

def RK3(s0, j0, xi10, xi20):
    ks3 = h*equations(s0 +RK2(s0, j0, xi10, xi20)[0]/2, j0, xi10, xi20)[0]
    kj3 = h*equations(s0, j0 + RK2(s0, j0, xi10, xi20)[1]/2, xi10, xi20)[1]
    return [ks3, kj3]

def RK4(s0, j0, xi10, xi20):
    ks4 = h*equations(s0  + RK3(s0, j0, xi10, xi20)[0], j0, xi10, xi20)[0]
    kj4 = h*equations(s0, j0 + RK3(s0, j0, xi10, xi20)[1], xi10, xi20)[1]
    return [ks4, kj4]

#set initial conditions 
s[0] = 1/n
j[0] = 2/n
s_det[0] = 1/n
j_det[0] = 2/n
i = 0                   #first iteration
while i<N-1:
    #solve  the determinitic SIR
    s_det[i+1] = s_det[i] + (1/6)*(rK1(s[i], j[i])[0] + 2*rK2(s[i], j[i])[0] + 2*rK3(s[i], j[i])[0] + rK4(s[i], j[i])[0])
    j_det[i+1] = j_det[i] + (1/6)*(rK1(s[i], j[i])[1] + 2*rK2(s[i], j[i])[1] + 2*rK3(s[i], j[i])[1] + rK4(s[i], j[i])[1])
    r_det[i]= 1 - j_det[i] - s_det[i]
    #solve the stochastic SIR
    s[i+1] = s[i] + (1/6)*(RK1(s[i], j[i], xi1[i], xi2[i])[0] + 2*RK2(s[i], j[i], xi1[i], xi2[i])[0] + 2*RK3(s[i], j[i], xi1[i], xi2[i])[0] + RK4(s[i], j[i], xi1[i], xi2[i])[0])
    j[i+1] = j[i] + (1/6)*(RK1(s[i], j[i], xi1[i], xi2[i])[1] + 2*RK2(s[i], j[i], xi1[i], xi2[i])[1] + 2*RK3(s[i], j[i], xi1[i], xi2[i])[1] + RK4(s[i], j[i], xi1[i], xi2[i])[1])
    r[i]= 1 - j[i] - s[i]
    i=i+1

end_time = time.time() - start_time
print('CPU Time: ',end_time)
t = np.linspace(0, N, N)
plt.grid()
plt.loglog(t, s_det, t, j_det,t, r_det,  t, s, t , j, t, r)
plt.title(r'\textbf{Numerical solution to stochastic and deterministic SIR}, $\displaystyle\alpha=30$,\\ $\displaystyle\beta=10$ and $\displaystyle N=10000$')
plt.ylabel(r'\textit{sjr}')
plt.xlabel(r'\textit{time [s]}')
#plt.xlim(0, 100)
plt.legend(['s - deterministic', 'j - deterministic', 'r - deterministic', 's -stochastic', 'j - stochastic', 'r - stochastic'])
plt.show()
plt.grid()
plt.loglog(np.linspace(0, N, N), ((n-n*(s_det+j_det+r_det))/n),np.linspace(0, N, N),  (n-n*(s+j+r))/n)
plt.title(r'\textbf{Error growth of the integrator}')
plt.ylim(0, 10**(-10))
plt.legend(['Deterministic', 'Stochastic'])
plt.ylabel(r'\textit{Error}')
plt.xlabel(r'\textit{Iteration number}')
plt.show()
'''
plt.scatter(t, xi1,s=0.3)
plt.scatter(t, xi2, s=0.3)
plt.show()
'''