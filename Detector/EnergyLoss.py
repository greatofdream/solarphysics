import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator
I = 75*1e-6 # MeV
N_H, Z_H, A_H, n_H = 1, 1, 1, 2
N_O, Z_O, A_O, n_O = 1, 8, 16, 1
N_A = 6.02*1e23
e = 1.6*1e-19
J2MeV = 6.24*1e12
epislon_0 = 8.854*1e-12
m_0 = 0.511
rho = 1000 #kg/m^3
N = N_A*rho*1000/(A_H*n_H+A_O*n_O)
Z = Z_H + Z_O
N_H, N_O = N*n_H, N*n_O
omegas = np.array([A_H*n_H,A_O*n_O])/(A_H*n_H+A_O*n_O)
Zs = np.array([Z_H, Z_O])
Ns = np.array([N_H, N_O])
n = 1.33
if __name__=="__main__":
    psr = argparse.ArgumentParser()
    psr.add_argument('-o', dest='opt')
    args = psr.parse_args()
    energys = np.arange(1,10,0.1)
    betas = np.sqrt(energys**2-m_0**2)/energys
    n_betas = 1-betas**2
    # Ionization
    coeff = 1/8/np.pi/epislon_0**2*e**4/m_0*J2MeV**2/100
    coeff_ion = coeff*N*Z
    dE_ion = coeff_ion/betas**2*(np.log(m_0*betas**2*energys/2/I**2/n_betas)-np.log(2)*(2*np.sqrt(n_betas)-n_betas)+n_betas+(1-np.sqrt(n_betas))**2/8)
    # Bremsstralung
    coeff_bre = coeff/2/np.pi/137*Ns*Zs*(Zs+1)
    dE_bre = coeff_bre*energys.reshape((-1,1))*(4*np.log(2*energys.reshape(-1,1)/m_0)-4/3)@omegas
    # Cherenkov
    coeff_che = 1/8/np.pi/epislon_0*e**2/(3*1e8)**2*J2MeV/100
    delta_beta_2 = 27*10*1e30/4/np.pi/(1/betas**2-1)
    dE_che = coeff_che*(1-1/n**2/betas**2)*delta_beta_2
    with PdfPages(args.opt) as pdf:
        fig, ax =plt.subplots(dpi=150)
        ax.plot(energys, dE_ion, label='Ionization')
        ax.plot(energys, dE_bre, label='Bremsstrahlung')
        ax.plot(energys, dE_che, label='Cherenkov')
        ax.set_xlabel('Energy(MeV)')
        ax.set_ylabel('dE/dx(MeV/cm)')
        ax.legend()
        pdf.savefig(fig)
        ax.set_yscale('log')
        pdf.savefig(fig)
