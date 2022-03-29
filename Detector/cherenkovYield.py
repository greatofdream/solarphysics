from EnergyLoss import *
from matplotlib.ticker import MultipleLocator
omega_0 = 6*1e15
lambda_0 = 3*1e8*2*np.pi/omega_0 # m
lambda_beta = 5.5*1e-7 #m
if __name__=="__main__":
    psr = argparse.ArgumentParser()
    psr.add_argument('-o', dest='opt')
    args = psr.parse_args()
    binw = 0.001
    energys = np.arange(0.776, 10, binw)
    energys_k = energys - m_0
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

    # Integration using bins data
    dEdx = dE_ion + dE_bre + dE_che
    dx = binw/dEdx
    # calculate the photons number
    dPhotons = 1/137.035*(1-1/n**2/betas**2)*(2*np.pi/lambda_0-2*np.pi/lambda_beta)/100
    photons = np.cumsum(dPhotons*dx)
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

        fig, ax = plt.subplots(dpi=150)
        ax.plot(energys_k, dPhotons)
        ax.set_xlabel('Kinetic Energy(MeV)')
        ax.set_ylabel('Photons/cm')
        ax.grid(axis='both')
        ax.xaxis.set_major_locator(MultipleLocator(1))
        pdf.savefig(fig)

        fig, ax = plt.subplots(dpi=150)
        ax.plot(energys_k, photons)
        ax.set_xlabel('Kinetic Energy(MeV)')
        ax.set_ylabel('Photons Number')
        ax.grid(axis='both')
        ax.xaxis.set_major_locator(MultipleLocator(1))
        pdf.savefig(fig)
