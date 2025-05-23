'''
Calculate the cherenkov photons direction distribution with setted electron direction distribution
'''
import numpy as np
import argparse
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
plt.style.use('../journal.mplstyle')
from scipy.special import iv
from MCMC import mcmc, gaus_sin, expGaus, expGaus_cos, expGaus_sin, samplePhotonDirection
import scipy.integrate as integrate
from scipy.stats import norm
def rotate(v, n, phis):
    return n

# theory calculation
def ceren_pdf_alpha(alpha, theta_p, theta_c, sigma=np.pi/6):
    # integrate from [0, pi]
    coeff = np.cos(alpha)*np.sin(theta_p)*np.sin(theta_c) + np.cos(theta_p)*np.cos(theta_c)
    return np.exp((coeff - 1) / sigma**2) / np.pi / np.sqrt(1 - (coeff)**2)

def ceren_pdf(p, theta_p, theta_c, sigma=np.pi/6):
    # integrate from [0, pi]
    coeff = p *np.sin(theta_p)*np.sin(theta_c) + np.cos(theta_p)*np.cos(theta_c)
    return np.exp((coeff - 1) / sigma**2) / np.pi / np.sqrt(1 - p**2) / np.sqrt(1 - (coeff)**2)

def ceren_expGauscos_pdf_alpha(alpha, theta_p, theta_c, sigma=np.pi/6):
    A = norm.cdf(1, 0, sigma/2) - norm.cdf(0, 0, sigma/2)
    coeff = np.cos(alpha)*np.sin(theta_p)*np.sin(theta_c) + np.cos(theta_p)*np.cos(theta_c)
    return np.exp((coeff - 1) / sigma**2) / np.pi / np.sqrt(1 - coeff) / 2 / A 

def numint_expGaus(thetas):
    alphas = np.arange(1, 10000)/10000*np.pi
    binwidth = np.pi / 10000
    sin_theta_c = np.sqrt(1-cos_theta_c**2)
    s_theta_c_alpha = (np.sin(thetas)*sin_theta_c)[:, np.newaxis]*np.cos(alphas)[np.newaxis, :]
    c_thetac_c_theta = np.cos(thetas)*cos_theta_c
    P_c_theta = np.sum(np.exp((s_theta_c_alpha+c_thetac_c_theta[:, np.newaxis] - 1)/(np.pi/6)**2)/np.sqrt(1-(s_theta_c_alpha+c_thetac_c_theta[:, np.newaxis])**2), axis=1) * binwidth
    return P_c_theta, s_theta_c_alpha, c_thetac_c_theta

def numint_expGaus_cos(thetas, sigma=np.pi/6):
    A = norm.cdf(1, 0, sigma/2) - norm.cdf(0, 0, sigma/2)
    alphas = np.arange(1, 10000)/10000*np.pi
    binwidth = np.pi / 10000
    sin_theta_c = np.sqrt(1-cos_theta_c**2)
    s_theta_c_alpha = (np.sin(thetas)*sin_theta_c)[:, np.newaxis]*np.cos(alphas)[np.newaxis, :]
    c_thetac_c_theta = np.cos(thetas)*cos_theta_c
    P_c_theta = np.sum(
            np.exp((s_theta_c_alpha+c_thetac_c_theta[:, np.newaxis] - 1)/(np.pi/6)**2) / np.sqrt((1-(s_theta_c_alpha+c_thetac_c_theta[:, np.newaxis]))) / 2,
            axis=1)
    return P_c_theta / A * binwidth / np.pi, s_theta_c_alpha, c_thetac_c_theta

def numint_expGaus_sin(thetas, sigma=np.pi/6):
    A = sigma**2 * (1 - np.exp(-2 / sigma**2))
    alphas = np.arange(1, 10000)/10000*np.pi
    binwidth = np.pi / 10000
    sin_theta_c = np.sqrt(1-cos_theta_c**2)
    s_theta_c_alpha = (np.sin(thetas)*sin_theta_c)[:, np.newaxis]*np.cos(alphas)[np.newaxis, :]
    c_thetac_c_theta = np.cos(thetas)*cos_theta_c
    P_c_theta = np.sum(
            np.exp((s_theta_c_alpha+c_thetac_c_theta[:, np.newaxis] - 1)/(sigma)**2),
            axis=1)
    return P_c_theta / A * binwidth / np.pi, s_theta_c_alpha, c_thetac_c_theta

psr = argparse.ArgumentParser()
psr.add_argument('-i', dest='e_d', help='electron direction distribution')
psr.add_argument('-o', dest='opt', help='output figure file')
psr.add_argument('-N', dest='N', type=int, default=100000, help='number of electron')
psr.add_argument('--int', action='store_true', default=False, help='use MCMC or numeric integration method')
args = psr.parse_args()
N = args.N
N_burn = 5000
theta_c = 42
cos_theta_c = np.cos(theta_c / 180 * np.pi)
# sample electron direction
thetas_p = np.arange(1, 180, 0.1)/180*np.pi
c_thetas_p = np.cos(thetas_p)
dir_sigma = np.pi / 6 # consistent with the value used in expGaus
if not args.int:
    if args.e_d == 'gaus':
        # use gaussian; without consider the boundary [0, 2pi]
        thetas = np.abs(np.random.normal(0, 30, N)) / 180 * np.pi
        Pe_thetas_p = np.exp(-thetas_p**2/2/dir_sigma**2)/np.sqrt(2*np.pi)/np.pi*6 *2 # normalize, *2 is due to half axis
        Pe_c_thetas_p = Pe_thetas_p / np.sin(thetas_p)
        Pp_c_thetas_p, s_theta_c_alpha, c_thetac_c_theta = numint_expGaus(thetas_p)

    elif args.e_d == 'gaus_sin':
        thetas = mcmc(N + N_burn, gaus_sin, 0.5, 0, jump=lambda x, y: (x + y)%np.pi)[N_burn:]
    
    elif args.e_d == 'expGaus':
        # sample the direction of electron used in the ToyMC
        thetas = mcmc(N + N_burn, expGaus, 0.5, 0, jump=lambda x, y: (x + y)%np.pi)[N_burn:]
        # use numerical method to calculate the distribution
        Pe_thetas_p = np.exp((c_thetas_p-1)/dir_sigma**2)
        Pe_thetas_p = Pe_thetas_p / np.sum(Pe_thetas_p) / (np.pi/180*0.1)
        Pe_c_thetas_p = Pe_thetas_p / np.sin(thetas_p)
        Pp_c_thetas_p, s_theta_c_alpha, c_thetac_c_theta = numint_expGaus(thetas_p)

    elif args.e_d == 'expGaus_cos':
        thetas = mcmc(N + N_burn, expGaus_cos, 0.5, 0, jump=lambda x, y: (x + y)%np.pi)[N_burn:]
        Pe_thetas_p = np.exp((c_thetas_p-1)/dir_sigma**2) * np.cos(thetas_p/2)
        Pe_thetas_p = Pe_thetas_p / np.sum(Pe_thetas_p) / (np.pi/180*0.1)
        Pe_c_thetas_p = Pe_thetas_p / np.sin(thetas_p)
        Pp_c_thetas_p, s_theta_c_alpha, c_thetac_c_theta = numint_expGaus_cos(thetas_p)

    elif args.e_d == 'expGaus_sin':
        dir_sigma = 15 / 180 * np.pi
        thetas = mcmc(N + N_burn, expGaus_sin, 0.5, 0, jump=lambda x, y: (x + y)%np.pi)[N_burn:]
        Pe_thetas_p = np.exp((c_thetas_p-1)/dir_sigma**2) * np.sin(thetas_p)
        Pe_thetas_p = Pe_thetas_p / np.sum(Pe_thetas_p) / (np.pi/180*0.1)
        Pe_c_thetas_p = Pe_thetas_p / np.sin(thetas_p)
        Pp_c_thetas_p, s_theta_c_alpha, c_thetac_c_theta = numint_expGaus_sin(thetas_p, dir_sigma)

else:
    thetas = mcmc(N + N_burn, expGaus_cos, 0.5, 0, jump=lambda x, y: (x + y)%np.pi)[N_burn:]
    Pe_thetas_p = np.exp((c_thetas_p-1)/dir_sigma**2) * np.cos(thetas_p/2)
    Pe_thetas_p = Pe_thetas_p / np.sum(Pe_thetas_p) / (np.pi/180*0.1)
    Pe_c_thetas_p = Pe_thetas_p / np.sin(thetas_p)
    f_cos = np.zeros((thetas_p.shape[0], 2))
    for i, theta in enumerate(thetas_p):
        # f_cos[i] = integrate.quad(lambda x: ceren_pdf(x, theta, theta_c / 180 * np.pi, dir_sigma), -1, 1)
        f_cos[i] = integrate.quad(lambda x: ceren_expGauscos_pdf_alpha(x, theta, theta_c / 180 * np.pi, dir_sigma), 0, np.pi)
    print(f_cos)

# sample photon direction use ToyMC
phis  = np.random.rand(N) * np.pi
n_init = np.array([0, 0, 1])
photon_ns = samplePhotonDirection(thetas, phis, cos_theta_c, n_init, N, 10)

'''
N_theta, N_c = 18000, 4300
e_theta_degree = np.arange(N_theta)/N_theta*180
e_theta = e_theta_degree*np.pi/180
p_e = np.exp(-(e_theta_degree/30)**2/2)
p_phase_theta = np.sin(e_theta) * p_e
p_photon, p_photon_p = np.zeros(N_theta), np.zeros(N_theta)
p_phase_theta_cumsum = np.insert(np.cumsum(p_phase_theta), 0, 0)
p_theta_cumsum = np.insert(np.cumsum(p_e), 0, 0)

p_photon[:N_c] = p_phase_theta_cumsum[np.arange(N_c)+N_c] - p_phase_theta_cumsum[N_c-np.arange(N_c)]
p_photon[N_c:(N_theta-N_c)] = p_phase_theta_cumsum[np.arange(N_c, N_theta-N_c)+N_c] - p_phase_theta_cumsum[np.arange(N_c, N_theta-N_c)-N_c]
p_photon[(N_theta-N_c):] = p_phase_theta_cumsum[N_theta*2-N_c-np.arange(N_theta-N_c, N_theta)] - p_phase_theta_cumsum[np.arange(N_theta-N_c, N_theta)-N_c]
p_photon = p_photon/np.sum(p_photon)
p_photon_p[:N_c] = p_theta_cumsum[np.arange(N_c)+N_c] - p_theta_cumsum[N_c-np.arange(N_c)]
p_photon_p[N_c:(N_theta-N_c)] = p_theta_cumsum[np.arange(N_c, N_theta-N_c)+N_c] - p_theta_cumsum[np.arange(N_c, N_theta-N_c)-N_c]
p_photon_p[(N_theta-N_c):] = p_theta_cumsum[N_theta*2-N_c-np.arange(N_theta-N_c, N_theta)] - p_theta_cumsum[np.arange(N_theta-N_c, N_theta)-N_c]
p_photon_p = p_photon_p/np.sum(p_photon_p)
'''
# Wrong calculation using sin_theta

'''
Pp_c_thetas_p = iv(0, (np.pi/6)**2/np.sqrt(1-cos_theta_c)/np.sin(thetas_p))
'''
with PdfPages(args.opt) as pdf:
    fig, ax = plt.subplots()
    h = ax.hist(thetas, range=[0, np.pi], bins=1000, histtype='step', density=True, label='e')
    bin_width = np.pi/1000
    ax.plot(thetas_p, Pe_thetas_p, label='parameterized') # /np.sum(Pe_thetas_p)/2/(500*5/6/1500)/bin_width
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel('PDF')
    ax.set_title('electron direction distribution')
    ax.legend()
    pdf.savefig(fig)
    ax.set_yscale('log')
    pdf.savefig(fig)

    fig, ax = plt.subplots()
    ax.hist(np.cos(thetas), range=[-1, 1], bins=1000, histtype='step', density=True, label='e')
    ax.plot(c_thetas_p, Pe_c_thetas_p, label='paramterized') # /(1000*5/6/1500)/(2/1000)*2
    ax.set_xlabel(r'cos$\theta$')
    ax.set_ylabel('PDF')
    ax.set_title('electron direction distribution')
    ax.legend()
    pdf.savefig(fig)
    ax.set_yscale('log')
    pdf.savefig(fig)

    fig, ax = plt.subplots()
    ax.hist(np.arccos(photon_ns @ n_init), range=[0, np.pi], bins=1000, histtype='step', density=True, label='photon')
    ax.legend()
    ax.set_yscale('log')
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel('PDF')
    ax.set_title('photon direction distribution')
    pdf.savefig(fig)

    fig, ax = plt.subplots()
    h = ax.hist(photon_ns @ n_init, range=[-1, 1], bins=1000, histtype='step', density=True, label='photon(MC)')
    print(np.sum(h[0]))
    if not args.int:
        # ax.plot(c_thetas_p, Pp_c_thetas_p/np.sum(Pp_c_thetas_p)/(1000*5/6/1500)/(2/1000), alpha=0.6, label='photon(theory)')
        ax.plot(c_thetas_p, Pp_c_thetas_p, alpha=0.6, label='photon (theory)')
    else:
        print(np.sum(f_cos[:, 0]))
        ax.plot(c_thetas_p, f_cos[:, 0], label='photon (theory)')
    ax.legend()
    ax.set_xlabel(r'cos$\theta$')
    ax.set_ylabel('PDF')
    ax.set_title('photon direction distribution')
    pdf.savefig(fig)
    ax.set_yscale('log')
    pdf.savefig(fig)

'''
    fig, ax = plt.subplots()
    h = ax.hist(np.arccos(photon_ns@n_init), range=[0, np.pi], bins=1000, density=True, histtype='step', label='photon')
    print(np.sum(h[0])*np.pi/1000)
    ax.plot(e_theta, p_photon / (np.pi/1000) * N_theta/1000, label='predict')
    # ax.plot(e_theta, p_photon_p / (np.pi/1000) * N_theta/1000, label='predict')
    ax.legend()
    ax.set_xlabel(r'$\theta$')
    pdf.savefig(fig)
    ax.set_yscale('log')
    pdf.savefig(fig)
'''
