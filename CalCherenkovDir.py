'''
Calculate the cherenkov photons direction distribution with setted electron direction distribution
'''
import numpy as np
import argparse
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.special import iv
from MCMC import mcmc, gaus_sin, expGaus, expGaus_cos
def rotate(v, n, phis):
    return n
# theory calculation
def numint_expGaus(thetas):
    alphas = np.arange(1, 10000)/10000*np.pi
    sin_theta_c = np.sqrt(1-cos_theta_c**2)
    s_theta_c_alpha = (np.sin(thetas)*sin_theta_c)[:, np.newaxis]*np.cos(alphas)[np.newaxis, :]
    c_thetac_c_theta = np.cos(thetas)*cos_theta_c
    P_c_theta = np.sum(np.exp((s_theta_c_alpha+c_thetac_c_theta[:, np.newaxis])/(np.pi/6)**2)/np.sqrt(1-(s_theta_c_alpha+c_thetac_c_theta[:, np.newaxis])**2), axis=1)
    return P_c_theta, s_theta_c_alpha, c_thetac_c_theta
def numint_expGaus_cos(thetas):
    alphas = np.arange(1, 10000)/10000*np.pi
    sin_theta_c = np.sqrt(1-cos_theta_c**2)
    s_theta_c_alpha = (np.sin(thetas)*sin_theta_c)[:, np.newaxis]*np.cos(alphas)[np.newaxis, :]
    c_thetac_c_theta = np.cos(thetas)*cos_theta_c
    P_c_theta = np.sum(np.exp((s_theta_c_alpha+c_thetac_c_theta[:, np.newaxis])/(np.pi/6)**2)/np.sqrt((1-(s_theta_c_alpha+c_thetac_c_theta[:, np.newaxis]))/2)/2, axis=1)
    return P_c_theta, s_theta_c_alpha, c_thetac_c_theta
psr = argparse.ArgumentParser()
psr.add_argument('-i', dest='e_d', help='electron direction distribution')
psr.add_argument('-o', dest='opt', help='output figure file')
psr.add_argument('-N', dest='N', type=int, default=100000, help='number of electron')
args = psr.parse_args()
N = args.N
N_burn = 5000
cos_theta_c = np.cos(42/180*np.pi)
# sample electron direction
thetas_p = np.arange(1, 180, 0.1)/180*np.pi
c_thetas_p = np.cos(thetas_p)

if args.e_d == 'gaus':
    # use gaussian; without consider the boundary [0, 2pi]
    thetas = np.abs(np.random.normal(0, 30, N)) / 180 * np.pi
    Pe_thetas_p = np.exp(-thetas_p**2/2/(np.pi/6)**2)/np.sqrt(2*np.pi)/np.pi*6 *2 # normalize, *2 is due to half axis
    Pe_c_thetas_p = Pe_thetas_p / np.sin(thetas_p)
    Pp_c_thetas_p, s_theta_c_alpha, c_thetac_c_theta = numint_expGaus(thetas_p)
elif args.e_d == 'gaus_sin':
    thetas = mcmc(N + N_burn, gaus_sin, 0.5, 0, jump=lambda x, y: (x + y)%np.pi)[N_burn:]

elif args.e_d == 'expGaus':
    thetas = mcmc(N + N_burn, expGaus, 0.5, 0, jump=lambda x, y: (x + y)%np.pi)[N_burn:]
    Pe_thetas_p = np.exp((c_thetas_p-1)/(np.pi/6)**2)
    Pe_thetas_p = Pe_thetas_p / np.sum(Pe_thetas_p) / (np.pi/180*0.1)
    Pe_c_thetas_p = Pe_thetas_p / np.sin(thetas_p)
    Pp_c_thetas_p, s_theta_c_alpha, c_thetac_c_theta = numint_expGaus(thetas_p)
elif args.e_d == 'expGaus_cos':
    thetas = mcmc(N + N_burn, expGaus_cos, 0.5, 0, jump=lambda x, y: (x + y)%np.pi)[N_burn:]
    Pe_thetas_p = np.exp((c_thetas_p-1)/(np.pi/6)**2) * np.cos(thetas_p/2)
    Pe_thetas_p = Pe_thetas_p / np.sum(Pe_thetas_p) / (np.pi/180*0.1)
    Pe_c_thetas_p = Pe_thetas_p / np.sin(thetas_p)
    Pp_c_thetas_p, s_theta_c_alpha, c_thetac_c_theta = numint_expGaus_cos(thetas_p)
# sample phi
phis  = np.random.rand(N) * np.pi
n_init = np.array([0, 0, 1])
e_n_normal = np.zeros((N, 3))
e_n_normal[:, 0] = np.cos(phis)
e_n_normal[:, 1] = np.sin(phis)
e_ns = n_init * np.cos(thetas)[:, np.newaxis] + e_n_normal * np.sin(thetas)[:, np.newaxis]
# generate photons
photon_yield = 10
phis_photon = np.random.rand(N * photon_yield) * np.pi
e_n_auxiliary = np.cross(n_init, e_n_normal)
photon_n_init = e_ns * cos_theta_c + e_n_auxiliary * np.sqrt(1-cos_theta_c**2)
rotation = Rotation.from_rotvec(phis_photon[:, np.newaxis] * 2 * np.repeat(e_ns, photon_yield, 0))
photon_ns = rotation.apply(np.repeat(photon_n_init, photon_yield, 0))


    
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
    ax.hist(photon_ns @ n_init, range=[-1, 1], bins=1000, histtype='step', density=True, label='photon')
    ax.plot(c_thetas_p, Pp_c_thetas_p/np.sum(Pp_c_thetas_p)/(1000*5/6/1500)/(2/1000), label='parameterized')
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
