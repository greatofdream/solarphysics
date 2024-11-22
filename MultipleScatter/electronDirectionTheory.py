'''
Theory distribution of cherenkov photon
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
plt.style.use('../journal.mplstyle')
from MCMC import mcmc_transfer, samplePhotonDirection
import h5py
import argparse

def rotate(v, n, phis):
    return n
psr = argparse.ArgumentParser()
psr.add_argument('--angle', dest='angle', default=90, type=int, help='electron angle degree')
psr.add_argument('-o', dest='opt', help='output file name')
psr.add_argument('--entries', type=int, default=100000, help='entries')
args = psr.parse_args()

theta_c = 42
N = args.entries
# former using divide_N decide the electron direction; now use angle ratio instead above divide_N
angle = args.angle / 180 * np.pi
# Cerenkov angle using 42 degree
cos_theta_c = np.cos(theta_c / 180 * np.pi)
thetas, phis = np.ones(N) * angle, np.random.rand(N, 2)*np.pi
n_init = np.array([0, 0, 1])
# ToyMC calculation
photon_ns = samplePhotonDirection(thetas, phis[:, 1], cos_theta_c, n_init, N, 1)

# theory calculation
N_burn = 10000
samples = mcmc_transfer(N+N_burn, angle, np.arccos(photon_ns[0, 2]), theta_c / 180 * np.pi)
with h5py.File(args.opt, 'w') as opt:
    opt.attrs['theta'] = args.angle
    opt.create_dataset('mc', data=photon_ns@n_init, compression='gzip')
    opt.create_dataset('theory', data=samples[N_burn:], compression='gzip')

with PdfPages(args.opt + '.pdf') as pdf:
    fig, ax = plt.subplots()
    ax.hist(np.cos(np.abs(thetas)), range=[-2, 2], bins=1000, histtype='step', label='e')
    ax.set_xlabel(r'cos$\theta$')
    ax.legend()
    ax.set_yscale('log')
    pdf.savefig(fig)

    fig, ax = plt.subplots()
    ax.hist(photon_ns@n_init, range=[-1, 1], bins=1000, histtype='step', label='photon(MC)')
    ax.hist(np.cos(samples[N_burn:]), range=[-1, 1], bins=1000, histtype='step', alpha=0.5, label='photon(theory)')
    ax.set_xlim([-1, 1])
    ax.legend()
    ax.set_xlabel(r'cos$\theta$')
    pdf.savefig(fig)
    ax.set_yscale('log')
    pdf.savefig(fig)

    fig, ax = plt.subplots()
    ax.hist(np.abs(thetas), range=[0, np.pi], bins=1000, histtype='step', label='e')
    ax.set_xlabel(r'$\theta$')
    ax.legend()
    pdf.savefig(fig)

    fig, ax = plt.subplots()
    h = ax.hist(np.arccos(photon_ns@n_init) / np.pi * 180, range=[0, 180], bins=1000, histtype='step', label='photon')
    #ax.hist(np.abs(np.arccos(samples[N_burn:])), range=[0, np.pi], bins=1000, histtype='step', alpha=0.5, label='predict')
    ax.hist(np.abs(samples[N_burn:]) / np.pi * 180, range=[0, 180], bins=1000, histtype='step', alpha=0.5, label='predict')
    ax.set_xlim([0, 180])
    ax.legend()
    ax.set_xlabel(r'$\theta[\degree]$')
    pdf.savefig(fig)
    ax.set_yscale('log')
    pdf.savefig(fig)

    fig, ax = plt.subplots()
    h = ax.hist(np.arccos(photon_ns@n_init), range=[0, np.pi], bins=1000, density=True, histtype='step', label='photon')
    ax.set_xlim([0, np.pi])
    print(np.sum(h[0])*np.pi/1000)
    ax.legend()
    ax.set_xlabel(r'$\theta$')
    pdf.savefig(fig)
    ax.set_yscale('log')
    pdf.savefig(fig)

