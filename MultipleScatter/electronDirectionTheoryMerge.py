'''
Merge the theory distribution of cherenkov photon
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator
plt.style.use('../journal.mplstyle')
import h5py
import argparse

def loadH5(f):
    with h5py.File(f, 'r') as ipt:
        mc = ipt['mc'][:]
        angle = ipt.attrs['theta']
    return mc, angle
psr = argparse.ArgumentParser()
psr.add_argument('-i', dest='ipt', nargs='+', help='hdf5 files')
psr.add_argument('-o', dest='opt', help='output file name')
args = psr.parse_args()

N = len(args.ipt)
res = [loadH5(args.ipt[i]) for i in range(N)]
with PdfPages(args.opt) as pdf:
    fig, ax = plt.subplots(figsize=(12, 6))
    for i in range(N):
       ax.hist(res[i][0], range=[-1, 1], bins=1000, histtype='step', label='{}$^\degree$'.format(res[i][1]))
    ax.set_xlim([-1, 1])
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.legend(ncols=3)
    ax.set_xlabel(r'cos$\theta$')
    ax.set_ylabel('Entries')
    pdf.savefig(fig)
    ax.set_yscale('log')
    pdf.savefig(fig)

 
