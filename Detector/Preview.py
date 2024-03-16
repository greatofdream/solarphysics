from ast import arg
from cProfile import label
import numpy as np
import argparse
from Reader import JUNOReader, JinpingReader, SKReader
import WDSim
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator
plt.style.use('../journal.mplstyle')
from scipy.constants import physical_constants
junoAbsorptionFile = '/cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc830/Pre-Release/J21v1r0-Pre2/data/Simulation/DetSim/Material/Water/ABSLENGTH'
junoScaleFile = '/cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc830/Pre-Release/J21v1r0-Pre2/data/Simulation/DetSim/Material/Water/scale'
junoRindexFile = '/cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc830/Pre-Release/J21v1r0-Pre2/data/Simulation/DetSim/Material/Water/RINDEX'
h = 4.13567
if __name__=="__main__":
    psr = argparse.ArgumentParser()
    psr.add_argument('-o', dest='opt')
    args = psr.parse_args()
    junor = JUNOReader()
    jinpingr = JinpingReader()
    skr = SKReader()

    junor.read(junoAbsorptionFile, 'abs')
    junor.read(junoScaleFile, 'scale')
    junor.read(junoRindexFile, 'rindex')
    jinpingr.read('Jinping/Water.xml')
    skr.readPMTQE()
    with PdfPages(args.opt) as pdf:
        # water paramter
        fig, ax = plt.subplots(dpi=150, figsize=(10,6))
        ax2 = ax.twiny()
        ax.plot(junor.abs['E'], junor.abs['abs']*junor.abs_scale, label='JUNO offline')
        ax.plot(WDSim.ENERGY_water, WDSim.ABSORPTION_water, label='SuperK')
        ax.plot(jinpingr.abs[:,0], jinpingr.abs[:,1], label='Jinping')
        ax.set_xlabel('Energy/eV')
        ax.set_ylabel('Absorption Length/cm')
        ax2.set_xlim(ax.get_xlim())
        energy = np.arange(1,11)
        ax2.set_xticks(energy)
        ax2.set_xticklabels(["%.1f" % z for z in h*3/energy*100])
        ax2.set_xlabel('Wavelength/nm')
        ax.yaxis.set_minor_locator(MultipleLocator(5000))
        ax.set_ylim([0,90000])
        ax.legend()
        ax.set_title('water parameter')
        pdf.savefig(fig)
        # refraction
        fig, ax = plt.subplots(dpi=150)
        ax.plot(skr.pmtQE['lambda'], skr.pmtQE['QE'])
        secax = ax.secondary_xaxis('top', functions=(lambda x: physical_constants['speed of light in vacuum'][0]/x*physical_constants['Planck constant in eV/Hz'][0]*1E9, lambda x: physical_constants['speed of light in vacuum'][0]*(physical_constants['reduced Planck constant in eV s'][0]*1E9)/x,),)
        secax.set_xlabel('E/eV')
        ax.set_xlabel('$\lambda$/nm')
        ax.set_ylabel('QE/%')
        ax.xaxis.set_minor_locator(MultipleLocator(10))
        ax.yaxis.set_minor_locator(MultipleLocator(1))
        pdf.savefig(fig)
