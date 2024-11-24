'''
Theory mc for the scattering with prior distribution of point source
'''
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator
from tqdm import tqdm
from MCMC import mcmc, expGaus_cos, samplePhotonDirection
np.seterr(all='raise')
class Tub:
    def __init__(self, R, halfZ):
        self.R = R
        self.halfZ = halfZ
    def inter(self, v, n):
        pos = self.interPlane(v, n)
        if np.sqrt(pos[0]**2 + pos[1]**2) > self.R:
            pos = self.interTub(v, n)
        return pos
    def interTub(self, v, n):
        a, b, c = n[0]**2 + n[1]**2, 2 * (v[0]*n[0] + v[1]*n[1]), v[0]**2+v[1]**2-self.R**2
        t = (-b + np.sqrt(b**2-4*a*c))/2/a
        v_new = v + t * n
        return v_new
    def interPlane(self, v, n):
        if n[2]>0:
            t = (self.halfZ - v[2]) / n[2]
        else:
            t = (-self.halfZ - v[2]) / n[2]
        v_new = v + t * n
        return v_new

class Sphere:
    def __init__(self, R):
        self.R = R
    def inter(self, v, n):
        b, c = v@n, v@v - self.R**2
        t = np.sqrt(b**2 - c) - b
        v_new = v + t * n
        return v_new
class Photon():
    def __init__(self):
        pass
    def iso(self, N):
        # generate random direction
        random = np.random.rand(N, 2)
        c_theta_p0, phi_p0 = random[:, 0] * 2 - 1, random[:, 1] * np.pi * 2
        s_theta_p0 = np.sqrt(1 - c_theta_p0**2)
        px_p0, py_p0, pz_p0 = s_theta_p0 * np.cos(phi_p0), s_theta_p0 * np.sin(phi_p0), c_theta_p0
        return px_p0, py_p0, pz_p0
    def cerenkov(self, N, n_v, N_burn = 5000, cos_theta_c = np.cos(42/180*np.pi)):
        thetas = mcmc(N + N_burn, expGaus_cos, 0.5, 0, jump=lambda x, y: (x + y)%np.pi)[N_burn:]
        phis  = np.random.rand(N) * np.pi
        photon_ns = samplePhotonDirection(thetas, phis, cos_theta_c, n_v, N, 1)
        return photon_ns[:, 0], photon_ns[:, 1], photon_ns[:, 2]

psr = argparse.ArgumentParser()
psr.add_argument('-o', dest='opt', help='output file name')
psr.add_argument('--vertex', dest='vertex', nargs='+', default=['0', '0', '-1600'], help='vertex')
psr.add_argument('--dir', dest='dir', nargs='+', default=['0', '0', '-1'], help='vertex')
psr.add_argument('-R', dest='R', type=int, default=1690, help='radius')
psr.add_argument('-hz', dest='hz', type=int, default=1810, help='half of Z')
psr.add_argument('--geo', dest='geo', default='Tub', help='geometry')
args = psr.parse_args()

scatterL = 10000#cm
N = 1000000
N_p = 10
v = np.array([float(i) for i in args.vertex])
n_v = np.array([float(i) for i in args.dir])
R, halfZ = args.R, args.hz
if args.geo == 'Tub':
    geo = Tub(R, halfZ)
else:
    geo = Sphere(R)


# photon emit from v[:], with uniform distribution
photonDirGenerator = Photon()
def scatter(geo, N, v, ps):
    px_p0, py_p0, pz_p0 = ps
    # scaterring direction
    Ls, is_scatter = np.zeros(N), np.zeros(N, dtype=bool)
    for i in tqdm(range(N)):
        p = geo.inter(v, np.array([px_p0[i], py_p0[i], pz_p0[i]]))
        Ls[i] = np.sqrt(np.sum((v-p)**2))
    # scatter probability = 1 - exp(-L/scatterL)
    scatterP = 1 - np.exp(-Ls/scatterL)
    random = np.random.rand(N, 2)
    is_scatter = scatterP > random[:, 0]
    t_Ls = random[is_scatter, 1] * Ls[is_scatter]
    scatterN = np.sum(is_scatter)
    print('scatter/total: {}/{}'.format(scatterN, N))

    x_p1 = v[0] + t_Ls * px_p0[is_scatter]
    y_p1 = v[1] + t_Ls * py_p0[is_scatter]
    z_p1 = v[2] + t_Ls * pz_p0[is_scatter]

    # scatter distribution is uniform
    random_p = np.random.rand(scatterN, 2)
    c_theta_p, phi_p = random_p[:, 0] *2 -1, random_p[:, 1] * np.pi * 2
    s_theta_p = np.sqrt(1 - c_theta_p**2)
    px_v, py_v, pz_v = s_theta_p * np.cos(phi_p), s_theta_p * np.sin(phi_p), c_theta_p

    photon_res = np.empty((scatterN,), dtype=[('ID', np.int32), ('x_scatter', np.float64), ('y_scatter', np.float64), ('z_scatter', np.float64), ('x', np.float64), ('y', np.float64), ('z', np.float64), ('c_theta', np.float64)])
    photon_res['x_scatter'] = x_p1
    photon_res['y_scatter'] = y_p1
    photon_res['z_scatter'] = z_p1
    for i in tqdm(range(scatterN)):
        p = geo.inter(np.array([x_p1[i], y_p1[i], z_p1[i]]), np.array([px_v[i], py_v[i], pz_v[i]]))
        photon_res[i] = (i, x_p1[i], y_p1[i], z_p1[i], p[0], p[1], p[2], (p-v)@n_v/np.sqrt(np.sum((p-v)**2)))
    return photon_res
## isotropic direction
px_p0, py_p0, pz_p0 = photonDirGenerator.iso(N)
photon_iso = scatter(geo, N, v, [px_p0, py_p0, pz_p0])
## cherenkov direction
px_p0, py_p0, pz_p0 = photonDirGenerator.cerenkov(N, n_v)
photon_cerenkov = scatter(geo, N, v, [px_p0, py_p0, pz_p0])

with h5py.File(args.opt, 'w') as opt:
    opt.create_dataset('iso', data=photon_iso, compression='gzip')
    opt.create_dataset('cerenkov', data=photon_cerenkov, compression='gzip')

with PdfPages(args.opt+'.pdf') as pdf:
    fig, ax = plt.subplots()
    ax.hist(photon_iso['c_theta'], bins=100, range=[-1, 1], density=True, histtype='step', label='isotropic')
    ax.hist(photon_cerenkov['c_theta'], bins=100, range=[-1, 1], density=True, histtype='step', label='cerenkov')
    ax.set_xlabel(r'$\cos{\theta}$')
    ax.set_ylabel('PDF')
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax.legend()
    pdf.savefig(fig)
 
