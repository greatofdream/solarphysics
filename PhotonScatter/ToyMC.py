'''
Theory mc for the scattering with prior distribution of point source
'''
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
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

psr = argparse.ArgumentParser()
psr.add_argument('-o', dest='opt', help='output file name')
args = psr.parse_args()

scatterL = 10000#cm
N = 1000000
N_p = 10
v = np.array([0, 0, -1600])
n_v = np.array([0, 0, -1])
R, halfZ = 1690, 1810
tub = Tub(R, halfZ)


# photon emit from v[:], with uniform distribution
random = np.random.rand(N, 4)
c_theta_p0, phi_p0 = random[:, 0] * 2 - 1, random[:, 1] * np.pi * 2
s_theta_p0 = np.sqrt(1 - c_theta_p0**2)
px_p0, py_p0, pz_p0 = s_theta_p0 * np.cos(phi_p0), s_theta_p0 * np.sin(phi_p0), c_theta_p0
Ls, is_scatter = np.zeros(N), np.zeros(N, dtype=bool)
for i in tqdm(range(N)):
    p = tub.inter(v, np.array([px_p0[i], py_p0[i], pz_p0[i]]))
    Ls[i] = np.sqrt(np.sum((v-p)**2))
# scatter probability = 1 - exp(-L/scatterL)
scatterP = 1 - np.exp(-Ls/scatterL)
is_scatter = scatterP>random[:,2]
t_Ls = random[is_scatter,3] * Ls[is_scatter]
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
    p = tub.inter(np.array([x_p1[i], y_p1[i], z_p1[i]]), np.array([px_v[i], py_v[i], pz_v[i]]))
    photon_res[i] = (i, x_p1[i], y_p1[i], z_p1[i], p[0], p[1], p[2], (p-v)@n_v/np.sqrt(np.sum((p-v)**2)))

with h5py.File(args.opt, 'w') as opt:
    opt.create_dataset('res', data=photon_res, compression='gzip')

with PdfPages(args.opt+'.pdf') as pdf:
    fig, ax = plt.subplots()
    ax.hist(photon_res['c_theta'], bins=100, histtype='step')
    ax.set_xlabel(r'$\cos{\theta}$')
    pdf.savefig(fig)
 
