import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
def rotate(v, n, phis):
    return n
N = 100000
cos_theta_c = 43/180*np.pi
n_init = np.array([0, 0, 1])
thetas, phis = np.random.normal(0, 30, N)/180*np.pi, np.random.rand(N, 2)*np.pi
e_n_normal = np.zeros((N, 3))
e_n_normal[:, 0] = np.cos(phis[:, 0])
e_n_normal[:, 1] = np.sin(phis[:, 0])
e_ns = n_init * np.cos(thetas)[:, np.newaxis] + e_n_normal * np.sin(thetas)[:, np.newaxis]
# generate photons
e_n_auxiliary = np.cross(n_init, e_n_normal)
photon_n_init = e_ns * cos_theta_c + e_n_auxiliary * np.sqrt(1-cos_theta_c**2)
rotation = Rotation.from_rotvec(phis[:, 1][:, np.newaxis]*2*e_ns)
photon_ns = rotation.apply(photon_n_init)
with PdfPages('direction.pdf') as pdf:
    fig, ax = plt.subplots()
    ax.hist(np.cos(np.abs(thetas)), range=[-2, 2], bins=1000, histtype='step', label='e')
    ax.set_xlabel(r'cos$\theta$')
    ax.legend()
    ax.set_yscale('log')
    pdf.savefig(fig)
    fig, ax = plt.subplots()
    ax.hist(photon_ns@n_init, range=[-2,2], bins=1000, histtype='step', label='photon')
    ax.legend()
    ax.set_yscale('log')
    pdf.savefig(fig)

