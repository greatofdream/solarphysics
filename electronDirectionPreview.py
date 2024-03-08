'''
Store the results to h5
Preview the electorn direction distribution using Monte Carlo simulation 
'''
import argparse

import uproot, h5py, numpy as np
import duckdb, pandas as pd
# https://duckdb.org/docs/guides/python/sql_on_pandas.html
import matplotlib.pyplot as plt
from matplotlib import colors
plt.style.use('./journal.mplstyle')
from matplotlib.backends.backend_pdf import PdfPages 
psr = argparse.ArgumentParser()
psr.add_argument('-i', dest='ipt', help='MC hdf5 file')
psr.add_argument('-o', dest='opt', help='output file name')
psr.add_argument('--energy', type=int, help='energy in MeV')
args = psr.parse_args()
E_mc = args.energy
E_k_cut = 0.264 # Cherenkov threshold
print('begin to read the root file')
with h5py.File(args.ipt, 'r') as ipt:
    simtruth = ipt['SimTruth'][:]
    simtrack = ipt['SimTrack'][:]
    simstep = ipt['SimStep'][:]
simtrack = pd.DataFrame(simtrack[simtrack['E_k']>E_k_cut])
# electron step array: The result array is hierarchical structure, have to use awkward array
e_xyzs = simstep[simstep["PdgId"]==11]
e_xyzs_cut = e_xyzs[e_xyzs["E_k"]>E_k_cut]
e_Ps_initialstep = e_xyzs[e_xyzs["TrackId"]==1]
# Merge array in different entries
e_xs, e_ys, e_zs, e_ls = e_xyzs['StepPoint_Pre_x'], e_xyzs['StepPoint_Pre_y'], e_xyzs['StepPoint_Pre_z'], e_xyzs['StepLength']
e_xs_cut, e_ys_cut, e_zs_cut, e_ls_cut, e_pxs_cut, e_pys_cut, e_pzs_cut, e_Eks_cut, e_dEs_cut = e_xyzs_cut['StepPoint_Pre_x'], e_xyzs_cut['StepPoint_Pre_y'], e_xyzs_cut['StepPoint_Pre_z'], e_xyzs_cut['StepLength'], e_xyzs_cut['Px'], e_xyzs_cut['Py'], e_xyzs_cut['Pz'], e_xyzs_cut['E_k'], e_xyzs_cut['dE']
print('finish read file and data selection')
e_prs_cut = np.sqrt(e_pxs_cut**2 + e_pys_cut**2)
e_pthetas_cut = np.pi/2 + np.arctan2(e_pzs_cut, e_prs_cut)
e_rs_cut = np.sqrt(e_xs_cut**2+e_ys_cut**2)
# theta variance
E_intervals = np.arange(0, E_mc+0.4, 0.2)
bins = np.digitize(e_Eks_cut, E_intervals)
df = pd.DataFrame(e_xyzs_cut)
df['ptheta'] = e_pthetas_cut
df['Interval'] = bins
ptheta_result = df.groupby('Interval')['ptheta'].agg(['mean', 'std', 'count'])
def weighted(xs):
    w = xs['StepLength']/np.sum(xs['StepLength'])
    x_mu = np.sum(xs['ptheta']*w)
    x_std = np.sqrt(np.sum(w*(xs['ptheta']-x_mu)**2))
    return pd.Series([x_mu, x_std], index=['mean', 'std'])
ptheta_w_result = df.groupby('Interval').apply(weighted)
ptheta_result['E'] = E_intervals[ptheta_result.index.values]
ptheta_w_result['E'] = E_intervals[ptheta_w_result.index.values]
# first step theta
ptheta_initial = pd.DataFrame(df).groupby('EventID').apply(lambda x: x.sort_values('T').head(1))
# ptheta_inital = ptheta_group.nth(0)
with h5py.File(args.opt, 'w') as opt:
    opt.attrs['E'] = E_mc
    opt.attrs['sigma_theta'] = np.std(ptheta_initial['ptheta'])
    opt.create_dataset('theta', data=ptheta_result.to_records(), compression='gzip')
    opt.create_dataset('theta_w', data=ptheta_w_result.to_records(), compression='gzip')
#     opt.create_dataset('theta_initial', data=ptheta_initial.to_records(), compression='gzip')

with PdfPages(args.opt+'.pdf') as pdf:
    # track deposition and check

    # position distribution projection on x-y and x-z plane in steps
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    h_xy = axs[0].hist2d(e_xs, e_ys, bins=[100, 100], norm=colors.LogNorm())
    fig.colorbar(h_xy[3], ax=axs[0])
    h_rz = axs[1].hist2d(np.sqrt(e_xs**2+e_ys**2), e_zs, bins=[100, 100], norm=colors.LogNorm())
    fig.colorbar(h_rz[3], ax=axs[1])
    axs[0].set_xlabel('x/mm')
    axs[0].set_ylabel('y/mm')
    axs[1].set_xlabel('r/mm')
    axs[1].set_ylabel('z/mm')
    plt.tight_layout()
    pdf.savefig(fig)

    # position distribution with energy cut projection on x-y and x-z plane in steps
    # step length weighted position distribution with cut projection on x-y and x-z plane in steps
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    h_xy = axs[0, 0].hist2d(e_xs_cut, e_ys_cut, bins=[100, 100], norm=colors.LogNorm())
    fig.colorbar(h_xy[3], ax=axs[0, 0])
    h_rz = axs[0, 1].hist2d(np.sqrt(e_xs_cut**2+e_ys_cut**2), e_zs_cut, bins=[100, 100], norm=colors.LogNorm())
    fig.colorbar(h_rz[3], ax=axs[0, 1])
    h_xy = axs[1, 0].hist2d(e_xs_cut, e_ys_cut, bins=[100, 100], weights=e_ls_cut, norm=colors.LogNorm())
    fig.colorbar(h_xy[3], ax=axs[1, 0])
    h_rz = axs[1, 1].hist2d(e_rs_cut, e_zs_cut, bins=[100, 100], weights=e_ls_cut, norm=colors.LogNorm())
    fig.colorbar(h_rz[3], ax=axs[1,1])
    axs[0, 0].set_title('Cherenkov Energy cut')
    axs[0, 0].set_xlabel('x/mm')
    axs[0, 0].set_ylabel('y/mm')
    axs[0, 1].set_xlabel('r/mm')
    axs[0, 1].set_ylabel('z/mm')
    axs[1, 0].set_title('Cherenkov Energy cut and weighted')
    axs[1, 0].set_xlabel('x/mm')
    axs[1, 0].set_ylabel('y/mm')
    axs[1, 1].set_xlabel('r/mm')
    axs[1, 1].set_ylabel('z/mm')
    plt.tight_layout()
    pdf.savefig(fig)

    print('check parent pdg')
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    e_xyzs_cut = pd.DataFrame(e_xyzs_cut)
    e_track_df = duckdb.sql(
    'select c.EventID, c.StepPoint_Pre_x, c.StepPoint_Pre_y, c.StepPoint_Pre_z, c.TrackId, p.ParentTrackId, p.PdgId, p.ParentPdgId from e_xyzs_cut c join (select d.EventID, d.TrackId, d.PdgId, d.ParentTrackId, q.PdgId as ParentPdgId from simtrack d join simtrack q on d.ParentTrackId = q.TrackId and c.EventID = q.EventID) p on c.TrackId = p.TrackId and c.EventID = p.EventID'
    ).df()
    print('get parent pdg')
    #e_xyzs_cuts['ParentTrackId'] = pd.DataFrame(simtrack).set_index('TrackId').loc[e_xyzs_cut['TrackId']]['ParentTrackId']
    #e_xyzs_cuts['ParentPdgId'] = pd.DataFrame(simtrack).set_index('TrackId').loc[e_xyzs_cut['ParentTrackId']]['PdgId']
    print('begin plot')
    for eid, e_xyz_cut in e_track_df.groupby('EventID'):
        for j, e_xyz in e_xyz_cut.groupby('TrackId'):
            color = 'k'
            if e_xyz.iloc[0]['ParentPdgId']!=11:
                if e_xyz.iloc[0]['ParentPdgId']==22:# parent is photon
                    color = 'r'
                else:
                    color = 'g'
            axs[0].plot(e_xyz['StepPoint_Pre_x'], e_xyz['StepPoint_Pre_y'], color=color)
            axs[1].plot(e_xyz['StepPoint_Pre_x'], e_xyz['StepPoint_Pre_z'], color=color)
    axs[0].set_title('Cherenkov Energy cut scatter(red parent Id is photon)')
    axs[0].set_xlabel('x/mm')
    axs[0].set_ylabel('y/mm')
    axs[1].set_xlabel('x/mm')
    axs[1].set_ylabel('z/mm')
    plt.tight_layout()
    pdf.savefig(fig)
    print('finished check parent Pdg')

    # step length weighted direction distribution with cut projection on x-y and x-z plane in steps
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    h_xy = axs[0, 0].hist2d(e_pxs_cut, e_pys_cut, bins=[100, 100], norm=colors.LogNorm())
    fig.colorbar(h_xy[3], ax=axs[0, 0])
    h_rz = axs[0, 1].hist2d(e_prs_cut, e_pzs_cut, bins=[100, 100], norm=colors.LogNorm())
    fig.colorbar(h_rz[3], ax=axs[0, 1])
    h_xy = axs[1, 0].hist2d(e_pxs_cut, e_pys_cut, bins=[100, 100], weights=e_ls_cut, norm=colors.LogNorm())
    fig.colorbar(h_xy[3], ax=axs[1, 0])
    h_rz = axs[1, 1].hist2d(e_prs_cut, e_pzs_cut, bins=[100, 100], weights=e_ls_cut, norm=colors.LogNorm())
    fig.colorbar(h_rz[3], ax=axs[1, 1])
    axs[0, 0].set_xlabel('Px/MeV')
    axs[0, 0].set_ylabel('Py/MeV')
    axs[0, 1].set_xlabel('Pr/MeV')
    axs[0, 1].set_ylabel('Pz/MeV')
    axs[0, 0].set_title('energy cut')
    axs[1, 0].set_xlabel('Px/MeV')
    axs[1, 0].set_ylabel('Py/MeV')
    axs[1, 1].set_xlabel('Pr/MeV')
    axs[1, 1].set_ylabel('Pz/MeV')
    axs[1, 0].set_title('steplength weighted')
    plt.tight_layout()
    pdf.savefig(fig)

    # no-weighted and step length weighted direction distribution with cut projection on E-theta plane in steps
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    h_xy = axs[0].hist2d(e_Eks_cut, e_pthetas_cut, bins=[100, 100], norm=colors.LogNorm())
    fig.colorbar(h_xy[3], ax=axs[0])
    h_rz = axs[1].hist2d(e_Eks_cut, e_pthetas_cut, bins=[100, 100], weights=e_ls_cut, norm=colors.LogNorm())
    fig.colorbar(h_rz[3], ax=axs[1])
    axs[0].set_xlabel('Ek/MeV')
    axs[0].set_ylabel(r'$\theta_P/^\circ$')
    axs[1].set_xlabel('Ek/MeV')
    axs[1].set_ylabel(r'$\theta_P/^\circ$')
    plt.tight_layout()
    pdf.savefig(fig)

    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    h_xy = axs[0].hist2d(e_Eks_cut, e_pthetas_cut, bins=[100, 100])
    fig.colorbar(h_xy[3], ax=axs[0])
    axs[0].errorbar(E_intervals[ptheta_result.index.values], ptheta_result['mean'], ptheta_result['std'])
    h_rz = axs[1].hist2d(e_Eks_cut, e_pthetas_cut, bins=[100, 100], weights=e_ls_cut)
    fig.colorbar(h_rz[3], ax=axs[1])
    axs[1].errorbar(E_intervals[ptheta_w_result.index.values], ptheta_w_result['mean'], ptheta_w_result['std'])
    axs[0].set_xlabel('Ek/MeV')
    axs[0].set_ylabel(r'$\theta_P/^\circ$')
    axs[1].set_xlabel('Ek/MeV')
    axs[1].set_ylabel(r'$\theta_P/^\circ$')
    plt.tight_layout()
    pdf.savefig(fig)
    # theta, phi of direction distribution step 
    # Energy vs weighted r, z
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    h_xy = axs[0].hist2d(e_Eks_cut, e_rs_cut, bins=[100, 100], weights=e_ls_cut, norm=colors.LogNorm())
    fig.colorbar(h_xy[3], ax=axs[0])
    h_rz = axs[1].hist2d(e_Eks_cut, e_zs_cut, bins=[100, 100], weights=e_ls_cut, norm=colors.LogNorm())
    fig.colorbar(h_rz[3], ax=axs[1])
    axs[0].set_xlabel('Ek/MeV')
    axs[0].set_ylabel('r/mm')
    axs[1].set_xlabel('Ek/MeV')
    axs[1].set_ylabel('z/mm')
    plt.tight_layout()
    pdf.savefig(fig)
    # Energy vs stepLength, dE
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    h_xy = axs[0].hist2d(e_Eks_cut, e_ls_cut, bins=[100, 100], norm=colors.LogNorm())
    fig.colorbar(h_xy[3], ax=axs[0])
    h_rz = axs[1].hist2d(e_Eks_cut, e_dEs_cut, bins=[100, 100], norm=colors.LogNorm())
    fig.colorbar(h_rz[3], ax=axs[1])
    axs[0].set_xlabel('Ek/MeV')
    axs[0].set_ylabel('stepLength/mm')
    axs[1].set_xlabel('Ek/MeV')
    axs[1].set_ylabel('dE/MeV')
    plt.tight_layout()
    pdf.savefig(fig)
    # Ek vs position
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    axs[0].hist(e_Eks_cut, bins=100, histtype='step')
    axs[0].hist(e_Eks_cut, bins=100, weights=e_ls_cut, histtype='step', label='weight')
    axs[1].hist(e_rs_cut, bins=100, range=[0, 200], histtype='step')
    axs[1].hist(e_rs_cut, bins=100, range=[0, 200], weights=e_ls_cut, histtype='step', label='weight')
    axs[0].set_xlabel('Ek/MeV')
    axs[1].set_xlabel('r/mm')
    axs[0].legend()
    axs[1].legend()
    plt.tight_layout()
    pdf.savefig(fig)
    # Energy vs stepLength, dE
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    h_xy = axs[0].hist2d(e_Eks_cut, e_rs_cut, range=[[0, E_mc], [0, 200]], bins=[100, 100], norm=colors.LogNorm())
    fig.colorbar(h_xy[3], ax=axs[0])
    h_rz = axs[1].hist2d(e_rs_cut, e_pthetas_cut, weights=e_ls_cut, range=[[0, 200], [0, np.pi]], bins=[100, 100], norm=colors.LogNorm())
    fig.colorbar(h_rz[3], ax=axs[1])
    axs[0].set_xlabel('Ek/MeV')
    axs[0].set_ylabel('r/mm')
    axs[1].set_xlabel('r/mm')
    axs[1].set_ylabel(r'$\theta_p/^\circ$')
    plt.tight_layout()
    pdf.savefig(fig)
    # Energy vs stepLength, dE
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    axs[0].hist(e_pthetas_cut/np.pi*180, weights=e_ls_cut, range=[0, 180], bins=100, histtype='step', label='weighted')
    axs[0].hist(e_pthetas_cut/np.pi*180, range=[0, 180], bins=100, histtype='step', label='origin')
    h_rz = axs[1].hist2d(e_rs_cut, e_pthetas_cut/np.pi*180, range=[[0, 200], [0, 180]], bins=[100, 100], norm=colors.LogNorm())
    fig.colorbar(h_rz[3], ax=axs[1])
    axs[0].set_xlabel(r'$\theta_p/^\circ$')
    axs[0].legend()
    axs[1].set_xlabel('r/mm')
    axs[1].set_ylabel(r'$\theta_p/^\circ$')
    plt.tight_layout()
    pdf.savefig(fig)

    # Energy vs stepLength, dE
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    axs[0].hist(np.cos(e_pthetas_cut), weights=e_ls_cut, range=[-1, 1], bins=1000, histtype='step', label='weighted')
    axs[0].hist(np.cos(e_pthetas_cut), range=[-1, 1], bins=100, histtype='step', label='origin')
    h_rz = axs[1].hist2d(e_rs_cut, np.cos(e_pthetas_cut), range=[[0, 200], [-1, 1]], bins=[100, 100], norm=colors.LogNorm())
    fig.colorbar(h_rz[3], ax=axs[1])
    axs[0].set_xlabel(r'cos$\theta_p$')
    axs[0].legend()
    axs[1].set_xlabel('r/mm')
    axs[1].set_ylabel(r'cos$\theta_p$')
    plt.tight_layout()
    pdf.savefig(fig)

    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    axs[0].hist(ptheta_initial['ptheta']/np.pi*180, range=[0, 180], bins=100, histtype='step')
    axs[0].set_xlabel(r'$\theta_p/^\circ$')
    axs[1].hist(np.cos(ptheta_initial['ptheta']), range=[-1, 1], bins=100, histtype='step')
    axs[1].set_xlabel(r'cos$\theta_p$')
    pdf.savefig(fig)
