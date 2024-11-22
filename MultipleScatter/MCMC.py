import numpy as np
# 采样
def gaus_sin(x, sigma=30/180*np.pi):
    return np.exp(-x**2/2/sigma**2) * np.sin(x)
def expGaus(x, sigma=30/180*np.pi):
    return np.exp((np.cos(x)-1)/sigma**2)
def expGaus_cos(x, sigma=30/180*np.pi):
    return np.exp((np.cos(x)-1)/sigma**2) * np.cos(x/2)
def mcmc(N, distribution, wander_sigma, start_v, jump=lambda x, y: x + y):
    # use distribution to do MCMC sample
    wander, accept = np.random.normal(0, wander_sigma, N), np.random.rand(N)
    samples = np.zeros(N)
    samples[0] = start_v
    for i in range(N-1):
        v_next = jump(samples[i], wander[i])
        p_n_0 = distribution(v_next)/distribution(samples[i])
        if accept[i]<p_n_0:
            samples[i+1] = v_next
        else:
            samples[i+1] = samples[i]
    return samples
def mc(N, theta_e=np.pi/3, theta_c=43/180*np.pi):
    wander, accept = np.random.normal(0, 0.5, N), np.random.rand(N)
    theta_s = np.zeros(N)
    theta_s[0] = theta_e
    for i in range(N-1):
        theta_next = theta_s[i] + wander[i]
        cos_phi_0 = (np.cos(theta_c)-np.cos(theta_s[i])*np.cos(theta_e))/np.sin(theta_s[i])/np.sin(theta_e)
        cos_phi_n = (np.cos(theta_c)-np.cos(theta_next)*np.cos(theta_e))/np.sin(theta_next)/np.sin(theta_e)
        if abs(cos_phi_n) -1 >-1e-6:
            theta_s[i+1] = theta_s[i]
            continue
        p_n_0 = np.sqrt(((np.cos(theta_c)*np.cos(theta_next)-np.cos(theta_e))**2+(1-cos_phi_n**2)*np.sin(theta_e)**2*np.sin(theta_next)**2)/((np.cos(theta_c)*np.cos(theta_s[i])-np.cos(theta_e))**2+(1-cos_phi_0**2)*np.sin(theta_e)**2*np.sin(theta_s[i])**2))*np.sqrt((1-cos_phi_0**2)/(1-cos_phi_n**2))/np.sin(theta_next)*np.sin(theta_s[i])
        if accept[i]<p_n_0:
            theta_s[i+1] = theta_next
        else:
            theta_s[i+1] = theta_s[i]
    return theta_s
def mcmc_transfer(N, theta_e=np.pi/3, theta_0=np.pi/3, theta_c=43/180*np.pi):
    # use mcmc sample transfer probability
    wander, accept = np.random.normal(0, 0.5, N), np.random.rand(N)
    theta_s = np.zeros(N)
    theta_s[0] = theta_0
    for i in range(N-1):
        theta_next = theta_s[i] + wander[i]
        cos_phi_n = (np.cos(theta_c)-np.cos(theta_next)*np.cos(theta_e))/np.sin(theta_next)/np.sin(theta_e)
        if abs(cos_phi_n) -1 >-1e-6:
            theta_s[i+1] = theta_s[i]
            continue
        p_n_0 = np.sin(theta_next)/np.sin(theta_s[i])*np.sqrt(((np.sin(theta_c)*np.sin(theta_s[i]))**2-(np.cos(theta_e)-np.cos(theta_c)*np.cos(theta_s[i]))**2)/((np.sin(theta_c)*np.sin(theta_next))**2-(np.cos(theta_e)-np.cos(theta_c)*np.cos(theta_next))**2))
        if accept[i]<p_n_0:
            theta_s[i+1] = theta_next
        else:
            theta_s[i+1] = theta_s[i]
    return theta_s

# 按照cos采样
def mc_c(N, theta_e=np.pi/3, theta_c=43/180*np.pi):
    wander, accept = np.random.normal(0, 0.1, N), np.random.rand(N)
    theta_s = np.zeros(N)
    theta_s[0] = np.cos(theta_e)
    for i in range(N-1):
        theta_next = theta_s[i] + wander[i]
        cos_phi_0 = (np.cos(theta_c)-theta_s[i]*np.cos(theta_e))/np.sqrt(1-theta_s[i]**2)/np.sin(theta_e)
        cos_phi_n = (np.cos(theta_c)-theta_next*np.cos(theta_e))/np.sqrt(1-theta_next**2)/np.sin(theta_e)
        if abs(cos_phi_n)-1>-1e-3:
            theta_s[i+1] = theta_s[i]
            continue
        p_n_0 = np.sqrt(((np.cos(theta_c)*theta_next-np.cos(theta_e))**2+(1-cos_phi_n**2)*np.sin(theta_e)**2)/((np.cos(theta_c)*theta_s[i]-np.cos(theta_e))**2+(1-cos_phi_0**2)*np.sin(theta_e)**2))/(1-theta_next**2)*(1-theta_s[i]**2)*np.sqrt((1-cos_phi_0**2)/(1-cos_phi_n**2))
        if accept[i]<p_n_0: 
            theta_s[i+1] = theta_next
        else:
            theta_s[i+1] = theta_s[i]
    return theta_s
