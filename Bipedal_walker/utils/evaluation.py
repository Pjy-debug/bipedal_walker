import numpy as np
from scipy.stats import norm
from tqdm import tqdm
from operator import itemgetter
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')


def var_next(n, x_n, mean_old, var_old):
    var_new = var_old * (n-1) / n + (x_n - mean_old)**2 * (n-1) / n**2
    return var_new

def mean_next_batch(m, n, mean_batch, mean_old):
    mean_new = mean_old + m*(mean_batch - mean_old)/n
    return mean_new

def var_next_batch(m, n, mean_batch, var_batch, mean_old, var_old):
    var_new = (n-m)*var_old/n + m*(n-m)*(mean_batch-mean_old)**2/n**2 + m*var_batch/n
    return var_new

def get_crash_rate(crash):
    num = len(crash)
    crash_rate = np.zeros_like(crash)
    crash_accum = 0
    for i in range(num):
        crash_accum += crash[i]
        crash_rate[i] = crash_accum / (i + 1)
    return crash_rate

def get_crash_rate_batch(crash, sep=10**4):
    m = sep
    n = 0
    mean= 0
    num = len(crash) // sep
    crash_rate = np.zeros(num)
    for i in range(num):
        n += m
        mean_batch = np.mean(crash[i*sep:(i+1)*sep])
        mean = mean_next_batch(m, n, mean_batch, mean)
        crash_rate[i] = mean
    return crash_rate

def get_rela_half_width(crash, confidence_alpha = 0.05):
    z_alpha = norm.ppf(1 - confidence_alpha)
    num = len(crash)
    half_width = np.zeros(num)
    var_tmp = 0
    mean_tmp = 0
    for i in range(num):
        var_tmp = var_next(i + 1, crash[i], mean_tmp, var_tmp)
        mean_tmp += (crash[i] - mean_tmp) / (i + 1)
        I_alpha = z_alpha * np.sqrt(var_tmp / (i + 1))
        half_width[i] = I_alpha / (mean_tmp + 1e-30)
    return half_width

def get_rela_half_width_batch(crash, sep=10**4, confidence_alpha=0.05):
    m = sep
    n = 0
    mean, var = 0, 0
    num = len(crash) // sep
    half_width = np.zeros(num)
    z_alpha = norm.ppf(1 - confidence_alpha)
    for i in range(num):
        n += m
        mean_batch = np.mean(crash[i*sep:(i+1)*sep])
        var_batch = np.var(crash[i*sep:(i+1)*sep])
        var = var_next_batch(m, n, mean_batch, var_batch, mean, var)
        mean = mean_next_batch(m, n, mean_batch, mean)
        half_width[i] = z_alpha * np.sqrt(var / n) / (mean + 1e-30)
    return half_width

def get_rela_half_width_i(crash, confidence_alpha = 0.05):
    z_alpha = norm.ppf(1 - confidence_alpha)
    var = np.var(crash)
    mean = np.average(crash)
    I_alpha = z_alpha * np.sqrt(var / len(crash))
    half_width = I_alpha / (mean + 1e-30)
    return half_width

def get_rela_half_width_i_recur(crash, confidence_alpha = 0.05):
    z_alpha = norm.ppf(1 - confidence_alpha)
    num = len(crash)
    var_tmp = 0
    mean_tmp = 0
    for i in range(num):
        var_tmp = var_next(i + 1, crash[i], mean_tmp, var_tmp)
        mean_tmp += (crash[i] - mean_tmp) / (i + 1)
    I_alpha = z_alpha * np.sqrt(var_tmp / (i + 1))
    half_width = I_alpha / (mean_tmp + 1e-30)
    return half_width

def get_rela_half_width_cv(i, X, control_step, CV, index_permuted):
    if i == 0: return 0
    X_CV = {}
    estimator = Ridge(alpha=1e-8)
    X_i = X[:i]
    control_steps = np.array(list(set(control_step[:i]))).astype(int)
    for l in control_steps:
        if l == 0:
            critical_set = (control_step[:i] == l)
            X_CV[l] = X_i[critical_set]
        else:
            critical_set = (control_step[:i] == l)
            index = index_permuted[:i]
            index = index[critical_set]
            Y = np.array(itemgetter(*index)(CV))
            Z = Y - Y.mean(axis=0)
            if Z.ndim == 1: Z = Z.reshape(1,-1)
            estimator.fit(Z, X_i[critical_set])
            beta = estimator.coef_
            X_CV[l] = X_i[critical_set] - beta.dot(Z.T)
    XX = np.hstack([X_CV[l] for l in control_steps])
    return get_rela_half_width_i(XX)

def eval_SCV_i(n, new_AV=False, th=0):
    """Evaluate SCV of the front `n` testing results.
    """
    if new_AV:
        crash_NADE_CV = np.load(f'data/SCV_crash_NADE_1e7_av3_{th}.npy')
        control_step = np.load(f'data/SCV_CS_1e7_av3_{th}.npy')
        crash_NADE_CV = crash_NADE_CV.flatten()
        control_step = control_step.flatten()
        control_variares = np.load(f'data/SCV_CV_1e7_av3_{th}.npy', allow_pickle=True)
    else:
        crash_NADE_CV = np.load(f'data/SCV_crash_NADE_1e7_{th}.npy')
        control_step = np.load(f'data/SCV_CS_1e7_{th}.npy')
        crash_NADE_CV = crash_NADE_CV.flatten()
        control_step = control_step.flatten()
        control_variares = np.load(f'data/SCV_CV_1e7_{th}.npy', allow_pickle=True)
    CV = {}
    for i in range(len(control_variares)):
        for j in range(len(control_variares[0])):
            CV[j + len(control_variares[0]) * i] = control_variares[i][j]

    estimator = Ridge(alpha=1e-8)
    X = crash_NADE_CV
    X_CV = {}
    X_CS = {}
    X_i = X[:n]
    control_steps = np.array(list(set(control_step[:n]))).astype(int)
    for l in np.delete(control_steps, 0):
        critical_set = (control_step[:n] == l)
        index = np.arange(n)
        index = index[critical_set]
        Y = np.array(itemgetter(*index)(CV))
        Z = Y - Y.mean(axis=0)
        if Z.ndim == 1: Z = Z.reshape(1,-1)
        estimator.fit(Z, X_i[critical_set])
        beta = estimator.coef_
        X_CV[l] = X_i[critical_set] - beta.dot(Z.T)
        X_CS[l] = X_i[critical_set]
    return control_step, X_CS, X_CV

def swap_crash(X, index_permuted, control_step):
    idx1 = 8330811
    idx11 = 18330811
    idx2 = 8673240
    idx22 = 18673240
    X[idx1], X[idx11] = X[idx11], X[idx1]
    X[idx2], X[idx22] = X[idx2], X[idx22]
    index_permuted[idx1], index_permuted[idx11] = index_permuted[idx11], index_permuted[idx1]
    index_permuted[idx2], index_permuted[idx22] = index_permuted[idx22], index_permuted[idx2]
    control_step[idx1], control_step[idx11] = control_step[idx11], control_step[idx1]
    control_step[idx2], control_step[idx22] = control_step[idx22], control_step[idx2]
    return X, index_permuted, control_step

def eval_SCV_shuffle(start, num, sep=1000, seed=42, half_width_threshold=0.3, swap=False):
    crash_NADE = np.load('data/crash_NADE_3e7.npy')
    control_step = np.load('data/control_step_3e7.npy')
    CV = np.load('data/control_variares_3e7.npy', allow_pickle=True)
    CV = CV[0]
    num = int(num / sep)
    estimator = Ridge(alpha=1e-8)
    rng = np.random.default_rng(seed=seed)
    control_step = rng.permutation(control_step)
    rng = np.random.default_rng(seed=seed)
    index_permuted = rng.permutation(np.arange(len(CV)))
    rng = np.random.default_rng(seed=seed)
    X = rng.permutation(crash_NADE)
    if swap:
        idx1 = 8330811
        idx11 = 18330811
        idx2 = 8673240
        idx22 = 18673240
        X[idx1], X[idx11] = X[idx11], X[idx1]
        X[idx2], X[idx22] = X[idx2], X[idx22]
        index_permuted[idx1], index_permuted[idx11] = index_permuted[idx11], index_permuted[idx1]
        index_permuted[idx2], index_permuted[idx22] = index_permuted[idx22], index_permuted[idx2]
        control_step[idx1], control_step[idx11] = control_step[idx11], control_step[idx1]
        control_step[idx2], control_step[idx22] = control_step[idx22], control_step[idx2]
    X_CV = {}
    half_width_SCV = np.zeros(num)
    for k in tqdm(range(num)):
        i = int(start) + k * sep
        X_i = X[:i+1]
        control_steps = np.array(list(set(control_step[:i+1]))).astype(int)
        for l in control_steps:
            if l == 0:
                critical_set = (control_step[:i+1] == l)
                X_CV[l] = X_i[critical_set]
            else:
                critical_set = (control_step[:i+1] == l)
                index = index_permuted[:i+1]
                index = index[critical_set]
                Y = np.array(itemgetter(*index)(CV))
                Z = Y - Y.mean(axis=0)
                if Z.ndim == 1: Z = Z.reshape(1,-1)
                estimator.fit(Z, X_i[critical_set])
                beta = estimator.coef_
                X_CV[l] = X_i[critical_set] - beta.dot(Z.T)
        XX = np.hstack([X_CV[l] for l in control_steps])
        half_width_SCV[k] = get_rela_half_width_i(XX)
    if half_width_SCV.max() > half_width_threshold:
        num_SCV_th = np.where(half_width_SCV > half_width_threshold)[0][-1] + 1
    else:
        num_SCV_th = 0
    return num_SCV_th, half_width_SCV

def eval_SCV_shuffle_ablation(start, num, sep=10**4, seed=168, swap=False):
    """for ablation study
    """
    crash_NADE = np.load('data/crash_NADE_3e7.npy')
    control_step = np.load('data/control_step_3e7.npy')
    CV = np.load('data/control_variares_3e7.npy', allow_pickle=True)
    CV = CV[0]
    num = int(num / sep)
    estimator = Ridge(alpha=1e-8)
    rng = np.random.default_rng(seed=seed)
    control_step = rng.permutation(control_step)
    rng = np.random.default_rng(seed=seed)
    index_permuted = rng.permutation(np.arange(len(CV)))
    rng = np.random.default_rng(seed=seed)
    X = rng.permutation(crash_NADE)
    if swap:
        idx1 = 8330811
        idx11 = 18330811
        idx2 = 8673240
        idx22 = 18673240
        X[idx1], X[idx11] = X[idx11], X[idx1]
        X[idx2], X[idx22] = X[idx2], X[idx22]
        index_permuted[idx1], index_permuted[idx11] = index_permuted[idx11], index_permuted[idx1]
        index_permuted[idx2], index_permuted[idx22] = index_permuted[idx22], index_permuted[idx2]
        control_step[idx1], control_step[idx11] = control_step[idx11], control_step[idx1]
        control_step[idx2], control_step[idx22] = control_step[idx22], control_step[idx2]
    X_CV = {}
    X_CS = {}
    rhw_w = {}
    rhw_wo = {}
    for l in range(1,10):
        rhw_w[l] = np.zeros(num)
        rhw_wo[l] = np.zeros(num)
        X_CV[l] = []
        X_CS[l] = []
    for k in tqdm(range(num)):
        i = int(start) + k * sep
        X_i = X[:i+1]
        control_steps = np.array(list(set(control_step[:i+1]))).astype(int)
        for l in control_steps:
            if l == 0:
                critical_set = (control_step[:i+1] == l)
                X_CV[l] = X_i[critical_set]
                X_CS[l] = X_i[critical_set]
            else:
                critical_set = (control_step[:i+1] == l)
                index = index_permuted[:i+1]
                index = index[critical_set]
                Y = np.array(itemgetter(*index)(CV))
                Z = Y - Y.mean(axis=0)
                if Z.ndim == 1: Z = Z.reshape(1,-1)
                estimator.fit(Z, X_i[critical_set])
                beta = estimator.coef_
                X_CV[l] = X_i[critical_set] - beta.dot(Z.T)
                X_CS[l] = X_i[critical_set]
        for l in range(1, 10):
            XX_w = np.hstack((np.hstack([X_CS[i] for i in np.delete(range(10), l)]), X_CV[l]))
            XX_wo = np.hstack((np.hstack([X_CV[i] for i in np.delete(range(10), l)]), X_CS[l]))
            rhw_w[l][k] = get_rela_half_width_i(XX_w)
            rhw_wo[l][k] = get_rela_half_width_i(XX_wo)
    return rhw_w, rhw_wo

def eval_SCV_shuffle_i(n, seed=42, swap=False):
    crash_NADE_CV = np.load('data/crash_NADE_3e7.npy')
    control_step = np.load('data/control_step_3e7.npy')
    CV = np.load('data/control_variares_3e7.npy', allow_pickle=True)
    CV = CV[0]
    estimator = Ridge(alpha=1e-8)
    rng = np.random.default_rng(seed=seed)
    control_step = rng.permutation(control_step)
    rng = np.random.default_rng(seed=seed)
    index_permuted = rng.permutation(np.arange(len(CV)))
    rng = np.random.default_rng(seed=seed)
    X = rng.permutation(crash_NADE_CV)
    if swap:
        idx1 = 8330811
        idx11 = 18330811
        idx2 = 8673240
        idx22 = 18673240
        X[idx1], X[idx11] = X[idx11], X[idx1]
        X[idx2], X[idx22] = X[idx2], X[idx22]
        index_permuted[idx1], index_permuted[idx11] = index_permuted[idx11], index_permuted[idx1]
        index_permuted[idx2], index_permuted[idx22] = index_permuted[idx22], index_permuted[idx2]
        control_step[idx1], control_step[idx11] = control_step[idx11], control_step[idx1]
        control_step[idx2], control_step[idx22] = control_step[idx22], control_step[idx2]
    X_CV = {}
    X_CS = {}
    X_i = X[:n]
    control_steps = np.array(list(set(control_step[:n]))).astype(int)
    for l in control_steps:
        if l == 0:
            critical_set = (control_step[:n] == l)
            X_CV[l] = X_i[critical_set]
        else:
            critical_set = (control_step[:n] == l)
            index = index_permuted[:n]
            index = index[critical_set]
            Y = np.array(itemgetter(*index)(CV))
            Z = Y - Y.mean(axis=0)
            if Z.ndim == 1: Z = Z.reshape(1,-1)
            estimator.fit(Z, X_i[critical_set])
            beta = estimator.coef_
            X_CV[l] = X_i[critical_set] - beta.dot(Z.T)
            X_CS[l] = X_i[critical_set]
    return control_step, X_CS, X_CV

def eval_SCV(start, num, sep=10000, seed=42, AV=13, th=0):
    crash_NADE_CV = np.load(f'data/SCV_crash_NADE_3e7_av{AV}_{th}.npy')
    control_step = np.load(f'data/SCV_CS_3e7_av{AV}_{th}.npy')
    crash_NADE_CV = crash_NADE_CV.flatten()
    control_step = control_step.flatten()
    control_variares = np.load(f'data/SCV_CV_3e7_av{AV}_{th}.npy', allow_pickle=True)
    CV = {}
    for i in range(len(control_variares)):
        for j in range(len(control_variares[0])):
            CV[j + len(control_variares[0]) * i] = control_variares[i][j]

    rng = np.random.default_rng(seed=seed)
    control_step = rng.permutation(control_step)
    rng = np.random.default_rng(seed=seed)
    index_permuted = rng.permutation(np.arange(len(CV)))
    rng = np.random.default_rng(seed=seed)
    X = rng.permutation(crash_NADE_CV)

    num = int(num / sep)
    estimator = Ridge(alpha=1e-8)
    X_CV = {}
    half_width_SCV = np.zeros(num)
    for k in tqdm(range(num)):
        i = int(start) + k * sep
        X_i = X[:i+1]
        control_steps = np.array(list(set(control_step[:i+1]))).astype(int)
        for l in control_steps:
            if l == 0:
                critical_set = (control_step[:i+1] == l)
                X_CV[l] = X_i[critical_set]
            else:
                critical_set = (control_step[:i+1] == l)
                index = index_permuted[:i+1]
                index = index[critical_set]
                Y = np.array(itemgetter(*index)(CV))
                Z = Y - Y.mean(axis=0)
                if Z.ndim == 1: Z = Z.reshape(1,-1)
                estimator.fit(Z, X_i[critical_set])
                beta = estimator.coef_
                X_CV[l] = X_i[critical_set] - beta.dot(Z.T)
        XX = np.hstack([X_CV[l] for l in control_steps])
        half_width_SCV[k] = get_rela_half_width_i(XX)
    return half_width_SCV
