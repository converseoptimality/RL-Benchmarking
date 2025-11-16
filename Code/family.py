import numpy as np
from dataclasses import dataclass

# ---------- small LA helpers ----------
def _sym(A): return 0.5*(A + A.T)

def spd_solve(SPD, B):
    return np.linalg.solve(_sym(SPD), B)

def spd_sqrt(A, eps=1e-12):
    w, V = np.linalg.eigh(_sym(A))
    w = np.clip(w, eps, None)
    return (V * np.sqrt(w)) @ V.T

def spd_invsqrt(A, eps=1e-12):
    w, V = np.linalg.eigh(_sym(A))
    w = np.clip(w, eps, None)
    return (V * (1.0/np.sqrt(w))) @ V.T

# ---------- orthogonal S_p via Givens (per-joint gravity-like term in local coords) ----------
def givens(n, i, j, phi):
    G = np.eye(n)
    c, s = np.cos(phi), np.sin(phi)
    G[i, i] = c; G[j, j] = c
    G[i, j] = -s; G[j, i] = s
    return G

def S_p_matrix(
    s, p, n_links,
    alpha0=0.25, beta0=0.20,
    kappa1=1.0, kappa2=1.0,
    grav0=0.9, swirl_gain=0.5
):
    """
    Orthogonal S_p(s) built as a product of Givens rotations.

    This is the 'earlier' gravity version (local, per-joint):
      alpha_i = swirl_gain * alpha0 * p * tanh(kappa1 * theta_i * omega_i) + grav0 * p * sin(theta_i)
      beta_i  = swirl_gain * beta0  * p * tanh(kappa2 * (theta_{i+1} - theta_i))
    """
    d = 2*n_links
    S = np.eye(d)
    theta = s[0::2]; omega = s[1::2]

    # per-joint rotation in the (theta_i, omega_i) plane (gravity-like + swirl)
    for i in range(n_links):
        alpha = swirl_gain*alpha0*p*np.tanh(kappa1*theta[i]*omega[i]) + grav0*p*np.sin(theta[i])
        S = givens(d, 2*i, 2*i+1, alpha) @ S

    # neighbor coupling in (omega_i, omega_{i+1}) plane (swirl-only)
    for i in range(n_links-1):
        beta = swirl_gain*beta0*p*np.tanh(kappa2*(theta[i+1]-theta[i]))
        S = givens(d, 2*i+1, 2*(i+1)+1, beta) @ S

    return S  # orthogonal

def g_matrix(s, kappa=None):
    n = s.size//2
    if kappa is None: kappa = np.ones(n)
    G = np.zeros((2*n, n))
    th = s[0::2]
    for i in range(n):
        G[2*i+1, i] = kappa[i]*np.cos(th[i])  # torque -> velocity
    return G

@dataclass
class ConverseFamily:
    n: int
    P: np.ndarray      # (2n×2n) SPD
    Q: np.ndarray      # (2n×2n) with Q = tau P
    R: np.ndarray      # (n×n) SPD
    gamma: float
    Sigma: np.ndarray  # (2n×2n) PSD
    sqrt_PmQ: np.ndarray
    sqrt_Sigma: np.ndarray
    c_cost: float
    rng: np.random.Generator

    # S_p knobs
    alpha0: float = 0.25
    beta0: float  = 0.20
    kappa1: float = 1.0
    kappa2: float = 1.0
    grav0:  float = 0.8
    swirl_gain: float = 0.5
    grav_always_on: bool = True  # kept for API compatibility (unused in this S_p version)

    def H_gamma_p(self, s, p):
        G = g_matrix(s)
        S = self.R + self.gamma*(p**2) * (G.T @ self.P @ G)   # SPD
        M = self.P @ G @ spd_solve(S, G.T @ self.P)
        return _sym(self.gamma * (self.P - self.gamma*(p**2)*M))

    def f_p(self, s, p):
        H = self.H_gamma_p(s, p)
        Hm12 = spd_invsqrt(H)
        S = S_p_matrix(s, p, self.n, self.alpha0, self.beta0, self.kappa1, self.kappa2,
                       self.grav0, self.swirl_gain)
        return Hm12 @ (S @ (self.sqrt_PmQ @ s))

    def a_star_p(self, s, p):
        G = g_matrix(s)
        S = self.R + self.gamma*(p**2) * (G.T @ self.P @ G)
        rhs = self.gamma * p * (G.T @ self.P @ self.f_p(s, p))
        return - spd_solve(S, rhs)

    def step(self, s, a, p, stochastic=True):
        mean = self.f_p(s, p) + p * (g_matrix(s) @ a)
        if stochastic and np.any(self.Sigma):
            z = self.rng.normal(size=s.size)
            return mean + self.sqrt_Sigma @ z
        return mean

def build_family(n=6, gamma=0.99, r_u=10.0, tau=0.05,
                 sigma_theta=0.0, sigma_omega=1e-4, seed=7,
                 grav0=0.8, swirl_gain=0.5, grav_always_on=True):
    rng = np.random.default_rng(seed)
    d = 2*n

    # banded P
    M = np.zeros((d, d))
    for i in range(d):
        M[i, i] = 1.0 + 0.5*rng.random()
        if i+1 < d: M[i, i+1] = M[i+1, i] = 0.15*rng.random()
        if i+2 < d: M[i, i+2] = M[i+2, i] = 0.05*rng.random()
    P = _sym(M.T @ M) + 1e-6*np.eye(d)
    Q = tau * P
    R = r_u * np.eye(n)

    Sigma = np.zeros((d, d))
    Sigma[0::2, 0::2] = (sigma_theta**2) * np.eye(n)
    Sigma[1::2, 1::2] = (sigma_omega**2) * np.eye(n)

    sqrt_PmQ = spd_sqrt((1.0 - tau) * P)
    sqrt_Sigma = spd_sqrt(Sigma) if np.any(Sigma) else Sigma
    c_cost = float(gamma/(1.0 - gamma) * np.trace(P @ Sigma))

    fam = ConverseFamily(
        n=n, P=P, Q=Q, R=R, gamma=gamma, Sigma=Sigma,
        sqrt_PmQ=sqrt_PmQ, sqrt_Sigma=sqrt_Sigma, c_cost=c_cost, rng=rng,
        grav0=grav0, swirl_gain=swirl_gain, grav_always_on=grav_always_on
    )
    
    return fam
