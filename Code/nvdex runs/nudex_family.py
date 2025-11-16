import numpy as np
from dataclasses import dataclass

# ---------- SPD helpers ----------
def _sym(A): return 0.5 * (A + A.T)

def spd_sqrt(A, eps=1e-12):
    w, V = np.linalg.eigh(_sym(A))
    w = np.clip(w, eps, None)
    return (V * np.sqrt(w)) @ V.T

def spd_invsqrt(A, eps=1e-12):
    w, V = np.linalg.eigh(_sym(A))
    w = np.clip(w, eps, None)
    return (V * (1.0/np.sqrt(w))) @ V.T

def spd_solve(SPD, B):
    return np.linalg.solve(_sym(SPD), B)

# ---------- structure helpers ----------
def module_size(r_v: int, r_w: int) -> int:
    return 3 + r_v + r_w  # [x, y, phi] + v-chain + w-chain

def indices_module(i: int, r_v: int, r_w: int):
    base = i * module_size(r_v, r_w)
    return {
        "x": base + 0,
        "y": base + 1,
        "phi": base + 2,
        "v0": base + 3,
        "v_chain": list(range(base + 3, base + 3 + r_v)),
        "w0": base + 3 + r_v,
        "w_chain": list(range(base + 3 + r_v, base + 3 + r_v + r_w)),
    }

def build_G(K: int, r_v: int, r_w: int, tau: float):
    n = K * module_size(r_v, r_w)
    m = 2 * K
    G = np.zeros((n, m), dtype=float)
    for i in range(K):
        idx = indices_module(i, r_v, r_w)
        G[idx["v_chain"][-1], 2*i + 0] = tau
        G[idx["w_chain"][-1], 2*i + 1] = tau
    return G

# ---------- orthogonal field ----------
def apply_S_field(s: np.ndarray, vec: np.ndarray,
                  K: int, r_v: int, r_w: int,
                  alpha: float, kappa: float,
                  alpha_xy: float = 0.30) -> np.ndarray:
    out = vec.copy()
    for i in range(K):
        idx = indices_module(i, r_v, r_w)

        # (x,y) rotation by heading
        phi = float(s[idx["phi"]])
        c, si = np.cos(phi), np.sin(phi)
        X, Y = out[idx["x"]], out[idx["y"]]
        out[idx["x"]] = c * X - si * Y
        out[idx["y"]] = si * X + c * Y

        # (v0, w0) coupling
        v0 = float(s[idx["v0"]]); w0 = float(s[idx["w0"]])
        nu1 = alpha * np.tanh(kappa * v0 * w0)
        c1, s1 = np.cos(nu1), np.sin(nu1)
        V0, W0 = out[idx["v0"]], out[idx["w0"]]
        out[idx["v0"]] = c1 * V0 - s1 * W0
        out[idx["w0"]] = s1 * V0 + c1 * W0

        # bounded cross couplings so action influences pose
        nu2 = alpha_xy * np.tanh(kappa * phi)
        c2, s2 = np.cos(nu2), np.sin(nu2)
        X, V0 = out[idx["x"]], out[idx["v0"]]
        out[idx["x"]] = c2 * X - s2 * V0
        out[idx["v0"]] = s2 * X + c2 * V0

        Y, W0 = out[idx["y"]], out[idx["w0"]]
        out[idx["y"]] = c2 * Y - s2 * W0
        out[idx["w0"]] = s2 * Y + c2 * W0
    return out

# ---------- calibration ----------
def _rho_base(invsqrt_H: np.ndarray, P: np.ndarray) -> float:
    M = invsqrt_H @ spd_sqrt(P)
    return float(np.max(np.abs(np.linalg.eigvals(M))))

def _precompute_metric_parts(P, R, G, gamma, p):
    Pinv = np.linalg.inv(P)
    Rinv = np.linalg.inv(R)
    X = Pinv + gamma * (p**2) * (G @ Rinv @ G.T)           # SPD
    invsqrt_H = (1.0/np.sqrt(gamma)) * spd_sqrt(X)         # H^{-1/2}
    return Pinv, Rinv, invsqrt_H

def _calibrate_R_for_target_rho(P, R, G, gamma, p, rho_target,
                                max_iter=12, scale=0.6):
    R_new = R.copy()
    for _ in range(max_iter):
        Pinv, Rinv, invsqrt_H = _precompute_metric_parts(P, R_new, G, gamma, p)
        rb = _rho_base(invsqrt_H, P)
        if rb > 1.01 * rho_target:
            return R_new, Pinv, Rinv, invsqrt_H, rb
        R_new = np.diag(np.diag(R_new) * scale) + 1e-12 * np.eye(R.shape[0])
    Pinv, Rinv, invsqrt_H = _precompute_metric_parts(P, R_new, G, gamma, p)
    rb = _rho_base(invsqrt_H, P)
    return R_new, Pinv, Rinv, invsqrt_H, rb

def _choose_beta(invsqrt_H, P, rho_target, beta_clip=(1e-6, 0.999)):
    rb = _rho_base(invsqrt_H, P)
    beta = 0.0 if rb <= rho_target else 1.0 - (rho_target / rb)**2
    return float(np.clip(beta, *beta_clip)), rb

# ---------- family ----------
@dataclass
class NUDExFamily:
    # dimensions/knobs
    K: int; r_v: int; r_w: int; tau: float
    gamma: float; p: float
    n: int; m: int
    # QG params
    P: np.ndarray; Q: np.ndarray; R: np.ndarray; Sigma: np.ndarray
    # field
    alpha: float; kappa: float
    # precomputations
    G: np.ndarray; invsqrt_H: np.ndarray
    sqrt_PmQ: np.ndarray; S_mat: np.ndarray; S_chol: np.ndarray
    sqrt_Sigma: np.ndarray; b_value: float
    rng: np.random.Generator

    # back-compat for your callback
    @property
    def c_cost(self):  # b = γ/(1-γ) tr(PΣ)
        return self.b_value

    @property
    def state_dim(self): return self.n
    @property
    def action_dim(self): return self.m

    # dynamics
    def f_p(self, s: np.ndarray) -> np.ndarray:
        t = self.sqrt_PmQ @ s
        t = apply_S_field(s, t, self.K, self.r_v, self.r_w, self.alpha, self.kappa)
        return self.invsqrt_H @ t

    def a_star_p(self, s: np.ndarray) -> np.ndarray:
        rhs = self.G.T @ self.P @ self.f_p(s)
        y = np.linalg.solve(self.S_chol, rhs)
        x = np.linalg.solve(self.S_chol.T, y)
        return -(self.gamma * self.p) * x

    def step(self, s: np.ndarray, a: np.ndarray, p: float, stochastic=True) -> np.ndarray:
        mean = self.f_p(s) + (self.G @ a) # mean = self.f_p(s) + p * (self.G @ a) 
        if stochastic and np.any(self.Sigma):
            z = self.rng.normal(size=self.n)
            return mean + self.sqrt_Sigma @ z
        return mean

def build_family(
    # structure
    K=1, r_v=1, r_w=1, tau=0.01,
    # discount/control
    gamma=0.99, p=1.0,
    # spectra
    weight_pose=2.0, weight_v=1.0, weight_w=1.0,
    r_u_v=0.4, r_u_w=0.4,
    # noise
    sigma_pose=1e-6, sigma_v=1e-3, sigma_w=1e-3,
    # field
    alpha=0.1, kappa=0.4,
    # instability target
    rho_target=1.05, allow_R_rescale=True, R_scale=0.6, max_rescale_iter=10,
    seed=7
) -> NUDExFamily:
    rng = np.random.default_rng(seed)

    n_mod = module_size(r_v, r_w)
    n = K * n_mod
    m = 2 * K

    # geometry
    G = build_G(K, r_v, r_w, tau)

    # P, R
    diag_P, diag_R = [], []
    for _ in range(K):
        diag_P.extend([weight_pose, weight_pose, weight_pose])
        diag_P.extend([weight_v] * r_v)
        diag_P.extend([weight_w] * r_w)
        diag_R.extend([r_u_v, r_u_w])
    P = np.diag(np.array(diag_P, dtype=float)) + 1e-12*np.eye(n)
    R0 = np.diag(np.array(diag_R, dtype=float)) + 1e-12*np.eye(m)

    # metric + R calibration
    if allow_R_rescale:
        R, Pinv, Rinv, invsqrt_H, rb = _calibrate_R_for_target_rho(
            P, R0, G, gamma, p, rho_target, max_iter=max_rescale_iter, scale=R_scale
        )
    else:
        R = R0
        Pinv, Rinv, invsqrt_H = _precompute_metric_parts(P, R, G, gamma, p)
        rb = _rho_base(invsqrt_H, P)

    # choose beta for target rho(A0)
    beta, _ = _choose_beta(invsqrt_H, P, rho_target)
    Q = beta * P

    # Sigma + b
    diag_Sig = []
    for _ in range(K):
        diag_Sig.extend([sigma_pose, sigma_pose, sigma_pose])
        diag_Sig.extend([sigma_v] * r_v)
        diag_Sig.extend([sigma_w] * r_w)
    Sigma = np.diag(np.array(diag_Sig, dtype=float))
    sqrt_Sigma = spd_sqrt(Sigma) if np.any(Sigma) else Sigma
    b_value = float(gamma / (1.0 - gamma) * np.trace(P @ Sigma))

    # sqrt(P-Q)
    PQ = P - Q
    sqrt_PmQ = spd_sqrt(PQ)

    # S = R + γ p^2 G^T P G
    S_mat = R + gamma * (p**2) * (G.T @ P @ G)
    S_chol = np.linalg.cholesky(_sym(S_mat))

    return NUDExFamily(
        K=K, r_v=r_v, r_w=r_w, tau=tau,
        gamma=gamma, p=p, n=n, m=m,
        P=P, Q=Q, R=R, Sigma=Sigma,
        alpha=alpha, kappa=kappa,
        G=G, invsqrt_H=invsqrt_H,
        sqrt_PmQ=sqrt_PmQ, S_mat=S_mat, S_chol=S_chol,
        sqrt_Sigma=sqrt_Sigma, b_value=b_value, rng=rng
    )
