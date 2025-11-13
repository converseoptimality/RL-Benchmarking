import sympy as sp
import numpy as np
from sympy import Matrix, symbols
import sympy as sp
from IPython.display import display
import os
import numpy as np
import matplotlib
# use TkAgg so plt.show() pops up a window
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from crank_system import N
import pathlib
import os, numpy as np, matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyArrowPatch
from matplotlib import cm

#import matplotlib.patheffects as pe

def ensure_parent_dir(fname: str):
    """Create the parent folder(s) for `fname` if they don’t exist."""
    pathlib.Path(fname).parent.mkdir(parents=True, exist_ok=True)
    


# --------------------------------------------------------------------------- #
# Helper: tip positions                                                       #
# --------------------------------------------------------------------------- #
def crank_tip_positions(traj, L=1.0, D=2.5):
    N   = traj.shape[1] // 2
    th  = traj[:, 0::2]
    xt  = np.arange(N)*D + L*np.cos(th)
    yt  =                  L*np.sin(th)
    return xt, yt


# --------------------------------------------------------------------------- #
# High-resolution summary figure                                              #
# --------------------------------------------------------------------------- #
def visual_description_plot(t, traj, forces, rot, L, D, X_MAX,
                            out_path, title="Motion envelope",
                            n_arrows=12):
    T, N2 = traj.shape
    N     = N2 // 2
    x_tip = np.arange(N)*D + L*np.cos(traj[:, 0::2])
    y_tip =                  L*np.sin(traj[:, 0::2])

    if forces is None:
        forces = np.ones_like(x_tip)

    colours = cm.get_cmap("viridis")(np.linspace(0, 1, T))
    fig, ax = plt.subplots(figsize=(4, 6), dpi=400, facecolor="white")
    ax.set_aspect("equal","box"); ax.axis("off")
    ax.set_xlim(-L-.6, L+.6); ax.set_ylim(-.5, X_MAX+.5)
    ax.set_title(title, fontsize=13, pad=6)
    ax.axvline(0, ls=":", lw=.6, c="k")

    # coloured tip trajectories
    for i in range(N):
        xs, ys = rot(x_tip[:, i], y_tip[:, i])
        ax.scatter(xs, ys, c=colours, s=.8, linewidths=0, zorder=1)

    # start / end configurations
    for idx, col in zip([0, -1], ["tab:red", "tab:green"]):
        for i in range(N):
            bx, by = rot(0,0); by -= i*D
            tx, ty = rot(x_tip[idx,i], y_tip[idx,i])
            ax.plot([bx, tx], [by, ty], lw=2.2,
                    color=col, solid_capstyle="round", zorder=3)
            ax.plot(tx, ty, ".", ms=6, color=col, zorder=4)

    # sparse arrows showing force direction
    arrow_idx = np.linspace(0, T-1, n_arrows, dtype=int)
    r_arm, r_head = 0.8*L, 0.12*0.8*L
    for j in arrow_idx:
        for i in range(N):
            tx, ty = rot(x_tip[j,i], y_tip[j,i])
            end_x  = tx + r_arm             # shaft to right
            sign   = np.sign(forces[j,i]) or 1
            arr_start = (end_x - sign*r_head, ty) if sign>0 else (end_x, ty)
            arr_end   = (end_x, ty)         if sign>0 else (end_x - 2*r_head, ty)
            ax.plot([tx, end_x], [ty, ty], lw=1.0,
                    color="0.55", zorder=5)
            ax.add_patch(FancyArrowPatch(arr_start, arr_end,
                                         arrowstyle="-|>",
                                         mutation_scale=6, lw=0,
                                         color="k", zorder=6))
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)


# --------------------------------------------------------------------------- #
# Main routine:  GIF  +  fading snapshot  +  description figure               #
# --------------------------------------------------------------------------- #
def animate_and_save(
        t, traj, title, out_prefix,
        forces=None,                 # (T,N) signed torque / action
        skip: int = 1,
        static_states: int = 120,
        L: float = 1.0, D: float = 2.5):

    r_arm, r_head = 0.8*L, 0.15*0.8*L
    os.makedirs("plots", exist_ok=True)

    T, N2 = traj.shape
    N     = N2 // 2
    x, y  = crank_tip_positions(traj, L, D)
    if forces is None:
        forces = np.ones_like(x)

    # 90° clockwise rotation
    X_MAX = (N-1)*D + L + .5
    rot   = lambda x0, y0: (y0, X_MAX - x0)

    # indices
    anim_idx  = np.arange(0, T, skip, dtype=int)
    if anim_idx[-1] != T-1: anim_idx = np.append(anim_idx, T-1)
    static_states = max(2, min(static_states, T))
    fade_idx = np.linspace(0, T-1, static_states, dtype=int)

    crank_cols = cm.get_cmap("tab10").colors
    decay_base = 0.96

    # ----------------------------------------------------------------------- #
    # STATIC FADING PNG (no shaft, transparent background)                    #
    # ----------------------------------------------------------------------- #
    fig_s, ax_s = plt.subplots(figsize=(4, 6), dpi=350, facecolor="none")
    ax_s.set_aspect("equal","box"); ax_s.axis("off")
    ax_s.set_xlim(-L-.6, L+.6); ax_s.set_ylim(-.5, X_MAX+.5)
    ax_s.set_title(title, fontsize=14, pad=4)
    ax_s.axvline(0, ls=":", lw=.7, c="k")

    for k, j in enumerate(fade_idx):                # oldest → newest
        alpha = decay_base**(static_states-1-k)
        for i in range(N):
            bx, by = rot(0,0);  by -= i*D
            tx, ty = rot(x[j,i], y[j,i])

            ax_s.plot([bx, tx], [by, ty], lw=1.8,
                      color=crank_cols[i%10], alpha=alpha,
                      solid_capstyle="round", zorder=1)
            ax_s.plot(bx, by, '.', ms=3, color="k",
                      alpha=alpha, zorder=2)
            ax_s.plot(tx, ty, '.', ms=4.5, color="k",
                      alpha=alpha, zorder=3)

    for i in range(N):
        _, yb = rot(0,0)
        # ax_s.text(-L-.35, yb-i*D, f"Crank {i+1}",
        #           ha="right", va="center", fontsize=7)

    static_path = f"plots/{out_prefix}_static.svg"
    fig_s.savefig(static_path, bbox_inches="tight", transparent=True)
    plt.close(fig_s)

    # ----------------------------------------------------------------------- #
    # GIF (shaft fixed right, arrow flips)                                    #
    # ----------------------------------------------------------------------- #
    x_s, y_s, F_s = x[anim_idx], y[anim_idx], forces[anim_idx]
    fig, ax = plt.subplots(figsize=(4, 6), dpi=250)
    ax.set_aspect("equal","box"); ax.axis("off")

    def draw(frame):
        ax.clear(); ax.set_aspect("equal","box"); ax.axis("off")
        ax.set_xlim(-L-.6, L+.6); ax.set_ylim(-.5, X_MAX+.5)
        ax.set_title(title, fontsize=12, pad=4)
        ax.axvline(0, ls=":", lw=.7, c="k")

        for i in range(N):
            bx, by = rot(0,0);     by -= i*D
            tx, ty = rot(x_s[frame,i], y_s[frame,i])

            ax.plot([bx, tx], [by, ty], "o-",
                    lw=3, markersize=6, color=crank_cols[i%10])
            if i < N-1:
                nx, ny = rot(x_s[frame,i+1], y_s[frame,i+1])
                ax.plot([tx, nx], [ty, ny], "r--", lw=1)

            end_x = tx + r_arm
            ax.plot([tx, end_x], [ty, ty], lw=2,
                    color=crank_cols[i%10],
                    solid_capstyle="round")
            sign = np.sign(F_s[frame,i]) or 1
            arr_start, arr_end = ((end_x - r_head, ty), (end_x, ty)) if sign>0 \
                                 else ((end_x, ty), (end_x - r_head, ty))
            ax.add_patch(FancyArrowPatch(arr_start, arr_end,
                                         arrowstyle="-|>", mutation_scale=8,
                                         lw=0, color="k", zorder=10))
        return []

    fps = 50 / max(1, skip)
    ani = FuncAnimation(fig, draw, frames=len(anim_idx),
                        blit=True, interval=1000/fps)
    gif_path = f"plots/{out_prefix}.gif"
    ani.save(gif_path, writer=PillowWriter(fps=fps))
    plt.close(fig)

    # ----------------------------------------------------------------------- #
    # VISUAL DESCRIPTION FIGURE                                               #
    # ----------------------------------------------------------------------- #
    desc_path = f"plots/{out_prefix}_description.png"
    visual_description_plot(t, traj, forces,
                            rot, L, D, X_MAX,
                            out_path=desc_path,
                            title=f"{title} – motion envelope")

    print("Saved:",
          gif_path, ",", static_path, "and", desc_path)



def plot_stage_cost(t, traj, acts, stage_cost, prefix):
    """
    Plot and save the instantaneous running cost c(s,u) over time.
    
    Args:
      t          : 1-D array of times, shape (T,)
      traj       : 2-D array of states, shape (T, dim_state)
      acts       : 2-D array of actions, shape (T, dim_inputs)
      stage_cost : callable (s_vec, a_vec) -> float
      prefix     : filename prefix, e.g. "myrun"
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    os.makedirs("plots", exist_ok=True)
    os.makedirs("plots/data", exist_ok=True)

    # 1) Compute cost at each step
    costs = np.array([stage_cost(s, a) for s, a in zip(traj, acts)],
                     dtype=np.float32)

    # 2) Plot
    fig, ax = plt.subplots()
    ax.plot(t, costs, label="c(s,u)")
    ax.set_xlabel("time")
    ax.set_ylabel("stage cost")
    ax.legend(loc="upper right")
    fig.savefig(f"plots/{prefix}_stage_cost.svg", dpi=350)

    # 3) Save CSV
    data = np.column_stack([t, costs])
    header = "t,c"
    np.savetxt(f"plots/data/{prefix}_stage_cost.csv",
               data, delimiter=",", header=header, comments="")

    # 4) Interactive display
    plt.show()
    input("Press Enter to close the stage‐cost plot…")
    plt.close(fig)


def plot_value(t, V_vals, prefix):
    import os, matplotlib.pyplot as plt, numpy as np

    os.makedirs("plots", exist_ok=True)
    os.makedirs("plots/data", exist_ok=True)

    # 1) line plot
    fig, ax = plt.subplots()
    ax.plot(t, V_vals, label="V(s)")
    ax.set_xlabel("time"); ax.set_ylabel("V(s)")
    ax.legend(loc="upper right")
    fig.savefig(f"plots/{prefix}_value.svg", dpi=350)

    # 2) save CSV
    data = np.column_stack([t, V_vals])
    header = "t,V"
    np.savetxt(f"plots/data/{prefix}_value.csv", data,
               delimiter=",", header=header, comments="")

    # 3) interactive
    plt.show()
    input("Press Enter to close the V(s) plot…")
    plt.close(fig)


def plot_signals(t, traj, acts, prefix):
    # ensure folders exist
    os.makedirs("plots", exist_ok=True)
    os.makedirs("plots/data", exist_ok=True)

    # ─── 1) STATES ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots()
    for i in range(traj.shape[1]):
        ax.plot(t, traj[:, i], label=f"x{i}")
    ax.set_xlabel("time"); ax.set_ylabel("state")
    ax.legend(loc="upper right")
    # save vector graphics
    fig.savefig(f"plots/{prefix}_states.svg", dpi=350)
    # save the data
    data_states = np.column_stack([t, traj])
    header_states = ",".join(["t"] + [f"x{i}" for i in range(traj.shape[1])])
    np.savetxt(f"plots/data/{prefix}_states.csv",
               data_states, delimiter=",",
               header=header_states, comments="")
    # show interactively
    plt.show()
    input("Press Enter to close the state plot…")
    plt.close(fig)

    # ─── 2) ACTIONS ─────────────────────────────────────────────────────────
    if acts is not None:
        fig, ax = plt.subplots()
        for i in range(acts.shape[1]):
            ax.plot(t, acts[:, i], label=f"u{i}")
        ax.set_xlabel("time"); ax.set_ylabel("action")
        ax.legend(loc="upper right")
        fig.savefig(f"plots/{prefix}_actions.svg", dpi=350)
        data_acts = np.column_stack([t, acts])
        header_acts = ",".join(["t"] + [f"u{i}" for i in range(acts.shape[1])])
        np.savetxt(f"plots/data/{prefix}_actions.csv",
                   data_acts, delimiter=",",
                   header=header_acts, comments="")
        plt.show()
        input("Press Enter to close the action plot…")
        plt.close(fig)

def generate_g_matrix(dim_state, dim_inputs, N, Per_mag):
    """Generates a symbolic matrix 'g' with elements depending on state variables."""
    state_vars = Matrix(sp.symbols(f's1:{dim_state + 1}'))
    
    c = np.random.normal(size=dim_state**(N + 1)).reshape(tuple([dim_state] * (N + 1)))
    
    V = state_vars.dot(state_vars)
    xi = (Matrix([V]).jacobian(state_vars) * 1.2).T
    g = sp.Matrix.zeros(dim_state, dim_inputs)
    
    side = np.array([sp.cos(state_var) for state_var in state_vars])

    for i in range(dim_state):
        res = c[i]
        for _ in range(N):
            res = np.tensordot(res, side, axes=1)
        g[i, i] = 3 * (1+0.5**2) * 2**(sum([-(state_var/10)**2 for state_var in state_vars])) + Per_mag * res[None][0]
    
    return g, state_vars, c, V, xi


x, y, u = sp.symbols('x y u')
a, b, d, e, p, q  = sp.symbols('a b d e p q')


def lie(field, F):
    vars = sorted(list(filter(lambda s: 'x' in s.name or 'y' in s.name, F.free_symbols)),
                  key=lambda s: s.name)
    xs, ys = vars[:len(vars)//2], vars[len(vars)//2:]
    vars = [val for pair in zip(xs, ys) for val in pair]
    print(vars)
    return (sp.Matrix([F]).jacobian(vars) @ field)

def fc_from_VGH(V, G, H, u):
    """
    Works for either scalar or multi‑dimensional control input u.
    """
    f = sp.Rational(1, 4) * G @ lie(G, V).T - H          # ← unchanged
    u = sp.Matrix(u)                                     # make sure it's a column
    c = lie(H, V)[0] + (u.T @ u)[0]                      # ‖u‖²
    return f, c.simplify().collect(list(u.free_symbols))

def peq(var, rhss):
    """pretty prints expresions of the kind $A = F(X, Y, Z)$"""
    if not isinstance(var, list):
        var = var.split()
    var_symbs = sp.symbols(" ".join(var))
    if len(var) == 1:
        var_symbs = [var_symbs]
        rhss = [rhss]
    for var_symb, rhs in zip(var_symbs, rhss):
        with sp.evaluate(False):
            display(sp.Eq(var_symb, rhs))


    # assumes we're dealing with "concatenated cranks"
def print_problem(V, f, g, c, u=sp.Matrix([u])):
    peq('V', V)
    if not len(f) == 2:
        peq(" ".join([f'\dot{{x}}_{{{i}}} \dot{{y}}_{{{i}}}' for i in range(len(f) // 2)]), f + g @ u)
    else:
        peq("\dot{x} \dot{y}", f + g @ u)
    peq('c', c)

