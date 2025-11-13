####################################################################################
import sympy as sp
from IPython.display import display
import numpy as np
#from tqdm import tqdm

# fix the seed
#np.random.seed(5)
np.random.seed(1)

x, y, u = sp.symbols('x y u')
a, b, d, e, p, q  = sp.symbols('a b d e p q')


def lie(field, F):
    vars = sorted(list(filter(lambda s: 'x' in s.name or 'y' in s.name, F.free_symbols)),
                  key=lambda s: s.name)
    xs, ys = vars[:len(vars)//2], vars[len(vars)//2:]
    vars = [val for pair in zip(xs, ys) for val in pair]
    print(vars)
    return (sp.Matrix([F]).jacobian(vars) @ field)

def fc_from_VGH(V, G, H, u=sp.Matrix([u])):
    f = sp.Rational(1, 4) * G @ lie(G, V).T - H
    c = lie(H, V)[0] + (u.T @ u)[0]
    return f, c.simplify().collect([y])
####################################################################################


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




def not_(relations):
    return list(map(lambda s: ~s, relations))


def remove_control(expr): ## assuming SISO
    return expr.subs(u, 0)






M, R = sp.symbols('M R')
a, b, d = sp.symbols('a b d')


V = a * x ** 2 + 2 * b * x * y +  d * y ** 2
G = sp.Matrix([0, sp.cos(x)])
H = sp.Matrix([-y,  M*x + R * y + (sp.Rational(1, 4) * G @ G.T @ sp.Matrix([V]).jacobian([x, y]).T)[1]])

print(sp.Matrix([V]).jacobian([x, y]).T)

peq('H', H)

f, c = fc_from_VGH(V, G, H)

print_problem(V, f, G, c)


c_stripped = c.subs(u, 0).subs(sp.cos(x), 0).expand().collect([x, y])
peq("c_{stripped}", c_stripped)

m11 = c_stripped.coeff(x, 2)
m12 = c_stripped.coeff(x, 1).diff(y) / 2
m21 = m12
m22 = c_stripped.coeff(x, 0).diff(y).diff(y) / 2

criteria = [m11 > 0, m11 * m22 - m21 * m12 > 0,
            a > 0, a * d - b ** 2 > 0]

print(criteria)

big_rel = (m11 * m22 - m21 * m12).expand().collect(d)
C = big_rel.coeff(d, 0)
B = big_rel.coeff(d, 1)
A = big_rel.coeff(d, 2)

D = (B**2 - 4 * A * C).simplify()

peq('D', D)

D_root = sp.symbols('D_root')


better_criteria = [M > 0, R > 0, b > 0, a > M * b / R,
                   d > (-B + D_root) / (2 * A),
                   d < (-B - D_root) / (2 * A),
                   sp.Eq(D_root * D_root, D)]
print(better_criteria)



bottom_d = sp.lambdify([a, b, M, R], (-B + sp.sqrt(D)) / (2 * A))
length_d = sp.lambdify([a, b, M, R], -sp.sqrt(D) / A)


############################################################################## down

N = 4

def symb_array(names, count=N):
    res = []
    for name in names.split():
        res.append(sp.symbols(" ".join([name + str(i) for i in range(count)])))
    return res

as_, bs, ds, Ms, Rs, Ks = symb_array('a b d M R K')
Ks = Ks[:-1]
xs, ys, us = symb_array('x y u')

state = [s for x_, y_ in zip(xs, ys) for s in (x_, y_)]

M, R = sp.symbols('M R')
a, b, d = sp.symbols('a b d')
K = sp.symbols('K')

V_n = a * x ** 2 + 2 * b * x * y +  d * y ** 2
G_n = sp.Matrix([0, sp.cos(x)])
H_n = sp.Matrix([-y,  M*x + R * y + (sp.Rational(1, 4) * G_n @ G_n.T @ sp.Matrix([V_n]).jacobian([x, y]).T)[1]])




V = \
  sum([V_n.subs([(x, xs[i]), (y, ys[i]), (a, as_[i]), (b, bs[i]), (d, ds[i])])
       for i in range(N)])
G = sp.diag(*[G_n.subs(x, xs[i]) for i in range(N)])
H = sp.Matrix.vstack(*[H_n.subs([(x, xs[i]), (y, ys[i]), (a, as_[i]),
                                 (b, bs[i]), (d, ds[i]), (M, Ms[i]), (R, Rs[i])]) \
                     for i in range(N)])
pad = sp.symbols('pad')
for i in range(N):
    padded_xs = list(xs) + [pad]
    padded_Ks = list(Ks) + [pad]
    H[2 * i + 1] += padded_Ks[i - 1] * (padded_xs[i] - padded_xs[i - 1]) # enacted by previous crank
    H[2 * i + 1] += padded_Ks[i] * (padded_xs[i] - padded_xs[i + 1]) # enacted by next crank
H = H.subs(pad, 0)

import pickle
import os

# ─── Automatically compute or load cached (f, c) ─────────────────────────
cache_path = "fc_converse.pkl"
if os.path.exists(cache_path):
    with open(cache_path, "rb") as pf:
        f, c = pickle.load(pf)
else:
    # compute for the first time
    f, c = fc_from_VGH(V, G, H, u=sp.Matrix(us))
    # save to disk for future runs
    with open(cache_path, "wb") as pf:
        pickle.dump((f, c), pf)

# now f and c are available (either loaded or freshly computed)

print_problem(V, f, G, c, u=sp.Matrix(us))

c_stripped = c
for i in range(N):
    c_stripped = c_stripped.subs(us[i], 0).subs(sp.cos(xs[i]), 0)

Q = sp.Matrix([c_stripped]).jacobian(state).T.jacobian(state) / 2

Q_no_springs = Q.subs(list(zip(Ks, [0] * (N - 1))))
DeltaQ = Q - Q_no_springs


peq('c_{stripped|same|K}', c_stripped.subs(list(zip(Ks, [K] * (N - 1)))).expand().collect(K))

peq("\Delta{Q}", DeltaQ)


# making a function that transforms the unit ball into a region of permitted ellastic moduli (Ks)
# after generating the systems, this is how we link them
############################################################################## down

DeltaQ_vectorized = DeltaQ.reshape(len(DeltaQ), 1)
DeltaQ_F2 = DeltaQ_vectorized.dot(DeltaQ_vectorized)
peq("||\Delta{Q}||_F^2", DeltaQ_F2.expand().collect(Ks[0]))
P = sp.Matrix([DeltaQ_F2]).jacobian(Ks).T.jacobian(Ks) / 2
peq("P", P)
L, D = ((P + P.T)/2).LDLdecomposition(hermitian=False)
peq("LDL", (L, D))




print("The linear part is zero:")
peq("F", sp.Matrix([DeltaQ_F2]).jacobian(Ks).subs(list(zip(Ks, [0] * (N - 1)))))


# "different Ks can be generated"

## Let's compute the inverse
peq("Q_{initial}", Q_no_springs)
Q_no_springs_inv = sp.diag(*[Q_no_springs[2 * i: 2 * i + 2, 2 * i: 2 * i + 2].inv() for i in range(N)])
peq("Q_{initial|inverse}", Q_no_springs_inv)

Q_no_springs_inv_vec = Q_no_springs_inv.reshape(len(DeltaQ), 1)


rhs = 1 / Q_no_springs_inv_vec.dot(Q_no_springs_inv_vec)


# transform from unit ball that yields valid Ks
transform = L.T.inv() @ D.inv(method="LU").pow(sp.Rational(1, 2)) * sp.sqrt(rhs)


# For same K
K = sp.symbols('K')

peq("||\Delta{Q}||_F^2", DeltaQ_F2.subs(list(zip(Ks, [K] * (N - 1)))).expand().collect(K))

ub_single = sp.sqrt(rhs / (DeltaQ_F2.subs(list(zip(Ks, [K] * (N - 1)))).diff(K).diff(K) / 2))

display(K < ub_single)

# generating parameters for a single crank, used in the multiple cranks function
############################################################################## down


transform_matrix = sp.lambdify(as_ + bs + ds + Ms + Rs, transform)




def generate_abdMR(size, range_=10):
    res = np.random.random(size=size * 5).reshape(size, 5) * range_
    for i in range(size):
        res[i, -1] /= 10
        res[i, -1] += 0.25
        res[i, -2] /= 16
        res[i, -2] += 30
        _, b, _, M, R = res[i]

        res[i, 0] += M * b / R,
        a = res[i, 0]

        res[i, 2] *= length_d(a, b, M, R) / range_
        res[i, 2] += bottom_d(a, b, M, R)
    return res

# sampling from a unit ball

def sample_unit_ball(dim=N-1):
    while True:
        res = np.random.random(size=dim) * 2 - 1
        if res @ res < 1:
            return res


# for generating parameters for
############################################################################## down
def gen_params(size, range_=30):
    samples = []
    for i in range(size):
        subs = {}
        params = generate_abdMR(N, range_=range_)
        for i, param in enumerate(params):
            a, b, d, M, R = param
            subs.update({as_[i].name: a, bs[i].name: b, ds[i].name: d, Ms[i].name: M, Rs[i].name: R})
        transform_ = transform_matrix(**subs)
        p = sample_unit_ball()
        subs.update(dict(zip(Ks, (transform_ @ p).tolist())))
        samples.append(subs)
    return samples


num_tests = 1
params = gen_params(num_tests)
print(params)