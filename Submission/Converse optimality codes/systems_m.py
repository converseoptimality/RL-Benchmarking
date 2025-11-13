import numpy as np
import sympy as sp
from sympy import symbols, Matrix


class GenericODE:
    _system_type = "diff_eqn"

    def __init__(self, 
                 name, 
                 dim_state, 
                 dim_inputs, 
                 dim_observation, 
                 observation_naming, 
                 state_naming, 
                 inputs_naming, 
                 action_bounds, 
                 rhs, 
                 rhs_func):
        
        self._name = name
        self._dim_state = dim_state
        self._dim_inputs = dim_inputs
        self._dim_observation = dim_observation
        self._observation_naming = observation_naming
        self._state_naming = state_naming
        self._inputs_naming = inputs_naming
        self._action_bounds = action_bounds
        self.rhs = rhs
        self.rhs_func=rhs_func

    def compute_state_dynamics(self, state, inputs):
        return self.rhs_func(state.T, inputs)[0]
        #return self.rhs(state, inputs)
    
    
class BenchmarkSysType1(GenericODE):
    def __init__(self, dim_state=3, dim_inputs=1, truncation_depth=5, V=None, xi=None, g=None):
        self.dim_state = dim_state
        self.dim_inputs = dim_inputs
        self.N = truncation_depth

        # SymPy variables
        state_vars = sp.Matrix(sp.symbols(f's1:{self.dim_state + 1}')).T  # [s1, s2, ..., sd]
        action_vars = sp.Matrix(sp.symbols(f'a1:{self.dim_inputs + 1}')).T  # [a1, ..., ad]
        c = sp.symbols(" ".join([f"c{i}" for i in range(self.N + 1)]))


        # Default V(s) if not provided
        if V is None:
            V = state_vars.dot(state_vars)  # ||s||^2

        # Compute gradient of V
        #grad_V = V.jacobian(state_vars)
        grad_V = Matrix([V]).jacobian(state_vars)

        # Default xi(s) if not provided
        if xi is None:
            xi = (grad_V * (1 + 0.5**2 )).T

        # Default g(s) if not provided
        if g is None:
            g = sp.Matrix([
                3 * (1 + 0.5**2) * sp.exp(-((state_vars[i] / 10)**2)) +
                0.2 * sp.Sum(sp.cos(n * sp.Sum(state_vars)), (n, 0, self.N)).doit()
                for i,n in (range(dim_state), range(self.N + 1)) ])


        f = -xi + (1 / 4) * (g @ g.T) @ grad_V.T


        # Define the right-hand side (rhs) of the system
        rhs = f + g @ action_vars.T # transpose it to reach the size of m*1

        #sample vars and substitute
        np.random.seed(4)
        sampled_parameters = np.random.normal(size=3 + self.N)
        c_ = sampled_parameters[2:]
        rhs = rhs.subs(zip(c, c_))

        self._state_vars = state_vars
        self._action_vars = action_vars
        self._rhs_sympy = rhs

        # Convert rhs to a numerical function
        rhs_func = sp.lambdify([state_vars, action_vars], rhs, modules="numpy")

        super().__init__(
            "Type1",
            self.dim_state,
            self.dim_inputs,
            self.dim_state,
            [f"x_{i+1}" for i in range(self.dim_state)],
            [f"x_{i+1}" for i in range(self.dim_state)],
            [f"u_{i+1}" for i in range(self.dim_inputs)],
            [[-1, 1] for _ in range(self.dim_inputs)],
            rhs,
            rhs_func
        )

