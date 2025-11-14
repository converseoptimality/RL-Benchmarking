# NUDEx / K1 / easy (v1)

**ID:** `NUDEx/K1/easy/v1`  
**Family:** NUDEx  
**Tags:** nonholonomic, unstable, oracle, CRN-ready

## Summary
Single-vehicle NUDEx instance. Provides analytic QG structure (drop in your system builder), difficulty controlled by `rho_target`, and CRN-ready initialization.

## State & Action
- **State (dim=5):** (x, y, phi, v0, w0)
- **Action (dim=2):** (u_v, u_w)

## Dynamics & Cost
- Control-affine QG dynamics with drift `f_p(s)` and quadratic costs `c(s,a)=s^T Q s + a^T R a`.

## Oracle
- Expected analytic oracle: `a_star_p(s, p)` and value `V_star(s)`. Provide artifacts for `P, Q, R` if available.

## Notes
- Add Bellman sanity checks and CRN evaluation hooks in your training script.
