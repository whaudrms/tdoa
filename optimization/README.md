# Solver history

All `estimate_position` solvers accept `return_history=False` by default.
Enabling it appends a NumPy history array to the usual `(position, r0)` result:

```python
position, r0, history = estimate_position(sensors, tdoa, return_history=True)
```

- Gradient descent, Newton, Gauss–Newton and LM: `(K + 1, 2)` positions,
  including the initial position and every completed update. Gradient-descent
  line-search trial points are excluded.
- SQP: initial state followed by SLSQP major-iteration states. States have
  two coordinates for the unconstrained OCP and three `[x, y, r0]` for the
  constrained OCP. Intermediate states can violate constraints.
- Linear least squares: `(1, 3)` containing the final `[x, y, r0]` only;
  this is a direct solve with no iterative trajectory.
- Active set, penalty, barrier, augmented Lagrangian and interior point:
  `(K + 1, 3)` states. Only the initial state and completed outer iterations
  are recorded; inner optimization, working-set, and line-search trials are excluded.

Gradient descent, Newton and Gauss–Newton also retain `return_iterations`.
When both flags are enabled, the result is
`(position, r0, iterations, history)`. These solvers satisfy
`len(history) == iterations + 1`.

History is returned in memory, not automatically written to disk. Existing
failure behavior is unchanged: failed solves raise rather than returning history.

## Constrained methods

The five additional methods default to `ocp='constrained_least_square'`.
For the CLI select the OCP explicitly:

```bash
python3 tdoa.py --solver augmented_lagrangian --ocp constrained_least_square --no-show
python3 -m visualization.plot_constrained_least_square --solver interior_point
```

All accept `max_iterations` (outer iterations), `tolerance=1e-7`, and
`return_history=False`. The objective is quadratic but the range equality
is nonlinear, so these are local nonlinear constrained methods, not generic
convex QP solvers. No global optimum is certified.

| Method | Outer update | Inner computation |
| --- | --- | --- |
| `active_set` | Accepted SQP state | Active-set linearized QP KKT solve, regularized Lagrangian Hessian, L1 merit line search |
| `penalty` | Multiply penalty by 10 | Analytic-Jacobian least squares on objective and equality/negative-range penalty residuals |
| `augmented_lagrangian` | Update equality multiplier; increase penalty if feasibility stalls | Bounded least squares on shifted equality residual |
| `barrier` | Multiply log-barrier parameter by 0.1 | Equality-constrained SLSQP on f - mu log(r0) |
| `interior_point` | Multiply central-path parameter by 0.1 | Bounded least-squares solution of primal-dual KKT equations, with positive range and inequality multiplier |

The barrier method intentionally uses SLSQP as its inner equality solver;
the interior-point method solves central-path KKT residuals with SciPy
`least_squares` (numerical Jacobian), rather than a hand-written predictor-corrector.
Penalty and augmented-Lagrangian inner solves also use SciPy `least_squares`.
Success requires feasibility and first-order stationarity within tolerance;
interior point also checks complementarity using its explicit multiplier.
Continuation parameters are currently fixed in the corresponding modules.

The range equality is nonsmooth at the cone apex (position equals the reference).
The non-barrier methods use a zero norm subgradient there; their smooth KKT
checks do not certify nonsmooth apex optima. Barrier/interior point approach
the apex only in the continuation limit. These implementations are intended
for smooth positive-range solutions. Position limits in plots are display
bounds, not optimization bounds.
