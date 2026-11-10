# OCP selection

Plot the `error_least_square` objective using the current sensor configuration
and measured TDOA (including the pipeline's pulse-echo correction):

```bash
python3 -m ocp.plot_error_least_square
python3 -m ocp.plot_error_least_square --no-show --xlim -0.1 0.6 --ylim -0.1 0.5
```

The 3D surface uses `(x, y, F(x,y))`; the adjacent panel shows contours.
The red marker is the LM solver estimate, not a certified global minimum.
Output defaults to `results/error_least_square_3d.png`.

`tdoa.py` and `run.py` accept `--ocp NAME`. The default is
`error_least_square`. `linear_least_squares` always uses
`linear_least_square`, regardless of this option.

Available OCPs are `error_least_square` and `constrained_least_square`.

`constrained_least_square` supports `sqp`, `active_set`, `penalty`, `barrier`,
`augmented_lagrangian`, and `interior_point`. Its state is
`[x, y, r0]`, objective is `0.5 ||A state - b||^2`, and constraints are
`r0^2 = ||position - reference||^2` and `r0 >= 0`. SQP enforces the equivalent
`r0 - ||position - reference|| = 0` equality to avoid a zero constraint
Jacobian at the reference-sensor initial state. Unconstrained solvers
explicitly reject this OCP because they do not enforce constraints.

Example: `python3 tdoa.py --solver sqp --ocp constrained_least_square`.

Nonlinear solvers accept `estimate_position(..., ocp='error_least_square')`.
Providers receive `(position, sensors, delta_r)` with distances in meters.
All nonlinear solvers share the sensor-centroid initial position. A deterministic
small offset is used if it coincides with a sensor. Constrained SQP initializes
r0 to the geometric distance from that position to the reference sensor.

Provider functions:

- `objective(...) -> value`: common evaluation and gradient-descent line search.
- `objective_gradient(...) -> (value, gradient)`: gradient descent and SQP.
- `objective_gradient_hessian(...) -> (value, gradient, exact_hessian)`: Newton.
- `gauss_newton_terms(...) -> (value, gradient, approximate_hessian, jacobian)`:
  Gauss–Newton and LM.

Register a new provider in `OCP_NAMES` after implementing the applicable functions.
Use `SUPPORTED_SOLVERS` to restrict providers that need special solver support.
Implementing constrained objectives also requires explicit constraint support
in the solvers. Selecting the name alone does not enforce any constraints.

The comparison CSV records the solver's `ocp` and the common `evaluation_ocp`.
The legacy CSV column `objective_m2` holds the selected evaluation objective.
It has units m^2 for `error_least_square`, and m^4 for the linearized
`constrained_least_square`; use `evaluation_ocp` when interpreting that column.
