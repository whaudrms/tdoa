# OCP selection

`tdoa.py` and `run.py` accept `--ocp NAME`. The default is
`error_least_square`. `linear_least_squares` always uses
`linear_least_square`, regardless of this option.

`noise_least_square` is a registered placeholder and raises
`NotImplementedError` until implemented.

`constrained_least_square` is implemented for SQP only. Its state is
`[x, y, r0]`, objective is `0.5 ||A state - b||^2`, and constraints are
`r0^2 = ||position - reference||^2` and `r0 >= 0`. SQP enforces the equivalent
`r0 - ||position - reference|| = 0` equality to avoid a zero constraint
Jacobian at the reference-sensor initial state. Other nonlinear solvers
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

Set `IMPLEMENTED = True` only after the applicable functions are implemented.
Use `SUPPORTED_SOLVERS` to restrict providers that need special solver support.
Implementing constrained objectives also requires explicit constraint support
in the solvers. Selecting the name alone does not enforce any constraints.

The comparison CSV records the solver's `ocp` and the common `evaluation_ocp`.
The legacy CSV column `objective_m2` holds the selected evaluation objective.
It has units m^2 for `error_least_square`, and m^4 for the linearized
`constrained_least_square`; use `evaluation_ocp` when interpreting that column.
