"""TDOA 비선형 위치 최적화 문제 정의 및 gradient descent 솔버.

입력
----
s_i = (x_i, y_i): i번째 센서의 알려진 위치 [m], i = 1, ..., N.
S1을 기준 센서로 사용한다.
delta_t_i = t_i - t_1: 측정 TDOA [s], i = 2, ..., N.
c > 0: 알려진 음속 [m/s].
코드의 입력 배열에서는 sensors[0]이 s_1이고,
tdoa[i - 2]가 delta_t_i에 해당한다.

최적화 변수
-----------
p = (x, y) in R^2: 추정할 음원의 위치 [m].
기준 센서까지의 거리 r0 = ||p - sensors[0]||_2는 p로 결정되므로
별도의 독립 변수로 두지 않는다.

측정 모델과 잔차
----------------
예측 거리차:
    h_i(p) = ||p - s_i||_2 - ||p - s_1||_2

측정 거리차:
    delta_r_i = c * delta_t_i

거리차 잔차 [m]:
    e_i(p) = h_i(p) - delta_r_i
           = ||p - s_i||_2 - ||p - s_1||_2 - c * delta_t_i

목적함수와 최적화 문제
---------------------
    p* in argmin_{p in R^2} F(p)

    F(p) = (1/2) * sum_{i=2}^{N} e_i(p)^2
         = (1/2) * ||e(p)||_2^2

모든 잔차에 동일한 가중치를 사용하는 비선형 최소제곱 문제다.
상수 1/2은 미분 표현을 간단하게 하며 최소점에는 영향을 주지 않는다.
추가적인 위치 영역 제약은 두지 않는다.

성질
----
거리의 비선형성 때문에 일반적으로 비볼록이며, 해의 유일성이나
전역 최적해 도달은 보장되지 않는다. 목적함수는 센서 위치에서도
정의되지만 거리 노름은 p = s_i에서 미분 불가능하다.
구현에서는 공통 목적함수의 1 nm 거리 평활화로 해당 지점의 미분을 정의한다.

센서 중심점을 공통 초기값으로 사용하며 Armijo backtracking으로
스텝 크기를 정한다. 그래디언트 노름으로 수렴을 판정한다.
"""

import numpy as np
from ocp.registry import DEFAULT_OCP, get_ocp

from optimization.linear_least_squares import (
    SOUND_SPEED, initial_position,
)




def estimate_position(sensors, tdoa, c=SOUND_SPEED, *, ocp=DEFAULT_OCP,
                      max_iterations=10_000, gradient_tolerance=1e-8):
    """Refine the sensor-centroid initial position; return (position, distance to the reference sensor).

    Raises RuntimeError on non-convergence rather than returning an
    unconverged iterate as a successful estimate. A stationary point need
    not be a global minimum of this nonconvex objective.
    """
    problem = get_ocp(ocp, solver='gradient_descent')
    if not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError('max_iterations must be a positive integer')
    if not np.isfinite(gradient_tolerance) or gradient_tolerance <= 0:
        raise ValueError('gradient_tolerance must be positive and finite')
    position, _ = initial_position(sensors, tdoa, c)
    sensors = np.asarray(sensors, dtype=float)
    delta_r = c * np.asarray(tdoa, dtype=float)
    for iteration in range(max_iterations + 1):
        value, gradient = problem.objective_gradient(position, sensors, delta_r)
        if not np.isfinite(value) or not np.isfinite(gradient).all():
            raise RuntimeError('Gradient descent encountered non-finite values')
        gradient_norm = np.linalg.norm(gradient)
        if gradient_norm <= gradient_tolerance:
            return position, np.linalg.norm(position - sensors[0])
        if iteration == max_iterations:
            break
        step = 1.0
        for _ in range(60):
            candidate = position - step * gradient
            distances = np.linalg.norm(candidate - sensors, axis=1)
            candidate_value = problem.objective(candidate, sensors, delta_r)
            if (np.all(distances > 0) and np.isfinite(candidate_value)
                    and candidate_value <= value - 1e-4 * step * gradient_norm**2):
                position = candidate
                break
            step *= 0.5
        else:
            raise RuntimeError('Gradient descent line search failed')
    raise RuntimeError(f'Gradient descent did not converge in {max_iterations} iterations')
