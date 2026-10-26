"""TDOA 비선형 위치 최적화 문제 정의 및 순수 Gauss–Newton 솔버.

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
기준 센서까지의 거리 r_1 = ||p - s_1||_2는 p로 결정되므로
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
구현에서는 해당 지점에서 명시적으로 실패를 보고한다.

미분과 Gauss–Newton 방향
------------------
u_i = (p - s_i) / ||p - s_i||_2
J_i = (u_i - u_1)^T
g = J^T e
H_approx = J^T J

J는 각 잔차의 gradient를 행으로 쌓은 (N-1, 2) 야코비안이다.
정확한 헤시안의 잔차 2차 미분 항을 생략하고 J^T J로 근사한다.
(J^T J) * delta_p = -J^T e를 풀고 p_next = p + delta_p로 갱신한다.
즉 delta_p = -(J^T J)^(-1) * J^T e이다. 이 역행렬 식은 J의 열들이
선형 독립일 때 성립한다. 감쇠나 스텝 크기 조절은 적용하지 않는다.
선형 최소제곱 해를 초기값으로 사용하며 그래디언트 노름으로 수렴을 판정한다.
순수 Gauss–Newton 업데이트는 목적함수 감소나 수렴을 보장하지 않는다.
"""

import numpy as np

from optimization.linear_least_squares import (
    SOUND_SPEED, estimate_position as linear_estimate,
)


def _residual_jacobian(position, sensors, delta_r):
    offsets = position - sensors
    distances = np.linalg.norm(offsets, axis=1)
    if np.any(distances == 0):
        raise ValueError('Gauss-Newton Jacobian is undefined at a sensor position')
    residual = distances[1:] - distances[0] - delta_r
    directions = offsets / distances[:, None]
    jacobian = directions[1:] - directions[0]
    return residual, jacobian


def estimate_position(sensors, tdoa, c=SOUND_SPEED, *,
                      max_iterations=10_000, gradient_tolerance=1e-8):
    """Refine the linear estimate with full-step Gauss-Newton updates.

    Return (position, distance to S1). Raise RuntimeError on failure.
    Gradient convergence does not guarantee a local or global minimum.
    """
    if not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError('max_iterations must be a positive integer')
    if not np.isfinite(gradient_tolerance) or gradient_tolerance <= 0:
        raise ValueError('gradient_tolerance must be positive and finite')
    position, _ = linear_estimate(sensors, tdoa, c)
    sensors = np.asarray(sensors, dtype=float)
    delta_r = c * np.asarray(tdoa, dtype=float)
    for iteration in range(max_iterations + 1):
        residual, jacobian = _residual_jacobian(position, sensors, delta_r)
        value = 0.5 * (residual @ residual)
        gradient = jacobian.T @ residual
        hessian = jacobian.T @ jacobian
        if (not np.isfinite(value) or not np.isfinite(gradient).all()
                or not np.isfinite(hessian).all()):
            raise RuntimeError('Gauss-Newton method encountered non-finite values')
        if np.linalg.norm(gradient) <= gradient_tolerance:
            return position, np.linalg.norm(position - sensors[0])
        if iteration == max_iterations:
            break
        if np.linalg.matrix_rank(jacobian) < 2:
            raise RuntimeError('Gauss-Newton Jacobian is rank deficient')
        try:
            delta_position = np.linalg.solve(hessian, -gradient)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError('Gauss-Newton normal matrix is singular; cannot compute update') from exc
        if not np.isfinite(delta_position).all():
            raise RuntimeError('Gauss-Newton method encountered a non-finite update')
        position = position + delta_position
    raise RuntimeError(f'Gauss-Newton method did not converge in {max_iterations} iterations')
