"""Nonlinear active-set SQP with regularized Lagrangian Hessian.

The range lower bound enters/leaves the QP working set. History records
accepted outer SQP steps, excluding working-set and line-search trials.
"""
import numpy as np
from ._common import SOUND_SPEED, prepare, geometry, optimality, finish


def estimate_position(sensors, tdoa, c=SOUND_SPEED, *, ocp='constrained_least_square',
                      max_iterations=500, tolerance=1e-7, return_history=False):
    z, sensors, a, b = prepare(sensors, tdoa, c, ocp, 'active_set', max_iterations, tolerance)
    history = [z.copy()] if return_history else None
    lam = 0.
    for _ in range(max_iterations):
        feasibility, stationarity = optimality(z, sensors, a, b)
        if feasibility <= tolerance and stationarity <= tolerance:
            return finish(z, history, return_history)
        h, j = geometry(z, sensors)
        g = a.T @ (a @ z - b)
        H = a.T @ a
        offset = z[:2] - sensors[0]
        r = np.linalg.norm(offset)
        if r > 1e-12:
            u = offset / r
            H[:2, :2] -= lam * (np.eye(2) - np.outer(u, u)) / r
        H += max(0., 1e-6 - np.linalg.eigvalsh(H).min()) * np.eye(3)
        active = False
        for _working in range(4):
            C = np.vstack((j, [0., 0., -1.])) if active else j[None, :]
            rhs = np.r_[-h, z[2]] if active else np.array([-h])
            kkt = np.block([[H, C.T], [C, np.zeros((len(C), len(C)))]])
            solution = np.linalg.lstsq(kkt, np.r_[-g, rhs], rcond=None)[0]
            d, multipliers = solution[:3], solution[3:]
            if not active and z[2] + d[2] < 0:
                active = True
                continue
            if active and multipliers[-1] < 0:
                active = False
                continue
            break
        else:
            raise RuntimeError('Active-set working set failed to stabilize')
        lam = multipliers[0]
        penalty = max(1., 2 * np.max(np.abs(multipliers)))
        e = a @ z - b
        merit = 0.5 * (e @ e) + penalty * abs(h)
        slope = g @ d - penalty * abs(h)
        step = 1.
        for _line in range(60):
            trial = z + step * d
            et = a @ trial - b
            mt = 0.5 * (et @ et) + penalty * abs(geometry(trial, sensors)[0])
            if trial[2] >= -1e-14 and mt <= merit + 1e-4 * step * slope:
                z = trial
                break
            step *= 0.5
        else:
            raise RuntimeError('Active-set merit line search failed')
        if return_history:
            history.append(z.copy())
    raise RuntimeError('Active-set method did not converge')
