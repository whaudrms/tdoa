"""Constrained solver integration and outer-history regression checks."""
import unittest
import numpy as np
from tdoa import SOLVERS, SENSORS
from ocp.constrained_least_square import build_system
from optimization.constrained_solver._common import optimality
from optimization.unconstrained_solver.linear_least_squares import initial_position

NAMES = ('active_set', 'penalty', 'barrier', 'augmented_lagrangian', 'interior_point')


class ConstrainedSolvers(unittest.TestCase):
    def measurement(self, truth):
        ranges = np.linalg.norm(np.asarray(truth) - SENSORS, axis=1)
        return (ranges[1:] - ranges[0]) / 343.

    def test_exact_measurements_and_outer_history(self):
        for truth in ([.4, .2], [.1, .1], [.3, .3]):
            tdoa = self.measurement(truth)
            a, b = build_system(SENSORS, tdoa, 343.)
            initial, r0 = initial_position(SENSORS, tdoa)
            for name in NAMES:
                with self.subTest(truth=truth, solver=name):
                    solver = SOLVERS[name]
                    p, r, history = solver(SENSORS, tdoa, return_history=True)
                    self.assertEqual(history.shape[1], 3)
                    np.testing.assert_allclose(history[0], np.r_[initial, r0])
                    np.testing.assert_allclose(history[-1], np.r_[p, r])
                    self.assertTrue(np.isfinite(history).all())
                    np.testing.assert_allclose(p, truth, atol=5e-6, rtol=0)
                    feasibility, stationarity = optimality(np.r_[p, r], SENSORS, a, b)
                    self.assertLessEqual(feasibility, 1e-7)
                    self.assertLessEqual(stationarity, 1e-7)
                    plain = solver(SENSORS, tdoa)
                    self.assertEqual(len(plain), 2)
                    np.testing.assert_allclose(plain[0], p)

    def test_one_outer_iteration_limit(self):
        tdoa = self.measurement([.4, .2])
        # Tight tolerance prevents one outer solve from satisfying continuation criteria.
        for name in ('barrier', 'interior_point', 'active_set'):
            with self.subTest(solver=name):
                with self.assertRaises(RuntimeError):
                    SOLVERS[name](SENSORS, tdoa, max_iterations=1, tolerance=1e-12)

    def test_noisy_measurements_against_sqp(self):
        tdoa = self.measurement([.4, .2]) + np.array([2., -1., 1.5]) * 1e-6
        a, b = build_system(SENSORS, tdoa, 343.)
        p, r = SOLVERS['sqp'](SENSORS, tdoa, ocp='constrained_least_square')
        reference = 0.5 * np.linalg.norm(a @ np.r_[p, r] - b)**2
        for name in NAMES:
            with self.subTest(solver=name):
                p, r, history = SOLVERS[name](SENSORS, tdoa, return_history=True)
                feasibility, stationarity = optimality(np.r_[p, r], SENSORS, a, b)
                self.assertLessEqual(feasibility, 1e-7)
                self.assertLessEqual(stationarity, 1e-7)
                cost = 0.5 * np.linalg.norm(a @ np.r_[p, r] - b)**2
                self.assertLess(abs(cost - reference), 1e-9)
                if name in ('barrier', 'interior_point'):
                    self.assertTrue((history[:, 2] > 0).all())

    def test_wrong_ocp_and_invalid_controls(self):
        tdoa = self.measurement([.4, .2])
        for name in NAMES:
            with self.subTest(solver=name):
                for kwargs in ({'ocp': 'error_least_square'}, {'max_iterations': 0},
                               {'max_iterations': True}, {'tolerance': float('nan')}):
                    with self.assertRaises(ValueError):
                        SOLVERS[name](SENSORS, tdoa, **kwargs)


if __name__ == '__main__':
    unittest.main()
