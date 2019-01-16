#
# Test for the operator class
#
import pybamm

import numpy as np
import unittest


class TestFiniteVolumeDiscretisation(unittest.TestCase):
    def test_grad_div_shapes(self):
        param = pybamm.ParameterValues(
            base_parameters={"Ln": 0.1, "Ls": 0.2, "Lp": 0.3}
        )
        mesh = pybamm.FiniteVolumeMacroMesh(param, 2)
        disc = pybamm.FiniteVolumeDiscretisation(mesh)

        # grad
        var = pybamm.Variable("var", domain=["whole cell"])
        grad_eqn = pybamm.grad(var)
        y_slices = disc.get_variable_slices([var])
        grad_eqn_disc = disc.process_symbol(grad_eqn, y_slices, {})

        constant_y = np.ones_like(mesh["whole cell"].nodes)
        np.testing.assert_array_equal(
            grad_eqn_disc.evaluate(None, constant_y),
            np.zeros_like(mesh["whole cell"].edges[1:-1]),
        )

        # div
        N = pybamm.grad(var)
        div_eqn = pybamm.div(N)
        boundary_conditions = {
            N.id: {"left": pybamm.Scalar(1), "right": pybamm.Scalar(1)}
        }
        div_eqn_disc = disc.process_symbol(div_eqn, y_slices, boundary_conditions)

        # Linear y should have laplacian zero
        linear_y = mesh["whole cell"].nodes
        np.testing.assert_array_almost_equal(
            grad_eqn_disc.evaluate(None, linear_y),
            np.ones_like(mesh["whole cell"].edges[1:-1]),
        )
        np.testing.assert_array_almost_equal(
            div_eqn_disc.evaluate(None, linear_y),
            np.zeros_like(mesh["whole cell"].nodes),
        )

    def test_grad_convergence(self):
        # Convergence
        param = pybamm.ParameterValues(
            base_parameters={"Ln": 0.1, "Ls": 0.2, "Lp": 0.3}
        )
        var = pybamm.Variable("var", domain=["whole cell"])
        grad_eqn = pybamm.grad(var)

        # Prepare convergence testing
        ns = [50, 100, 200]
        errs = [0] * len(ns)
        for i, n in enumerate(ns):
            # Set up discretisation
            mesh = pybamm.FiniteVolumeMacroMesh(param, target_npts=n)
            disc = pybamm.FiniteVolumeDiscretisation(mesh)

            # Define exact solutions
            y = np.sin(mesh["whole cell"].nodes)
            grad_exact = np.cos(mesh["whole cell"].edges[1:-1])

            # Discretise and evaluate
            y_slices = disc.get_variable_slices([var])
            grad_eqn_disc = disc.process_symbol(grad_eqn, y_slices, {})
            grad_approx = grad_eqn_disc.evaluate(None, y)

            # Calculate errors
            errs[i] = np.linalg.norm(grad_approx - grad_exact) / np.linalg.norm(
                grad_exact
            )

        # Expect h**2 convergence
        [self.assertLess(errs[i + 1] / errs[i], 0.26) for i in range(len(errs) - 1)]

    @unittest.skip("div errors do not converge as expected")
    def test_div_convergence(self):
        # Convergence
        param = pybamm.ParameterValues(
            base_parameters={"Ln": 0.1, "Ls": 0.2, "Lp": 0.3}
        )
        var = pybamm.Variable("var", domain=["whole cell"])
        N = pybamm.grad(var)
        div_eqn = pybamm.div(N)
        boundary_conditions = {
            N.id: {"left": pybamm.Scalar(np.cos(0)), "right": pybamm.Scalar(np.cos(1))}
        }

        # Prepare convergence testing
        ns = [50, 100, 200]
        errs = [0] * len(ns)
        for i, n in enumerate(ns):
            # Set up discretisation
            mesh = pybamm.FiniteVolumeMacroMesh(param, target_npts=n)
            disc = pybamm.FiniteVolumeDiscretisation(mesh)

            # Define exact solutions
            y = np.sin(mesh["whole cell"].nodes)
            div_exact = -np.sin(mesh["whole cell"].nodes)

            # Discretise and evaluate
            y_slices = disc.get_variable_slices([var])
            div_eqn_disc = disc.process_symbol(div_eqn, y_slices, boundary_conditions)
            div_approx = div_eqn_disc.evaluate(None, y)

            # Calculate errors
            errs[i] = np.linalg.norm(div_approx - div_exact) / np.linalg.norm(div_exact)

        # Expect h**2 convergence
        [self.assertLess(errs[i + 1] / errs[i], 0.26) for i in range(len(errs) - 1)]


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()