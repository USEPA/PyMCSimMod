import jax.numpy as jnp
import numpy as np
import pytest
import torch

from pymcsimmod.model import MathematicalExpression, Number
from pymcsimmod.ode import JaxModel, ScipyModel, TorchModel


@pytest.fixture
def math_expr_add():
    return MathematicalExpression(operator="+", lhs=Number(value=3), rhs=Number(value=4))


@pytest.fixture
def math_expr_nested():
    # (3 + 4) * 2
    inner = MathematicalExpression(operator="+", lhs=Number(value=3), rhs=Number(value=4))
    return MathematicalExpression(operator="*", lhs=inner, rhs=Number(value=2))


@pytest.fixture
def math_expr_pow():
    # pow(3, 2)
    return MathematicalExpression(operator="pow", lhs=Number(value=3), rhs=Number(value=2))


class TestScipyModel:
    def test_model_runs_and_updates(self):
        # Minimal ODE: dA/dt = -k * A
        model_str = """
        States = { A };
        Initialize { A = 10; }
        Dynamics {
            dt(A) = -k * A;
        }
        k = 0.5;
        End.
        """
        model = ScipyModel(model=model_str)
        times = np.linspace(0, 5, 10)
        sol = model.run_model(times)
        # Check ComputedModel interface
        assert sol.states.shape == (10, 1)
        assert sol.times.shape == (10,)
        assert sol.var_names == ["A"]
        # Access by index and name
        np.testing.assert_allclose(sol[0], sol.states[:, 0])
        np.testing.assert_allclose(sol["A"], sol.states[:, 0])
        # Plotting (do not show in CI)
        ax = sol.plot_results()
        assert ax is not None

        # Update parameter and check new solution is different
        model.update_constants(k=1.0)
        sol2 = model.run_model(times)
        assert not np.allclose(sol.states, sol2.states)

        # Update y0 and check new solution is different
        model.update_Y0(A=1.0)
        sol3 = model.run_model(times)
        assert not np.allclose(sol2.states, sol3.states)
        assert model.Y0["A"] == 1.0
        assert model.parameters["k"] == 1.0

    def test_mathematical_expression_eval(self, math_expr_add):
        # Evaluate using ScipyModel's evaluate_expression
        model_str = """
        States = { X };
        Initialize { X = 0; }
        Dynamics { dt(X) = 0; }
        End.
        """
        model = ScipyModel(model=model_str)
        context = {}
        result = model.evaluate_expression(math_expr_add, context)
        assert result == 7

    def test_nested_mathematical_expression_eval(self, math_expr_nested):
        model_str = """
        States = { X };
        Initialize { X = 0; }
        Dynamics { dt(X) = 0; }
        End.
        """
        model = ScipyModel(model=model_str)
        context = {}
        result = model.evaluate_expression(math_expr_nested, context)
        assert result == 14

    def test_pow_mathematical_expression_eval(self, math_expr_pow):
        model_str = """
        States = { X };
        Initialize { X = 0; }
        Dynamics { dt(X) = 0; }
        End.
        """
        model = ScipyModel(model=model_str)
        context = {}
        result = model.evaluate_expression(math_expr_pow, context)
        assert result == 9


class TestJaxModel:
    def test_model_runs_and_updates(self):
        model_str = """
        States = { A };
        Initialize { A = 10; }
        Dynamics {
            dt(A) = -k * A;
        }
        k = 0.5;
        End.
        """
        model = JaxModel(model=model_str)
        times = np.linspace(0, 5, 10)
        sol = model.run_model(times)
        # Check ComputedModel interface
        assert sol.states.shape == (10, 1)
        assert sol.times.shape == (10,)
        assert sol.var_names == ["A"]
        # Access by index and name
        np.testing.assert_allclose(sol[0], sol.states[:, 0])
        np.testing.assert_allclose(sol["A"], sol.states[:, 0])
        # Plotting (do not show in CI)
        ax = sol.plot_results()
        assert ax is not None

        # Update parameter and check new solution is different
        model.update_constants(k=1.0)
        sol2 = model.run_model(times)
        assert not np.allclose(sol.states, sol2.states)

        # Update y0 and check new solution is different
        model.update_Y0(A=1.0)
        sol3 = model.run_model(times)
        assert not np.allclose(sol2.states, sol3.states)
        assert model.Y0["A"] == 1.0
        assert model.parameters["k"] == 1.0

    def test_mathematical_expression_eval(self, math_expr_add):
        model_str = """
        States = { X };
        Initialize { X = 0; }
        Dynamics { dt(X) = 0; }
        End.
        """
        model = JaxModel(model=model_str)
        # For JaxModel, all_vars must be a jnp array, but math_expr_add uses only constants
        all_vars = jnp.array([])
        result = model.evaluate_expression(math_expr_add, all_vars)
        assert result == 7

    def test_nested_mathematical_expression_eval(self, math_expr_nested):
        model_str = """
        States = { X };
        Initialize { X = 0; }
        Dynamics { dt(X) = 0; }
        End.
        """
        model = JaxModel(model=model_str)
        all_vars = jnp.array([])
        result = model.evaluate_expression(math_expr_nested, all_vars)
        assert result == 14

    def test_pow_mathematical_expression_eval(self, math_expr_pow):
        model_str = """
        States = { X };
        Initialize { X = 0; }
        Dynamics { dt(X) = 0; }
        End.
        """
        model = JaxModel(model=model_str)
        all_vars = jnp.array([])
        result = model.evaluate_expression(math_expr_pow, all_vars)
        assert result == 9


class TestTorchModel:
    def test_model_runs_and_updates(self):
        model_str = """
        States = { A };
        Initialize { A = 10; }
        Dynamics {
            dt(A) = -k * A;
        }
        k = 0.5;
        End.
        """
        model = TorchModel(model=model_str)
        model.update_Y0(A=10.0)
        times = torch.linspace(0, 5, steps=10)
        sol = model.run_model(times)
        # Check ComputedModel interface
        assert sol.states.shape == (10, 1)
        assert sol.times.shape == (10,)
        assert sol.var_names == ["A"]
        # Access by index and name
        np.testing.assert_allclose(sol[0], sol.states[:, 0])
        np.testing.assert_allclose(sol["A"], sol.states[:, 0])
        # Plotting (do not show in CI)
        ax = sol.plot_results()
        assert ax is not None

        # Update parameter and check new solution is different
        model.update_constants(k=1.0)
        sol2 = model.run_model(times)
        assert not np.allclose(sol.states, sol2.states)

        # Update y0 and check new solution is different
        model.update_Y0(A=1.0)
        sol3 = model.run_model(times)
        assert not np.allclose(sol2.states, sol3.states)
        assert model.Y0["A"] == 1.0
        assert model.parameters["k"] == 1.0
