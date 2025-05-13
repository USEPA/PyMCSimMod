import jax.numpy as jnp
import numpy as np
import pytest

from pymcsimmod.model import MathematicalExpression, Number
from pymcsimmod.ode import JaxModel, ScipyModel


@pytest.fixture
def math_expr_add():
    return MathematicalExpression(operator="+", lhs=Number(value=3), rhs=Number(value=4))


@pytest.fixture
def math_expr_nested():
    # (3 + 4) * 2
    inner = MathematicalExpression(operator="+", lhs=Number(value=3), rhs=Number(value=4))
    return MathematicalExpression(operator="*", lhs=inner, rhs=Number(value=2))


class TestScipyModel:
    def test_scipy_model_runs_and_updates(self):
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
        assert sol.y.shape[0] == 1

        # Update parameter and check new solution is different
        model.update_constants(k=1.0)
        sol2 = model.run_model(times)
        assert not np.allclose(sol.y, sol2.y)

        # Update y0 and check new solution is different
        model.update_Y0(A=1.0)
        sol3 = model.run_model(times)
        assert not np.allclose(sol2.y, sol3.y)
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


class TestJaxModel:
    def test_jax_model_runs_and_updates(self):
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
        assert sol.ys.shape[1] == 1

        # Update parameter and check new solution is different
        model.update_constants(k=1.0)
        sol2 = model.run_model(times)
        assert not jnp.allclose(sol.ys, sol2.ys)

        # Update y0 and check new solution is different
        model.update_Y0(A=1.0)
        sol3 = model.run_model(times)
        assert not jnp.allclose(sol2.ys, sol3.ys)
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
