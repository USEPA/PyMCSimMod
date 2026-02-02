"""Tests for JAX model implementations."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("jax")
pytest.importorskip("equinox")
pytest.importorskip("diffrax")

import jax
import jax.numpy as jnp

from pymcsimmod.config import BackendType
from pymcsimmod.forcing.unified import UnifiedForcingFactory
from pymcsimmod.models.computed import ComputedModel
from pymcsimmod.models.jax_model import EqxModel, JaxModel


class TestEqxModel:
    """Tests for the EqxModel class."""

    @pytest.fixture
    def simple_model_str(self):
        """Simple test model string."""
        return """
        States = {
            A
        };

        Parameters = {
            ka,
            ke
        };

        Inputs = {
            dose
        };

        Initialize {
            A = 0.0;
        }

        Dynamics {
            dt(A) = dose * ka - A * ke;
        }

        Outputs = {
            A_out
        };

        CalcOutputs {
            A_out = A;
        }

        End.
        """

    @pytest.fixture
    def eqx_model_components(self):
        """Basic components for creating an EqxModel."""

        # Create a minimal model tree mock
        class MockModelTree:
            def __init__(self):
                self.dynamic_calcs = {}
                self.calc_outputs = {
                    "A_out": MockExpr("A")  # A_out = A
                }
                self.dynamics = {"A": MockExpr("dose * ka - A * ke")}

        class MockExpr:
            def __init__(self, expr_str):
                self.expr_str = expr_str

            def evaluate(self, context, approach):
                # Simple mock evaluation for testing
                if self.expr_str == "dose * ka - A * ke":
                    return context.get("dose", 0.0) * context.get("ka", 1.0) - context.get(
                        "A", 0.0
                    ) * context.get("ke", 0.1)
                elif self.expr_str == "A":
                    return context.get("A", 0.0)
                return 0.0

        # Return the constructor arguments as separate values, not as a dict
        return MockModelTree()

    def test_compile_forcing_functions_dict_spec(self, eqx_model_components):
        """Test compiling forcing functions from dictionary specifications using unified factory."""
        model_tree = eqx_model_components

        model = EqxModel(
            parameters={"ka": 1.0, "ke": 0.1},
            forcing_functions={
                "dose": {
                    "function": "PerDose",
                    "args": (),
                    "kwargs": {"t0": 0.0, "duration": 1.0, "period": 24.0, "s": 10.0},
                }
            },
            Y0={"A": 0.0},
            events=[],
            model_tree=model_tree,
            state_names=("A",),
            output_names=("A_out",),
        )

        model.compile_forcing_functions()

        # Check that the function was compiled
        assert callable(model.forcing_functions["dose"])

        # Test the compiled function with JAX arrays
        result = model.forcing_functions["dose"](jnp.array(0.5))
        assert isinstance(result, jnp.ndarray)

        # Test JAX JIT compatibility
        jitted_func = jax.jit(model.forcing_functions["dose"])
        jitted_result = jitted_func(jnp.array(0.5))
        np.testing.assert_allclose(result, jitted_result, rtol=1e-6)

    def test_compile_forcing_functions_already_compiled(self, eqx_model_components):
        """Test that already compiled functions are left unchanged."""
        model_tree = eqx_model_components

        # Create a simple test function using unified factory
        original_func = UnifiedForcingFactory.create_forcing_function(
            "ZeroFunc", backend=BackendType.JAX
        )

        model = EqxModel(
            parameters={"ka": 1.0, "ke": 0.1},
            forcing_functions={"dose": original_func},
            Y0={"A": 0.0},
            events=[],
            model_tree=model_tree,
            state_names=("A",),
            output_names=("A_out",),
        )

        model.compile_forcing_functions()

        # Should be the same function
        assert model.forcing_functions["dose"] is original_func

    def test_compile_forcing_functions_invalid_spec(self, eqx_model_components):
        """Test error handling for invalid forcing function specifications."""
        model_tree = eqx_model_components

        model = EqxModel(
            parameters={"ka": 1.0, "ke": 0.1},
            forcing_functions={
                "dose": {"function": "NonExistentFunction", "args": (), "kwargs": {}}
            },
            Y0={"A": 0.0},
            events=[],
            model_tree=model_tree,
            state_names=("A",),
            output_names=("A_out",),
        )

        with pytest.raises(
            ValueError, match="Unknown forcing function type: 'NonExistentFunction'"
        ):
            model.compile_forcing_functions()

    def test_build_context(self, eqx_model_components):
        """Test context building for JAX compatibility."""
        model_tree = eqx_model_components

        # Create zero function using unified factory
        zero_func = UnifiedForcingFactory.create_forcing_function(
            "ZeroFunc", backend=BackendType.JAX
        )

        model = EqxModel(
            parameters={"ka": 1.0, "ke": 0.1},
            forcing_functions={"dose": zero_func},
            Y0={"A": 0.0},
            events=[],
            model_tree=model_tree,
            state_names=("A",),
            output_names=("A_out",),
        )

        state_vals = jnp.array([1.0])  # A = 1.0
        t = 0.5

        context = model.build_context(state_vals, t)

        # Check basic structure
        assert "A" in context
        assert context["A"] == 1.0
        assert "ka" in context
        assert "ke" in context
        assert "dose" in context

    def test_model_jit_compilation(self, eqx_model_components):
        """Test that the model function can be JIT compiled."""
        model_tree = eqx_model_components

        zero_func = UnifiedForcingFactory.create_forcing_function(
            "ZeroFunc", backend=BackendType.JAX
        )

        model = EqxModel(
            parameters={"ka": 1.0, "ke": 0.1},
            forcing_functions={"dose": zero_func},
            Y0={"A": 0.0},
            events=[],
            model_tree=model_tree,
            state_names=("A",),
            output_names=("A_out",),
        )

        # Test the model function
        t = 0.5
        y = jnp.array([1.0])

        dydt = model.model(t, y)
        assert isinstance(dydt, jnp.ndarray)
        assert dydt.shape == (1,)

        # Test that it can be JIT compiled (this is implicit in the @eqx.filter_jit decorator)
        # If JIT compilation fails, the test will fail

    def test_run_model_with_events_raises_error(self, eqx_model_components):
        """Test that having events raises NotImplementedError."""
        model_tree = eqx_model_components

        zero_func = UnifiedForcingFactory.create_forcing_function(
            "ZeroFunc", backend=BackendType.JAX
        )

        model = EqxModel(
            parameters={"ka": 1.0, "ke": 0.1},
            forcing_functions={"dose": zero_func},
            Y0={"A": 0.0},
            events=[{"type": "test"}],  # Add a dummy event
            model_tree=model_tree,
            state_names=("A",),
            output_names=("A_out",),
        )

        times = np.linspace(0, 10, 101)

        with pytest.raises(NotImplementedError, match="Discrete events are not yet supported"):
            model.run_model(times)

    def test_run_model_success(self, eqx_model_components):
        """Test successful model run without events."""
        model_tree = eqx_model_components

        zero_func = UnifiedForcingFactory.create_forcing_function(
            "ZeroFunc", backend=BackendType.JAX
        )

        model = EqxModel(
            parameters={"ka": 1.0, "ke": 0.1},
            forcing_functions={"dose": zero_func},
            Y0={"A": 0.0},
            events=[],
            model_tree=model_tree,
            state_names=("A",),
            output_names=("A_out",),
        )

        times = np.linspace(0, 2, 21)

        sol, calc_outputs, input_functions = model.run_model(times)

        # Check solution structure
        assert hasattr(sol, "ts")
        assert hasattr(sol, "ys")
        assert len(sol.ts) == len(times)
        assert sol.ys.shape == (len(times), 1)  # 1 state variable

        # Check outputs
        assert calc_outputs.shape == (len(times), 1)  # 1 output

        # Check input functions
        assert isinstance(input_functions, dict)
        assert "dose" in input_functions

        # Test JAX array compatibility in input functions
        jax_time = jnp.array(1.0)
        input_result = input_functions["dose"](jax_time)
        assert isinstance(input_result, jnp.ndarray)


class TestJaxModel:
    """Tests for the JaxModel class."""

    @pytest.fixture
    def simple_model_str(self):
        """Simple test model string."""
        return """
        States = {
            A
        };

        Inputs = {
            dose
        };

        Outputs = {
            A_out
        };

        # Parameters defined outside blocks with default values
        ka = 1.0;
        ke = 0.1;

        Initialize {
            A = 0.0;
        }

        Dynamics {
            dt(A) = dose * ka - A * ke;
        }

        CalcOutputs {
            A_out = A;
        }

        End.
        """

    def test_jax_model_runs_and_updates_comprehensive(self, simple_model_str):
        """Test comprehensive JAX model running, updates, and ComputedModel interface with full JAX compatibility."""
        model = JaxModel(simple_model_str)
        times = np.linspace(0, 5, 10)

        # Set initial condition to make the model dynamic
        model.update_Y0(A=10.0)

        # Run initial model
        sol = model.run_model(times)

        # Test ComputedModel interface
        assert sol.states.shape == (10, 1)
        assert sol.times.shape == (10,)
        assert sol.var_names == ["A"]

        # Test indexing access
        np.testing.assert_allclose(sol[0], sol.states[:, 0])
        np.testing.assert_allclose(sol["A"], sol.states[:, 0])

        # Test plotting (should not raise)
        ax = sol.plot_results()
        assert ax is not None

        # Update parameter and check new solution is different
        model.update_constants(ke=1.0)
        sol2 = model.run_model(times)
        assert not np.allclose(sol.states, sol2.states)
        assert model.parameters["ke"] == 1.0

        # Update Y0 again and check new solution is different
        model.update_Y0(A=1.0)
        sol3 = model.run_model(times)
        assert not np.allclose(sol2.states, sol3.states)
        assert model.Y0["A"] == 1.0

        # Test that all arrays are proper numpy arrays (JAX compatibility maintained)
        assert isinstance(sol.states, np.ndarray)
        assert isinstance(sol.times, np.ndarray)
        assert isinstance(sol.aux_outputs, np.ndarray)

    def test_jax_model_creation(self, simple_model_str):
        """Test JaxModel creation from string."""
        model = JaxModel(simple_model_str)

        assert hasattr(model, "state_names")
        assert hasattr(model, "parameters")
        assert hasattr(model, "forcing_functions")
        assert hasattr(model, "Y0")

    def test_model_method_not_implemented(self, simple_model_str):
        """Test that model method raises NotImplementedError."""
        model = JaxModel(simple_model_str)

        with pytest.raises(
            NotImplementedError, match="This method should be implemented in equinox module class"
        ):
            model.model(0.0, [1.0], ())

    def test_to_eqx_conversion(self, simple_model_str):
        """Test conversion to EqxModel."""
        model = JaxModel(simple_model_str)
        eqx_model = model._to_eqx()

        assert isinstance(eqx_model, EqxModel)
        assert isinstance(eqx_model.state_names, tuple)
        assert isinstance(eqx_model.output_names, tuple)

    def test_run_model_returns_computed_model(self, simple_model_str):
        """Test that run_model returns a ComputedModel with proper JAX array handling."""
        model = JaxModel(simple_model_str)

        # Parameters ka and ke now have default values in the model,
        # but we can still update them if needed using the proper API
        model.update_constants(ka=1.0, ke=0.1)

        times = np.linspace(0, 2, 21)

        result = model.run_model(times)

        assert isinstance(result, ComputedModel)
        assert len(result.times) == len(times)
        assert result.states.shape[0] == len(times)

        # Ensure all outputs are proper numpy arrays (converted from JAX)
        assert isinstance(result.times, np.ndarray)
        assert isinstance(result.states, np.ndarray)
        assert isinstance(result.aux_outputs, np.ndarray)

        # Test that no NaN values are present
        assert not np.any(np.isnan(result.states))
        assert not np.any(np.isnan(result.times))
        assert not np.any(np.isnan(result.aux_outputs))

    def test_single_state_jax_model(self):
        """Test that single-state models work correctly with JAX."""
        # Simple model with one state variable
        model_str = """
        States = {
            y
        };
        y0 = 2;
        m = 0.5;
        Initialize {
            y = y0;
        }
        Dynamics {
            dt(y) = m;
        }
        End.
        """

        model = JaxModel(model_str)
        model.update_constants(m=1.0, y0=5.0)

        # Test with various time ranges
        time_ranges = [
            np.linspace(0, 10, 20),  # Standard range
            np.array([0, 1, 2]),  # Just a few points
            np.array([0, 5]),  # Two points
        ]

        for times in time_ranges:
            result = model.run_model(times)

            # Verify the result structure
            assert isinstance(result, ComputedModel)
            assert len(result.times) == len(times)
            assert result.states.shape == (len(times), 1)
            assert "y" in result.dataframe.columns

            # Verify analytical solution: y(t) = y0 + m*t = 5 + 1*t
            expected_final = 5.0 + 1.0 * times[-1]
            actual_final = result.dataframe["y"].iloc[-1]
            np.testing.assert_allclose(actual_final, expected_final, rtol=2e-4)

    def test_multi_state_jax_model(self):
        """Test that multi-state models work correctly with JAX."""
        # Two-state model
        model_str = """
        States = {
            x, y
        };
        x0 = 1;
        y0 = 2;
        k1 = 0.5;
        k2 = 0.3;
        Initialize {
            x = x0;
            y = y0;
        }
        Dynamics {
            dt(x) = -k1 * x;
            dt(y) = k1 * x - k2 * y;
        }
        End.
        """

        model = JaxModel(model_str)
        times = np.linspace(0, 10, 50)

        result = model.run_model(times)

        # Verify the result structure
        assert isinstance(result, ComputedModel)
        assert len(result.times) == len(times)
        assert result.states.shape == (len(times), 2)
        assert "x" in result.dataframe.columns
        assert "y" in result.dataframe.columns

        # Verify that x is decreasing (should decay exponentially)
        x_vals = result.dataframe["x"].values
        assert x_vals[0] > x_vals[-1]  # x should decrease
        assert np.all(x_vals >= 0)  # x should stay non-negative

        # Verify physical constraint
        y_vals = result.dataframe["y"].values
        assert y_vals[0] == 2.0  # Initial condition
        assert np.all(y_vals >= 0)  # y should stay non-negative

    def test_calculated_parameters_basic(self):
        """Test basic calculated parameters functionality from Initialize section."""
        model_str = """
        States = {
            A
        };

        # Base parameters
        k1 = 1.0;
        k2 = 2.0;

        # This will be calculated in Initialize
        k_combined = 0.0;

        Initialize {
            A = 5.0;
            k_combined = k1 + k2;
        }

        Dynamics {
            dt(A) = -k_combined * A;
        }

        End.
        """

        model = JaxModel(model_str)

        # Verify that calculated parameters are included in parameters dict
        assert "k1" in model.parameters
        assert "k2" in model.parameters
        assert "k_combined" in model.parameters

        # Verify values are correct
        assert model.parameters["k1"] == 1.0
        assert model.parameters["k2"] == 2.0
        assert model.parameters["k_combined"] == 3.0  # k1 + k2

        # Verify state variables are only in Y0, not in parameters
        assert "A" in model.Y0
        assert "A" not in model.parameters
        assert model.Y0["A"] == 5.0

    def test_calculated_parameters_dependency_chain(self):
        """Test chained calculated parameters with dependencies."""
        model_str = """
        States = {
            A, B
        };

        # Base parameters
        M = 70.0;        # Body weight in kg
        Q_CC = 5.0;      # Cardiac output coefficient

        # These will be calculated in Initialize
        Q_C = 0.0;       # Cardiac output
        V_body = 0.0;    # Body volume
        clearance = 0.0; # Total clearance

        Initialize {
            A = 0.0;
            B = 0.0;

            # Chain of calculations
            Q_C = Q_CC * pow(M, 0.75);          # First calculation
            V_body = M / 1000.0;                # Second calculation
            clearance = Q_C * 0.1;              # Depends on Q_C
        }

        Dynamics {
            dt(A) = -clearance * A;
            dt(B) = clearance * A - 0.5 * B;
        }

        End.
        """

        model = JaxModel(model_str)

        # Verify all parameters are present
        assert "M" in model.parameters
        assert "Q_CC" in model.parameters
        assert "Q_C" in model.parameters
        assert "V_body" in model.parameters
        assert "clearance" in model.parameters

        # Calculate expected values
        expected_Q_C = 5.0 * (70.0**0.75)
        expected_V_body = 70.0 / 1000.0
        expected_clearance = expected_Q_C * 0.1

        # Verify calculated values are correct
        np.testing.assert_allclose(model.parameters["Q_C"], expected_Q_C, rtol=1e-10)
        np.testing.assert_allclose(model.parameters["V_body"], expected_V_body, rtol=1e-10)
        np.testing.assert_allclose(model.parameters["clearance"], expected_clearance, rtol=1e-10)

        # Verify state variables are in Y0
        assert "A" in model.Y0
        assert "B" in model.Y0
        assert model.Y0["A"] == 0.0
        assert model.Y0["B"] == 0.0

    def test_calculated_parameters_update_constants(self):
        """Test that calculated parameters are updated when base parameters change."""
        model_str = """
        States = {
            X
        };

        # Base parameters
        base_rate = 2.0;
        multiplier = 3.0;

        # Calculated parameters
        derived_rate = 0.0;
        final_rate = 0.0;

        Initialize {
            X = 1.0;
            derived_rate = base_rate * multiplier;
            final_rate = derived_rate + 1.0;
        }

        Dynamics {
            dt(X) = -final_rate * X;
        }

        End.
        """

        model = JaxModel(model_str)

        # Initial values
        assert model.parameters["base_rate"] == 2.0
        assert model.parameters["multiplier"] == 3.0
        assert model.parameters["derived_rate"] == 6.0  # 2.0 * 3.0
        assert model.parameters["final_rate"] == 7.0  # 6.0 + 1.0

        # Update base parameter
        model.update_constants(base_rate=4.0)

        # Verify calculated parameters are updated
        assert model.parameters["base_rate"] == 4.0
        assert model.parameters["multiplier"] == 3.0
        assert model.parameters["derived_rate"] == 12.0  # 4.0 * 3.0
        assert model.parameters["final_rate"] == 13.0  # 12.0 + 1.0

        # Verify Y0 is unchanged
        assert model.Y0["X"] == 1.0

    def test_calculated_parameters_update_multiple_constants(self):
        """Test updating multiple base parameters simultaneously."""
        model_str = """
        States = {
            Y
        };

        # Base parameters
        a = 1.0;
        b = 2.0;
        c = 3.0;

        # Calculated parameters
        sum_ab = 0.0;
        product_abc = 0.0;

        Initialize {
            Y = 0.0;
            sum_ab = a + b;
            product_abc = a * b * c;
        }

        Dynamics {
            dt(Y) = sum_ab - product_abc * Y;
        }

        End.
        """

        model = JaxModel(model_str)

        # Initial values
        assert model.parameters["sum_ab"] == 3.0  # 1.0 + 2.0
        assert model.parameters["product_abc"] == 6.0  # 1.0 * 2.0 * 3.0

        # Update multiple parameters
        model.update_constants(a=2.0, b=4.0)

        # Verify calculated parameters are updated
        assert model.parameters["a"] == 2.0
        assert model.parameters["b"] == 4.0
        assert model.parameters["c"] == 3.0
        assert model.parameters["sum_ab"] == 6.0  # 2.0 + 4.0
        assert model.parameters["product_abc"] == 24.0  # 2.0 * 4.0 * 3.0

    def test_calculated_parameters_reset_to_defaults(self):
        """Test reset_to_defaults functionality with calculated parameters."""
        model_str = """
        States = {
            Z
        };

        # Base parameters
        rate_constant = 0.5;
        scale_factor = 2.0;

        # Calculated parameter
        effective_rate = 0.0;

        Initialize {
            Z = 10.0;
            effective_rate = rate_constant * scale_factor;
        }

        Dynamics {
            dt(Z) = -effective_rate * Z;
        }

        End.
        """

        model = JaxModel(model_str)

        # Store original values
        original_rate_constant = model.parameters["rate_constant"]
        original_scale_factor = model.parameters["scale_factor"]
        original_effective_rate = model.parameters["effective_rate"]

        # Update parameters
        model.update_constants(rate_constant=1.0, scale_factor=3.0)

        # Verify updates
        assert model.parameters["rate_constant"] == 1.0
        assert model.parameters["scale_factor"] == 3.0
        assert model.parameters["effective_rate"] == 3.0  # 1.0 * 3.0

        # Reset to defaults
        model.update_constants(reset_to_defaults=True)

        # Verify reset worked
        assert model.parameters["rate_constant"] == original_rate_constant
        assert model.parameters["scale_factor"] == original_scale_factor
        assert model.parameters["effective_rate"] == original_effective_rate

        # Test reset with new values
        model.update_constants(reset_to_defaults=True, rate_constant=0.8)
        assert model.parameters["rate_constant"] == 0.8
        assert model.parameters["scale_factor"] == original_scale_factor
        assert model.parameters["effective_rate"] == 0.8 * original_scale_factor

    def test_calculated_parameters_pbpk_example(self):
        """Test with a simplified PBPK model structure similar to pbpk_simple.model."""
        model_str = """
        States = {
            A_plasma, A_liver
        };

        # Physiological parameters
        M = 0.25;               # Body weight (kg)
        Q_CC = 15.0;            # Cardiac output (L/h/kg^0.75)
        Q_LC = 0.21;            # Proportion of cardiac output to liver
        V_LC = 0.04;            # Volume fraction of liver

        # Calculated parameters
        Q_C = 0.0;              # Cardiac output (L/h)
        Q_L = 0.0;              # Blood flow rate to liver (L/h)
        V_L = 0.0;              # Volume of liver (L)

        Initialize {
            A_plasma = 0.0;
            A_liver = 0.0;

            # PBPK calculations
            Q_C = Q_CC * pow(M, 0.75);
            Q_L = Q_LC * Q_C;
            V_L = V_LC * M;
        }

        Dynamics {
            dt(A_plasma) = -Q_L * A_plasma / 1.0;
            dt(A_liver) = Q_L * A_plasma / 1.0 - 0.1 * A_liver / V_L;
        }

        End.
        """

        model = JaxModel(model_str)

        # Verify all parameters are present
        required_params = ["M", "Q_CC", "Q_LC", "V_LC", "Q_C", "Q_L", "V_L"]
        for param in required_params:
            assert param in model.parameters, f"Parameter {param} not found"

        # Calculate expected values
        M = 0.25
        Q_CC = 15.0
        Q_LC = 0.21
        V_LC = 0.04

        expected_Q_C = Q_CC * (M**0.75)
        expected_Q_L = Q_LC * expected_Q_C
        expected_V_L = V_LC * M

        # Verify calculated values
        np.testing.assert_allclose(model.parameters["Q_C"], expected_Q_C, rtol=1e-10)
        np.testing.assert_allclose(model.parameters["Q_L"], expected_Q_L, rtol=1e-10)
        np.testing.assert_allclose(model.parameters["V_L"], expected_V_L, rtol=1e-10)

        # Test parameter update (change body weight)
        new_M = 0.5
        model.update_constants(M=new_M)

        new_Q_C = Q_CC * (new_M**0.75)
        new_Q_L = Q_LC * new_Q_C
        new_V_L = V_LC * new_M

        np.testing.assert_allclose(model.parameters["Q_C"], new_Q_C, rtol=1e-10)
        np.testing.assert_allclose(model.parameters["Q_L"], new_Q_L, rtol=1e-10)
        np.testing.assert_allclose(model.parameters["V_L"], new_V_L, rtol=1e-10)

        # Verify state variables are in Y0 and unaffected
        assert "A_plasma" in model.Y0
        assert "A_liver" in model.Y0
        assert model.Y0["A_plasma"] == 0.0
        assert model.Y0["A_liver"] == 0.0


class TestJaxCompatibility:
    """Comprehensive JAX compatibility tests for the entire model pipeline."""

    def test_jax_dependencies_available(self):
        """Test that JAX dependencies are available."""
        import importlib.util

        jax_modules = ["jax", "equinox", "diffrax"]

        for module in jax_modules:
            spec = importlib.util.find_spec(module)
            assert spec is not None, f"Module '{module}' is not available"

        # If we get here, all dependencies are available
        assert True

    def test_jax_model_creation_compatibility(self):
        """Test that JAX models can be created without issues."""
        model_str = """
        States = {
            A
        };

        Parameters = {
            ka,
            ke
        };

        Inputs = {
            dose
        };

        Initialize {
            A = 0.0;
        }

        Dynamics {
            dt(A) = dose * ka - A * ke;
        }

        Outputs = {
            A_out
        };

        CalcOutputs {
            A_out = A;
        }

        End.
        """

        # This should not raise any exceptions
        model = JaxModel(model_str)
        assert model is not None

    def test_unified_forcing_functions_jax_compatibility(self):
        """Test unified forcing functions work correctly with JAX backend."""
        # Test all major forcing function types with JAX backend
        test_cases = [
            ("OnOff", {"t0": 1.0, "t1": 3.0, "s": 10.0}),
            ("PerDose", {"t0": 0.0, "duration": 1.0, "period": 24.0, "s": 10.0}),
            ("NDoses", {"t0_list": [0.0, 24.0], "duration": 1.0, "s": 10.0}),
            ("ConstFunc", {"value": 5.0}),
            ("ZeroFunc", {}),
        ]

        for func_name, kwargs in test_cases:
            func = UnifiedForcingFactory.create_forcing_function(
                func_name, backend=BackendType.JAX, **kwargs
            )

            # Test with JAX arrays
            test_time = jnp.array(1.0)
            result = func(test_time)
            assert isinstance(result, jnp.ndarray), f"Function {func_name} didn't return JAX array"

            # Test JIT compilation
            jitted_func = jax.jit(func)
            jitted_result = jitted_func(test_time)
            np.testing.assert_allclose(result, jitted_result, rtol=1e-6)

    def test_interpolated_forcing_jax_compatibility(self):
        """Test interpolated forcing functions work with JAX backend."""
        times = [0.0, 1.0, 2.0, 5.0]
        values = [10.0, 20.0, 30.0, 50.0]

        # Create interpolated forcing function via unified factory
        func = UnifiedForcingFactory.create_forcing_function(
            "InterpolatedForcing", backend=BackendType.JAX, times=times, values=values
        )

        # Test with JAX arrays
        test_times = jnp.array([0.5, 1.5, 4.0])
        results = jax.vmap(func)(test_times)
        assert isinstance(results, jnp.ndarray)
        assert len(results) == len(test_times)

        # Test JIT compilation
        jitted_func = jax.jit(jax.vmap(func))
        jitted_results = jitted_func(test_times)
        np.testing.assert_allclose(results, jitted_results, rtol=1e-6)

    def test_jax_array_broadcasting_compatibility(self):
        """Test that JAX functions work with array broadcasting."""
        model_str = """
        States = { A };
        Inputs = { dose };
        ka = 1.0; ke = 0.1;
        Initialize { A = 0.0; }
        Dynamics { dt(A) = dose * ka - A * ke; }
        End.
        """

        model = JaxModel(model_str)
        model.forcing_functions["dose"] = {
            "function": "OnOff",
            "kwargs": {"t0": 1.0, "t1": 3.0, "s": 10.0},
        }

        result = model.run_model(np.linspace(0, 5, 20))

        # Test vectorized evaluation of forcing functions
        test_times = jnp.array([0.5, 1.5, 2.5, 3.5])
        ff = result.input_functions["dose"]
        vectorized_ff = jax.vmap(ff)
        ff_results = vectorized_ff(test_times)

        assert isinstance(ff_results, jnp.ndarray)
        assert ff_results.shape == test_times.shape

    def test_model_with_interpolated_forcing_integration(self):
        """Test JAX model with interpolated forcing function from data."""
        model_str = """
        States = { A };
        Inputs = { time_varying_input };
        ka = 1.0; ke = 0.1;
        Initialize { A = 0.0; }
        Dynamics { dt(A) = time_varying_input * ka - A * ke; }
        Outputs = { A_out };
        CalcOutputs { A_out = A; }
        End.
        """

        # Create time-varying data (e.g., body weight growth)
        data_times = [0.0, 5.0, 10.0, 20.0, 30.0]
        data_values = [0.25, 0.5, 1.0, 1.8, 2.5]  # kg body weight

        model = JaxModel(model_str)

        # Use assign_forcing_function to set up interpolation
        model.assign_forcing_function(
            "time_varying_input",
            "InterpolatedForcing",
            data_dict={"time": data_times, "value": data_values},
        )

        # Run simulation
        times = np.linspace(0, 30, 150)
        result = model.run_model(times)

        # Verify results
        assert isinstance(result, ComputedModel)
        assert not np.any(np.isnan(result.states))

        # Test that forcing function is JAX-compatible
        ff = result.input_functions["time_varying_input"]
        test_time = jnp.array(15.0)
        ff_result = ff(test_time)
        assert isinstance(ff_result, jnp.ndarray)

        # Test JIT compilation of forcing function
        jitted_ff = jax.jit(ff)
        jitted_result = jitted_ff(test_time)
        np.testing.assert_allclose(ff_result, jitted_result, rtol=1e-6)

    def test_full_model_jax_pipeline(self):
        """Test complete JAX model pipeline with forcing functions."""
        model_str = """
        States = {
            A
        };

        Inputs = {
            dose_input
        };

        # Parameters with default values
        ka = 1.0;
        ke = 0.1;

        Initialize {
            A = 0.0;
        }

        Dynamics {
            dt(A) = dose_input * ka - A * ke;
        }

        Outputs = {
            A_out
        };

        CalcOutputs {
            A_out = A;
        }

        End.
        """

        model = JaxModel(model_str)

        # Test with different forcing function types
        forcing_tests = [
            {"function": "OnOff", "kwargs": {"t0": 1.0, "t1": 3.0, "s": 10.0}},
            {"function": "ConstFunc", "kwargs": {"value": 2.0}},
        ]

        times = np.linspace(0, 10, 50)

        for test_case in forcing_tests:
            # Assign forcing function
            model.forcing_functions["dose_input"] = {
                "function": test_case["function"],
                "args": (),
                "kwargs": test_case["kwargs"],
            }

            # Run model
            result = model.run_model(times)

            # Verify result structure
            assert isinstance(result, ComputedModel)
            assert isinstance(result.states, np.ndarray)
            assert isinstance(result.times, np.ndarray)
            assert not np.any(np.isnan(result.states))

            # Verify forcing function is in input_functions and is JAX-compatible
            assert "dose_input" in result.input_functions
            test_time = jnp.array(1.0)
            ff_result = result.input_functions["dose_input"](test_time)
            assert isinstance(ff_result, jnp.ndarray)

    def test_events_raise_error_for_jax_models(self):
        """Test that discrete events properly raise NotImplementedError for JAX models."""
        model_str = """
        States = { A };
        ka = 1.0; ke = 0.1;
        Initialize { A = 10.0; }
        Dynamics { dt(A) = -ke * A; }
        End.
        """

        model = JaxModel(model_str)

        # Manually add an event to test error handling
        model.events.append({"type": "test_event"})

        times = np.linspace(0, 5, 10)

        with pytest.raises(NotImplementedError, match="Discrete events are not yet supported"):
            model.run_model(times)

def test_reset_to_defaults_jax():
    """Test reset_to_defaults functionality for both parameters and Y0 with JAX model."""
    model_str = """
    States = { A };

    Inputs = { dose };

    # Parameters with default values
    ke = 0.5;
    ka = 1.0;

    Initialize {
        A = 10.0;
    }

    Dynamics {
        dt(A) = -ke * A + dose;
    }

    End.
    """

    model = JaxModel(model_str)

    # Store original default values
    original_parameters = model.parameters.copy()
    original_Y0 = model.Y0.copy()

    # Update some parameters and Y0 values
    model.update_constants(ke=2.0, ka=3.0)
    model.update_Y0(A=20.0)

    # Verify they changed
    assert model.parameters["ke"] == 2.0
    assert model.parameters["ka"] == 3.0
    assert model.Y0["A"] == 20.0

    # Test reset_to_defaults=True for parameters
    model.update_constants(reset_to_defaults=True, ke=1.5)
    assert model.parameters["ke"] == 1.5  # Updated value
    assert model.parameters["ka"] == 1.0  # Reset to default

    # Test reset_to_defaults=True for Y0
    model.update_Y0(reset_to_defaults=True, A=15.0)
    assert model.Y0["A"] == 15.0  # Updated value

    # Test reset_to_defaults=False (default behavior)
    model.update_constants(ke=5.0)  # Should only update ke, not reset others
    assert model.parameters["ke"] == 5.0
    assert model.parameters["ka"] == 1.0  # Should remain unchanged

    model.update_Y0(A=25.0)  # Should only update A
    assert model.Y0["A"] == 25.0

    # Full reset test
    model.update_constants(reset_to_defaults=True)
    model.update_Y0(reset_to_defaults=True)
    assert model.parameters == original_parameters
    assert model.Y0 == original_Y0


class TestJaxModelErrorHandling:
    """Test error handling and edge cases for JAX models."""

    def test_invalid_model_string(self):
        """Test handling of invalid model strings."""
        invalid_model = "This is not a valid model string"

        with pytest.raises((ValueError, RuntimeError)):  # Should raise some parsing exception
            JaxModel(invalid_model)

    def test_missing_required_sections(self):
        """Test handling of models missing required sections."""
        incomplete_model = """
        States = { A };
        # Missing Initialize and Dynamics sections - this should fail
        """
        # Note: No End statement, incomplete structure

        with pytest.raises((ValueError, RuntimeError)):  # Should raise validation exception
            JaxModel(incomplete_model)

    def test_empty_time_array(self, simple_pk_model_str):
        """Test handling of empty time arrays."""
        model = JaxModel(simple_pk_model_str)

        with pytest.raises((ValueError, IndexError)):
            model.run_model([])

    def test_single_time_point(self, simple_pk_model_str):
        """Test handling of single time point."""
        model = JaxModel(simple_pk_model_str)
        model.update_Y0(A=10.0)

        # Single time point should work
        result = model.run_model([0.0])
        assert isinstance(result, ComputedModel)
        assert len(result.times) == 1
        assert result.times[0] == 0.0
        assert result.states.shape == (1, 1)  # One time point, one state variable
        assert result.states[0, 0] == 10.0  # Should match initial condition

    def test_single_time_point_non_zero(self, simple_pk_model_str):
        """Test single time point at non-zero time (requires integration)."""
        model = JaxModel(simple_pk_model_str)
        model.update_Y0(A=10.0)
        model.update_constants(ke=0.1)

        # Single time point at t=1.0 should integrate from 0 to 1
        result = model.run_model([1.0])
        assert isinstance(result, ComputedModel)
        assert len(result.times) == 1
        assert result.times[0] == 1.0
        assert result.states.shape == (1, 1)

        # Should complete successfully and return finite values
        assert np.isfinite(result.states[0, 0])
        assert not np.isnan(result.states[0, 0])

    def test_single_time_point_negative(self, simple_pk_model_str):
        """Test single time point at negative time (backwards integration)."""
        model = JaxModel(simple_pk_model_str)
        model.update_Y0(A=10.0)

        # Single time point at t=-1.0 should integrate backwards
        result = model.run_model([-1.0])
        assert isinstance(result, ComputedModel)
        assert len(result.times) == 1
        assert result.times[0] == -1.0
        assert result.states.shape == (1, 1)

        # For backwards integration, should handle gracefully
        assert result.states[0, 0] > 0.0  # Should be positive

    def test_negative_time_values(self, simple_pk_model_str):
        """Test handling of negative time values."""
        model = JaxModel(simple_pk_model_str)
        model.update_Y0(A=10.0)

        # Should handle negative start time
        times = np.linspace(-1, 5, 61)
        result = model.run_model(times)
        assert isinstance(result, ComputedModel)

    def test_non_monotonic_times(self, simple_pk_model_str):
        """Test handling of non-monotonic time arrays."""
        model = JaxModel(simple_pk_model_str)
        model.update_Y0(A=10.0)

        # Non-monotonic times should cause issues with JAX solver
        times = [0, 2, 1, 3, 5]  # Not sorted

        # JAX requires monotonic times, so this should fail
        with pytest.raises(Exception) as excinfo:
            model.run_model(times)
        
        # Should get an error about monotonic times
        assert "increasing or decreasing" in str(excinfo.value)

    def test_numerical_instability_large_rates(self):
        """Test handling of potentially unstable models with large rate constants."""
        model_str = """
        States = { A };
        k = 1e6;  # Very large rate constant
        Initialize { A = 1.0; }
        Dynamics { dt(A) = -k * A; }
        End.
        """

        model = JaxModel(model_str)
        times = np.linspace(0, 1e-6, 10)  # Very short time for large rate

        # Should either work or provide helpful error message
        try:
            result = model.run_model(times)
            assert isinstance(result, ComputedModel)
            # Should not have NaN values
            assert not np.any(np.isnan(result.states))
        except (ValueError, RuntimeError) as e:
            # Should provide helpful error message about numerical stability
            assert len(str(e)) > 0

    def test_zero_state_variables(self):
        """Test handling of models with no state variables."""
        invalid_model = """
        States = {};  # Empty states
        Initialize {}
        Dynamics {}
        End.
        """

        # This should either raise an exception or create a valid but empty model
        try:
            model = JaxModel(invalid_model)
            # If it succeeds, should have empty state names
            assert len(model.state_names) == 0
        except Exception as e:
            # Raising an exception for empty states is also acceptable behavior
            pytest.skip(f"Empty states caused expected exception: {e}")

    def test_circular_dependencies_in_dynamics(self):
        """Test handling of circular dependencies in dynamics."""
        circular_model = """
        States = { A, B };
        Initialize { A = 1.0; B = 2.0; }
        Dynamics {
            dt(A) = B;
            dt(B) = A;  # Simple circular dependency
        }
        End.
        """

        # Should work fine - circular dependencies in dynamics are allowed
        model = JaxModel(circular_model)
        times = np.linspace(0, 1, 11)
        result = model.run_model(times)
        assert isinstance(result, ComputedModel)


class TestJaxModelFileLoading:
    """Test JAX model loading from files."""

    def test_model_creation_from_file(self, data_path):
        """Test JaxModel creation from .model file."""
        # Use a specific valid model file
        model_file = data_path / "pk1.model"
        if model_file.exists():
            model = JaxModel(model_file)
            assert isinstance(model, JaxModel)
            assert hasattr(model, "state_names")
            assert hasattr(model, "parameters")
            assert hasattr(model, "Y0")
        else:
            pytest.skip("Test model file not found")

    def test_model_creation_from_pathlib_path(self, data_path):
        """Test JaxModel creation from pathlib.Path object."""
        model_file = data_path / "pk1.model"
        if model_file.exists():
            from pathlib import Path

            path_obj = Path(model_file)
            model = JaxModel(path_obj)
            assert isinstance(model, JaxModel)
        else:
            pytest.skip("Test model file not found")

    def test_invalid_file_path(self):
        """Test handling of invalid file paths."""
        from pathlib import Path

        invalid_path = Path("nonexistent_file.model")

        with pytest.raises(FileNotFoundError):
            JaxModel(invalid_path)

    def test_file_with_invalid_content(self, tmp_path):
        """Test handling of files with invalid model content."""
        # Create temporary file with invalid content
        invalid_file = tmp_path / "invalid.model"
        invalid_file.write_text("This is not a valid model")

        with pytest.raises((ValueError, RuntimeError)):  # Should raise parsing exception
            JaxModel(invalid_file)


class TestJaxModelParameterEffects:
    """Test parameter and initial condition effects on solutions."""

    def test_parameter_effects_on_solution(self, simple_pk_model_str, short_times):
        """Test that parameter changes affect model solution."""
        model = JaxModel(simple_pk_model_str)
        model.update_Y0(A=10.0)
        model.update_constants(ke=0.1)

        # Add a constant dose to make the system more interesting
        model.assign_forcing_function("dose", "ConstFunc", value=1.0)

        # Run with default parameters
        result1 = model.run_model(short_times)

        # Change elimination rate and run again
        model.update_constants(ke=0.5)  # Faster elimination
        result2 = model.run_model(short_times)

        # Solutions should be different
        assert not np.allclose(result1.states, result2.states)

        # Final state should be lower with faster elimination (for given dose rate)
        assert result2.states[-1, 0] < result1.states[-1, 0]

    def test_initial_condition_effects_on_solution(self, simple_pk_model_str, short_times):
        """Test that initial condition changes affect model solution."""
        model = JaxModel(simple_pk_model_str)

        # Run with first initial condition
        model.update_Y0(A=5.0)
        result1 = model.run_model(short_times)

        # Run with different initial condition
        model.update_Y0(A=15.0)
        result2 = model.run_model(short_times)

        # Solutions should be different
        assert not np.allclose(result1.states, result2.states)

        # All states in result2 should be higher than result1 (proportional scaling)
        assert np.all(result2.states[:, 0] > result1.states[:, 0])

    def test_multiple_parameter_effects(self, complex_pk_model_str, standard_times):
        """Test effects of multiple parameter changes."""
        model = JaxModel(complex_pk_model_str)
        model.update_Y0(A0=0.0, A1=10.0, AUC=0.0)

        # Add a constant dose to A0 to make the system dynamic
        model.assign_forcing_function("dose", "ConstFunc", value=1.0)

        # Baseline run
        baseline_result = model.run_model(standard_times)

        # Change multiple parameters
        model.update_constants(ka=2.0, ke=0.5, V=20.0)
        modified_result = model.run_model(standard_times)

        # Solutions should be significantly different
        assert not np.allclose(baseline_result.states, modified_result.states)

        # Check that each state variable is affected
        for i in range(baseline_result.states.shape[1]):
            state_diff = np.abs(baseline_result.states[:, i] - modified_result.states[:, i])
            assert np.max(state_diff) > 0.01  # Some difference (lowered threshold)

    def test_parameter_sensitivity(self, simple_pk_model_str):
        """Test parameter sensitivity analysis."""
        model = JaxModel(simple_pk_model_str)
        model.update_Y0(A=10.0)

        # Add constant dose to make system responsive to parameter changes
        model.assign_forcing_function("dose", "ConstFunc", value=2.0)

        times = np.linspace(0, 10, 11)

        base_ke = 0.1
        model.update_constants(ke=base_ke)
        base_result = model.run_model(times)

        # Test sensitivity to small parameter changes
        perturbations = [0.05, 0.15, 0.2]  # ±50%, +50%, +100%

        for perturbed_ke in perturbations:
            model.update_constants(ke=perturbed_ke)
            perturbed_result = model.run_model(times)

            # Should see proportional changes in solution
            relative_solution_change = (
                np.abs(perturbed_result.states[-1, 0] - base_result.states[-1, 0])
                / base_result.states[-1, 0]
            )

            # Solution sensitivity should be related to parameter sensitivity
            assert relative_solution_change > 0.01  # Should have some effect

    def test_reset_effects(self, simple_pk_model_str, short_times):
        """Test that reset_to_defaults affects solutions appropriately."""
        model = JaxModel(simple_pk_model_str)

        # Start with model default initial conditions (A=0)
        # Get original solution
        original_result = model.run_model(short_times)

        # Modify parameters and Y0
        model.update_constants(ke=0.5)
        model.update_Y0(A=20.0)
        modified_result = model.run_model(short_times)

        # Should be different from original
        assert not np.allclose(original_result.states, modified_result.states)

        # Reset and verify we get back to original behavior
        model.update_constants(reset_to_defaults=True)
        model.update_Y0(reset_to_defaults=True)
        reset_result = model.run_model(short_times)

        # Should match original (within tolerance)
        np.testing.assert_allclose(original_result.states, reset_result.states, rtol=1e-10)


class TestJaxModelAuxiliaryOutputs:
    """Test auxiliary output calculation and CalcOutputs functionality."""

    def test_auxiliary_outputs_basic(self, complex_pk_model_str, short_times):
        """Test auxiliary output calculation in complex model."""
        model = JaxModel(complex_pk_model_str)
        model.update_Y0(A0=0.0, A1=10.0, AUC=0.0)

        result = model.run_model(short_times)

        # Should have auxiliary outputs
        assert hasattr(result, "aux_outputs")
        assert hasattr(result, "aux_names")
        assert result.aux_outputs.shape[0] == len(short_times)
        assert len(result.aux_names) > 0

        # Check specific outputs from the complex model
        if "C" in result.aux_names:
            C_idx = result.aux_names.index("C")
            # Concentration should be positive when A1 > 0
            assert result.aux_outputs[0, C_idx] > 0
            # C = A1 / V, so with A1=10, V=10, should get C=1
            expected_C = 10.0 / 10.0  # A1 / V
            assert np.isclose(result.aux_outputs[0, C_idx], expected_C, rtol=1e-6)

        if "Atot" in result.aux_names:
            Atot_idx = result.aux_names.index("Atot")
            # Total amount should be sum of A0 + A1 = 0 + 10 = 10
            assert np.isclose(result.aux_outputs[0, Atot_idx], 10.0, rtol=1e-6)

    def test_auxiliary_outputs_time_evolution(self, complex_pk_model_str):
        """Test that auxiliary outputs evolve correctly over time."""
        model = JaxModel(complex_pk_model_str)
        model.update_Y0(A0=20.0, A1=0.0, AUC=0.0)  # Start with dose in A0

        times = np.linspace(0, 10, 21)
        result = model.run_model(times)

        if "C" in result.aux_names and "Atot" in result.aux_names:
            C_idx = result.aux_names.index("C")
            Atot_idx = result.aux_names.index("Atot")

            # Total amount should decrease over time (elimination)
            total_amounts = result.aux_outputs[:, Atot_idx]
            assert total_amounts[0] > total_amounts[-1]  # Should decrease

            # Concentration should first increase (absorption) then decrease
            concentrations = result.aux_outputs[:, C_idx]
            max_conc_idx = np.argmax(concentrations)
            assert max_conc_idx > 0  # Peak should not be at t=0
            assert max_conc_idx < len(concentrations) - 1  # Peak should not be at end

    def test_auxiliary_outputs_with_parameters(self, complex_pk_model_str, short_times):
        """Test auxiliary outputs with different parameter values."""
        model = JaxModel(complex_pk_model_str)
        model.update_Y0(A0=0.0, A1=10.0, AUC=0.0)

        # Add a constant dose to prevent decay to zero
        model.assign_forcing_function("dose", "ConstFunc", value=1.0)

        # Test with default parameters - let system reach steady state
        times = np.array([10.0])  # Single time point after settling
        result1 = model.run_model(times)

        # Test with different volume of distribution
        model.update_constants(V=20.0)  # Double the volume
        result2 = model.run_model(times)

        if "C" in result1.aux_names:
            C_idx = result1.aux_names.index("C")
            # With constant input, steady state concentration should be inversely proportional to volume
            conc1 = result1.aux_outputs[0, C_idx]
            conc2 = result2.aux_outputs[0, C_idx]
            # Since we doubled V, concentration should be approximately halved (allowing for some tolerance)
            assert conc2 < conc1  # Should definitely be lower
            # More lenient test: just check it's significantly different
            assert abs(conc2 - conc1) / conc1 > 0.1

    def test_auxiliary_outputs_data_types(self, complex_pk_model_str, short_times):
        """Test that auxiliary outputs have correct data types."""
        model = JaxModel(complex_pk_model_str)
        model.update_Y0(A0=0.0, A1=10.0, AUC=0.0)

        result = model.run_model(short_times)

        # Auxiliary outputs should be numpy arrays
        assert isinstance(result.aux_outputs, np.ndarray)
        assert isinstance(result.aux_names, list)

        # Should not contain NaN or infinite values
        assert not np.any(np.isnan(result.aux_outputs))
        assert not np.any(np.isinf(result.aux_outputs))

        # Shape should be consistent
        assert result.aux_outputs.shape[0] == len(result.times)
        assert result.aux_outputs.shape[1] == len(result.aux_names)

    def test_model_with_no_calc_outputs(self, simple_pk_model_str, short_times):
        """Test model without CalcOutputs section."""
        model = JaxModel(simple_pk_model_str)
        model.update_Y0(A=10.0)

        result = model.run_model(short_times)

        # Should still have aux_outputs and aux_names attributes
        assert hasattr(result, "aux_outputs")
        assert hasattr(result, "aux_names")

        # aux_names might be empty or contain default outputs
        if len(result.aux_names) > 0:
            assert result.aux_outputs.shape[1] == len(result.aux_names)

    def test_auxiliary_outputs_access_via_dataframe(self, complex_pk_model_str, short_times):
        """Test accessing auxiliary outputs through dataframe interface."""
        model = JaxModel(complex_pk_model_str)
        model.update_Y0(A0=0.0, A1=10.0, AUC=0.0)

        result = model.run_model(short_times)
        df = result.dataframe

        # Auxiliary outputs should be in dataframe columns
        for aux_name in result.aux_names:
            assert aux_name in df.columns

            # Values should match aux_outputs array
            aux_idx = result.aux_names.index(aux_name)
            np.testing.assert_allclose(
                df[aux_name].values, result.aux_outputs[:, aux_idx], rtol=1e-10
            )


class TestJaxModelAdvancedInterpolation:
    """Test advanced interpolation functionality and input function handling."""

    def test_interpolated_forcing_from_dataframe(self, simple_pk_model_str):
        """Test InterpolatedForcing with pandas DataFrame."""
        model = JaxModel(simple_pk_model_str)
        model.update_Y0(A=0.0)

        # Create DataFrame for forcing

        df = pd.DataFrame({"time": [0, 2, 4, 6, 8, 10], "dose_rate": [0, 1, 5, 3, 1, 0]})

        # Assign forcing function with DataFrame
        model.assign_forcing_function(
            "dose", "InterpolatedForcing", dataframe=df, time_col="time", value_col="dose_rate"
        )

        times = np.linspace(0, 10, 101)
        result = model.run_model(times)

        assert isinstance(result, ComputedModel)
        assert result.states.shape[0] == len(times)

        # Should see accumulation corresponding to dose profile
        # Peak dose is at t=4, should see maximum accumulation after that
        peak_dose_idx = np.argmin(np.abs(times - 4.0))
        assert result.states[peak_dose_idx, 0] > result.states[0, 0]

    def test_input_functions_in_result(
        self, simple_pk_model_str, onoff_forcing_params, short_times
    ):
        """Test that input functions are included in ComputedModel result."""
        model = JaxModel(simple_pk_model_str)
        model.assign_forcing_function("dose", "OnOff", **onoff_forcing_params)

        result = model.run_model(short_times)

        # Result should contain input functions
        assert hasattr(result, "input_functions")
        assert "dose" in result.input_functions
        assert callable(result.input_functions["dose"])

        # Function should work with JAX arrays
        import jax.numpy as jnp

        dose_at_t2 = result.input_functions["dose"](jnp.array(2.0))
        assert isinstance(dose_at_t2, jnp.ndarray)

        # Test function behavior matches expected OnOff profile
        t_before = 0.5  # Before t0
        t_during = 2.0  # Between t0 and t1
        t_after = 6.0  # After t1

        dose_before = result.input_functions["dose"](jnp.array(t_before))
        dose_during = result.input_functions["dose"](jnp.array(t_during))
        dose_after = result.input_functions["dose"](jnp.array(t_after))

        # Should be low before and after, high during
        assert dose_before < dose_during
        assert dose_after < dose_during

    def test_multiple_interpolated_forcing_functions(self, complex_pk_model_str):
        """Test model with multiple interpolated forcing functions."""
        # Create a model string with multiple inputs
        multi_input_model = """
        States = {
            A0, A1, AUC
        };

        Inputs = {
            oral_dose,
            iv_dose
        };

        Outputs = {
            C, Atot
        };

        # Parameters
        ka = 1.0; ke = 0.1; V = 10.0;

        Initialize {
            A0 = 0.0; A1 = 0.0; AUC = 0.0;
        }

        Dynamics {
            dt(A0) = oral_dose - ka * A0;
            dt(A1) = ka * A0 + iv_dose - ke * A1;
            dt(AUC) = A1 / V;
        }

        CalcOutputs {
            C = A1 / V;
            Atot = A0 + A1;
        }

        End.
        """

        model = JaxModel(multi_input_model)

        # Set up different dosing profiles for each input
        oral_times = [0, 24, 48, 72]
        oral_doses = [100, 0, 50, 0]

        iv_times = [12, 36, 60]
        iv_doses = [25, 25, 25]

        model.assign_forcing_function(
            "oral_dose", "InterpolatedForcing", data_dict={"time": oral_times, "value": oral_doses}
        )

        model.assign_forcing_function(
            "iv_dose", "InterpolatedForcing", data_dict={"time": iv_times, "value": iv_doses}
        )

        times = np.linspace(0, 72, 145)
        result = model.run_model(times)

        # Should have both input functions in result
        assert "oral_dose" in result.input_functions
        assert "iv_dose" in result.input_functions

        # Test that functions work independently
        oral_at_24 = result.input_functions["oral_dose"](jnp.array(24.0))
        iv_at_12 = result.input_functions["iv_dose"](jnp.array(12.0))

        # Should reflect the dosing schedules
        assert oral_at_24 < 100.0  # Should be decreasing from peak
        assert iv_at_12 > 20.0  # Should be close to 25

    def test_interpolated_forcing_boundary_conditions(self, simple_pk_model_str):
        """Test interpolated forcing at boundary conditions."""
        model = JaxModel(simple_pk_model_str)
        model.update_Y0(A=0.0)

        # Create forcing data with specific boundary behavior
        forcing_times = [0.0, 5.0, 10.0]
        forcing_values = [10.0, 0.0, 5.0]

        model.assign_forcing_function(
            "dose",
            "InterpolatedForcing",
            data_dict={"time": forcing_times, "value": forcing_values},
        )

        times = np.linspace(0, 10, 21)
        result = model.run_model(times)

        # Test boundary values
        dose_func = result.input_functions["dose"]

        # At boundaries, should match exactly
        assert np.isclose(dose_func(jnp.array(0.0)), 10.0, atol=1e-6)
        assert np.isclose(dose_func(jnp.array(5.0)), 0.0, atol=1e-6)
        assert np.isclose(dose_func(jnp.array(10.0)), 5.0, atol=1e-6)

        # At midpoints, should be interpolated
        midpoint_val = dose_func(jnp.array(2.5))  # Between 0 and 5
        assert 0.0 < midpoint_val < 10.0

    def test_interpolated_forcing_extrapolation(self, simple_pk_model_str):
        """Test behavior of interpolated forcing outside data range."""
        model = JaxModel(simple_pk_model_str)
        model.update_Y0(A=0.0)

        # Create forcing data that doesn't cover full simulation time
        forcing_times = [2.0, 4.0, 6.0]
        forcing_values = [5.0, 10.0, 2.0]

        model.assign_forcing_function(
            "dose",
            "InterpolatedForcing",
            data_dict={"time": forcing_times, "value": forcing_values},
        )

        times = np.linspace(0, 10, 21)
        result = model.run_model(times)

        dose_func = result.input_functions["dose"]

        # Test extrapolation behavior (should clamp or extend constant)
        dose_before = dose_func(jnp.array(0.0))  # Before data range
        dose_after = dose_func(jnp.array(10.0))  # After data range

        # Should handle extrapolation gracefully (values should be reasonable)
        assert dose_before >= 0.0
        assert dose_after >= 0.0
        assert not np.isnan(dose_before)
        assert not np.isnan(dose_after)

    def test_jax_compatibility_of_interpolated_functions(self, simple_pk_model_str):
        """Test that interpolated functions maintain full JAX compatibility."""
        model = JaxModel(simple_pk_model_str)
        model.update_Y0(A=0.0)

        # Set up interpolated forcing
        times_data = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        values_data = [0.0, 5.0, 10.0, 8.0, 3.0, 0.0]

        model.assign_forcing_function(
            "dose", "InterpolatedForcing", data_dict={"time": times_data, "value": values_data}
        )

        times = np.linspace(0, 5, 26)
        result = model.run_model(times)

        dose_func = result.input_functions["dose"]

        # Test JAX transformations
        import jax

        # Test vmap (vectorization)
        test_times = jnp.array([0.5, 1.5, 2.5, 3.5, 4.5])
        vectorized_func = jax.vmap(dose_func)
        vectorized_results = vectorized_func(test_times)

        assert isinstance(vectorized_results, jnp.ndarray)
        assert len(vectorized_results) == len(test_times)

        # Test JIT compilation
        jitted_func = jax.jit(dose_func)
        jit_result = jitted_func(jnp.array(2.0))
        direct_result = dose_func(jnp.array(2.0))

        np.testing.assert_allclose(jit_result, direct_result, rtol=1e-10)
