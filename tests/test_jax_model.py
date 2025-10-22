"""Tests for JAX model implementations."""

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("equinox")
pytest.importorskip("diffrax")

import jax
import jax.numpy as jnp

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

    def test_static_onoff_function(self):
        """Test the static OnOff forcing function."""
        t = jnp.array(1.0)
        t0 = 0.5
        t1 = 2.0

        result = EqxModel.OnOff(t, t0, t1)
        assert isinstance(result, jnp.ndarray)
        assert 0 <= result <= 1  # Should be between 0 and 1

        # Test JAX JIT compatibility
        jitted_onoff = jax.jit(EqxModel.OnOff)
        jitted_result = jitted_onoff(t, t0, t1)
        np.testing.assert_allclose(result, jitted_result, rtol=1e-6)

    def test_static_perdose_function(self):
        """Test the static PerDose forcing function."""
        perdose_func = EqxModel.PerDose(0.0, 1.0, 24.0)

        # Test at different time points
        t_on = jnp.array(0.5)  # During first dose
        t_off = jnp.array(12.0)  # Between doses
        t_second = jnp.array(24.5)  # During second dose

        result_on = perdose_func(t_on)
        result_off = perdose_func(t_off)
        result_second = perdose_func(t_second)

        assert result_on > 0.5  # Should be active
        assert result_off < 0.5  # Should be inactive
        assert result_second > 0.5  # Should be active again

        # Test JAX JIT compatibility
        jitted_perdose = jax.jit(perdose_func)
        jitted_result = jitted_perdose(t_on)
        np.testing.assert_allclose(result_on, jitted_result, rtol=1e-6)

    def test_static_ndoses_function(self):
        """Test the static NDoses forcing function."""
        t0_list = [0.0, 24.0, 48.0]
        duration = 1.0

        ndoses_func = EqxModel.NDoses(t0_list, duration)

        # Test at different time points
        t_first = jnp.array(0.5)  # During first dose
        t_between = jnp.array(12.0)  # Between doses
        t_second = jnp.array(24.5)  # During second dose

        result_first = ndoses_func(t_first)
        result_between = ndoses_func(t_between)
        result_second = ndoses_func(t_second)

        assert result_first > 0.5  # Should be active
        assert result_between < 0.1  # Should be inactive
        assert result_second > 0.5  # Should be active again

        # Test JAX JIT compatibility
        jitted_ndoses = jax.jit(ndoses_func)
        jitted_result = jitted_ndoses(t_first)
        np.testing.assert_allclose(result_first, jitted_result, rtol=1e-6)

    def test_static_zerofunc(self):
        """Test the static ZeroFunc function."""
        zero_func = EqxModel.ZeroFunc()

        result = zero_func(jnp.array(1.0))
        assert result == 0.0

        # Test JAX JIT compatibility
        jitted_zero = jax.jit(zero_func)
        jitted_result = jitted_zero(jnp.array(1.0))
        assert result == jitted_result

    def test_compile_forcing_functions_dict_spec(self, eqx_model_components):
        """Test compiling forcing functions from dictionary specifications."""
        # Test with dictionary specification for PerDose (which is a proper factory)
        model_tree = eqx_model_components

        model = EqxModel(
            parameters={"ka": 1.0, "ke": 0.1},
            forcing_functions={
                "dose": {"function": "PerDose", "args": (0.0, 1.0, 24.0), "kwargs": {"s": 10.0}}
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

        # Test the compiled function
        result = model.forcing_functions["dose"](jnp.array(0.5))
        assert isinstance(result, jnp.ndarray)

    def test_compile_forcing_functions_already_compiled(self, eqx_model_components):
        """Test that already compiled functions are left unchanged."""
        model_tree = eqx_model_components
        original_func = EqxModel.ZeroFunc()

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
            AttributeError, match="Forcing function 'NonExistentFunction' not found"
        ):
            model.compile_forcing_functions()

    def test_build_context(self, eqx_model_components):
        """Test context building for JAX compatibility."""
        model_tree = eqx_model_components

        model = EqxModel(
            parameters={"ka": 1.0, "ke": 0.1},
            forcing_functions={"dose": EqxModel.ZeroFunc()},
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

        model = EqxModel(
            parameters={"ka": 1.0, "ke": 0.1},
            forcing_functions={"dose": EqxModel.ZeroFunc()},
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

        model = EqxModel(
            parameters={"ka": 1.0, "ke": 0.1},
            forcing_functions={"dose": EqxModel.ZeroFunc()},
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

        model = EqxModel(
            parameters={"ka": 1.0, "ke": 0.1},
            forcing_functions={"dose": EqxModel.ZeroFunc()},
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
        """Test comprehensive JAX model running, updates, and ComputedModel interface."""
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
        """Test that run_model returns a ComputedModel."""
        model = JaxModel(simple_model_str)

        # Parameters ka and ke now have default values in the model,
        # but we can still update them if needed using the proper API
        model.update_constants(ka=1.0, ke=0.1)

        times = np.linspace(0, 2, 21)

        result = model.run_model(times)

        assert isinstance(result, ComputedModel)
        assert len(result.times) == len(times)
        assert result.states.shape[0] == len(times)

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
    """JAX compatibility tests integrated into the test suite."""

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

    def test_jax_jit_compatibility(self):
        """Test that JAX model functions are JIT-compatible."""
        import jax.numpy as jnp

        # Test OnOff function
        t = jnp.array(1.0)
        result = EqxModel.OnOff(t, 0.5, 2.0)
        jitted_onoff = jax.jit(EqxModel.OnOff)
        jitted_result = jitted_onoff(t, 0.5, 2.0)

        np.testing.assert_allclose(result, jitted_result, rtol=1e-6)

        # Test PerDose function
        perdose_func = EqxModel.PerDose(0.0, 1.0, 24.0)
        jitted_perdose = jax.jit(perdose_func)
        result1 = perdose_func(t)
        result2 = jitted_perdose(t)

        np.testing.assert_allclose(result1, result2, rtol=1e-6)

    def test_forcing_functions_jax_compatibility(self):
        """Test JAX forcing function implementations."""
        import jax.numpy as jnp

        from pymcsimmod.forcing.jax_functions import create_ndoses, create_onoff, create_perdose

        # Test create_onoff
        onoff_func = create_onoff(0.5, 2.0, 10.0)
        result = onoff_func(jnp.array(1.0))
        assert isinstance(result, jnp.ndarray)

        # Test create_perdose
        perdose_func = create_perdose(0.0, 1.0, 24.0, 10.0)
        result = perdose_func(jnp.array(12.0))
        assert isinstance(result, jnp.ndarray)

        # Test create_ndoses
        ndoses_func = create_ndoses([0.0, 24.0, 48.0], 1.0, 10.0)
        result = ndoses_func(jnp.array(25.0))
        assert isinstance(result, jnp.ndarray)


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
