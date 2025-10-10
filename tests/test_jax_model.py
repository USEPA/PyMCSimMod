"""Tests for JAX model implementations."""

import pytest
import numpy as np

pytest.importorskip("jax")
pytest.importorskip("equinox") 
pytest.importorskip("diffrax")

import jax
import jax.numpy as jnp
import equinox as eqx

from pymcsimmod.models.jax_model import EqxModel, JaxModel
from pymcsimmod.models.computed import ComputedModel


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
        from pymcsimmod.parser import ModelParser
        from pymcsimmod.lexer import ModelLexer
        
        # Create a minimal model tree mock
        class MockModelTree:
            def __init__(self):
                self.dynamic_calcs = {}
                self.calc_outputs = {
                    'A_out': MockExpr("A")  # A_out = A
                }
                self.dynamics = {
                    'A': MockExpr("dose * ka - A * ke")
                }
        
        class MockExpr:
            def __init__(self, expr_str):
                self.expr_str = expr_str
            
            def evaluate(self, context, approach):
                # Simple mock evaluation for testing
                if self.expr_str == "dose * ka - A * ke":
                    return context.get('dose', 0.0) * context.get('ka', 1.0) - context.get('A', 0.0) * context.get('ke', 0.1)
                elif self.expr_str == "A":
                    return context.get('A', 0.0)
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
        t_on = jnp.array(0.5)   # During first dose
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
        t_first = jnp.array(0.5)   # During first dose
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
            parameters={'ka': 1.0, 'ke': 0.1},
            forcing_functions={
                'dose': {
                    'function': 'PerDose',
                    'args': (0.0, 1.0, 24.0),
                    'kwargs': {'s': 10.0}
                }
            },
            Y0={'A': 0.0},
            events=[],
            model_tree=model_tree,
            state_names=('A',),
            output_names=('A_out',)
        )
        
        model.compile_forcing_functions()
        
        # Check that the function was compiled
        assert callable(model.forcing_functions['dose'])
        
        # Test the compiled function
        result = model.forcing_functions['dose'](jnp.array(0.5))
        assert isinstance(result, jnp.ndarray)

    def test_compile_forcing_functions_already_compiled(self, eqx_model_components):
        """Test that already compiled functions are left unchanged."""
        model_tree = eqx_model_components
        original_func = EqxModel.ZeroFunc()
        
        model = EqxModel(
            parameters={'ka': 1.0, 'ke': 0.1},
            forcing_functions={'dose': original_func},
            Y0={'A': 0.0},
            events=[],
            model_tree=model_tree,
            state_names=('A',),
            output_names=('A_out',)
        )
        
        model.compile_forcing_functions()
        
        # Should be the same function
        assert model.forcing_functions['dose'] is original_func

    def test_compile_forcing_functions_invalid_spec(self, eqx_model_components):
        """Test error handling for invalid forcing function specifications."""
        model_tree = eqx_model_components
        
        model = EqxModel(
            parameters={'ka': 1.0, 'ke': 0.1},
            forcing_functions={
                'dose': {
                    'function': 'NonExistentFunction',
                    'args': (),
                    'kwargs': {}
                }
            },
            Y0={'A': 0.0},
            events=[],
            model_tree=model_tree,
            state_names=('A',),
            output_names=('A_out',)
        )
        
        with pytest.raises(AttributeError, match="Forcing function 'NonExistentFunction' not found"):
            model.compile_forcing_functions()

    def test_build_context(self, eqx_model_components):
        """Test context building for JAX compatibility."""
        model_tree = eqx_model_components
        
        model = EqxModel(
            parameters={'ka': 1.0, 'ke': 0.1},
            forcing_functions={'dose': EqxModel.ZeroFunc()},
            Y0={'A': 0.0},
            events=[],
            model_tree=model_tree,
            state_names=('A',),
            output_names=('A_out',)
        )
        
        state_vals = jnp.array([1.0])  # A = 1.0
        t = 0.5
        
        context = model.build_context(state_vals, t)
        
        # Check basic structure
        assert 'A' in context
        assert context['A'] == 1.0
        assert 'ka' in context
        assert 'ke' in context
        assert 'dose' in context

    def test_model_jit_compilation(self, eqx_model_components):
        """Test that the model function can be JIT compiled."""
        model_tree = eqx_model_components
        
        model = EqxModel(
            parameters={'ka': 1.0, 'ke': 0.1},
            forcing_functions={'dose': EqxModel.ZeroFunc()},
            Y0={'A': 0.0},
            events=[],
            model_tree=model_tree,
            state_names=('A',),
            output_names=('A_out',)
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
            parameters={'ka': 1.0, 'ke': 0.1},
            forcing_functions={'dose': EqxModel.ZeroFunc()},
            Y0={'A': 0.0},
            events=[{'type': 'test'}],  # Add a dummy event
            model_tree=model_tree,
            state_names=('A',),
            output_names=('A_out',)
        )
        
        times = np.linspace(0, 10, 101)
        
        with pytest.raises(NotImplementedError, match="Discrete events are not yet supported"):
            model.run_model(times)

    def test_run_model_success(self, eqx_model_components):
        """Test successful model run without events."""
        model_tree = eqx_model_components
        
        model = EqxModel(
            parameters={'ka': 1.0, 'ke': 0.1},
            forcing_functions={'dose': EqxModel.ZeroFunc()},
            Y0={'A': 0.0},
            events=[],
            model_tree=model_tree,
            state_names=('A',),
            output_names=('A_out',)
        )
        
        times = np.linspace(0, 2, 21)
        
        sol, calc_outputs, input_functions = model.run_model(times)
        
        # Check solution structure
        assert hasattr(sol, 'ts')
        assert hasattr(sol, 'ys')
        assert len(sol.ts) == len(times)
        assert sol.ys.shape == (len(times), 1)  # 1 state variable
        
        # Check outputs
        assert calc_outputs.shape == (len(times), 1)  # 1 output
        
        # Check input functions
        assert isinstance(input_functions, dict)
        assert 'dose' in input_functions


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

    def test_jax_model_creation(self, simple_model_str):
        """Test JaxModel creation from string."""
        model = JaxModel(simple_model_str)
        
        assert hasattr(model, 'state_names')
        assert hasattr(model, 'parameters')
        assert hasattr(model, 'forcing_functions')
        assert hasattr(model, 'Y0')

    def test_model_method_not_implemented(self, simple_model_str):
        """Test that model method raises NotImplementedError."""
        model = JaxModel(simple_model_str)
        
        with pytest.raises(NotImplementedError, match="This method should be implemented in equinox module class"):
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


class TestJaxCompatibility:
    """JAX compatibility tests integrated into the test suite."""

    def test_jax_dependencies_available(self):
        """Test that JAX dependencies are available."""
        import jax
        import equinox
        import diffrax
        
        # If we get here, dependencies are available
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
        from pymcsimmod.forcing.jax_functions import create_onoff, create_perdose, create_ndoses
        import jax.numpy as jnp
        
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