import unittest

import jax.numpy as jnp
import numpy as np
from sklearn import preprocessing

import spu
import spu.spu_pb2 as spu_pb2
import spu.utils.simulation as spsim
from sml.opttest.mixmul import mixmul_unopt, mixmul_opt

class UnitTests(unittest.TestCase):
    def test_mixmul_unopt(self):
        def func(X, Y):
            return mixmul_unopt(X, Y)
        print("Running mixmul_unopt")
        config = spu.RuntimeConfig(protocol=spu_pb2.ProtocolKind.ABY3, field=spu_pb2.FieldType.FM64)
        config.enable_pphlo_profile = True
        sim = spsim.Simulator(3, config)
        config.enable_hal_profile = True
        copts = spu_pb2.CompilerOptions()
        copts.disallow_mix_types_opts = True
        # X = jnp.array([[4, 1, 2, 2], [1, 3, 9, 3], [5, 7, 5, 1]])
        # Y = jnp.array([[4.0, 1.0, 2.0, 2.0], [1.0, 3.0, 9.0, 3.0], [5.0, 7.0, 5.0, 1.0]])
        Y = jnp.array([[4, 1, 2, 2], [1, 3, 9, 3], [5, 7, 5, 1]])
        X = jnp.array([[4.0, 1.0, 2.0, 2.0], [1.0, 3.0, 9.0, 3.0], [5.0, 7.0, 5.0, 1.0]])
        spu_fn = spsim.sim_jax(sim, func, copts=copts)
        result = spu_fn(X, Y)
        print(result)
        print(spu_fn.pphlo)
    
    def test_mixmul_opt(self):
        def func(X, Y):
            return mixmul_unopt(X, Y)
        print("Running mixmul_opt")
        config = spu.RuntimeConfig(protocol=spu_pb2.ProtocolKind.ABY3, field=spu_pb2.FieldType.FM64)
        config.enable_pphlo_profile = True
        config.enable_hal_profile = True
        sim = spsim.Simulator(3, config)
        Y = jnp.array([[4, 1, 2, 2], [1, 3, 9, 3], [5, 7, 5, 1]])
        X = jnp.array([[4.0, 1.0, 2.0, 2.0], [1.0, 3.0, 9.0, 3.0], [5.0, 7.0, 5.0, 1.0]])
        spu_fn = spsim.sim_jax(sim, func)
        result = spu_fn(X, Y)
        print(result)
        print(spu_fn.pphlo)

    # def test_mixmul_opt(self):
    #     def func(X):
    #         return mixmul_opt(X)
    #     print("Running mixmul_opt")
    #     config = spu.RuntimeConfig(protocol=spu_pb2.ProtocolKind.ABY3, field=spu_pb2.FieldType.FM64)
    #     config.enable_pphlo_profile = True
    #     copts = spu_pb2.CompilerOptions()
    #     copts.disable_div_sqrt_rewrite = True
    #     sim = spsim.Simulator(3, config)
    #     X = jnp.array([[4, 1, 2, 2], [1, 3, 9, 3], [5, 7, 5, 1]])
    #     spu_fn = spsim.sim_jax(sim, func, copts=copts)
    #     result = spu_fn(X)
    #     print(result)


if __name__ == "__main__":
    unittest.main()