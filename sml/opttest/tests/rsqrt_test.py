import unittest

import jax.numpy as jnp
import numpy as np
from sklearn import preprocessing

import spu
import spu.spu_pb2 as spu_pb2
import spu.utils.simulation as spsim
from sml.opttest.rsqrt import rsqrt_unopt, rsqrt_opt

class UnitTests(unittest.TestCase):
    def test_rsqrt_unopt(self):
        def func(X):
            return rsqrt_unopt(X)
        print("Running rsqrt_unopt")
        config = spu.RuntimeConfig(protocol=spu_pb2.ProtocolKind.ABY3, field=spu_pb2.FieldType.FM64)
        config.enable_pphlo_profile = True
        sim = spsim.Simulator(3, config)
        copts = spu_pb2.CompilerOptions()
        copts.disable_div_sqrt_rewrite = True
        X = jnp.array([[4, 1, 2, 2], [1, 3, 9, 3], [5, 7, 5, 1]])
        spu_fn = spsim.sim_jax(sim, func, copts=copts)
        result = spu_fn(X)
        print(result)
    
    def test_rsqrt_opt(self):
        def func(X):
            return rsqrt_unopt(X)
        print("Running rsqrt_opt")
        config = spu.RuntimeConfig(protocol=spu_pb2.ProtocolKind.ABY3, field=spu_pb2.FieldType.FM64)
        config.enable_pphlo_profile = True
        copts = spu_pb2.CompilerOptions()
        copts.disable_div_sqrt_rewrite = False
        sim = spsim.Simulator(3, config)
        X = jnp.array([[4, 1, 2, 2], [1, 3, 9, 3], [5, 7, 5, 1]])
        spu_fn = spsim.sim_jax(sim, func, copts=copts)
        result = spu_fn(X)
        print(result)

    # def test_rsqrt_opt(self):
    #     def func(X):
    #         return rsqrt_opt(X)
    #     print("Running rsqrt_opt")
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