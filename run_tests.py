import sys
sys.path.insert(0, '.')

from SimplCore.tests.verify_eqp import (
    test_1_primitive_pool,
    test_2_composition_router,
    test_3_eqp_linear,
    test_4_model_integration,
    test_5_parameter_efficiency,
    test_6_weight_approximation,
    test_7_incremental_scaling,
    test_8_end_to_end,
    benchmark_ab_comparison,
)

print("=" * 60)
print("EQP 等效参数技术 — 完整验证套件")
print("=" * 60)

results = []
results.append(("PrimitivePool", test_1_primitive_pool()))
results.append(("CompositionRouter", test_2_composition_router()))
results.append(("EQPLinear", test_3_eqp_linear()))
results.append(("ModelIntegration", test_4_model_integration()))
results.append(("ParameterEfficiency", test_5_parameter_efficiency()))
results.append(("WeightApproximation", test_6_weight_approximation()))
results.append(("IncrementalScaling", test_7_incremental_scaling()))
results.append(("EndToEnd", test_8_end_to_end()))

benchmark_ab_comparison()

print("\n" + "=" * 60)
print("EQP 验证汇总")
print("=" * 60)
all_pass = True
for name, passed in results:
    status = "✅" if passed else "❌"
    print(f"  {status} {name}")
    all_pass = all_pass and passed

print(f"\n  {'ALL PASS' if all_pass else 'SOME FAILED'}")
print("=" * 60)

sys.exit(0 if all_pass else 1)
