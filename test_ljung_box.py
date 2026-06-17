import numpy as np
from autocorrelation_service import AutocorrelationService

np.random.seed(42)
service = AutocorrelationService()

print("=" * 60)
print("测试 1: 白噪声序列 (应为白噪声)")
print("=" * 60)
white_noise = np.random.normal(0, 1, 200).tolist()
result = service.analyze_from_list(white_noise, include_stationarity=False)
lb = result['ljung_box']
print(f"is_white_noise: {lb['is_white_noise']}")
print(f"conclusion: {lb['conclusion']}")
if lb.get('overall'):
    print(f"overall p-value: {lb['overall']['p_value']:.4f}")
assert lb['is_white_noise'], "白噪声序列应被识别为白噪声"
print("通过 ✓")
print()

print("=" * 60)
print("测试 2: AR(1) 序列 (不应为白噪声)")
print("=" * 60)
n = 200
ar1 = np.zeros(n)
for i in range(1, n):
    ar1[i] = 0.7 * ar1[i-1] + np.random.normal(0, 1)
result2 = service.analyze_from_list(ar1.tolist(), include_stationarity=False)
lb2 = result2['ljung_box']
print(f"is_white_noise: {lb2['is_white_noise']}")
print(f"significant_lags: {lb2.get('significant_lags', [])[:10]}")
print(f"conclusion: {lb2['conclusion']}")
assert not lb2['is_white_noise'], "AR(1) 序列不应被识别为白噪声"
print("通过 ✓")
print()

print("=" * 60)
print("测试 3: 指定自定义滞后阶数列表")
print("=" * 60)
result3 = service.analyze_from_list(
    ar1.tolist(),
    include_stationarity=False,
    ljung_box_lags=[4, 8, 12, 16, 20]
)
lb3 = result3['ljung_box']
print(f"tested_lags: {lb3['tested_lags']}")
print(f"results_by_lag 数量: {len(lb3['results_by_lag'])}")
for r in lb3['results_by_lag']:
    print(f"  滞后 {r['lag']:2d}: Q={r['ljung_box_statistic']:.4f}, p={r['p_value']:.4f}")
assert lb3['tested_lags'] == [4, 8, 12, 16, 20], "测试阶数应匹配"
print("通过 ✓")
print()

print("=" * 60)
print("测试 4: Box-Pierce 检验")
print("=" * 60)
result4 = service.analyze_from_list(
    ar1.tolist(),
    include_stationarity=False,
    ljung_box_lags=10,
    ljung_box_boxpierce=True
)
lb4 = result4['ljung_box']
print(f"has overall box_pierce: {'box_pierce_statistic' in lb4.get('overall', {})}")
if 'overall' in lb4:
    ov = lb4['overall']
    print(f"Ljung-Box Q: {ov['ljung_box_statistic']:.4f}")
    print(f"Box-Pierce Q: {ov.get('box_pierce_statistic', 'N/A')}")
print("通过 ✓")
print()

print("=" * 60)
print("测试 5: 不执行 Ljung-Box 检验")
print("=" * 60)
result5 = service.analyze_from_list(
    ar1.tolist(),
    include_stationarity=False,
    include_ljung_box=False
)
print(f"'ljung_box' in result: {'ljung_box' in result5}")
assert 'ljung_box' not in result5, "不应包含 ljung_box 键"
print("通过 ✓")
print()

print("=" * 60)
print("测试 6: CLI 输出展示 (pretty text)")
print("=" * 60)
print(service.to_pretty_text(result))
print()

print("=" * 60)
print("所有 Ljung-Box 检验测试通过！ ✓✓✓")
print("=" * 60)
