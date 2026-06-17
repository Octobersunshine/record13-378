import numpy as np
from autocorrelation_service import AutocorrelationService

np.random.seed(99)
n = 20
data = np.random.normal(0, 1, n).tolist()
service = AutocorrelationService()

print('=== 测试 1: nlags = N (正好等于样本量，应限制为 N-1) ===')
result1 = service.analyze_from_list(data, nlags=n, include_stationarity=False)
print(f'N={n}, 请求 nlags={n}')
print(f'ACF nlags: {result1["acf"]["nlags"]}')
print(f'PACF nlags: {result1["pacf"]["nlags"]}')
print(f'被截断: {result1.get("nlags_info", {}).get("was_truncated", False)}')
assert result1["acf"]["nlags"] == n - 1, "ACF nlags 应等于 N-1"
assert result1.get("nlags_info", {}).get("was_truncated", False), "应标记为已截断"
print('通过 ✓')
print()

print('=== 测试 2: nlags = N-1 (边界值，应正常使用) ===')
result2 = service.analyze_from_list(data, nlags=n-1, include_stationarity=False)
print(f'N={n}, 请求 nlags={n-1}')
print(f'ACF nlags: {result2["acf"]["nlags"]}')
print(f'PACF nlags: {result2["pacf"]["nlags"]}')
print(f'被截断: {result2.get("nlags_info", {}).get("was_truncated", False)}')
assert result2["acf"]["nlags"] == n - 1, "ACF nlags 应等于 N-1"
print('通过 ✓')
print()

print('=== 测试 3: nlags = 0 (应自动调整为 1) ===')
result3 = service.analyze_from_list(data, nlags=0, include_stationarity=False)
print(f'请求 nlags=0')
print(f'ACF nlags: {result3["acf"]["nlags"]}')
print(f'PACF nlags: {result3["pacf"]["nlags"]}')
print(f'被截断: {result3.get("nlags_info", {}).get("was_truncated", False)}')
assert result3["acf"]["nlags"] == 1, "ACF nlags 应自动调整为 1"
assert result3.get("nlags_info", {}).get("was_truncated", False), "应标记为已截断"
print('通过 ✓')
print()

print('=== 测试 4: nlags 为负数 (应自动调整为 1) ===')
result4 = service.analyze_from_list(data, nlags=-5, include_stationarity=False)
print(f'请求 nlags=-5')
print(f'ACF nlags: {result4["acf"]["nlags"]}')
print(f'PACF nlags: {result4["pacf"]["nlags"]}')
print(f'被截断: {result4.get("nlags_info", {}).get("was_truncated", False)}')
assert result4["acf"]["nlags"] == 1, "ACF nlags 应自动调整为 1"
assert result4.get("nlags_info", {}).get("was_truncated", False), "应标记为已截断"
print('通过 ✓')
print()

print('=== 测试 5: 默认 nlags (不指定，应正常) ===')
result5 = service.analyze_from_list(data, include_stationarity=False)
print(f'未指定 nlags')
print(f'ACF nlags: {result5["acf"]["nlags"]}')
print(f'PACF nlags: {result5["pacf"]["nlags"]}')
print(f'被截断: {result5.get("nlags_info", {}).get("was_truncated", False)}')
expected_default = min(int(n / 4), 40, n - 1)
assert result5["acf"]["nlags"] == expected_default, f"默认 nlags 应为 {expected_default}"
print('通过 ✓')
print()

print('=== 测试 6: PACF 方法限制 (ywadjusted 要求 nlags <= N//2 - 1) ===')
np.random.seed(42)
n_small = 12
data_small = np.random.normal(0, 1, n_small).tolist()
result6 = service.analyze_from_list(data_small, nlags=10, include_stationarity=False)
print(f'N={n_small}, 请求 nlags=10')
print(f'ACF nlags: {result6["acf"]["nlags"]} (请求 10，N-1={n_small-1})')
print(f'PACF nlags: {result6["pacf"]["nlags"]} (ywadjusted 限制 N//2-1={n_small//2-1})')
assert result6["acf"]["nlags"] == 10, "ACF nlags 应使用请求的 10 (小于 N-1)"
assert result6["pacf"]["nlags"] == n_small // 2 - 1, f"PACF nlags 应等于 {n_small//2-1}"
assert result6.get("nlags_info", {}).get("was_truncated", False), "PACF 被截断，整体应标记为已截断"
print('通过 ✓')
print()

print('=' * 60)
print('所有边界情况测试通过！ ✓')
print('=' * 60)

print()
print('=== 测试 7: nlags 在合理范围内 (不应被截断) ===')
n = 50
data_ok = np.random.normal(0, 1, n).tolist()
result7 = service.analyze_from_list(data_ok, nlags=10, include_stationarity=False)
print(f'N={n}, 请求 nlags=10')
print(f'ACF nlags: {result7["acf"]["nlags"]}')
print(f'PACF nlags: {result7["pacf"]["nlags"]}')
print(f'被截断: {result7.get("nlags_info", {}).get("was_truncated", False)}')
assert result7["acf"]["nlags"] == 10, "ACF nlags 应为 10"
assert result7["pacf"]["nlags"] == 10, f"PACF nlags 应为 10 (N//2-1={n//2-1})"
assert "nlags_info" not in result7, "不应有 nlags_info"
print('通过 ✓')

print()
print('=' * 60)
print('全部测试完成，所有断言通过！ ✓✓✓')
print('=' * 60)
