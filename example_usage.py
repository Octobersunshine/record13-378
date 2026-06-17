import numpy as np
from autocorrelation_service import AutocorrelationService


def example_1_basic_list():
    print("=" * 60)
    print("示例 1: 使用列表数据进行基本分析")
    print("=" * 60)

    np.random.seed(42)
    n = 200
    ar_data = np.zeros(n)
    for i in range(1, n):
        ar_data[i] = 0.7 * ar_data[i-1] + np.random.normal(0, 1)

    service = AutocorrelationService()
    result = service.analyze_from_list(ar_data.tolist(), alpha=0.05)

    print(service.to_pretty_text(result))


def example_2_json_input():
    print("\n" + "=" * 60)
    print("示例 2: 使用 JSON 格式输入")
    print("=" * 60)

    np.random.seed(123)
    n = 150
    ma_data = np.zeros(n)
    error = np.random.normal(0, 1, n)
    for i in range(1, n):
        ma_data[i] = error[i] + 0.5 * error[i-1]

    import json
    json_data = json.dumps({
        "data": ma_data.tolist(),
        "alpha": 0.05,
        "nlags": 20
    })

    service = AutocorrelationService()
    result = service.analyze_from_json(json_data)

    print(service.to_pretty_text(result))


def example_3_white_noise():
    print("\n" + "=" * 60)
    print("示例 3: 白噪声序列（应无显著自相关）")
    print("=" * 60)

    np.random.seed(789)
    white_noise = np.random.normal(0, 1, 200)

    service = AutocorrelationService(default_alpha=0.05)
    result = service.analyze_from_list(white_noise.tolist())

    print(service.to_pretty_text(result))


def example_4_arma_process():
    print("\n" + "=" * 60)
    print("示例 4: ARMA(2,1) 过程")
    print("=" * 60)

    np.random.seed(456)
    n = 300
    arma_data = np.zeros(n)
    error = np.random.normal(0, 1, n)
    for i in range(2, n):
        arma_data[i] = 0.6 * arma_data[i-1] - 0.3 * arma_data[i-2] + \
                       error[i] + 0.4 * error[i-1]

    service = AutocorrelationService()
    result = service.analyze_from_list(arma_data.tolist(), nlags=25)

    print(service.to_pretty_text(result))


def example_5_json_output():
    print("\n" + "=" * 60)
    print("示例 5: 输出 JSON 格式结果")
    print("=" * 60)

    np.random.seed(999)
    n = 100
    ar1_data = np.zeros(n)
    for i in range(1, n):
        ar1_data[i] = -0.5 * ar1_data[i-1] + np.random.normal(0, 1)

    service = AutocorrelationService()
    result = service.analyze_from_list(
        ar1_data.tolist(),
        include_stationarity=False
    )

    json_output = service.to_json(result)
    print(json_output[:800] + "...")
    print("\n(输出已截断，完整 JSON 包含所有分析数据)")


def example_6_custom_params():
    print("\n" + "=" * 60)
    print("示例 6: 自定义参数的高级分析")
    print("=" * 60)

    np.random.seed(321)
    n = 250
    data = np.zeros(n)
    for i in range(1, n):
        data[i] = 0.8 * data[i-1] + np.random.normal(0, 1)

    service = AutocorrelationService(default_alpha=0.01)
    result = service.analyze_from_list(
        data.tolist(),
        alpha=0.01,
        nlags=30,
        include_stationarity=True
    )

    print(service.to_pretty_text(result))


if __name__ == "__main__":
    example_1_basic_list()
    example_2_json_input()
    example_3_white_noise()
    example_4_arma_process()
    example_5_json_output()
    example_6_custom_params()
