import json
import csv
import io
from typing import List, Dict, Union, Optional
from autocorrelation_analyzer import TimeSeriesAutocorrelation


class AutocorrelationService:
    def __init__(self, default_alpha: float = 0.05, default_nlags: Optional[int] = None):
        self.default_alpha = default_alpha
        self.default_nlags = default_nlags

    def analyze_from_list(self, data: List[float],
                          alpha: Optional[float] = None,
                          nlags: Optional[int] = None,
                          include_stationarity: bool = True) -> Dict:
        alpha = alpha if alpha is not None else self.default_alpha
        nlags = nlags if nlags is not None else self.default_nlags

        analyzer = TimeSeriesAutocorrelation(data, alpha=alpha, nlags=nlags)

        result = analyzer.analyze()

        if include_stationarity:
            result['stationarity'] = analyzer.stationarity_test()

        return result

    def analyze_from_json(self, json_input: str,
                          data_key: str = 'data',
                          alpha: Optional[float] = None,
                          nlags: Optional[int] = None,
                          include_stationarity: bool = True) -> Dict:
        try:
            parsed = json.loads(json_input)
        except json.JSONDecodeError as e:
            return {'error': f'JSON 解析错误: {str(e)}', 'success': False}

        if data_key not in parsed:
            return {'error': f'JSON 中未找到键: {data_key}', 'success': False}

        data = parsed[data_key]

        if not isinstance(data, list):
            return {'error': '数据必须是数组格式', 'success': False}

        alpha = alpha if alpha is not None else parsed.get('alpha', self.default_alpha)
        nlags = nlags if nlags is not None else parsed.get('nlags', self.default_nlags)

        result = self.analyze_from_list(data, alpha=alpha, nlags=nlags,
                                        include_stationarity=include_stationarity)
        result['success'] = True

        return result

    def analyze_from_csv(self, csv_input: str,
                         column: Union[str, int] = 0,
                         has_header: bool = True,
                         alpha: Optional[float] = None,
                         nlags: Optional[int] = None,
                         include_stationarity: bool = True) -> Dict:
        try:
            data = self._parse_csv(csv_input, column, has_header)
        except Exception as e:
            return {'error': f'CSV 解析错误: {str(e)}', 'success': False}

        result = self.analyze_from_list(data, alpha=alpha, nlags=nlags,
                                        include_stationarity=include_stationarity)
        result['success'] = True

        return result

    def _parse_csv(self, csv_input: str, column: Union[str, int],
                   has_header: bool) -> List[float]:
        data = []

        if hasattr(csv_input, 'read'):
            reader = csv.reader(csv_input)
        else:
            reader = csv.reader(io.StringIO(csv_input))

        rows = list(reader)

        if not rows:
            raise ValueError("CSV 文件为空")

        start_idx = 1 if has_header else 0

        if isinstance(column, str):
            if not has_header:
                raise ValueError("指定列名时必须有表头")
            header = rows[0]
            if column not in header:
                raise ValueError(f"列 '{column}' 不在表头中")
            col_idx = header.index(column)
        else:
            col_idx = column

        for row in rows[start_idx:]:
            if col_idx >= len(row):
                raise ValueError(f"列索引 {col_idx} 超出范围")
            try:
                value = float(row[col_idx].strip())
                data.append(value)
            except ValueError:
                raise ValueError(f"无法将 '{row[col_idx]}' 转换为数值")

        return data

    def analyze_from_file(self, file_path: str,
                          column: Union[str, int] = 0,
                          has_header: bool = True,
                          alpha: Optional[float] = None,
                          nlags: Optional[int] = None,
                          include_stationarity: bool = True) -> Dict:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except FileNotFoundError:
            return {'error': f'文件不存在: {file_path}', 'success': False}
        except Exception as e:
            return {'error': f'读取文件错误: {str(e)}', 'success': False}

        if file_path.endswith('.json'):
            return self.analyze_from_json(content, alpha=alpha, nlags=nlags,
                                          include_stationarity=include_stationarity)
        elif file_path.endswith('.csv'):
            return self.analyze_from_csv(content, column=column, has_header=has_header,
                                         alpha=alpha, nlags=nlags,
                                         include_stationarity=include_stationarity)
        else:
            try:
                data = [float(line.strip()) for line in content.splitlines()
                        if line.strip()]
                result = self.analyze_from_list(data, alpha=alpha, nlags=nlags,
                                                include_stationarity=include_stationarity)
                result['success'] = True
                return result
            except ValueError as e:
                return {'error': f'解析文件错误: {str(e)}', 'success': False}

    def to_json(self, result: Dict, indent: int = 2) -> str:
        return json.dumps(result, ensure_ascii=False, indent=indent)

    def to_pretty_text(self, result: Dict) -> str:
        if not result.get('success', True):
            return f"错误: {result.get('error', '未知错误')}"

        lines = []
        lines.append("=" * 60)
        lines.append("时间序列自相关分析报告")
        lines.append("=" * 60)

        if 'data_info' in result:
            di = result['data_info']
            lines.append("\n【数据信息】")
            lines.append(f"  数据长度: {di['length']}")
            lines.append(f"  均值: {di['mean']:.4f}")
            lines.append(f"  标准差: {di['std']:.4f}")
            lines.append(f"  最小值: {di['min']:.4f}")
            lines.append(f"  最大值: {di['max']:.4f}")

        if 'acf' in result:
            acf = result['acf']
            lines.append("\n【自相关函数 (ACF)】")
            lines.append(f"  最大滞后阶数: {acf['nlags']}")
            lines.append(f"  显著滞后阶数: {acf['significant_lags']}")
            lines.append(f"  显著阶数数量: {acf['significant_count']}")
            lines.append(f"  前 10 阶 ACF 值:")
            for i in range(min(11, len(acf['acf_values']))):
                val = acf['acf_values'][i]
                sig = "*" if i in acf['significant_lags'] else " "
                lines.append(f"    滞后 {i:2d}: {val:+.4f} {sig}")

        if 'pacf' in result:
            pacf = result['pacf']
            lines.append("\n【偏自相关函数 (PACF)】")
            lines.append(f"  最大滞后阶数: {pacf['nlags']}")
            lines.append(f"  显著滞后阶数: {pacf['significant_lags']}")
            lines.append(f"  显著阶数数量: {pacf['significant_count']}")
            lines.append(f"  前 10 阶 PACF 值:")
            for i in range(min(11, len(pacf['pacf_values']))):
                val = pacf['pacf_values'][i]
                sig = "*" if i in pacf['significant_lags'] else " "
                lines.append(f"    滞后 {i:2d}: {val:+.4f} {sig}")

        if 'summary' in result:
            s = result['summary']
            lines.append("\n【分析摘要】")
            lines.append(f"  ACF 衰减模式: {s['acf_decay_pattern']}")
            lines.append(f"  PACF 截尾阶数: {s['pacf_cutoff_lag']}")
            lines.append(f"  建议 AR 阶数: {s['suggested_ar_order']}")
            lines.append(f"  建议 MA 阶数: {s['suggested_ma_order']}")
            lines.append(f"  模型解读: {s['interpretation']}")

        if 'stationarity' in result:
            lines.append("\n【平稳性检验】")
            if 'adf_test' in result['stationarity']:
                adf = result['stationarity']['adf_test']
                lines.append("  ADF 检验:")
                lines.append(f"    检验统计量: {adf['test_statistic']:.4f}")
                lines.append(f"    p 值: {adf['p_value']:.4f}")
                lines.append(f"    是否平稳: {'是' if adf['is_stationary'] else '否'}")
                lines.append(f"    临界值: {adf['critical_values']}")

            if 'kpss_test' in result['stationarity']:
                kpss = result['stationarity']['kpss_test']
                lines.append("  KPSS 检验:")
                if kpss.get('error'):
                    lines.append(f"    错误: {kpss['error']}")
                else:
                    lines.append(f"    检验统计量: {kpss['test_statistic']:.4f}")
                    lines.append(f"    p 值: {kpss['p_value']:.4f}")
                    lines.append(f"    是否平稳: {'是' if kpss['is_stationary'] else '否'}")
                    lines.append(f"    临界值: {kpss['critical_values']}")

        lines.append("\n" + "=" * 60)
        lines.append("注: 标有 * 的滞后阶数表示在显著性水平下显著")
        lines.append("=" * 60)

        return "\n".join(lines)
