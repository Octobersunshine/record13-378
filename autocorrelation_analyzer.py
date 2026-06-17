import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.stattools import adfuller, kpss
from typing import List, Dict, Union, Optional, Tuple


class TimeSeriesAutocorrelation:
    def __init__(self, data: Union[List[float], np.ndarray, pd.Series],
                 alpha: float = 0.05,
                 nlags: Optional[int] = None):
        self.data = self._validate_data(data)
        self.alpha = alpha
        self.nlags = nlags if nlags is not None else min(int(len(self.data) / 4), 40)
        self._validate_params()

    def _validate_data(self, data: Union[List[float], np.ndarray, pd.Series]) -> np.ndarray:
        if isinstance(data, pd.Series):
            data = data.values
        elif isinstance(data, list):
            data = np.array(data, dtype=float)
        elif not isinstance(data, np.ndarray):
            raise TypeError("数据类型必须是 list、numpy.ndarray 或 pandas.Series")

        if data.ndim != 1:
            raise ValueError("时间序列数据必须是一维的")

        if len(data) < 10:
            raise ValueError("时间序列数据长度至少为 10 个观测值")

        if np.isnan(data).any():
            raise ValueError("时间序列数据包含 NaN 值，请先处理缺失值")

        if np.isinf(data).any():
            raise ValueError("时间序列数据包含无穷大值")

        return data.astype(float)

    def _validate_params(self):
        if self.alpha <= 0 or self.alpha >= 1:
            raise ValueError("显著性水平 alpha 必须在 (0, 1) 之间")

        if self.nlags < 1:
            raise ValueError("滞后阶数 nlags 必须至少为 1")

        if self.nlags >= len(self.data):
            raise ValueError("滞后阶数 nlags 必须小于数据长度")

    def compute_acf(self, adjusted: bool = False,
                    fft: bool = True) -> Dict[str, Union[np.ndarray, float, List]]:
        acf_values, confint = acf(
            self.data,
            nlags=self.nlags,
            alpha=self.alpha,
            adjusted=adjusted,
            fft=fft,
            missing='raise'
        )

        lags = np.arange(len(acf_values))
        lower_ci = confint[:, 0] - acf_values
        upper_ci = confint[:, 1] - acf_values

        significant = np.abs(acf_values) > np.abs(upper_ci)
        significant_lags = lags[significant].tolist()

        return {
            'acf_values': acf_values.tolist(),
            'lags': lags.tolist(),
            'confidence_interval': {
                'lower': lower_ci.tolist(),
                'upper': upper_ci.tolist(),
                'alpha': self.alpha
            },
            'significant_lags': significant_lags,
            'significant_count': len(significant_lags),
            'nlags': self.nlags
        }

    def compute_pacf(self, method: str = 'ywadjusted') -> Dict[str, Union[np.ndarray, float, List]]:
        pacf_values, confint = pacf(
            self.data,
            nlags=self.nlags,
            alpha=self.alpha,
            method=method
        )

        lags = np.arange(len(pacf_values))
        lower_ci = confint[:, 0] - pacf_values
        upper_ci = confint[:, 1] - pacf_values

        significant = np.abs(pacf_values) > np.abs(upper_ci)
        significant_lags = lags[significant].tolist()

        return {
            'pacf_values': pacf_values.tolist(),
            'lags': lags.tolist(),
            'confidence_interval': {
                'lower': lower_ci.tolist(),
                'upper': upper_ci.tolist(),
                'alpha': self.alpha
            },
            'significant_lags': significant_lags,
            'significant_count': len(significant_lags),
            'nlags': self.nlags,
            'method': method
        }

    def analyze(self) -> Dict[str, Union[Dict, List, float, int]]:
        acf_result = self.compute_acf()
        pacf_result = self.compute_pacf()

        summary = self._generate_summary(acf_result, pacf_result)

        return {
            'acf': acf_result,
            'pacf': pacf_result,
            'summary': summary,
            'data_info': {
                'length': len(self.data),
                'mean': float(np.mean(self.data)),
                'std': float(np.std(self.data)),
                'min': float(np.min(self.data)),
                'max': float(np.max(self.data))
            }
        }

    def _generate_summary(self, acf_result: Dict, pacf_result: Dict) -> Dict:
        acf_vals = np.array(acf_result['acf_values'])
        pacf_vals = np.array(pacf_result['pacf_values'])

        acf_significant = acf_result['significant_lags']
        pacf_significant = pacf_result['significant_lags']

        acf_decay_pattern = self._detect_decay_pattern(acf_vals[1:])
        pacf_cutoff = self._detect_cutoff_lag(pacf_vals[1:])

        ar_suggestion = self._suggest_ar_order(pacf_significant)
        ma_suggestion = self._suggest_ma_order(acf_significant)

        return {
            'acf_first_significant_lag': acf_significant[0] if acf_significant else None,
            'pacf_first_significant_lag': pacf_significant[0] if pacf_significant else None,
            'acf_significant_count': len(acf_significant),
            'pacf_significant_count': len(pacf_significant),
            'acf_decay_pattern': acf_decay_pattern,
            'pacf_cutoff_lag': pacf_cutoff,
            'suggested_ar_order': ar_suggestion,
            'suggested_ma_order': ma_suggestion,
            'interpretation': self._generate_interpretation(
                acf_decay_pattern, pacf_cutoff, ar_suggestion, ma_suggestion
            )
        }

    def _detect_decay_pattern(self, values: np.ndarray) -> str:
        if len(values) < 5:
            return 'insufficient_data'

        abs_values = np.abs(values)
        n = len(abs_values)
        half = n // 2

        first_half_mean = np.mean(abs_values[:half])
        second_half_mean = np.mean(abs_values[half:])

        if first_half_mean < 0.1:
            return 'very_weak'

        decay_ratio = second_half_mean / first_half_mean if first_half_mean > 0 else 1

        if decay_ratio < 0.3:
            if np.all(values[:5] > 0) or np.all(values[:5] < 0):
                return 'exponential_decay'
            else:
                return 'damped_oscillation'
        elif decay_ratio < 0.7:
            return 'gradual_decay'
        else:
            return 'slow_decay_or_no_decay'

    def _detect_cutoff_lag(self, values: np.ndarray) -> Optional[int]:
        abs_values = np.abs(values)
        threshold = 1.96 / np.sqrt(len(self.data))

        for i in range(len(abs_values) - 2):
            if abs_values[i] > threshold and np.all(abs_values[i+1:i+3] < threshold):
                return i + 1

        return None

    def _suggest_ar_order(self, significant_pacf_lags: List[int]) -> int:
        if not significant_pacf_lags:
            return 0

        max_lag = max(significant_pacf_lags)
        if max_lag <= 1:
            return 1
        elif max_lag <= 4:
            return min(max_lag, 3)
        else:
            return min(max_lag, 5)

    def _suggest_ma_order(self, significant_acf_lags: List[int]) -> int:
        if not significant_acf_lags:
            return 0

        positive_lags = [l for l in significant_acf_lags if l > 0]
        if not positive_lags:
            return 0

        max_lag = max(positive_lags)
        if max_lag <= 1:
            return 1
        elif max_lag <= 4:
            return min(max_lag, 3)
        else:
            return min(max_lag, 5)

    def _generate_interpretation(self, acf_decay: str, pacf_cutoff: Optional[int],
                                 ar_order: int, ma_order: int) -> str:
        interpretations = []

        if acf_decay in ['exponential_decay', 'damped_oscillation', 'gradual_decay']:
            interpretations.append("ACF 呈现衰减模式，提示序列存在自相关结构")

        if pacf_cutoff is not None:
            interpretations.append(f"PACF 在滞后 {pacf_cutoff} 阶后截尾，提示可能为 AR({pacf_cutoff}) 过程")

        if ar_order > 0 and ma_order == 0:
            interpretations.append(f"建议考虑 AR({ar_order}) 模型")
        elif ma_order > 0 and ar_order == 0:
            interpretations.append(f"建议考虑 MA({ma_order}) 模型")
        elif ar_order > 0 and ma_order > 0:
            interpretations.append(f"建议考虑 ARMA({ar_order}, {ma_order}) 模型")
        else:
            interpretations.append("序列可能为白噪声，无显著自相关结构")

        return "；".join(interpretations)

    def stationarity_test(self) -> Dict[str, Dict]:
        adf_result = self._adf_test()
        kpss_result = self._kpss_test()

        return {
            'adf_test': adf_result,
            'kpss_test': kpss_result
        }

    def _adf_test(self) -> Dict[str, Union[float, str, bool]]:
        result = adfuller(self.data, autolag='AIC')

        return {
            'test_statistic': float(result[0]),
            'p_value': float(result[1]),
            'used_lag': int(result[2]),
            'n_obs': int(result[3]),
            'critical_values': {k: float(v) for k, v in result[4].items()},
            'is_stationary': bool(result[1] < self.alpha),
            'ic_best': float(result[5])
        }

    def _kpss_test(self) -> Dict[str, Union[float, str, bool]]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = kpss(self.data, regression='c', nlags='auto')

            return {
                'test_statistic': float(result[0]),
                'p_value': float(result[1]),
                'used_lag': int(result[2]),
                'critical_values': {k: float(v) for k, v in result[3].items()},
                'is_stationary': bool(result[1] >= self.alpha)
            }
        except Exception:
            return {
                'test_statistic': None,
                'p_value': None,
                'used_lag': None,
                'critical_values': {},
                'is_stationary': None,
                'error': 'KPSS 测试执行失败'
            }
