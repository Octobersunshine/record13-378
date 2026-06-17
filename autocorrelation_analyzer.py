import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from typing import List, Dict, Union, Optional, Tuple


class TimeSeriesAutocorrelation:
    def __init__(self, data: Union[List[float], np.ndarray, pd.Series],
                 alpha: float = 0.05,
                 nlags: Optional[int] = None):
        self.data = self._validate_data(data)
        self.alpha = alpha
        n_samples = len(self.data)
        max_allowed_nlags = n_samples - 1

        if nlags is None:
            self.nlags = min(int(n_samples / 4), 40, max_allowed_nlags)
            self._nlags_truncated = False
            self._requested_nlags = None
        else:
            self._requested_nlags = nlags
            if nlags > max_allowed_nlags:
                self.nlags = max_allowed_nlags
                self._nlags_truncated = True
            elif nlags < 1:
                self.nlags = 1
                self._nlags_truncated = True
            else:
                self.nlags = nlags
                self._nlags_truncated = False

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

        n_samples = len(self.data)
        if self.nlags < 1 or self.nlags >= n_samples:
            raise ValueError(
                f"滞后阶数 nlags 必须在 [1, {n_samples - 1}] 范围内，"
                f"当前值为 {self.nlags}"
            )

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

        result = {
            'acf_values': acf_values.tolist(),
            'lags': lags.tolist(),
            'confidence_interval': {
                'lower': lower_ci.tolist(),
                'upper': upper_ci.tolist(),
                'alpha': self.alpha
            },
            'significant_lags': significant_lags,
            'significant_count': len(significant_lags),
            'nlags': self.nlags,
            'nlags_truncated': self._nlags_truncated
        }
        if self._requested_nlags is not None:
            result['requested_nlags'] = self._requested_nlags

        return result

    def compute_pacf(self, method: str = 'ywadjusted') -> Dict[str, Union[np.ndarray, float, List]]:
        n_samples = len(self.data)

        if method in ('ywadjusted', 'ywmle', 'ld'):
            max_pacf_nlags = n_samples // 2 - 1
        else:
            max_pacf_nlags = n_samples - 1

        effective_nlags = min(self.nlags, max_pacf_nlags)
        pacf_truncated = (effective_nlags < self.nlags)

        pacf_values, confint = pacf(
            self.data,
            nlags=effective_nlags,
            alpha=self.alpha,
            method=method
        )

        lags = np.arange(len(pacf_values))
        lower_ci = confint[:, 0] - pacf_values
        upper_ci = confint[:, 1] - pacf_values

        significant = np.abs(pacf_values) > np.abs(upper_ci)
        significant_lags = lags[significant].tolist()

        result = {
            'pacf_values': pacf_values.tolist(),
            'lags': lags.tolist(),
            'confidence_interval': {
                'lower': lower_ci.tolist(),
                'upper': upper_ci.tolist(),
                'alpha': self.alpha
            },
            'significant_lags': significant_lags,
            'significant_count': len(significant_lags),
            'nlags': effective_nlags,
            'nlags_truncated': self._nlags_truncated or pacf_truncated,
            'method': method
        }
        if self._requested_nlags is not None:
            result['requested_nlags'] = self._requested_nlags
        if pacf_truncated:
            result['pacf_nlags_limit'] = max_pacf_nlags

        return result

    def analyze(self) -> Dict[str, Union[Dict, List, float, int]]:
        acf_result = self.compute_acf()
        pacf_result = self.compute_pacf()

        summary = self._generate_summary(acf_result, pacf_result)

        result = {
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

        if self._nlags_truncated or acf_result.get('nlags_truncated') or \
                pacf_result.get('nlags_truncated'):
            result['nlags_info'] = {
                'requested_nlags': self._requested_nlags,
                'acf_nlags': acf_result['nlags'],
                'pacf_nlags': pacf_result['nlags'],
                'was_truncated': True,
                'max_allowed_nlags': len(self.data) - 1,
                'pacf_nlags_limit': pacf_result.get('pacf_nlags_limit')
            }

        return result

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

    def ljung_box_test(self, lags: Optional[Union[int, List[int]]] = None,
                       boxpierce: bool = False) -> Dict[str, Union[Dict, List, bool, str]]:
        n_samples = len(self.data)

        if lags is None:
            test_lags = min(self.nlags, n_samples - 1)
        elif isinstance(lags, int):
            test_lags = min(lags, n_samples - 1)
        else:
            test_lags = [min(l, n_samples - 1) for l in lags]

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lb_result = acorr_ljungbox(
                    self.data,
                    lags=test_lags,
                    boxpierce=boxpierce,
                    return_df=True
                )

            results_by_lag = []
            is_white_noise_all = True
            significant_lags = []

            for idx, row in lb_result.iterrows():
                lag = int(idx)
                lb_stat = float(row['lb_stat'])
                lb_pvalue = float(row['lb_pvalue'])
                is_white_noise_lag = bool(lb_pvalue >= self.alpha)

                if not is_white_noise_lag:
                    is_white_noise_all = False
                    significant_lags.append(lag)

                lag_result = {
                    'lag': lag,
                    'ljung_box_statistic': lb_stat,
                    'p_value': lb_pvalue,
                    'is_white_noise': is_white_noise_lag
                }

                if boxpierce and 'bp_stat' in row and 'bp_pvalue' in row:
                    lag_result['box_pierce_statistic'] = float(row['bp_stat'])
                    lag_result['box_pierce_p_value'] = float(row['bp_pvalue'])

                results_by_lag.append(lag_result)

            result = {
                'alpha': self.alpha,
                'tested_lags': test_lags if isinstance(test_lags, list) else
                               list(range(1, test_lags + 1)),
                'results_by_lag': results_by_lag,
                'is_white_noise': is_white_noise_all,
                'significant_lags': significant_lags,
                'conclusion': self._ljung_box_conclusion(is_white_noise_all, significant_lags)
            }

            if isinstance(test_lags, int):
                max_lag = test_lags
            else:
                max_lag = max(test_lags)

            if not isinstance(test_lags, list) and len(results_by_lag) > 0:
                overall = results_by_lag[-1]
                result['overall'] = {
                    'lag': max_lag,
                    'ljung_box_statistic': overall['ljung_box_statistic'],
                    'p_value': overall['p_value'],
                    'is_white_noise': overall['is_white_noise']
                }
                if boxpierce and 'box_pierce_statistic' in overall:
                    result['overall']['box_pierce_statistic'] = overall['box_pierce_statistic']
                    result['overall']['box_pierce_p_value'] = overall['box_pierce_p_value']

            return result

        except Exception as e:
            return {
                'alpha': self.alpha,
                'tested_lags': test_lags if isinstance(test_lags, list) else test_lags,
                'error': f'Ljung-Box 检验执行失败: {str(e)}',
                'is_white_noise': None,
                'conclusion': '检验失败'
            }

    def _ljung_box_conclusion(self, is_white_noise: bool,
                              significant_lags: List[int]) -> str:
        if is_white_noise:
            return (f"在显著性水平 α={self.alpha} 下，不能拒绝序列为白噪声的原假设，"
                    f"序列可视为纯随机序列")
        else:
            return (f"在显著性水平 α={self.alpha} 下，拒绝序列为白噪声的原假设，"
                    f"序列存在显著自相关结构（显著滞后阶数: {significant_lags}）")
