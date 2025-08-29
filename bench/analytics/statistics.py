"""Statistical analysis framework for evaluation results."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
from scipy import stats


class TestType(Enum):
    """Statistical test types."""

    T_TEST = "t_test"
    MANN_WHITNEY = "mann_whitney"
    WILCOXON = "wilcoxon"
    KRUSKAL_WALLIS = "kruskal_wallis"
    CHI_SQUARE = "chi_square"


@dataclass
class StatisticalResult:
    """Statistical test result with confidence metrics."""

    test_name: str
    statistic: float
    p_value: float
    confidence_interval: tuple[float, float]
    effect_size: float | None
    interpretation: str
    significant: bool
    sample_sizes: tuple[int, ...]
    assumptions_met: dict[str, bool]


class StatisticalAnalyzer:
    """Advanced statistical analysis for evaluation results."""

    def __init__(self, alpha: float = 0.05):
        """Initialize with significance level."""
        self.alpha = alpha

    def compare_providers(
        self,
        provider_a_scores: list[float],
        provider_b_scores: list[float],
        test_type: TestType | None = None,
    ) -> StatisticalResult:
        """Compare two providers with appropriate statistical test."""
        # Convert to numpy arrays
        a_scores = np.array(provider_a_scores)
        b_scores = np.array(provider_b_scores)

        # Check assumptions
        assumptions = self._check_assumptions(a_scores, b_scores)

        # Select appropriate test if not specified
        if test_type is None:
            test_type = self._select_test(assumptions)

        # Perform statistical test
        if test_type == TestType.T_TEST:
            return self._perform_t_test(a_scores, b_scores, assumptions)
        elif test_type == TestType.MANN_WHITNEY:
            return self._perform_mann_whitney(a_scores, b_scores, assumptions)
        else:
            raise ValueError(f"Unsupported test type: {test_type}")

    def trend_analysis(
        self, scores: list[float], timestamps: list[datetime]
    ) -> dict[str, Any]:
        """Detect trends in performance over time."""
        if len(scores) != len(timestamps):
            raise ValueError("Scores and timestamps must have same length")

        # Convert timestamps to numeric values (days since first timestamp)
        time_numeric = [
            (ts - timestamps[0]).total_seconds() / 86400 for ts in timestamps
        ]

        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            time_numeric, scores
        )

        # Calculate trend strength
        trend_strength = abs(r_value)
        trend_direction = (
            "improving" if slope > 0 else "declining" if slope < 0 else "stable"
        )

        # Detect seasonality using autocorrelation
        seasonality = self._detect_seasonality(scores)

        return {
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_value**2,
            "p_value": p_value,
            "trend_direction": trend_direction,
            "trend_strength": trend_strength,
            "seasonality": seasonality,
            "forecast_next_7_days": self._forecast_linear(
                slope, intercept, len(time_numeric), 7
            ),
        }

    def regression_detection(
        self, baseline_scores: list[float], current_scores: list[float]
    ) -> StatisticalResult:
        """Detect performance regression with statistical confidence."""
        baseline = np.array(baseline_scores)
        current = np.array(current_scores)

        # Perform one-tailed t-test (testing if current < baseline)
        statistic, p_value = stats.ttest_ind(current, baseline, alternative="less")

        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(
            (
                (len(baseline) - 1) * np.var(baseline, ddof=1)
                + (len(current) - 1) * np.var(current, ddof=1)
            )
            / (len(baseline) + len(current) - 2)
        )

        cohens_d = (
            (np.mean(current) - np.mean(baseline)) / pooled_std if pooled_std > 0 else 0
        )

        # Calculate confidence interval for difference in means
        diff_mean = np.mean(current) - np.mean(baseline)
        se_diff = pooled_std * np.sqrt(1 / len(current) + 1 / len(baseline))
        t_critical = stats.t.ppf(1 - self.alpha / 2, len(baseline) + len(current) - 2)
        ci_lower = diff_mean - t_critical * se_diff
        ci_upper = diff_mean + t_critical * se_diff

        # Interpret results
        significant = p_value < self.alpha
        if significant and cohens_d < -0.2:
            interpretation = "Significant performance regression detected"
        elif significant:
            interpretation = (
                "Statistically significant but small performance difference"
            )
        else:
            interpretation = "No significant performance regression"

        return StatisticalResult(
            test_name="regression_detection_t_test",
            statistic=statistic,
            p_value=p_value,
            confidence_interval=(ci_lower, ci_upper),
            effect_size=cohens_d,
            interpretation=interpretation,
            significant=significant,
            sample_sizes=(len(baseline), len(current)),
            assumptions_met=self._check_assumptions(baseline, current),
        )

    def effect_size_analysis(
        self, group_a: list[float], group_b: list[float]
    ) -> dict[str, float]:
        """Calculate Cohen's d and other effect size metrics."""
        a = np.array(group_a)
        b = np.array(group_b)

        # Cohen's d
        pooled_std = np.sqrt(
            ((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1))
            / (len(a) + len(b) - 2)
        )
        cohens_d = (
            float(np.mean(np.abs(b - np.median(b)))) / pooled_std
            if pooled_std > 0
            else 0
        )

        # Glass's delta (using group_b as control)
        glass_delta = (
            (np.mean(a) - np.mean(b)) / np.std(b, ddof=1)
            if np.std(b, ddof=1) > 0
            else 0
        )

        # Hedges' g (bias-corrected Cohen's d)
        j = 1 - (3 / (4 * (len(a) + len(b) - 2) - 1))
        hedges_g = cohens_d * j

        # Effect size interpretation available but not used in return value
        # "large" if abs(cohens_d) > 0.8, "medium" if > 0.5, "small" if > 0.2

        return {
            "cohens_d": float(cohens_d),
            "glass_delta": float(glass_delta),
            "hedges_g": float(hedges_g),
        }

    def multiple_comparison_correction(
        self, p_values: list[float], method: str = "bonferroni"
    ) -> dict[str, Any]:
        """Apply multiple comparison correction."""
        p_array = np.array(p_values)

        if method == "bonferroni":
            corrected_p = np.minimum(p_array * len(p_values), 1.0)
        elif method == "fdr_bh":  # Benjamini-Hochberg FDR
            corrected_p = self._fdr_correction(p_array)
        else:
            raise ValueError(f"Unsupported correction method: {method}")

        return {
            "original_p_values": p_values,
            "corrected_p_values": corrected_p.tolist(),
            "significant_after_correction": (corrected_p < self.alpha).tolist(),
            "method": method,
        }

    def _check_assumptions(
        self, group_a: np.ndarray, group_b: np.ndarray
    ) -> dict[str, bool]:
        """Check statistical test assumptions."""
        assumptions = {}

        # Normality test (Shapiro-Wilk for small samples, Anderson-Darling for larger)
        if len(group_a) < 50:
            _, p_a = stats.shapiro(group_a)
            _, p_b = stats.shapiro(group_b)
        else:
            _, p_a = stats.normaltest(group_a)
            _, p_b = stats.normaltest(group_b)

        assumptions["normality_a"] = p_a > 0.05
        assumptions["normality_b"] = p_b > 0.05
        assumptions["normality_both"] = (
            assumptions["normality_a"] and assumptions["normality_b"]
        )

        # Equal variances test (Levene's test)
        _, p_levene = stats.levene(group_a, group_b)
        assumptions["equal_variances"] = p_levene > 0.05

        # Sample size adequacy
        assumptions["adequate_sample_size"] = len(group_a) >= 5 and len(group_b) >= 5

        return assumptions

    def _select_test(self, assumptions: dict[str, bool]) -> TestType:
        """Select appropriate statistical test based on assumptions."""
        if assumptions["normality_both"] and assumptions["equal_variances"]:
            return TestType.T_TEST
        else:
            return TestType.MANN_WHITNEY

    def _perform_t_test(
        self, group_a: np.ndarray, group_b: np.ndarray, assumptions: dict[str, bool]
    ) -> StatisticalResult:
        """Perform t-test with confidence interval."""
        equal_var = assumptions["equal_variances"]
        statistic, p_value = stats.ttest_ind(group_a, group_b, equal_var=equal_var)

        # Calculate confidence interval for difference in means
        diff_mean = np.mean(group_a) - np.mean(group_b)

        if equal_var:
            pooled_std = np.sqrt(
                (
                    (len(group_a) - 1) * np.var(group_a, ddof=1)
                    + (len(group_b) - 1) * np.var(group_b, ddof=1)
                )
                / (len(group_a) + len(group_b) - 2)
            )
            se_diff = pooled_std * np.sqrt(1 / len(group_a) + 1 / len(group_b))
            df = len(group_a) + len(group_b) - 2
        else:
            se_diff = np.sqrt(
                np.var(group_a, ddof=1) / len(group_a)
                + np.var(group_b, ddof=1) / len(group_b)
            )
            # Welch's t-test degrees of freedom
            df = (
                np.var(group_a, ddof=1) / len(group_a)
                + np.var(group_b, ddof=1) / len(group_b)
            ) ** 2 / (
                (np.var(group_a, ddof=1) / len(group_a)) ** 2 / (len(group_a) - 1)
                + (np.var(group_b, ddof=1) / len(group_b)) ** 2 / (len(group_b) - 1)
            )

        t_critical = stats.t.ppf(1 - self.alpha / 2, df)
        ci_lower = diff_mean - t_critical * se_diff
        ci_upper = diff_mean + t_critical * se_diff

        # Calculate effect size
        effect_size = self.effect_size_analysis(group_a.tolist(), group_b.tolist())[
            "cohens_d"
        ]

        # Interpretation
        significant = p_value < self.alpha
        if significant:
            interpretation = (
                f"Significant difference detected "
                f"(p={p_value:.4f}, d={effect_size:.3f})"
            )
        else:
            interpretation = (
                f"No significant difference (p={p_value:.4f}, d={effect_size:.3f})"
            )

        return StatisticalResult(
            test_name="independent_t_test",
            statistic=statistic,
            p_value=p_value,
            confidence_interval=(ci_lower, ci_upper),
            effect_size=effect_size,
            interpretation=interpretation,
            significant=significant,
            sample_sizes=(len(group_a), len(group_b)),
            assumptions_met=assumptions,
        )

    def _perform_mann_whitney(
        self, group_a: np.ndarray, group_b: np.ndarray, assumptions: dict[str, bool]
    ) -> StatisticalResult:
        """Perform Mann-Whitney U test."""
        statistic, p_value = stats.mannwhitneyu(
            group_a, group_b, alternative="two-sided"
        )

        # Calculate effect size (rank-biserial correlation)
        n1, n2 = len(group_a), len(group_b)
        effect_size = 1 - (2 * statistic) / (n1 * n2)

        # Bootstrap confidence interval for median difference
        ci_lower, ci_upper = self._bootstrap_median_diff_ci(group_a, group_b)

        significant = p_value < self.alpha
        interpretation = (
            f"Mann-Whitney U test: "
            f"{'Significant' if significant else 'No significant'} "
            f"difference in distributions"
        )

        return StatisticalResult(
            test_name="mann_whitney_u",
            statistic=statistic,
            p_value=p_value,
            confidence_interval=(ci_lower, ci_upper),
            effect_size=effect_size,
            interpretation=interpretation,
            significant=significant,
            sample_sizes=(n1, n2),
            assumptions_met=assumptions,
        )

    def _detect_seasonality(self, scores: list[float]) -> dict[str, Any]:
        """Detect seasonal patterns in time series."""
        if len(scores) < 14:  # Need at least 2 weeks for weekly seasonality
            return {"detected": False, "period": None, "strength": 0}

        # Check for weekly seasonality (period = 7)
        autocorr_7 = self._autocorrelation(scores, 7)

        # Check for monthly seasonality (period = 30)
        autocorr_30 = self._autocorrelation(scores, 30) if len(scores) >= 60 else 0

        max_autocorr = max(abs(autocorr_7), abs(autocorr_30))

        if max_autocorr > 0.3:  # Threshold for significant seasonality
            period = 7 if abs(autocorr_7) > abs(autocorr_30) else 30
            return {
                "detected": True,
                "period": period,
                "strength": max_autocorr,
                "type": "weekly" if period == 7 else "monthly",
            }

        return {"detected": False, "period": None, "strength": max_autocorr}

    def _autocorrelation(self, series: list[float], lag: int) -> float:
        """Calculate autocorrelation at given lag."""
        if len(series) <= lag:
            return 0

        series_array = np.array(series)
        # Length available for future statistical calculations if needed
        # n = len(series_array)

        # Remove mean
        series_centered = series_array - np.mean(series_array)

        # Calculate autocorrelation
        numerator = np.sum(series_centered[:-lag] * series_centered[lag:])
        denominator = np.sum(series_centered**2)

        return float(numerator / denominator) if denominator > 0 else 0.0

    def _forecast_linear(
        self, slope: float, intercept: float, current_time: int, periods: int
    ) -> list[float]:
        """Generate linear forecast for next periods."""
        return [slope * (current_time + i) + intercept for i in range(1, periods + 1)]

    def _bootstrap_median_diff_ci(
        self, group_a: np.ndarray, group_b: np.ndarray, n_bootstrap: int = 1000
    ) -> tuple[float, float]:
        """Bootstrap confidence interval for median difference."""
        np.random.seed(42)  # For reproducibility

        bootstrap_diffs = []
        for _ in range(n_bootstrap):
            boot_a = np.random.choice(group_a, len(group_a), replace=True)
            boot_b = np.random.choice(group_b, len(group_b), replace=True)
            bootstrap_diffs.append(np.median(boot_a) - np.median(boot_b))

        ci_lower = np.percentile(bootstrap_diffs, (self.alpha / 2) * 100)
        ci_upper = np.percentile(bootstrap_diffs, (1 - self.alpha / 2) * 100)

        return ci_lower, ci_upper

    def _fdr_correction(self, p_values: np.ndarray) -> np.ndarray:
        """Benjamini-Hochberg FDR correction."""
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]

        # Calculate adjusted p-values
        n = len(p_values)
        adjusted_p = np.zeros_like(sorted_p)

        for i in range(n - 1, -1, -1):
            if i == n - 1:
                adjusted_p[i] = sorted_p[i]
            else:
                adjusted_p[i] = min(adjusted_p[i + 1], sorted_p[i] * n / (i + 1))

        # Restore original order
        corrected_p = np.zeros_like(p_values)
        corrected_p[sorted_indices] = adjusted_p

        return corrected_p

    def _interpret_effect_size(self, effect_size: float) -> str:
        """Interpret Cohen's d effect size."""
        if effect_size < 0.2:
            return "negligible"
        elif effect_size < 0.5:
            return "small"
        elif effect_size < 0.8:
            return "medium"
        else:
            return "large"
