"""Trend detection and forecasting engine for performance analysis."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


class TrendType(Enum):
    """Types of performance trends."""

    IMPROVING = "improving"
    DECLINING = "declining"
    STABLE = "stable"
    VOLATILE = "volatile"
    SEASONAL = "seasonal"


@dataclass
class TrendAnalysis:
    """Comprehensive trend analysis result."""

    trend_type: TrendType
    slope: float
    r_squared: float
    volatility: float
    seasonal_component: dict[str, float] | None
    forecast: list[float]
    confidence_bands: tuple[list[float], list[float]]
    changepoints: list[int]
    trend_strength: float


class TrendDetector:
    """Advanced trend detection and forecasting."""

    def __init__(self, volatility_threshold: float = 0.15):
        """Initialize trend detector with volatility threshold."""
        self.volatility_threshold = volatility_threshold

    def analyze_performance_trend(
        self, scores: list[float], timestamps: list[datetime]
    ) -> TrendAnalysis:
        """Comprehensive trend analysis with forecasting."""
        if len(scores) != len(timestamps):
            raise ValueError("Scores and timestamps must have same length")

        if len(scores) < 3:
            raise ValueError("Need at least 3 data points for trend analysis")

        scores_array = np.array(scores)

        # Convert timestamps to numeric (days since first timestamp)
        time_numeric = self._timestamps_to_numeric(timestamps)

        # Detect changepoints
        changepoints = self._detect_changepoints(scores_array)

        # Calculate basic trend metrics
        slope, r_squared = self._calculate_linear_trend(time_numeric, scores_array)

        # Calculate volatility
        volatility = self._calculate_volatility(scores_array)

        # Determine trend type
        trend_type = self._classify_trend(slope, r_squared, volatility)

        # Detect seasonality
        seasonal_component = self._analyze_seasonality(scores_array)

        # Generate forecast
        forecast, confidence_bands = self._generate_forecast(
            time_numeric, scores_array, forecast_periods=7
        )

        # Calculate trend strength
        trend_strength = self._calculate_trend_strength(slope, r_squared, volatility)

        return TrendAnalysis(
            trend_type=trend_type,
            slope=slope,
            r_squared=r_squared,
            volatility=volatility,
            seasonal_component=seasonal_component,
            forecast=forecast,
            confidence_bands=confidence_bands,
            changepoints=changepoints,
            trend_strength=trend_strength,
        )

    def detect_anomalies(
        self, scores: list[float], method: str = "isolation_forest"
    ) -> list[int]:
        """Detect anomalous performance points."""
        if len(scores) < 10:
            return []  # Need sufficient data for anomaly detection

        scores_array = np.array(scores).reshape(-1, 1)

        if method == "isolation_forest":
            detector = IsolationForest(contamination=0.1, random_state=42)
            anomaly_labels = detector.fit_predict(scores_array)
            return [i for i, label in enumerate(anomaly_labels) if label == -1]

        elif method == "z_score":
            z_scores = np.abs(stats.zscore(scores))
            return [i for i, z in enumerate(z_scores) if z > 3]

        elif method == "iqr":
            q1, q3 = np.percentile(scores, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            return [
                i
                for i, score in enumerate(scores)
                if score < lower_bound or score > upper_bound
            ]

        else:
            raise ValueError(f"Unsupported anomaly detection method: {method}")

    def seasonal_decomposition(
        self, scores: list[float], period: int = 7
    ) -> dict[str, list[float]]:
        """Decompose time series into trend, seasonal, and residual components."""
        if len(scores) < 2 * period:
            return {
                "trend": scores,
                "seasonal": [0] * len(scores),
                "residual": [0] * len(scores),
            }

        scores_array = np.array(scores)

        # Simple moving average for trend
        trend = self._moving_average(scores_array, window=period)

        # Detrend the series
        detrended = scores_array - trend

        # Calculate seasonal component
        seasonal = self._extract_seasonal_component(detrended, period)

        # Calculate residual
        residual = scores_array - trend - seasonal

        return {
            "trend": trend.tolist(),
            "seasonal": seasonal.tolist(),
            "residual": residual.tolist(),
            "original": scores,
        }

    def _timestamps_to_numeric(self, timestamps: list[datetime]) -> np.ndarray:
        """Convert timestamps to numeric values (days since first timestamp)."""
        first_timestamp = timestamps[0]
        return np.array(
            [(ts - first_timestamp).total_seconds() / 86400 for ts in timestamps]
        )

    def _detect_changepoints(self, scores: np.ndarray) -> list[int]:
        """Detect changepoints using PELT-like algorithm (simplified)."""
        if len(scores) < 10:
            return []

        changepoints = []
        window_size = max(5, len(scores) // 10)

        for i in range(window_size, len(scores) - window_size):
            # Compare means before and after potential changepoint
            before = scores[max(0, i - window_size) : i]
            after = scores[i : min(len(scores), i + window_size)]

            # Perform t-test
            try:
                _, p_value = stats.ttest_ind(before, after)
                if p_value < 0.01:  # Significant change
                    changepoints.append(i)
            except Exception:
                # Skip invalid statistical calculations
                # Skip invalid statistical calculations - logging would be ideal
                pass
                continue

        # Remove changepoints that are too close to each other
        filtered_changepoints: list[int] = []
        for cp in changepoints:
            if (
                not filtered_changepoints
                or cp - filtered_changepoints[-1] > window_size
            ):
                filtered_changepoints.append(cp)

        return filtered_changepoints

    def _calculate_linear_trend(
        self, time_numeric: np.ndarray, scores: np.ndarray
    ) -> tuple[float, float]:
        """Calculate linear trend slope and R-squared."""
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            time_numeric, scores
        )
        return slope, r_value**2

    def _calculate_volatility(self, scores: np.ndarray) -> float:
        """Calculate volatility as coefficient of variation."""
        mean_score = np.mean(scores)
        if mean_score == 0:
            return 0
        return float(np.std(scores, ddof=1) / abs(mean_score))

    def _classify_trend(
        self, slope: float, r_squared: float, volatility: float
    ) -> TrendType:
        """Classify trend based on slope, R-squared, and volatility."""
        # High volatility indicates volatile trend
        if volatility > self.volatility_threshold:
            return TrendType.VOLATILE

        # Low R-squared indicates no clear trend (stable)
        if r_squared < 0.1:
            return TrendType.STABLE

        # Classify based on slope direction
        if slope > 0.01:  # Threshold for meaningful improvement
            return TrendType.IMPROVING
        elif slope < -0.01:  # Threshold for meaningful decline
            return TrendType.DECLINING
        else:
            return TrendType.STABLE

    def _analyze_seasonality(self, scores: np.ndarray) -> dict[str, float] | None:
        """Analyze seasonal patterns in the data."""
        if len(scores) < 14:  # Need at least 2 weeks for weekly seasonality
            return None

        seasonal_info = {}

        # Test for weekly seasonality (period = 7)
        if len(scores) >= 14:
            weekly_autocorr = self._autocorrelation(scores, 7)
            seasonal_info["weekly_strength"] = abs(weekly_autocorr)
            seasonal_info["weekly_detected"] = abs(weekly_autocorr) > 0.3

        # Test for monthly seasonality (period = 30)
        if len(scores) >= 60:
            monthly_autocorr = self._autocorrelation(scores, 30)
            seasonal_info["monthly_strength"] = abs(monthly_autocorr)
            seasonal_info["monthly_detected"] = abs(monthly_autocorr) > 0.3

        # Return None if no seasonality detected
        if not any(
            seasonal_info.get(f"{period}_detected", False)
            for period in ["weekly", "monthly"]
        ):
            return None

        return seasonal_info

    def _generate_forecast(
        self, time_numeric: np.ndarray, scores: np.ndarray, forecast_periods: int = 7
    ) -> tuple[list[float], tuple[list[float], list[float]]]:
        """Generate forecast with confidence bands."""
        # Use polynomial regression for better fit
        poly_features = PolynomialFeatures(degree=min(2, len(scores) // 5))
        x_poly = poly_features.fit_transform(time_numeric.reshape(-1, 1))

        model = LinearRegression()
        model.fit(x_poly, scores)

        # Generate future time points
        last_time = time_numeric[-1]
        future_times = np.array([last_time + i for i in range(1, forecast_periods + 1)])
        future_x = poly_features.transform(future_times.reshape(-1, 1))

        # Generate predictions
        forecast = model.predict(future_x).tolist()

        # Calculate prediction intervals (simplified)
        residuals = scores - model.predict(x_poly)
        mse = float(np.mean(np.abs(residuals))) ** 2
        std_error = np.sqrt(mse)

        # 95% confidence intervals
        confidence_factor = 1.96 * std_error
        lower_band = [pred - confidence_factor for pred in forecast]
        upper_band = [pred + confidence_factor for pred in forecast]

        return forecast, (lower_band, upper_band)

    def _calculate_trend_strength(
        self, slope: float, r_squared: float, volatility: float
    ) -> float:
        """Calculate overall trend strength score (0-1)."""
        # Normalize slope impact (absolute value, capped at 1.0)
        slope_strength = min(abs(slope), 1.0)

        # R-squared indicates how well trend explains variance
        fit_strength = r_squared

        # Low volatility increases trend strength
        stability_strength = max(0, 1 - volatility)

        # Weighted combination
        return 0.4 * slope_strength + 0.4 * fit_strength + 0.2 * stability_strength

    def _moving_average(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate moving average with padding."""
        if len(data) < window:
            return data

        # Pad the beginning and end
        padded_data = np.pad(data, (window // 2, window // 2), mode="edge")

        # Calculate moving average
        ma = np.convolve(padded_data, np.ones(window) / window, mode="valid")

        # Ensure same length as original data
        if len(ma) != len(data):
            ma = ma[: len(data)]

        return ma

    def _extract_seasonal_component(
        self, detrended: np.ndarray, period: int
    ) -> np.ndarray:
        """Extract seasonal component from detrended data."""
        seasonal = np.zeros_like(detrended)

        for i in range(len(detrended)):
            # Average values at the same seasonal position
            seasonal_positions = list(range(i % period, len(detrended), period))
            if seasonal_positions:
                seasonal[i] = np.mean(detrended[seasonal_positions])

        return seasonal

    def _calculate_trend_volatility(self, residuals: list[float]) -> float:
        """Calculate trend volatility from residuals."""
        if not residuals:
            return 0.0

        return float(np.mean(np.abs(residuals)))

    def _autocorrelation(self, series: np.ndarray, lag: int) -> float:
        """Calculate autocorrelation at given lag."""
        if len(series) <= lag:
            return 0

        # Remove mean
        series_centered = series - np.mean(series)

        # Calculate autocorrelation
        numerator = np.sum(series_centered[:-lag] * series_centered[lag:])
        denominator = np.sum(series_centered**2)

        return float(numerator / denominator) if denominator > 0 else 0.0
