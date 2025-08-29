"""Advanced analytics and insights engine for evaluation harness."""

from .insights import Insight, InsightsEngine
from .regression import RegressionAlert, RegressionAnalyzer
from .statistics import StatisticalAnalyzer, StatisticalResult
from .trends import TrendAnalysis, TrendDetector, TrendType

__all__ = [
    "StatisticalAnalyzer",
    "StatisticalResult",
    "TrendDetector",
    "TrendAnalysis",
    "TrendType",
    "RegressionAnalyzer",
    "RegressionAlert",
    "InsightsEngine",
    "Insight",
]
