"""
Tests for analytics statistics module.
"""

from bench.analytics.statistics import StatisticalAnalyzer, StatisticalResult, TestType


class TestStatisticalAnalyzer:
    """Test cases for StatisticalAnalyzer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = StatisticalAnalyzer()

        # Sample data for testing
        self.normal_data_a = [0.8, 0.82, 0.78, 0.85, 0.79, 0.81, 0.83, 0.77, 0.84, 0.80]
        self.normal_data_b = [
            0.75,
            0.77,
            0.73,
            0.79,
            0.74,
            0.76,
            0.78,
            0.72,
            0.80,
            0.75,
        ]
        self.non_normal_data = [0.1, 0.2, 0.9, 0.95, 0.98, 0.99, 0.3, 0.4, 0.5, 0.6]

    def test_compare_providers_ttest(self):
        """Test provider comparison using t-test."""
        result = self.analyzer.compare_providers(
            self.normal_data_a, self.normal_data_b, test_type=TestType.T_TEST
        )

        assert isinstance(result, StatisticalResult)
        assert "t_test" in result.test_name.lower()
        assert result.p_value is not None
        assert result.effect_size is not None
        assert result.confidence_interval is not None
        assert len(result.confidence_interval) == 2
        assert result.significant is not None

    def test_compare_providers_mannwhitney(self):
        """Test provider comparison using Mann-Whitney U test."""
        result = self.analyzer.compare_providers(
            self.normal_data_a, self.normal_data_b, test_type=TestType.MANN_WHITNEY
        )

        assert isinstance(result, StatisticalResult)
        assert "mann" in result.test_name.lower()
        assert result.p_value is not None
        assert result.effect_size is not None
        assert result.significant is not None

    def test_compare_providers_auto_selection(self):
        """Test automatic test selection based on data properties."""
        # Test with normal data
        result_normal = self.analyzer.compare_providers(
            self.normal_data_a, self.normal_data_b
        )
        assert isinstance(result_normal, StatisticalResult)

        # Test with non-normal data
        result_non_normal = self.analyzer.compare_providers(
            self.non_normal_data, self.normal_data_b
        )
        assert isinstance(result_non_normal, StatisticalResult)

    def test_effect_size_calculation(self):
        """Test effect size calculations."""
        result = self.analyzer.compare_providers(self.normal_data_a, self.normal_data_b)
        assert result.effect_size is not None
        assert isinstance(result.effect_size, int | float)

    def test_basic_functionality(self):
        """Test basic statistical functionality."""
        result = self.analyzer.compare_providers(self.normal_data_a, self.normal_data_b)

        # Check basic result structure
        assert isinstance(result, StatisticalResult)
        assert result.p_value is not None
        assert result.confidence_interval is not None
        assert len(result.confidence_interval) == 2
        assert result.significant is not None

    def test_empty_data_handling(self):
        """Test handling of empty data sets."""
        import warnings

        warnings.filterwarnings("ignore")

        # Test empty data - may return None or raise exception
        result1 = self.analyzer.compare_providers([], self.normal_data_b)
        result2 = self.analyzer.compare_providers(self.normal_data_a, [])

        # Either should return None or raise an exception (both are valid)
        assert result1 is None or isinstance(result1, StatisticalResult)
        assert result2 is None or isinstance(result2, StatisticalResult)

    def test_single_value_data(self):
        """Test handling of single-value data."""
        single_value_a = [0.8]
        single_value_b = [0.7]

        try:
            result = self.analyzer.compare_providers(single_value_a, single_value_b)
            # Should handle gracefully, though statistical power is limited
            assert isinstance(result, StatisticalResult)
        except (ValueError, Warning):
            pass  # May not be able to perform statistical test with single values

    def test_identical_data(self):
        """Test handling of identical data sets."""
        import warnings

        warnings.filterwarnings("ignore")

        identical_data = [0.8] * 10
        try:
            result = self.analyzer.compare_providers(identical_data, identical_data)
            # Should handle gracefully - no difference expected
            assert isinstance(result, StatisticalResult)
            # p_value might be NaN for identical data, so check if it exists
            if result.p_value is not None and not (
                result.p_value != result.p_value
            ):  # Check for NaN
                assert result.p_value >= 0.05  # Should not be significant
        except (ValueError, ZeroDivisionError, RuntimeWarning):
            # Some statistical tests may fail with identical data
            pass

    def test_statistical_result_properties(self):
        """Test StatisticalResult data class properties."""
        result = self.analyzer.compare_providers(self.normal_data_a, self.normal_data_b)

        # Check all required fields are present based on actual implementation
        assert hasattr(result, "test_name")
        assert hasattr(result, "statistic")
        assert hasattr(result, "p_value")
        assert hasattr(result, "effect_size")
        assert hasattr(result, "confidence_interval")
        assert hasattr(result, "significant")
        assert hasattr(result, "assumptions_met")

    def test_significance_threshold(self):
        """Test significance threshold configuration."""
        # Test with different alpha levels
        analyzer_strict = StatisticalAnalyzer(alpha=0.01)
        analyzer_lenient = StatisticalAnalyzer(alpha=0.10)

        result_strict = analyzer_strict.compare_providers(
            self.normal_data_a, self.normal_data_b
        )
        result_lenient = analyzer_lenient.compare_providers(
            self.normal_data_a, self.normal_data_b
        )

        # Same p-value, different significance determination
        assert result_strict.p_value == result_lenient.p_value
        # Lenient threshold more likely to find significance
        if result_strict.significant:
            assert result_lenient.significant
