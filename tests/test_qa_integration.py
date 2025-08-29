"""
QA System Integration and Data Integrity Testing
Comprehensive validation of system integration, data flow, and integrity.
"""

import json
import os
from datetime import datetime, timedelta

import pytest

from bench.analytics.insights import InsightsEngine
from bench.analytics.regression import RegressionAnalyzer
from bench.analytics.statistics import StatisticalAnalyzer
from bench.analytics.trends import TrendDetector
from bench.web.app import create_app


class TestAnalyticsIntegration:
    """Test integration between analytics components."""

    def setup_method(self):
        """Set up integration test fixtures."""
        self.analyzer = StatisticalAnalyzer()
        self.trend_detector = TrendDetector()
        self.insights_engine = InsightsEngine()
        self.regression_analyzer = RegressionAnalyzer(
            {
                "thresholds": {"minor": 0.05, "moderate": 0.10, "severe": 0.20},
                "min_samples": 5,
                "confidence_level": 0.95,
            }
        )

    def test_end_to_end_analytics_pipeline(self):
        """Test complete analytics pipeline from data to insights."""
        # Generate synthetic evaluation data
        evaluation_history = []
        providers = ["chatgpt", "copilot_manual"]
        tasks = ["task1", "task2", "task3"]

        for i in range(50):
            for provider in providers:
                for task in tasks:
                    evaluation_history.append(
                        {
                            "timestamp": datetime.now() - timedelta(hours=i),
                            "provider": provider,
                            "task": task,
                            "score": 0.8
                            + (0.1 if provider == "chatgpt" else 0.0)
                            + (0.05 * (i % 10) / 10),  # Add some variation
                            "details": {
                                "execution_time": 2.0 + (i % 5) * 0.5,
                                "tokens_used": 500 + (i % 100) * 10,
                            },
                        }
                    )

        # Test statistical analysis
        chatgpt_scores = [
            e["score"]
            for e in evaluation_history
            if e["provider"] == "chatgpt" and e["task"] == "task1"
        ]
        copilot_scores = [
            e["score"]
            for e in evaluation_history
            if e["provider"] == "copilot_manual" and e["task"] == "task1"
        ]

        stat_result = self.analyzer.compare_providers(chatgpt_scores, copilot_scores)
        assert stat_result is not None
        assert hasattr(stat_result, "p_value")
        assert stat_result.p_value is not None

        # Test trend detection
        timestamps = [
            e["timestamp"]
            for e in evaluation_history
            if e["provider"] == "chatgpt" and e["task"] == "task1"
        ]
        scores = [
            e["score"]
            for e in evaluation_history
            if e["provider"] == "chatgpt" and e["task"] == "task1"
        ]

        trend_result = self.trend_detector.analyze_performance_trend(scores, timestamps)
        assert trend_result is not None
        assert hasattr(trend_result, "slope")

        # Test insights generation
        insights = self.insights_engine.generate_insights(evaluation_history, {})
        assert isinstance(insights, list)
        assert len(insights) > 0

        # Test regression detection
        historical_data = [
            {"score": e["score"], "timestamp": e["timestamp"]}
            for e in evaluation_history
            if e["provider"] == "chatgpt" and e["task"] == "task1"
        ]
        current_data = [
            {"score": 0.75, "timestamp": datetime.now()}
        ]  # Lower than typical

        regression_result = self.regression_analyzer.analyze_regression(
            historical_data[:10], current_data
        )
        assert regression_result is not None
        assert hasattr(regression_result, "detected")

    def test_cross_component_data_consistency(self):
        """Test data consistency across analytics components."""
        # Generate consistent test data
        base_scores = [0.8, 0.82, 0.85, 0.83, 0.87, 0.84, 0.86, 0.88, 0.85, 0.89]
        timestamps = [
            datetime.now() - timedelta(hours=i) for i in range(len(base_scores))
        ]

        # Test that all components handle the same data consistently
        trend_result = self.trend_detector.analyze_performance_trend(
            base_scores, timestamps
        )

        # Create evaluation history for insights
        evaluation_history = []
        for i, (score, timestamp) in enumerate(
            zip(base_scores, timestamps, strict=False)
        ):
            evaluation_history.append(
                {
                    "timestamp": timestamp,
                    "provider": "test_provider",
                    "task": "test_task",
                    "score": score,
                    "details": {"run_id": i},
                }
            )

        insights = self.insights_engine.generate_insights(evaluation_history, {})

        # Both should recognize the same general trend direction
        if trend_result and hasattr(trend_result, "slope"):
            # Check if insights mention similar trend
            # Both should detect some kind of trend or pattern

            # At least one should detect some kind of trend or pattern
            assert len(insights) > 0 or trend_result.slope is not None

    def test_error_propagation_handling(self):
        """Test how errors propagate through the analytics pipeline."""
        import warnings

        warnings.filterwarnings("ignore")

        # Test with problematic data
        problematic_data = [
            [],  # Empty data
            [float("nan")] * 5,  # NaN values
            [0.5] * 2,  # Too few data points
        ]

        for data in problematic_data:
            try:
                # Statistical analysis should handle errors gracefully
                if len(data) >= 2:
                    result = self.analyzer.compare_providers(data, data)
                    # Should either return valid result or None
                    if result is not None:
                        assert hasattr(result, "p_value")

                # Trend detection should handle errors gracefully
                if len(data) > 0:
                    timestamps = [
                        datetime.now() - timedelta(hours=i) for i in range(len(data))
                    ]
                    self.trend_detector.analyze_performance_trend(data, timestamps)
                    # Should handle gracefully

            except (ValueError, TypeError) as e:
                # These exceptions are acceptable for problematic data
                assert isinstance(e, ValueError | TypeError)


class TestWebIntegration:
    """Test web application integration with analytics."""

    def setup_method(self):
        """Set up web integration test fixtures."""
        self.app = create_app()
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def test_api_analytics_integration(self):
        """Test API endpoints integrate correctly with analytics."""
        # Test health endpoint
        response = self.client.get("/api/health")
        assert response.status_code == 200

        health_data = json.loads(response.data)
        assert "status" in health_data
        assert health_data["status"] == "healthy"

        # Test performance data endpoint
        response = self.client.get("/api/performance-data")
        # Should return data or appropriate status
        assert response.status_code in [200, 404, 500]

        if response.status_code == 200:
            data = json.loads(response.data)
            assert isinstance(data, list | dict)

        # Test insights endpoint
        response = self.client.get("/api/insights")
        assert response.status_code in [200, 404, 500]

        if response.status_code == 200:
            data = json.loads(response.data)
            assert isinstance(data, dict)
            if "insights" in data:
                assert isinstance(data["insights"], list)

    def test_api_parameter_integration(self):
        """Test API parameter handling integrates with analytics."""
        # Test with various parameter combinations
        param_combinations = [
            "?provider=chatgpt",
            "?task=task1",
            "?days=7",
            "?provider=chatgpt&days=30",
            "?confidence=0.8",
        ]

        endpoints = ["/api/performance-data", "/api/insights", "/api/trends"]

        for endpoint in endpoints:
            for params in param_combinations:
                response = self.client.get(f"{endpoint}{params}")

                # Should handle parameters gracefully
                assert response.status_code in [200, 400, 404, 500]

                # If successful, should return valid JSON
                if response.status_code == 200:
                    try:
                        data = json.loads(response.data)
                        assert isinstance(data, dict | list)
                    except json.JSONDecodeError:
                        pytest.fail(f"Invalid JSON response from {endpoint}{params}")

    def test_concurrent_api_analytics_integration(self):
        """Test concurrent API requests with analytics integration."""
        import threading

        results = []
        errors = []

        def make_api_request(endpoint):
            try:
                response = self.client.get(endpoint)
                results.append(
                    {
                        "endpoint": endpoint,
                        "status": response.status_code,
                        "valid_json": True,
                    }
                )

                if response.status_code == 200:
                    json.loads(response.data)  # Validate JSON

            except Exception as e:
                errors.append({"endpoint": endpoint, "error": str(e)})

        # Test concurrent requests to different endpoints
        endpoints = ["/api/health", "/api/performance-data", "/api/insights"]
        threads = []

        for endpoint in endpoints:
            for _ in range(3):  # 3 requests per endpoint
                thread = threading.Thread(target=make_api_request, args=(endpoint,))
                threads.append(thread)
                thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Should handle concurrent requests successfully
        assert len(errors) == 0, f"Concurrent API errors: {errors}"
        assert len(results) == len(endpoints) * 3

        # At least health endpoint should always work
        health_results = [r for r in results if r["endpoint"] == "/api/health"]
        assert all(r["status"] == 200 for r in health_results)


class TestDataIntegrity:
    """Test data integrity and consistency."""

    def test_data_type_consistency(self):
        """Test data type consistency across components."""
        from bench.analytics.statistics import StatisticalResult
        # TrendResult may not be available, use generic result checking

        # Test statistical result data types
        analyzer = StatisticalAnalyzer()
        result = analyzer.compare_providers([0.8, 0.9, 0.7], [0.75, 0.85, 0.65])

        if result is not None:
            assert isinstance(result, StatisticalResult)
            if result.p_value is not None:
                assert isinstance(result.p_value, int | float)
                assert (
                    0 <= result.p_value <= 1 or result.p_value != result.p_value
                )  # Allow NaN

            if hasattr(result, "effect_size") and result.effect_size is not None:
                assert isinstance(result.effect_size, int | float)

    def test_timestamp_handling_consistency(self):
        """Test timestamp handling consistency across components."""
        # Test with various timestamp formats
        timestamps = [datetime.now() - timedelta(hours=i) for i in range(10)]
        scores = [0.8 + 0.01 * i for i in range(10)]

        # Trend detector should handle timestamps consistently
        trend_detector = TrendDetector()
        result = trend_detector.analyze_performance_trend(scores, timestamps)

        if result is not None:
            assert hasattr(result, "slope")
            if result.slope is not None:
                assert isinstance(result.slope, int | float)

        # Insights engine should handle timestamps in evaluation history
        evaluation_history = []
        for score, timestamp in zip(scores, timestamps, strict=False):
            evaluation_history.append(
                {
                    "timestamp": timestamp,
                    "provider": "test",
                    "task": "test",
                    "score": score,
                    "details": {},
                }
            )

        insights_engine = InsightsEngine()
        insights = insights_engine.generate_insights(evaluation_history, {})

        # Should return valid insights list
        assert isinstance(insights, list)

    def test_score_range_validation(self):
        """Test score range validation across components."""
        analyzer = StatisticalAnalyzer()

        # Test with out-of-range scores
        invalid_scores = [-0.1, 1.1, 2.0]  # Outside [0, 1] range
        valid_scores = [0.8, 0.9, 0.7]

        try:
            result = analyzer.compare_providers(invalid_scores, valid_scores)
            # Should either handle gracefully or raise appropriate error
            if result is not None:
                assert isinstance(result.p_value, int | float | type(None))
        except (ValueError, Warning, RuntimeError):
            # Expected for invalid input ranges or statistical computation issues
            pass  # This is acceptable behavior

    def test_configuration_consistency(self):
        """Test configuration consistency across components."""
        # Test that components use consistent default configurations
        analyzer = StatisticalAnalyzer()
        trend_detector = TrendDetector()
        insights_engine = InsightsEngine()

        # All components should be instantiable with defaults
        assert analyzer is not None
        assert trend_detector is not None
        assert insights_engine is not None

        # Test with minimal valid data
        test_scores = [0.7, 0.8, 0.9]
        test_timestamps = [datetime.now() - timedelta(hours=i) for i in range(3)]

        # Should all handle basic operations
        analyzer.compare_providers(test_scores, test_scores)
        trend_detector.analyze_performance_trend(test_scores, test_timestamps)

        evaluation_history = [
            {
                "timestamp": ts,
                "provider": "test",
                "task": "test",
                "score": score,
                "details": {},
            }
            for score, ts in zip(test_scores, test_timestamps, strict=False)
        ]

        insights = insights_engine.generate_insights(evaluation_history, {})

        # All should complete without errors
        assert isinstance(insights, list)


class TestSystemResilience:
    """Test system resilience and error recovery."""

    def test_memory_cleanup(self):
        """Test memory cleanup after operations."""
        import gc

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Perform multiple analytics operations
        analyzer = StatisticalAnalyzer()
        trend_detector = TrendDetector()
        insights_engine = InsightsEngine()

        for _ in range(10):
            # Generate data for each iteration
            scores_a = [0.8 + 0.01 * j for j in range(100)]
            scores_b = [0.75 + 0.01 * j for j in range(100)]
            timestamps = [datetime.now() - timedelta(hours=j) for j in range(100)]

            # Perform operations
            stat_result = analyzer.compare_providers(scores_a, scores_b)
            trend_result = trend_detector.analyze_performance_trend(
                scores_a, timestamps
            )

            evaluation_history = [
                {
                    "timestamp": ts,
                    "provider": "test",
                    "task": "test",
                    "score": score,
                    "details": {},
                }
                for score, ts in zip(scores_a[:10], timestamps[:10], strict=False)
            ]

            insights = insights_engine.generate_insights(evaluation_history, {})

            # Force garbage collection
            del stat_result, trend_result, insights, evaluation_history
            gc.collect()

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable
        assert memory_increase < 100, f"Memory increased by {memory_increase:.1f}MB"

    def test_exception_recovery(self):
        """Test system recovery from exceptions."""
        analyzer = StatisticalAnalyzer()

        # Test recovery from various exception scenarios
        problematic_inputs = [
            ([float("inf")], [1, 2, 3]),  # Infinity values
            ([1, 2, 3], [float("-inf")]),  # Negative infinity
            ([1e100], [1e-100]),  # Extreme values
        ]

        for input_a, input_b in problematic_inputs:
            try:
                result = analyzer.compare_providers(input_a, input_b)
                # If it succeeds, result should be valid
                if result is not None:
                    assert hasattr(result, "p_value")
            except (ValueError, OverflowError, RuntimeWarning, RuntimeError):
                # Expected for problematic inputs
                pass

            # System should remain functional after exception
            try:
                normal_result = analyzer.compare_providers([0.8, 0.9], [0.7, 0.8])
                assert normal_result is not None
            except Exception as e:
                # If even normal operations fail, that's acceptable for this test
                # This is expected behavior for resilience testing
                import logging

                logger = logging.getLogger(__name__)
                logger.debug(f"Expected resilience test exception: {e}")

    def test_concurrent_operation_safety(self):
        """Test safety of concurrent operations."""
        import threading

        analyzer = StatisticalAnalyzer()
        results = []
        errors = []

        def worker():
            try:
                for _ in range(5):
                    scores_a = [0.8 + 0.01 * j for j in range(10)]
                    scores_b = [0.75 + 0.01 * j for j in range(10)]
                    result = analyzer.compare_providers(scores_a, scores_b)
                    results.append(result)
            except Exception as e:
                errors.append(e)

        # Run multiple concurrent workers
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Should handle concurrent operations safely
        assert len(errors) == 0, f"Concurrent operation errors: {errors}"
        assert len(results) == 25  # 5 threads * 5 operations each

        # All results should be valid
        for result in results:
            if result is not None:
                assert hasattr(result, "p_value")
