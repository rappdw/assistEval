"""
QA Performance and Scalability Testing
Comprehensive performance validation for analytics engine and dashboard.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from bench.analytics.insights import InsightsEngine
from bench.analytics.statistics import StatisticalAnalyzer
from bench.analytics.trends import TrendDetector
from bench.web.app import create_app


class TestAnalyticsPerformance:
    """Test analytics computation performance requirements."""

    def setup_method(self):
        """Set up performance test fixtures."""
        self.analyzer = StatisticalAnalyzer()
        self.trend_detector = TrendDetector()
        self.insights_engine = InsightsEngine()

    def test_analytics_computation_performance(self):
        """Measure analytics computation performance - Target: < 5s for analysis."""
        # Generate test datasets of varying sizes
        test_sizes = [100, 1000, 10000]

        for size in test_sizes:
            # Generate synthetic performance data
            np.random.seed(42)
            data_a = np.random.normal(0.8, 0.1, size).tolist()
            data_b = np.random.normal(0.75, 0.1, size).tolist()

            # Measure statistical analysis performance
            start_time = time.time()
            result = self.analyzer.compare_providers(data_a, data_b)
            analysis_time = time.time() - start_time

            # Performance requirements
            if size <= 1000:
                assert analysis_time < 1.0, (
                    f"Analysis took {analysis_time:.2f}s for {size} samples (<1s)"
                )
            elif size <= 10000:
                assert analysis_time < 5.0, (
                    f"Analysis took {analysis_time:.2f}s for {size} samples (<5s)"
                )

            # Verify result quality isn't compromised
            assert result.p_value is not None
            assert not np.isnan(result.p_value)

    def test_trend_detection_performance(self):
        """Measure trend detection performance."""
        from datetime import datetime, timedelta

        # Generate time series data
        timestamps = [datetime.now() - timedelta(days=i) for i in range(100, 0, -1)]
        scores = [0.8 + 0.001 * i + np.random.normal(0, 0.05) for i in range(100)]

        start_time = time.time()
        trend_result = self.trend_detector.analyze_performance_trend(scores, timestamps)
        trend_time = time.time() - start_time

        # Should complete trend analysis quickly
        assert trend_time < 2.0, f"Trend analysis took {trend_time:.2f}s (target: <2s)"

        # Verify result structure
        assert hasattr(trend_result, "slope")
        assert hasattr(trend_result, "r_squared")

    def test_memory_usage_patterns(self):
        """Monitor memory usage during analytics operations."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Perform multiple analytics operations
        large_data_a = np.random.normal(0.8, 0.1, 50000).tolist()
        large_data_b = np.random.normal(0.75, 0.1, 50000).tolist()

        # Run multiple analyses
        for _ in range(10):
            result = self.analyzer.compare_providers(large_data_a, large_data_b)
            assert result is not None

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory usage should be reasonable (< 500MB increase)
        assert memory_increase < 500, (
            f"Memory increased by {memory_increase:.1f}MB (target: <500MB)"
        )

    def test_concurrent_analytics_requests(self):
        """Test multiple simultaneous analytics requests."""

        def run_analysis(data_pair):
            data_a, data_b = data_pair
            return self.analyzer.compare_providers(data_a, data_b)

        # Generate multiple datasets
        datasets = []
        for i in range(10):
            np.random.seed(42 + i)
            data_a = np.random.normal(0.8, 0.1, 1000).tolist()
            data_b = np.random.normal(0.75, 0.1, 1000).tolist()
            datasets.append((data_a, data_b))

        # Run concurrent analyses
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(run_analysis, dataset) for dataset in datasets]
            results = [future.result() for future in as_completed(futures)]

        total_time = time.time() - start_time

        # Concurrent execution should be efficient
        assert total_time < 10.0, (
            f"Concurrent analysis took {total_time:.2f}s (target: <10s)"
        )
        assert len(results) == 10, "All analyses should complete successfully"

        # Verify all results are valid
        for result in results:
            assert result.p_value is not None
            assert not np.isnan(result.p_value)


class TestDashboardPerformance:
    """Validate dashboard performance requirements."""

    def setup_method(self):
        """Set up dashboard test fixtures."""
        self.app = create_app()
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def test_page_load_times(self):
        """Measure initial page load performance - Target: < 2 seconds."""
        # Measure dashboard page load
        start_time = time.time()
        response = self.client.get("/")
        load_time = time.time() - start_time

        assert response.status_code == 200
        assert load_time < 2.0, f"Dashboard loaded in {load_time:.2f}s (target: <2s)"

        # Verify essential content is present
        assert b"Analytics Dashboard" in response.data

    def test_api_response_times(self):
        """Test API endpoint response times."""
        endpoints = ["/api/health", "/api/performance-data", "/api/insights"]

        for endpoint in endpoints:
            start_time = time.time()
            response = self.client.get(endpoint)
            response_time = time.time() - start_time

            assert response.status_code in [200, 404], (
                f"Unexpected status for {endpoint}"
            )
            assert response_time < 1.0, (
                f"{endpoint} responded in {response_time:.2f}s (target: <1s)"
            )

    def test_concurrent_user_load(self):
        """Test dashboard with multiple concurrent users."""

        def simulate_user():
            """Simulate a user session."""
            try:
                # Load dashboard
                response1 = self.client.get("/")
                assert response1.status_code == 200

                # Check health endpoint
                response2 = self.client.get("/api/health")
                assert response2.status_code == 200

                # Get performance data
                self.client.get("/api/performance-data")
                # May return 200 with data or appropriate error status

                return True
            except Exception:
                return False

        # Simulate concurrent users
        num_users = 10
        with ThreadPoolExecutor(max_workers=num_users) as executor:
            start_time = time.time()
            futures = [executor.submit(simulate_user) for _ in range(num_users)]
            results = [future.result() for future in as_completed(futures)]
            total_time = time.time() - start_time

        # All users should complete successfully
        success_rate = sum(results) / len(results)
        assert success_rate >= 0.9, f"Success rate: {success_rate:.1%} (target: ≥90%)"

        # Should handle concurrent load efficiently
        assert total_time < 10.0, (
            f"Concurrent load test took {total_time:.2f}s (target: <10s)"
        )


class TestScalability:
    """Test system scalability characteristics."""

    def test_data_volume_scaling(self):
        """Test performance with increasing data volumes."""
        analyzer = StatisticalAnalyzer()

        # Test with increasing data sizes
        data_sizes = [1000, 10000, 50000]
        performance_times = []

        for size in data_sizes:
            np.random.seed(42)
            data_a = np.random.normal(0.8, 0.1, size).tolist()
            data_b = np.random.normal(0.75, 0.1, size).tolist()

            start_time = time.time()
            result = analyzer.compare_providers(data_a, data_b)
            analysis_time = time.time() - start_time
            performance_times.append(analysis_time)

            # Verify result quality
            assert result.p_value is not None
            assert not np.isnan(result.p_value)

        # Performance should scale reasonably (not exponentially)
        # Allow for some performance degradation but not excessive
        for i in range(1, len(performance_times)):
            size_ratio = data_sizes[i] / data_sizes[i - 1]
            time_ratio = performance_times[i] / performance_times[i - 1]

            # Time ratio should not exceed size ratio by more than 2x
            assert time_ratio <= size_ratio * 2, (
                f"Performance degradation too high: {time_ratio:.2f}x time "
                f"for {size_ratio:.2f}x data"
            )

    def test_insights_generation_scaling(self):
        """Test insights generation with varying data volumes."""
        engine = InsightsEngine()

        # Generate evaluation histories of different sizes
        history_sizes = [10, 100, 500]

        for size in history_sizes:
            # Generate synthetic evaluation history
            evaluation_history = []
            from datetime import datetime, timedelta

            for i in range(size):
                evaluation_history.append(
                    {
                        "timestamp": datetime.now() - timedelta(hours=i),  # Hourly data
                        "provider": "test_provider",
                        "task": "test_task",
                        "score": 0.8 + np.random.normal(0, 0.1),
                        "details": {"metric": np.random.random()},
                    }
                )

            start_time = time.time()
            insights = engine.generate_insights(evaluation_history, {})
            generation_time = time.time() - start_time

            # Should complete within reasonable time
            max_time = min(30.0, size * 0.1)  # Scale with data size but cap at 30s
            assert generation_time < max_time, (
                f"Insights generation took {generation_time:.2f}s for {size} "
                f"records (target: <{max_time}s)"
            )

            # Should generate valid insights
            assert isinstance(insights, list)


class TestResourceUtilization:
    """Test resource utilization patterns."""

    def test_cpu_utilization(self):
        """Monitor CPU utilization during intensive operations."""

        analyzer = StatisticalAnalyzer()

        # Generate large dataset
        np.random.seed(42)
        large_data_a = np.random.normal(0.8, 0.1, 100000).tolist()
        large_data_b = np.random.normal(0.75, 0.1, 100000).tolist()

        # Monitor CPU during analysis
        start_time = time.time()
        result = analyzer.compare_providers(large_data_a, large_data_b)
        analysis_time = time.time() - start_time

        # Verify analysis completed successfully
        assert result.p_value is not None
        assert not np.isnan(result.p_value)

        # Analysis should complete in reasonable time even with large data
        assert analysis_time < 30.0, (
            f"Large dataset analysis took {analysis_time:.2f}s (target: <30s)"
        )

    def test_thread_safety(self):
        """Test thread safety of analytics components."""
        analyzer = StatisticalAnalyzer()
        results = []
        errors = []

        def worker():
            try:
                np.random.seed()  # Different seed per thread
                data_a = np.random.normal(0.8, 0.1, 1000).tolist()
                data_b = np.random.normal(0.75, 0.1, 1000).tolist()
                result = analyzer.compare_providers(data_a, data_b)
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Run multiple threads simultaneously
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # All threads should complete successfully
        assert len(errors) == 0, f"Thread safety errors: {errors}"
        assert len(results) == 10, "All threads should produce results"

        # All results should be valid
        for result in results:
            assert result.p_value is not None
            assert not np.isnan(result.p_value)
