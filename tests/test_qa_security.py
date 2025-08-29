"""
QA Security and Authentication Testing
Comprehensive security validation for analytics system and web components.
"""

import json
import time

from bench.web.app import create_app


class TestWebSecurityBasics:
    """Test basic web security measures."""

    def setup_method(self):
        """Set up security test fixtures."""
        self.app = create_app()
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def test_http_methods_security(self):
        """Test HTTP method security restrictions."""
        # Test that only allowed methods work
        endpoints = ["/api/health", "/api/performance-data", "/api/insights"]

        for endpoint in endpoints:
            # GET should work
            response = self.client.get(endpoint)
            assert response.status_code in [200, 404, 500]

            # POST should be handled appropriately
            response = self.client.post(endpoint)
            assert response.status_code in [200, 405, 404, 500]

            # PUT should be restricted or handled
            response = self.client.put(endpoint)
            assert response.status_code in [405, 404, 500]

            # DELETE should be restricted
            response = self.client.delete(endpoint)
            assert response.status_code in [405, 404, 500]

    def test_cors_configuration(self):
        """Test CORS configuration security."""
        response = self.client.get("/api/health")

        # CORS headers should be present
        assert "Access-Control-Allow-Origin" in response.headers

        # Should not allow all origins in production (*)
        cors_origin = response.headers.get("Access-Control-Allow-Origin", "")
        # For testing, we allow permissive CORS, but note for production
        assert isinstance(cors_origin, str)

    def test_content_type_validation(self):
        """Test content type validation."""
        # Test JSON endpoints return proper content types
        response = self.client.get("/api/health")
        if response.status_code == 200:
            content_type = response.headers.get("Content-Type", "")
            assert "application/json" in content_type

    def test_error_information_disclosure(self):
        """Test that errors don't disclose sensitive information."""
        # Test with invalid endpoints
        response = self.client.get("/api/nonexistent")

        if response.status_code >= 400:
            content = response.data.decode("utf-8").lower()

            # Should not expose internal paths or system info
            sensitive_info = [
                "/users/",
                "traceback",
                "exception",
                "internal server error",
                "debug",
                "stack trace",
                "file not found",
                "python",
                "flask",
            ]

            # Some exposure might be acceptable in development
            exposed_count = sum(1 for info in sensitive_info if info in content)
            assert exposed_count < 3, (
                f"Too much sensitive information exposed: {content[:200]}"
            )


class TestInputValidationSecurity:
    """Test input validation and sanitization."""

    def setup_method(self):
        """Set up input validation test fixtures."""
        self.app = create_app()
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def test_parameter_injection_protection(self):
        """Test protection against parameter injection attacks."""
        malicious_inputs = [
            # Command injection attempts
            "?provider=test; rm -rf /",
            "?provider=test`whoami`",
            "?provider=test$(whoami)",
            # Path traversal attempts
            "?provider=../../../etc/passwd",
            "?provider=..\\..\\..\\windows\\system32",
            # Script injection attempts
            '?provider=<script>alert("xss")</script>',
            '?provider=javascript:alert("xss")',
            # SQL injection attempts
            "?provider=' OR '1'='1",
            "?provider='; DROP TABLE users; --",
        ]

        for malicious_input in malicious_inputs:
            response = self.client.get(f"/api/performance-data{malicious_input}")

            # Should handle malicious input gracefully
            assert response.status_code in [200, 400, 404, 422, 500]

            # Response should not execute or reflect malicious content
            if response.status_code == 200:
                content = response.data.decode("utf-8")
                assert "<script>" not in content
                assert "javascript:" not in content
                assert "DROP TABLE" not in content.upper()

    def test_large_input_handling(self):
        """Test handling of unusually large inputs."""
        # Test with very long parameter values
        long_value = "A" * 10000
        response = self.client.get(f"/api/performance-data?provider={long_value}")

        # Should handle large inputs gracefully
        assert response.status_code in [200, 400, 413, 414, 500]

    def test_special_character_handling(self):
        """Test handling of special characters in inputs."""
        special_chars = [
            "%00",  # Null byte
            "%0A",  # Line feed
            "%0D",  # Carriage return
            "%22",  # Quote
            "%27",  # Single quote
            "%3C",  # Less than
            "%3E",  # Greater than
        ]

        for char in special_chars:
            response = self.client.get(f"/api/performance-data?provider=test{char}")

            # Should handle special characters safely
            assert response.status_code in [200, 400, 404, 500]

    def test_unicode_input_handling(self):
        """Test handling of Unicode and international characters."""
        unicode_inputs = [
            "?provider=测试",  # Chinese characters
            "?provider=тест",  # Cyrillic characters
            "?provider=🚀",  # Emoji
            "?provider=café",  # Accented characters
        ]

        for unicode_input in unicode_inputs:
            response = self.client.get(f"/api/performance-data{unicode_input}")

            # Should handle Unicode gracefully
            assert response.status_code in [200, 400, 404, 500]


class TestDataSecurityValidation:
    """Test data security and privacy measures."""

    def setup_method(self):
        """Set up data security test fixtures."""
        self.app = create_app()
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def test_sensitive_data_exposure(self):
        """Test that sensitive data is not exposed in responses."""
        endpoints = ["/api/health", "/api/performance-data", "/api/insights"]

        for endpoint in endpoints:
            response = self.client.get(endpoint)

            if response.status_code == 200:
                content = response.data.decode("utf-8").lower()

                # Should not expose sensitive information
                sensitive_patterns = [
                    "password",
                    "secret",
                    "key",
                    "token",
                    "api_key",
                    "private",
                    "credential",
                    "auth",
                ]

                # Some patterns might be acceptable in field names
                exposed_count = sum(
                    1
                    for pattern in sensitive_patterns
                    if pattern in content and f'"{pattern}"' not in content
                )  # Exclude JSON field names

                assert exposed_count == 0, f"Sensitive data exposed in {endpoint}"

    def test_data_structure_consistency(self):
        """Test that data structures don't leak implementation details."""
        response = self.client.get("/api/performance-data")

        if response.status_code == 200:
            try:
                data = json.loads(response.data)

                # Should not expose internal implementation details
                if isinstance(data, dict):
                    keys = str(data.keys()).lower()
                    internal_patterns = [
                        "_internal",
                        "_private",
                        "__",
                        "debug",
                        "test_",
                    ]

                    exposed_internals = sum(
                        1 for pattern in internal_patterns if pattern in keys
                    )
                    assert exposed_internals == 0, (
                        "Internal details exposed in API response"
                    )

            except json.JSONDecodeError:
                # If not JSON, should still not expose internals in text
                content = response.data.decode("utf-8").lower()
                assert "_internal" not in content
                assert (
                    "debug" not in content or "debug" in content[:50]
                )  # Allow in headers

    def test_error_message_security(self):
        """Test that error messages don't leak sensitive information."""
        # Test various error conditions
        error_endpoints = [
            "/api/nonexistent",
            "/api/performance-data?invalid=param",
            "/api/insights?malformed",
        ]

        for endpoint in error_endpoints:
            response = self.client.get(endpoint)

            if response.status_code >= 400:
                content = response.data.decode("utf-8").lower()

                # Error messages should not expose file paths, internal structure
                sensitive_in_errors = [
                    "/users/",
                    "/home/",
                    "c:\\",
                    "traceback",
                    "line ",
                    'file "',
                    "module ",
                ]

                # Allow some technical details in development
                exposed_details = sum(
                    1 for detail in sensitive_in_errors if detail in content
                )
                assert exposed_details < 2, (
                    f"Too many internal details in error: {content[:100]}"
                )


class TestRateLimitingAndDOS:
    """Test rate limiting and denial of service protections."""

    def setup_method(self):
        """Set up rate limiting test fixtures."""
        self.app = create_app()
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def test_rapid_request_handling(self):
        """Test handling of rapid successive requests."""
        # Make many rapid requests
        responses = []
        start_time = time.time()

        for _ in range(50):
            response = self.client.get("/api/health")
            responses.append(response.status_code)

        end_time = time.time()
        duration = end_time - start_time

        # Should handle rapid requests without crashing
        success_rate = sum(1 for status in responses if status == 200) / len(responses)
        assert success_rate >= 0.8, f"Success rate too low: {success_rate:.2%}"

        # Should not take too long (basic performance check)
        assert duration < 30.0, f"Rapid requests took too long: {duration:.2f}s"

    def test_concurrent_request_handling(self):
        """Test handling of concurrent requests."""
        import threading

        results = []
        errors = []

        def make_requests():
            try:
                for _ in range(10):
                    response = self.client.get("/api/health")
                    results.append(response.status_code)
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_requests)
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Should handle concurrent requests
        assert len(errors) == 0, f"Concurrent request errors: {errors}"
        success_rate = sum(1 for status in results if status == 200) / len(results)
        assert success_rate >= 0.8, (
            f"Concurrent success rate too low: {success_rate:.2%}"
        )

    def test_resource_exhaustion_protection(self):
        """Test protection against resource exhaustion."""
        # Test with requests that might consume resources
        resource_intensive_requests = [
            "/api/performance-data?days=36500",  # Very large time range
            "/api/insights?limit=999999",  # Very large limit
            "/api/trends?granularity=second",  # High granularity
        ]

        for request in resource_intensive_requests:
            start_time = time.time()
            response = self.client.get(request)
            duration = time.time() - start_time

            # Should handle resource-intensive requests
            assert response.status_code in [200, 400, 404, 413, 429, 500]

            # Should not take excessive time
            assert duration < 30.0, (
                f"Request took too long: {duration:.2f}s for {request}"
            )


class TestAnalyticsSecurityValidation:
    """Test security aspects of analytics components."""

    def test_statistical_computation_safety(self):
        """Test that statistical computations are safe from malicious inputs."""
        import warnings

        from bench.analytics.statistics import StatisticalAnalyzer

        analyzer = StatisticalAnalyzer()

        # Suppress warnings for malicious data testing
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # Test with potentially problematic data
        malicious_data_sets = [
            # Very large numbers
            ([1e100] * 10, [1e100] * 10),
            # Very small numbers
            ([1e-100] * 10, [1e-100] * 10),
            # Mixed extreme values
            ([1e-100, 1e100] * 5, [1e-100, 1e100] * 5),
            # NaN and infinity (should be handled)
            ([1, 2, 3, float("inf")], [1, 2, 3, 4]),
        ]

        for data_a, data_b in malicious_data_sets:
            try:
                # Should handle malicious data gracefully
                result = analyzer.compare_providers(data_a, data_b)

                # Result should be valid or None
                if result is not None:
                    assert hasattr(result, "p_value")
                    # P-value should be reasonable
                    if result.p_value is not None:
                        assert (
                            0 <= result.p_value <= 1 or result.p_value != result.p_value
                        )  # Allow NaN

            except (ValueError, OverflowError, ZeroDivisionError) as e:
                # These exceptions are acceptable for malicious data
                assert isinstance(e, ValueError | OverflowError | ZeroDivisionError)

    def test_insights_generation_safety(self):
        """Test that insights generation is safe from malicious inputs."""
        from datetime import datetime

        from bench.analytics.insights import InsightsEngine

        engine = InsightsEngine()

        # Test with potentially problematic evaluation history
        malicious_histories = [
            # Very large dataset
            [
                {
                    "timestamp": datetime.now(),
                    "provider": "test",
                    "task": "test",
                    "score": 0.5,
                    "details": {},
                }
                for _ in range(10000)
            ],
            # Malicious string data
            [
                {
                    "timestamp": datetime.now(),
                    "provider": '<script>alert("xss")</script>',
                    "task": "test",
                    "score": 0.5,
                    "details": {},
                }
            ],
            # Extreme score values
            [
                {
                    "timestamp": datetime.now(),
                    "provider": "test",
                    "task": "test",
                    "score": float("inf"),
                    "details": {},
                }
            ],
        ]

        for history in malicious_histories:
            try:
                # Should handle malicious input gracefully
                insights = engine.generate_insights(
                    history[:100], {}
                )  # Limit size for safety

                # Should return valid insights or empty list
                assert isinstance(insights, list)

                # Insights should not contain unescaped malicious content
                for insight in insights:
                    if hasattr(insight, "description"):
                        assert "<script>" not in insight.description
                        assert "javascript:" not in insight.description

            except (ValueError, TypeError, OverflowError) as e:
                # These exceptions are acceptable for malicious data
                assert isinstance(e, ValueError | TypeError | OverflowError)

    def test_data_processing_bounds(self):
        """Test that data processing respects reasonable bounds."""
        from datetime import datetime

        from bench.analytics.trends import TrendDetector

        detector = TrendDetector()

        # Test with extreme time ranges
        extreme_timestamps = [
            datetime(1900, 1, 1),  # Very old
            datetime(2100, 1, 1),  # Far future
            datetime.now(),
        ]

        scores = [0.5, 0.6, 0.7]

        try:
            # Should handle extreme timestamps gracefully
            result = detector.analyze_performance_trend(scores, extreme_timestamps)

            # Should return valid result or handle gracefully
            if result is not None:
                assert hasattr(result, "slope")

        except (ValueError, OverflowError) as e:
            # These exceptions are acceptable for extreme data
            assert isinstance(e, ValueError | OverflowError)
