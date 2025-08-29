"""
Tests for analytics web API endpoints.
"""

import json
from unittest.mock import MagicMock, patch

from bench.web.app import create_app


class TestAnalyticsWebAPI:
    """Test cases for analytics web API."""

    def setup_method(self):
        """Set up test fixtures."""
        self.app = create_app()
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def test_dashboard_route(self):
        """Test dashboard HTML route."""
        response = self.client.get("/")

        assert response.status_code == 200
        assert b"Analytics Dashboard" in response.data
        assert b"dashboard.js" in response.data

    def test_health_check(self):
        """Test health check endpoint."""
        response = self.client.get("/api/health")

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["status"] == "healthy"
        assert "timestamp" in data

    @patch("bench.web.app.load_evaluation_results")
    def test_performance_data_endpoint(self, mock_load_results):
        """Test performance data API endpoint."""
        # Mock evaluation results
        mock_load_results.return_value = [
            {"provider": "chatgpt", "task": "task1", "score": 0.85},
            {"provider": "copilot", "task": "task1", "score": 0.75},
        ]

        response = self.client.get("/api/performance-data")

        assert response.status_code == 200
        data = json.loads(response.data)
        assert "data" in data
        assert "overall" in data["data"]
        assert "by_provider" in data["data"]
        assert "by_task" in data["data"]

    @patch("bench.web.app.load_evaluation_results")
    def test_provider_trends_endpoint(self, mock_load_results):
        """Test provider trends API endpoint."""
        mock_load_results.return_value = [
            {"provider": "chatgpt", "task": "task1", "score": 0.85},
        ]

        response = self.client.get("/api/trends/chatgpt")

        assert response.status_code == 200
        data = json.loads(response.data)
        assert "trends" in data

    def test_provider_trends_invalid_provider(self):
        """Test provider trends with invalid provider."""

        response = self.client.get("/api/trends/invalid_provider")

        assert response.status_code == 404
        data = json.loads(response.data)
        assert "message" in data

    @patch("bench.analytics.insights.InsightsEngine.generate_insights")
    def test_insights_endpoint(self, mock_generate_insights):
        """Test insights API endpoint."""
        # Mock insights
        mock_insight = MagicMock()
        mock_insight.title = "Test Insight"
        mock_insight.description = "Test description"
        mock_insight.category = "performance"
        mock_insight.severity = "warning"
        mock_insight.confidence = 0.8
        mock_insight.recommendations = ["Test recommendation"]
        mock_insight.evidence = {"test": "data"}
        mock_insight.affected_providers = ["chatgpt"]
        mock_insight.affected_tasks = ["task1"]
        mock_insight.timestamp.isoformat.return_value = "2024-01-01T00:00:00"

        mock_generate_insights.return_value = [mock_insight]

        response = self.client.get("/api/insights")

        assert response.status_code == 200
        data = json.loads(response.data)
        assert "insights" in data
        assert len(data["insights"]) == 1

    @patch("bench.web.app.load_evaluation_results")
    def test_provider_comparison_endpoint(self, mock_load_results):
        """Test provider comparison API endpoint."""

        # Mock separate calls for each provider
        def mock_load_side_effect(days=30, provider=None, task=None):
            if provider == "chatgpt":
                return [{"provider": "chatgpt", "task": "task1", "score": 0.85}]
            elif provider == "copilot":
                return [{"provider": "copilot", "task": "task1", "score": 0.75}]
            return []

        mock_load_results.side_effect = mock_load_side_effect

        response = self.client.get("/api/compare?provider_a=chatgpt&provider_b=copilot")

        assert response.status_code == 200
        data = json.loads(response.data)
        assert "comparison" in data

    def test_provider_comparison_missing_providers(self):
        """Test provider comparison with missing providers."""
        response = self.client.get("/api/compare")

        assert response.status_code == 400
        data = json.loads(response.data)
        assert "message" in data

    @patch("bench.web.app.load_evaluation_results")
    def test_regression_check_endpoint(self, mock_load_results):
        """Test regression check API endpoint."""
        mock_load_results.return_value = [
            {"provider": "chatgpt", "task": "task1", "score": 0.85},
        ]

        response = self.client.get("/api/regression-check")

        assert response.status_code == 200
        data = json.loads(response.data)
        assert "regression_alert" in data

    def test_cors_headers(self):
        """Test CORS headers are present."""
        response = self.client.get("/api/health")

        assert "Access-Control-Allow-Origin" in response.headers

    def test_json_content_type(self):
        """Test API endpoints return JSON content type."""
        response = self.client.get("/api/health")

        assert response.content_type == "application/json"

    def test_error_handling(self):
        """Test API error handling."""
        # Test 404 for non-existent endpoint
        response = self.client.get("/api/nonexistent")
        assert response.status_code == 404

    @patch("bench.web.app.load_evaluation_results")
    def test_empty_results_handling(self, mock_load_results):
        """Test handling of empty evaluation results."""
        mock_load_results.return_value = []

        response = self.client.get("/api/performance-data")

        assert response.status_code == 200
        data = json.loads(response.data)
        # Should handle empty results gracefully
        assert "data" in data

    def test_insights_confidence_filtering(self):
        """Test insights endpoint with confidence filtering."""
        with patch(
            "bench.analytics.insights.InsightsEngine.generate_insights"
        ) as mock_generate:
            # Mock insights with different confidence levels
            low_confidence = MagicMock()
            low_confidence.confidence = 0.5
            low_confidence.title = "Low Confidence"
            low_confidence.description = "Test"
            low_confidence.category = "test"
            low_confidence.severity = "info"
            low_confidence.recommendations = []
            low_confidence.evidence = {}
            low_confidence.affected_providers = ["chatgpt"]
            low_confidence.affected_tasks = ["task1"]
            low_confidence.timestamp.isoformat.return_value = "2024-01-01T00:00:00"

            high_confidence = MagicMock()
            high_confidence.confidence = 0.9
            high_confidence.title = "High Confidence"
            high_confidence.description = "Test"
            high_confidence.category = "test"
            high_confidence.severity = "warning"
            high_confidence.recommendations = []
            high_confidence.evidence = {}
            high_confidence.affected_providers = ["chatgpt"]
            high_confidence.affected_tasks = ["task1"]
            high_confidence.timestamp.isoformat.return_value = "2024-01-01T00:00:00"

            mock_generate.return_value = [low_confidence, high_confidence]

            # Test with confidence filter
            response = self.client.get("/api/insights?min_confidence=0.8")

            assert response.status_code == 200
            data = json.loads(response.data)
            # Should only return high confidence insight
            assert len(data["insights"]) == 1
            assert data["insights"][0]["title"] == "High Confidence"
