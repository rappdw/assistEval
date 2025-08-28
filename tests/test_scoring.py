"""Tests for scoring system components."""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest
import yaml

from bench.core.evaluators.base import EvaluationResult
from bench.core.scoring import (
    ProviderScore,
    ScoreManager,
    ScoringEngine,
    StabilityAnalyzer,
    TaskScore,
    WeightConfig,
)


class TestTaskScore:
    """Test TaskScore data structure."""

    def test_task_score_creation(self):
        """Test TaskScore creation with all fields."""
        task_score = TaskScore(
            task_id="test_task",
            evaluator_name="test_evaluator",
            raw_score=85.0,
            max_score=100.0,
            weighted_score=42.5,
            weight=0.5,
            sub_scores={"precision": 0.9, "recall": 0.8},
            details={"test": "data"},
            errors=["error1"],
            warnings=["warning1"],
        )

        assert task_score.task_id == "test_task"
        assert task_score.evaluator_name == "test_evaluator"
        assert task_score.raw_score == 85.0
        assert task_score.max_score == 100.0
        assert task_score.weighted_score == 42.5
        assert task_score.weight == 0.5
        assert task_score.sub_scores == {"precision": 0.9, "recall": 0.8}
        assert task_score.details == {"test": "data"}
        assert task_score.errors == ["error1"]
        assert task_score.warnings == ["warning1"]

    def test_score_percentage_calculation(self):
        """Test score percentage calculation."""
        task_score = TaskScore(
            task_id="test",
            evaluator_name="test",
            raw_score=75.0,
            max_score=100.0,
            weighted_score=37.5,
            weight=0.5,
            sub_scores={},
        )

        assert task_score.score_percentage == 75.0

    def test_score_percentage_zero_max(self):
        """Test score percentage with zero max score."""
        task_score = TaskScore(
            task_id="test",
            evaluator_name="test",
            raw_score=50.0,
            max_score=0.0,
            weighted_score=0.0,
            weight=1.0,
            sub_scores={},
        )

        assert task_score.score_percentage == 0.0


class TestProviderScore:
    """Test ProviderScore data structure."""

    def test_provider_score_creation(self):
        """Test ProviderScore creation."""
        task_scores = {
            "task1": TaskScore(
                task_id="task1",
                evaluator_name="eval1",
                raw_score=80.0,
                max_score=100.0,
                weighted_score=40.0,
                weight=0.5,
                sub_scores={},
            )
        }

        provider_score = ProviderScore(
            provider_name="test_provider",
            timestamp=datetime.now(),
            task_scores=list(task_scores.values()),
            total_score=40.0,
            max_score=105.0,
            stability_bonus=2.5,
            final_score=42.5,
            metadata={"version": "1.0"},
        )

        assert provider_score.provider_name == "test_provider"
        assert provider_score.total_score == 40.0
        assert provider_score.max_score == 105.0
        assert provider_score.stability_bonus == 2.5
        assert provider_score.final_score == 42.5
        assert provider_score.score_percentage == pytest.approx(40.476, rel=1e-3)

    def test_final_score_calculation(self):
        """Test final score includes stability bonus."""
        provider_score = ProviderScore(
            provider_name="test",
            timestamp=datetime.now(),
            task_scores=[],
            total_score=90.0,
            max_score=105.0,
            stability_bonus=3.0,
            final_score=93.0,
        )

        assert provider_score.final_score == 93.0


class TestWeightConfig:
    """Test WeightConfig functionality."""

    @pytest.fixture
    def sample_config_data(self):
        """Sample configuration data."""
        return {
            "version": "1.0",
            "total_points": 105,
            "tasks": {
                "task1_metrics": {
                    "precision": 10,
                    "recall": 10,
                    "f1": 10,
                    "accuracy": 5,
                    "confusion_matrix": {"tp": 1, "fp": 1, "fn": 1, "tn": 2},
                },
                "task2_ssn_regex": {"validity": 18, "line_matches": 12},
                "task3_exec_summary": {"structure": 12, "tone": 8},
            },
            "stability_bonus": {
                "enabled": True,
                "max_points": 5,
                "consistency_threshold": 0.95,
            },
        }

    def test_weight_config_creation(self, sample_config_data):
        """Test WeightConfig creation."""
        config = WeightConfig(sample_config_data)

        assert config.weights["total_points"] == 105
        assert config.weights["tasks"] == sample_config_data["tasks"]
        assert config.get_stability_config() == sample_config_data["stability_bonus"]

    def test_load_from_file(self, sample_config_data):
        """Test loading configuration from YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(sample_config_data, f)
            config_path = Path(f.name)

        try:
            config = WeightConfig.load_from_file(config_path)
            assert config.weights["total_points"] == 105
            assert "task1_metrics" in config.weights["tasks"]
        finally:
            config_path.unlink()

    def test_load_from_nonexistent_file(self):
        """Test loading from nonexistent file returns empty config."""
        config = WeightConfig.load_from_file(Path("nonexistent.yaml"))
        assert config.weights == {}

    def test_get_task_weights(self, sample_config_data):
        """Test getting task weights."""
        config = WeightConfig(sample_config_data)

        task1_weights = config.get_task_weights("task1_metrics")
        assert task1_weights["precision"] == 10
        assert task1_weights["recall"] == 10

    def test_get_task_weights_missing(self, sample_config_data):
        """Test getting weights for missing task returns empty dict."""
        config = WeightConfig(sample_config_data)

        weights = config.get_task_weights("nonexistent_task")
        assert weights == {}

    def test_get_total_possible_score(self, sample_config_data):
        """Test getting total possible score."""
        config = WeightConfig(sample_config_data)
        assert config.get_total_possible_score() == 95.0  # 40+30+20+5 stability bonus

    def test_get_stability_config(self, sample_config_data):
        """Test getting stability configuration."""
        config = WeightConfig(sample_config_data)
        stability_config = config.get_stability_config()

        assert stability_config["enabled"] is True
        assert stability_config["max_points"] == 5


class TestScoringEngine:
    """Test ScoringEngine functionality."""

    @pytest.fixture
    def weight_config(self):
        """Sample weight configuration."""
        config_data = {
            "total_points": 105,
            "tasks": {
                "task1_metrics": {
                    "precision": 10,
                    "recall": 10,
                    "f1": 10,
                    "accuracy": 5,
                    "confusion_matrix": {"tp": 1, "fp": 1, "fn": 1, "tn": 2},
                },
                "task2_ssn_regex": {"validity": 18, "line_matches": 12},
            },
            "stability_bonus": {
                "enabled": True,
                "max_points": 5,
                "consistency_threshold": 0.95,
            },
        }
        return WeightConfig(config_data)

    @pytest.fixture
    def scoring_engine(self, weight_config):
        """ScoringEngine instance."""
        return ScoringEngine(weight_config)

    def test_scoring_engine_creation(self, weight_config):
        """Test ScoringEngine creation."""
        engine = ScoringEngine(weight_config)
        assert engine.weight_config == weight_config
        assert isinstance(engine.stability_analyzer, StabilityAnalyzer)

    def test_calculate_task_score_metrics(self, scoring_engine):
        """Test calculating task score for metrics task."""
        eval_result = EvaluationResult(
            task_id="offline.task1.metrics_csv",
            total_score=35.0,
            max_score=40.0,
            sub_scores={
                "precision": 0.9,
                "recall": 0.8,
                "f1": 0.85,
                "accuracy": 0.95,
            },
            details={"confusion_matrix": {"tp": 90, "fp": 10, "fn": 20, "tn": 80}},
        )

        task_score = scoring_engine.calculate_task_score(eval_result)

        assert task_score.task_id == "offline.task1.metrics_csv"
        assert task_score.raw_score == 35.0
        assert task_score.max_score == 40.0
        # Weighted score: 0.9*10 + 0.8*10 + 0.85*10 + 0.95*5 = 30.25
        assert task_score.weighted_score == pytest.approx(30.25, rel=1e-3)

    def test_calculate_task_score_unknown_task(self, scoring_engine):
        """Test calculating score for unknown task."""
        eval_result = EvaluationResult(
            task_id="unknown.task",
            total_score=50.0,
            max_score=100.0,
            sub_scores={"score": 0.5},
        )

        task_score = scoring_engine.calculate_task_score(eval_result)

        assert task_score.task_id == "task"
        assert task_score.weighted_score == 50.0  # No weighting applied
        assert "Unknown task" in task_score.errors[0]

    def test_calculate_provider_score(self, scoring_engine):
        """Test calculating provider score."""
        eval_results = [
            EvaluationResult(
                task_id="offline.task1.metrics_csv",
                total_score=35.0,
                max_score=40.0,
                sub_scores={
                    "precision": 0.9,
                    "recall": 0.8,
                    "f1": 0.85,
                    "accuracy": 0.95,
                },
            ),
            EvaluationResult(
                task_id="offline.task2.ssn_regex",
                total_score=25.0,
                max_score=30.0,
                sub_scores={"validity": 0.9, "line_matches": 0.8},
            ),
        ]

        provider_score = scoring_engine.calculate_provider_score(
            eval_results, "test_provider"
        )

        assert provider_score.provider_name == "test_provider"
        assert provider_score.max_score == 75.0  # Based on actual weight calculation
        assert len(provider_score.task_scores) == 2
        assert provider_score.total_score > 0

    def test_validate_scores_valid(self, scoring_engine):
        """Test score validation with valid scores."""
        provider_score = ProviderScore(
            provider_name="test",
            timestamp=datetime.now(),
            task_scores=[],
            total_score=90.0,
            max_score=105.0,
            stability_bonus=2.5,
            final_score=92.5,
        )

        errors = scoring_engine.validate_scores(provider_score)
        assert len(errors) == 0

    def test_validate_scores_invalid(self, scoring_engine):
        """Test score validation with invalid scores."""
        provider_score = ProviderScore(
            provider_name="test",
            timestamp=datetime.now(),
            task_scores=[],
            total_score=-10.0,  # Invalid negative score
            max_score=105.0,
            stability_bonus=10.0,  # Invalid bonus > max
            final_score=-10.0,
        )

        errors = scoring_engine.validate_scores(provider_score)
        assert len(errors) >= 2
        assert any("negative" in error.lower() for error in errors)
        assert any("exceeds maximum" in error.lower() for error in errors)

    def test_extract_task_id_mapping(self, scoring_engine):
        """Test task ID extraction and mapping."""
        assert (
            scoring_engine._extract_task_id("offline.task1.metrics_csv")
            == "task1_metrics"
        )
        assert (
            scoring_engine._extract_task_id("offline.task2.ssn_regex")
            == "task2_ssn_regex"
        )
        assert scoring_engine._extract_task_id("unknown_task") == "unknown_task"

    def test_extract_evaluator_name(self, scoring_engine):
        """Test evaluator name extraction."""
        eval_result = EvaluationResult(
            task_id="offline.task1.metrics_csv",
            total_score=0,
            max_score=0,
            sub_scores={},
            metadata={"evaluator_name": "custom_evaluator"},
        )

        name = scoring_engine._extract_evaluator_name(eval_result)
        assert name == "custom_evaluator"

        # Test fallback to unknown
        eval_result.metadata = {}
        name = scoring_engine._extract_evaluator_name(eval_result)
        assert name == "unknown"


class TestStabilityAnalyzer:
    """Test StabilityAnalyzer functionality."""

    @pytest.fixture
    def weight_config(self):
        """Weight configuration with stability settings."""
        config_data = {
            "stability_bonus": {
                "enabled": True,
                "max_points": 5,
                "consistency_threshold": 0.95,
            }
        }
        return WeightConfig(config_data)

    @pytest.fixture
    def stability_analyzer(self, weight_config):
        """StabilityAnalyzer instance."""
        return StabilityAnalyzer(weight_config)

    def test_stability_analyzer_creation(self, weight_config):
        """Test StabilityAnalyzer creation."""
        analyzer = StabilityAnalyzer(weight_config)
        assert analyzer.weight_config == weight_config
        assert analyzer.analysis_details == {}

    def test_calculate_stability_bonus_perfect_consistency(self, stability_analyzer):
        """Test stability bonus with perfect consistency."""
        # Create identical results across multiple runs
        results = []
        for _ in range(3):
            results.append(
                EvaluationResult(
                    task_id="task1_metrics",
                    total_score=40.0,
                    max_score=40.0,
                    sub_scores={
                        "precision": 0.9,
                        "recall": 0.8,
                        "f1": 0.85,
                        "accuracy": 0.95,
                    },
                    details={
                        "confusion_matrix": {"tp": 90, "fp": 10, "fn": 20, "tn": 80}
                    },
                )
            )

        multi_run_results = {"task1_metrics": results}
        bonus = stability_analyzer.calculate_stability_bonus(multi_run_results)

        assert bonus == 5.0  # Perfect consistency = max bonus

    def test_calculate_stability_bonus_disabled(self, weight_config):
        """Test stability bonus when disabled."""
        config_data = weight_config.weights.copy()
        config_data["stability_bonus"]["enabled"] = False
        disabled_config = WeightConfig(config_data)
        analyzer = StabilityAnalyzer(disabled_config)

        bonus = analyzer.calculate_stability_bonus({})
        assert bonus == 0.0

    def test_analyze_task1_consistency_perfect(self, stability_analyzer):
        """Test Task 1 consistency analysis with perfect consistency."""
        results = []
        for _ in range(3):
            results.append(
                EvaluationResult(
                    task_id="task1",
                    total_score=40.0,
                    max_score=40.0,
                    sub_scores={
                        "precision": 0.9,
                        "recall": 0.8,
                        "f1": 0.85,
                        "accuracy": 0.95,
                    },
                    details={
                        "confusion_matrix": {"tp": 90, "fp": 10, "fn": 20, "tn": 80}
                    },
                )
            )

        consistency = stability_analyzer._analyze_task1_consistency(results)
        assert consistency == 1.0

    def test_analyze_task1_consistency_single_run(self, stability_analyzer):
        """Test Task 1 consistency with single run."""
        results = [
            EvaluationResult(
                task_id="task1",
                total_score=40.0,
                max_score=40.0,
                sub_scores={"precision": 0.9},
            )
        ]

        consistency = stability_analyzer._analyze_task1_consistency(results)
        assert consistency == 1.0  # Single run is perfectly consistent

    def test_analyze_structural_consistency_no_errors(self, stability_analyzer):
        """Test structural consistency with no schema errors."""
        results = []
        for _ in range(3):
            results.append(
                EvaluationResult(
                    task_id="task2",
                    total_score=30.0,
                    max_score=30.0,
                    sub_scores={},
                    errors=[],  # No errors
                )
            )

        multi_run_results = {"task2": results}
        consistency = stability_analyzer._analyze_structural_consistency(
            multi_run_results
        )
        assert consistency == 1.0

    def test_analyze_structural_consistency_with_errors(self, stability_analyzer):
        """Test structural consistency with schema errors."""
        results = [
            EvaluationResult(
                task_id="task2",
                total_score=30.0,
                max_score=30.0,
                sub_scores={},
                errors=["Schema error"],
            ),
            EvaluationResult(
                task_id="task2",
                total_score=30.0,
                max_score=30.0,
                sub_scores={},
                errors=[],  # No errors
            ),
        ]

        multi_run_results = {"task2": results}
        consistency = stability_analyzer._analyze_structural_consistency(
            multi_run_results
        )
        assert consistency == 0.0  # No consistency due to different error patterns

    def test_get_analysis_details(self, stability_analyzer):
        """Test getting analysis details."""
        # Run analysis to populate details with actual data
        multi_run_results = {
            "task1": [
                EvaluationResult(
                    task_id="task1",
                    total_score=40.0,
                    max_score=40.0,
                    sub_scores={},
                )
            ]
        }
        stability_analyzer.calculate_stability_bonus(multi_run_results)

        details = stability_analyzer.get_analysis_details()
        assert isinstance(details, dict)


class TestScoreManager:
    """Test ScoreManager functionality."""

    @pytest.fixture
    def score_manager(self):
        """ScoreManager instance."""
        import tempfile

        return ScoreManager(results_dir=Path(tempfile.mkdtemp()))

    @pytest.fixture
    def sample_provider_score(self):
        """Sample provider score for testing."""
        task_scores = [
            TaskScore(
                task_id="task1",
                evaluator_name="eval1",
                raw_score=80.0,
                max_score=100.0,
                weighted_score=40.0,
                weight=0.5,
                sub_scores={"precision": 0.8},
                details={"test": "data"},
                errors=["error1"],
                warnings=["warning1"],
            )
        ]

        return ProviderScore(
            provider_name="test_provider",
            timestamp=datetime(2024, 1, 1, 12, 0),
            task_scores=task_scores,
            total_score=90.0,
            max_score=105.0,
            stability_bonus=3.0,
            final_score=93.0,
            metadata={"version": "1.0"},
        )

    def test_save_and_load_provider_score(self, score_manager, sample_provider_score):
        """Test saving and loading provider scores."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results_dir = Path(temp_dir)

            # Save score
            score_manager.save_provider_score(sample_provider_score, results_dir)

            # Verify file was created
            score_file = results_dir / "scores" / "test_provider_score.json"
            assert score_file.exists()

            # Load scores
            loaded_scores = score_manager.load_provider_scores(results_dir)
            assert len(loaded_scores) == 1

            loaded_score = loaded_scores[0]
            assert loaded_score.provider_name == "test_provider"
            assert loaded_score.total_score == 90.0
            assert loaded_score.stability_bonus == 3.0
            assert len(loaded_score.task_scores) == 1

    def test_load_provider_scores_no_directory(self, score_manager):
        """Test loading scores when directory doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results_dir = Path(temp_dir) / "nonexistent"
            scores = score_manager.load_provider_scores(results_dir)
            assert scores == []

    def test_get_score_history(self, score_manager, sample_provider_score):
        """Test getting score history."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)

            # Create multiple run directories with scores
            for i in range(3):
                run_dir = base_dir / f"run_{i}"
                run_dir.mkdir()

                # Modify timestamp for each run
                score_copy = ProviderScore(
                    provider_name=sample_provider_score.provider_name,
                    timestamp=datetime(2024, 1, i + 1, 12, 0, 0),
                    task_scores=sample_provider_score.task_scores,
                    total_score=sample_provider_score.total_score + i,
                    max_score=sample_provider_score.max_score,
                    stability_bonus=sample_provider_score.stability_bonus,
                    final_score=sample_provider_score.final_score + i,
                    metadata=sample_provider_score.metadata,
                )

                score_manager.save_provider_score(score_copy, run_dir)

            # Get history (need to update score_manager's results_dir)
            score_manager.results_dir = base_dir
            history = score_manager.get_score_history("test_provider")
            assert len(history) == 3

            # Verify sorted by timestamp
            timestamps = [score.timestamp for score in history]
            assert timestamps == sorted(timestamps)

    def test_serialize_deserialize_provider_score(
        self, score_manager, sample_provider_score
    ):
        """Test serialization and deserialization."""
        # Serialize
        serialized = score_manager.serialize_provider_score(sample_provider_score)

        assert isinstance(serialized, str)

        # Deserialize
        deserialized = score_manager.deserialize_provider_score(serialized)

        assert deserialized.provider_name == sample_provider_score.provider_name
        assert deserialized.total_score == sample_provider_score.total_score
        assert deserialized.stability_bonus == sample_provider_score.stability_bonus
        assert len(deserialized.task_scores) == len(sample_provider_score.task_scores)

    def test_serialize_deserialize_invalid_data(self, score_manager):
        """Test deserialization with invalid data."""
        invalid_json = '{"invalid": "data"}'

        with pytest.raises(KeyError):
            score_manager.deserialize_provider_score(invalid_json)
