"""Test execution orchestration engine.

This module provides the core runner that orchestrates test execution across
providers, manages results collection, and handles error recovery.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from bench.adapters import create_provider
from bench.adapters.base import Provider, ProviderError
from bench.core.evaluators import EvaluationResult, EvaluatorRegistry
from bench.core.scoring import ProviderScore, ScoreManager, ScoringEngine, WeightConfig
from bench.core.validators import (
    ContentNormalizer,
    FieldExtractor,
    SchemaValidator,
    StructuralValidator,
)


class RunnerError(Exception):
    """Base exception for runner-related errors."""


class ConfigurationError(RunnerError):
    """Configuration loading or validation error."""


class TestExecutionError(RunnerError):
    """Test execution error."""


class ExecutionContext:
    """Manages test execution state and configuration."""

    def __init__(
        self,
        config_dir: Path = Path("configs"),
        results_dir: Path = Path("results"),
        tests_dir: Path = Path("bench/tests"),
    ):
        """Initialize execution context.

        Args:
            config_dir: Directory containing configuration files
            results_dir: Directory for storing results
            tests_dir: Directory containing test definitions
        """
        self.config_dir = config_dir
        self.results_dir = results_dir
        self.tests_dir = tests_dir
        self.console = Console()
        self.logger = logging.getLogger(__name__)

        # Loaded configurations
        self.providers_config: dict[str, Any] = {}
        self.weights_config: dict[str, Any] = {}
        self.matrix_config: dict[str, Any] = {}

        # Initialize validation components
        self.schema_validator = SchemaValidator(config_dir / "schemas")
        self.field_extractor = FieldExtractor()
        self.structural_validator = StructuralValidator()
        self.content_normalizer = ContentNormalizer()

    def load_configurations(
        self,
        providers_path: Path | None = None,
        weights_path: Path | None = None,
        matrix_path: Path | None = None,
    ) -> None:
        """Load all configuration files.

        Args:
            providers_path: Path to providers.yaml
            weights_path: Path to weights.yaml
            matrix_path: Path to runmatrix.yaml
        """
        try:
            # Load providers configuration
            providers_file = providers_path or self.config_dir / "providers.yaml"
            with open(providers_file, encoding="utf-8") as f:
                self.providers_config = yaml.safe_load(f)

            # Load weights configuration
            weights_file = weights_path or self.config_dir / "weights.default.yaml"
            with open(weights_file, encoding="utf-8") as f:
                self.weights_config = yaml.safe_load(f)

            # Load matrix configuration
            matrix_file = matrix_path or self.config_dir / "runmatrix.yaml"
            with open(matrix_file, encoding="utf-8") as f:
                self.matrix_config = yaml.safe_load(f)

        except FileNotFoundError as e:
            raise ConfigurationError(
                f"Configuration file not found: {e.filename}"
            ) from e
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML configuration: {e}") from e

    def load_test_definition(self, test_path: Path) -> dict[str, Any]:
        """Load and validate test definition.

        Args:
            test_path: Path to test YAML file

        Returns:
            Test definition dictionary
        """
        try:
            with open(test_path, encoding="utf-8") as f:
                test_def = yaml.safe_load(f)

            # Basic validation
            required_fields = ["id", "name", "category", "prompt", "expectation"]
            for field in required_fields:
                if field not in test_def:
                    raise ConfigurationError(
                        f"Test definition missing required field: {field}"
                    )

            return test_def  # type: ignore[no-any-return]

        except FileNotFoundError as e:
            raise ConfigurationError(
                f"Test definition file not found: {test_path}"
            ) from e
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML test definition: {e}") from e


class ResultsCollector:
    """Handles output collection and artifact storage."""

    def __init__(self, results_dir: Path):
        """Initialize results collector.

        Args:
            results_dir: Base directory for storing results
        """
        self.results_dir = results_dir
        self.current_run_dir: Path | None = None

    def create_run_directory(self) -> Path:
        """Create timestamped directory for current run.

        Returns:
            Path to created run directory
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.results_dir / f"run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (run_dir / "config").mkdir(exist_ok=True)
        (run_dir / "raw").mkdir(exist_ok=True)
        (run_dir / "parsed").mkdir(exist_ok=True)
        (run_dir / "metadata").mkdir(exist_ok=True)

        self.current_run_dir = run_dir

        # Update latest symlink
        latest_link = self.results_dir / "latest"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(run_dir.name)

        return run_dir

    def save_config_snapshot(self, context: ExecutionContext) -> None:
        """Save configuration snapshot for reproducibility.

        Args:
            context: Execution context with loaded configurations
        """
        if not self.current_run_dir:
            raise RunnerError("No active run directory")

        config_dir = self.current_run_dir / "config"

        # Save each configuration
        configs = {
            "providers.yaml": context.providers_config,
            "weights.yaml": context.weights_config,
            "runmatrix.yaml": context.matrix_config,
        }

        for filename, config in configs.items():
            with open(config_dir / filename, "w", encoding="utf-8") as f:
                yaml.dump(config, f, default_flow_style=False)

    def save_raw_output(
        self, provider_name: str, test_id: str, raw_output: str
    ) -> Path:
        """Save raw provider output.

        Args:
            provider_name: Name of the provider
            test_id: Test identifier
            raw_output: Raw text output from provider

        Returns:
            Path to saved file
        """
        if not self.current_run_dir:
            raise RunnerError("No active run directory")

        provider_dir = self.current_run_dir / "raw" / provider_name
        provider_dir.mkdir(parents=True, exist_ok=True)

        output_file = provider_dir / f"{test_id}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(raw_output)

        return output_file

    def save_parsed_output(
        self, provider_name: str, test_id: str, parsed_data: dict[str, Any]
    ) -> Path:
        """Save parsed and validated output.

        Args:
            provider_name: Name of the provider
            test_id: Test identifier
            parsed_data: Parsed JSON data

        Returns:
            Path to saved file
        """
        if not self.current_run_dir:
            raise RunnerError("No active run directory")

        provider_dir = self.current_run_dir / "parsed" / provider_name
        provider_dir.mkdir(parents=True, exist_ok=True)

        output_file = provider_dir / f"{test_id}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(parsed_data, f, indent=2)

        return output_file

    def save_execution_metadata(self, metadata: dict[str, Any]) -> Path:
        """Save execution metadata and timing information.

        Args:
            metadata: Execution metadata dictionary

        Returns:
            Path to saved metadata file
        """
        if not self.current_run_dir:
            raise RunnerError("No active run directory")

        metadata_file = self.current_run_dir / "metadata" / "execution.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return metadata_file


class TestRunner:
    """Core orchestration engine for benchmark execution."""

    def __init__(self, config_dir: Path = Path("configs")):
        """Initialize test runner.

        Args:
            config_dir: Directory containing configuration files
        """
        self.context = ExecutionContext(config_dir=config_dir)
        self.results_collector = ResultsCollector(Path("results"))
        self.console = Console()
        self.logger = logging.getLogger(__name__)

        # Initialize scoring components
        self.scoring_engine: ScoringEngine | None = None
        self.score_manager = ScoreManager(results_dir=Path("results"))

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    def _initialize_scoring_engine(self) -> None:
        """Initialize scoring engine with loaded weight configuration."""
        try:
            weights_path = self.context.config_dir / "weights.default.yaml"
            weight_config = WeightConfig.load_from_file(weights_path)
            self.scoring_engine = ScoringEngine(weight_config)
            self.logger.info("Scoring engine initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize scoring engine: {e}")
            self.scoring_engine = None

    def run_single(
        self, provider_name: str, test_path: Path, repetitions: int = 1
    ) -> dict[str, Any]:
        """Execute single test with specific provider.

        Args:
            provider_name: Name of provider to use
            test_path: Path to test definition file
            repetitions: Number of times to repeat the test

        Returns:
            Test execution results
        """
        try:
            # Load configurations
            self.context.load_configurations()

            # Initialize scoring engine
            self._initialize_scoring_engine()

            # Create run directory
            run_dir = self.results_collector.create_run_directory()
            self.results_collector.save_config_snapshot(self.context)

            # Load test definition
            test_def = self.context.load_test_definition(test_path)

            # Create provider
            provider = create_provider(provider_name)

            # Execute test repetitions
            results = []
            evaluation_results = []
            for rep in range(repetitions):
                self.console.print(
                    f"[blue]Running {test_def['name']} with {provider_name} "
                    f"(repetition {rep + 1}/{repetitions})[/blue]"
                )

                result = self._execute_single_test(provider, test_def, rep)
                results.append(result)

                # Collect evaluation results for scoring
                if "evaluation" in result and result["evaluation"]:
                    eval_result = self._create_evaluation_result_from_dict(
                        result["evaluation"], test_def["id"]
                    )
                    if eval_result:
                        evaluation_results.append(eval_result)

            # Calculate provider scores if scoring engine is available
            provider_score = None
            if self.scoring_engine and evaluation_results:
                provider_score = self._calculate_provider_scores(
                    evaluation_results, provider_name, run_dir
                )

            # Save execution metadata
            metadata = {
                "provider": provider_name,
                "test_path": str(test_path),
                "test_id": test_def["id"],
                "repetitions": repetitions,
                "run_directory": str(run_dir),
                "timestamp": datetime.now().isoformat(),
                "results": results,
            }

            # Add scoring information if available
            if provider_score:
                metadata["scoring"] = {
                    "total_score": provider_score.total_score,
                    "max_score": provider_score.max_score,
                    "final_score": provider_score.final_score,
                    "score_percentage": provider_score.score_percentage,
                    "stability_bonus": provider_score.stability_bonus,
                }

            self.results_collector.save_execution_metadata(metadata)

            self.console.print(
                f"[green]✓ Test completed. Results saved to {run_dir}[/green]"
            )
            return metadata

        except Exception as e:
            self.logger.error(f"Single test execution failed: {e}")
            raise TestExecutionError(f"Test execution failed: {e}") from e

    def run_matrix(self, matrix_path: Path | None = None) -> dict[str, Any]:
        """Execute full test matrix across providers and test sets.

        Args:
            matrix_path: Path to runmatrix.yaml file

        Returns:
            Matrix execution results
        """
        try:
            # Load configurations
            if matrix_path:
                self.context.load_configurations(matrix_path=matrix_path)
            else:
                self.context.load_configurations()

            # Initialize scoring engine
            self._initialize_scoring_engine()

            # Create run directory
            run_dir = self.results_collector.create_run_directory()
            self.results_collector.save_config_snapshot(self.context)

            matrix_config = self.context.matrix_config
            if "matrix" not in matrix_config:
                raise ConfigurationError(
                    "Matrix configuration missing 'matrix' section"
                )

            # Execute matrix
            matrix_results = []
            total_tests = len(matrix_config["matrix"])

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(),
                console=self.console,
            ) as progress:
                task = progress.add_task("Executing test matrix...", total=total_tests)

                for matrix_entry in matrix_config["matrix"]:
                    provider_name = matrix_entry["provider"]
                    test_set = matrix_entry["test_set"]
                    repetitions = matrix_entry.get("repetitions", 1)

                    progress.update(
                        task,
                        description=f"Running {provider_name} on {test_set} tests",
                        advance=0,
                    )

                    # Find test files for this test set
                    test_files = self._discover_test_files(test_set)

                    for test_file in test_files:
                        result = self.run_single(provider_name, test_file, repetitions)
                        matrix_results.append(result)

                    progress.advance(task)

            # Save matrix execution metadata
            metadata = {
                "matrix_config": matrix_config,
                "run_directory": str(run_dir),
                "timestamp": datetime.now().isoformat(),
                "total_tests": len(matrix_results),
                "results": matrix_results,
            }

            self.results_collector.save_execution_metadata(metadata)

            self.console.print(
                f"[green]✓ Matrix execution completed. "
                f"Results saved to {run_dir}[/green]"
            )
            return metadata

        except Exception as e:
            self.logger.error(f"Matrix execution failed: {e}")
            raise TestExecutionError(f"Matrix execution failed: {e}") from e

    def validate_test(self, test_path: Path) -> dict[str, Any]:
        """Validate test case definition.

        Args:
            test_path: Path to test definition file

        Returns:
            Validation results
        """
        try:
            # Load test definition
            test_def = self.context.load_test_definition(test_path)

            # Perform validation checks
            validation_results = {
                "test_path": str(test_path),
                "test_id": test_def.get("id", "unknown"),
                "valid": True,
                "errors": [],
                "warnings": [],
            }

            # Check required fields
            required_fields = [
                "id",
                "name",
                "category",
                "capability_profile",
                "prompt",
                "expectation",
                "scoring",
            ]

            for field in required_fields:
                if field not in test_def:
                    validation_results["errors"].append(
                        f"Missing required field: {field}"
                    )
                    validation_results["valid"] = False

            # Check capability profile
            if "capability_profile" in test_def:
                cap_profile = test_def["capability_profile"]
                if "web" not in cap_profile:
                    validation_results["warnings"].append(
                        "Missing 'web' capability specification"
                    )

            # Check prompt structure
            if "prompt" in test_def:
                prompt = test_def["prompt"]
                if not isinstance(prompt, dict):
                    validation_results["errors"].append("Prompt must be a dictionary")
                    validation_results["valid"] = False
                elif "user" not in prompt:
                    validation_results["errors"].append("Prompt missing 'user' field")
                    validation_results["valid"] = False

            return validation_results

        except Exception as e:
            return {
                "test_path": str(test_path),
                "valid": False,
                "errors": [f"Validation failed: {e}"],
                "warnings": [],
            }

    def _execute_single_test(
        self, provider: Provider, test_def: dict[str, Any], repetition: int
    ) -> dict[str, Any]:
        """Execute a single test instance.

        Args:
            provider: Provider instance
            test_def: Test definition
            repetition: Repetition number

        Returns:
            Test execution result
        """
        start_time = time.time()

        try:
            # Extract prompts and capabilities
            prompt = test_def["prompt"]
            system_prompt = prompt.get("system", "")
            user_prompt = prompt["user"]
            capabilities = test_def.get("capability_profile", {})

            # Invoke provider
            response = provider.invoke(
                system=system_prompt,
                user=user_prompt,
                options={},
                capabilities=capabilities,
            )

            # Save raw output
            raw_output = response.get("raw_text", "")
            self.results_collector.save_raw_output(
                provider.name, test_def["id"], raw_output
            )

            # Try to parse JSON if expected
            parsed_data = None
            if capabilities.get("json_required", False):
                try:
                    parsed_data = json.loads(raw_output)
                    self.results_collector.save_parsed_output(
                        provider.name, test_def["id"], parsed_data
                    )
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse JSON response: {e}")

            # Run evaluation if we have parsed data and scoring configuration
            evaluation_result = None
            if parsed_data and "scoring" in test_def:
                evaluation_result = self._evaluate_response(
                    parsed_data, test_def, provider.name
                )

            execution_time = time.time() - start_time

            result = {
                "repetition": repetition,
                "provider": provider.name,
                "test_id": test_def["id"],
                "execution_time": execution_time,
                "success": True,
                "raw_output_length": len(raw_output),
                "parsed_successfully": parsed_data is not None,
                "timestamp": datetime.now().isoformat(),
            }

            # Add evaluation results if available
            if evaluation_result:
                result["evaluation"] = {
                    "total_score": evaluation_result.total_score,
                    "max_score": evaluation_result.max_score,
                    "score_percentage": evaluation_result.score_percentage,
                    "sub_scores": evaluation_result.sub_scores,
                    "errors": evaluation_result.errors,
                    "warnings": evaluation_result.warnings,
                    "details": evaluation_result.details,
                    "metadata": evaluation_result.metadata,
                }

                # Save evaluation results
                self._save_evaluation_result(evaluation_result, provider.name)

            return result

        except ProviderError as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Provider error in test {test_def['id']}: {e}")

            return {
                "repetition": repetition,
                "provider": provider.name,
                "test_id": test_def["id"],
                "execution_time": execution_time,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def _discover_test_files(self, test_set: str) -> list[Path]:
        """Discover test files for a given test set.

        Args:
            test_set: Test set name (e.g., 'offline', 'online')

        Returns:
            List of test file paths
        """
        test_dir = self.context.tests_dir / test_set
        if not test_dir.exists():
            self.logger.warning(f"Test directory not found: {test_dir}")
            return []

        test_files = list(test_dir.glob("*.yaml"))
        test_files.extend(test_dir.glob("*.yml"))

        return sorted(test_files)

    def _evaluate_response(
        self, parsed_data: dict[str, Any], test_def: dict[str, Any], provider_name: str
    ) -> EvaluationResult | None:
        """Evaluate provider response using configured evaluator.

        Args:
            parsed_data: Parsed JSON response from provider
            test_def: Test definition with scoring configuration
            provider_name: Name of the provider

        Returns:
            Evaluation result or None if evaluation fails
        """
        try:
            scoring_config = test_def["scoring"]
            evaluator_name = scoring_config.get("evaluator")

            if not evaluator_name:
                self.logger.warning(f"No evaluator specified for test {test_def['id']}")
                return None

            if not EvaluatorRegistry.is_registered(evaluator_name):
                self.logger.error(f"Unknown evaluator: {evaluator_name}")
                return None

            # Create evaluator with configuration
            evaluator_config = scoring_config.get("config", {})
            evaluator = EvaluatorRegistry.create_evaluator(
                evaluator_name, evaluator_config
            )

            # Load answer key if available
            answer_key = self._load_answer_key(test_def["id"])

            # Run evaluation
            result = evaluator.evaluate(parsed_data, test_def, answer_key)

            self.logger.info(
                f"Evaluation completed for {test_def['id']} with {provider_name}: "
                f"{result.total_score:.2f}/{result.max_score:.2f} "
                f"({result.score_percentage:.1f}%)"
            )

            return result

        except Exception as e:
            self.logger.error(f"Evaluation failed for test {test_def['id']}: {e}")
            return None

    def _load_answer_key(self, test_id: str) -> dict[str, Any] | None:
        """Load answer key for a test.

        Args:
            test_id: Test identifier

        Returns:
            Answer key data or None if not found
        """
        try:
            # Determine test category from test_id
            # (offline.task1.metrics_csv -> offline)
            category = test_id.split(".")[0] if "." in test_id else "offline"

            answer_key_path = Path("answer_keys") / category / f"{test_id}.json"

            if not answer_key_path.exists():
                self.logger.debug(f"No answer key found at {answer_key_path}")
                return None

            with open(answer_key_path, encoding="utf-8") as f:
                data: dict[str, Any] = json.load(f)
                return data

        except Exception as e:
            self.logger.warning(f"Failed to load answer key for {test_id}: {e}")
            return None

    def _save_evaluation_result(
        self, evaluation_result: EvaluationResult, provider_name: str
    ) -> None:
        """Save evaluation result to results directory.

        Args:
            evaluation_result: Evaluation result to save
            provider_name: Name of the provider
        """
        try:
            if not self.results_collector.current_run_dir:
                self.logger.warning("No active run directory for saving evaluation")
                return

            eval_dir = (
                self.results_collector.current_run_dir / "evaluations" / provider_name
            )
            eval_dir.mkdir(parents=True, exist_ok=True)

            eval_file = eval_dir / f"{evaluation_result.task_id}.json"

            # Convert evaluation result to serializable format
            eval_data = {
                "task_id": evaluation_result.task_id,
                "total_score": evaluation_result.total_score,
                "max_score": evaluation_result.max_score,
                "score_percentage": evaluation_result.score_percentage,
                "sub_scores": evaluation_result.sub_scores,
                "details": evaluation_result.details,
                "errors": evaluation_result.errors,
                "warnings": evaluation_result.warnings,
                "metadata": evaluation_result.metadata,
            }

            with open(eval_file, "w", encoding="utf-8") as f:
                json.dump(eval_data, f, indent=2)

            self.logger.debug(f"Evaluation result saved to {eval_file}")

        except Exception as e:
            self.logger.error(f"Failed to save evaluation result: {e}")

    def _create_evaluation_result_from_dict(
        self, eval_dict: dict[str, Any], task_id: str
    ) -> EvaluationResult | None:
        """Create EvaluationResult from dictionary data.

        Args:
            eval_dict: Dictionary containing evaluation data
            task_id: Task identifier

        Returns:
            EvaluationResult instance or None if creation fails
        """
        try:
            return EvaluationResult(
                task_id=task_id,
                total_score=eval_dict.get("total_score", 0.0),
                max_score=eval_dict.get("max_score", 0.0),
                sub_scores=eval_dict.get("sub_scores", {}),
                details=eval_dict.get("details", {}),
                errors=eval_dict.get("errors", []),
                warnings=eval_dict.get("warnings", []),
                metadata=eval_dict.get("metadata", {}),
            )
        except Exception as e:
            self.logger.error(f"Failed to create EvaluationResult: {e}")
            return None

    def _calculate_provider_scores(
        self,
        evaluation_results: list[EvaluationResult],
        provider_name: str,
        run_dir: Path,
    ) -> ProviderScore | None:
        """Calculate and save provider scores.

        Args:
            evaluation_results: List of evaluation results
            provider_name: Name of the provider
            run_dir: Current run directory
        """
        try:
            if not self.scoring_engine:
                self.logger.warning("No scoring engine available for score calculation")
                return None

            # Calculate provider score
            provider_score = self.scoring_engine.calculate_provider_score(
                evaluation_results, provider_name
            )

            # Save provider score
            self.score_manager.save_provider_score(provider_score, run_dir)

            self.console.print(
                f"[green]✓ Provider score calculated: "
                f"{provider_score.final_score:.2f}/{provider_score.max_score:.2f} "
                f"({provider_score.score_percentage:.1f}%)[/green]"
            )

            return provider_score

        except Exception as e:
            self.logger.error(f"Failed to calculate provider scores: {e}")
            return None
