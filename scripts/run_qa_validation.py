#!/usr/bin/env python3
"""
Stage 12 Analytics QA Validation Script

This script runs comprehensive QA validation for the Advanced Analytics Engine,
including statistical accuracy, performance, dashboard, security, and integration.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_qa_automation import QATestRunner


def main() -> int:
    """Main QA validation entry point."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive QA validation for Stage 12 Analytics"
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=[
            "statistical_accuracy",
            "performance",
            "dashboard",
            "security",
            "integration",
        ],
        default=None,
        help="Test categories to run (default: all)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Test timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--performance-threshold",
        type=float,
        default=2.0,
        help="Performance threshold in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="qa_validation_report.json",
        help="Output report file (default: qa_validation_report.json)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Configure QA runner
    config = {
        "test_timeout": args.timeout,
        "parallel_jobs": 4,
        "coverage_threshold": 80.0,
        "performance_threshold": args.performance_threshold,
        "memory_threshold": 500,
    }

    if args.verbose:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        logger.info("🚀 Starting Stage 12 Analytics QA Validation")
        logger.info(f"📊 Test categories: {args.categories or 'all'}")
        logger.info(f"⏱️  Timeout: {args.timeout}s")
        logger.info(f"🎯 Performance threshold: {args.performance_threshold}s")
        logger.info("-" * 60)

    # Initialize and run QA validation
    runner = QATestRunner(config)

    try:
        results = runner.run_test_suite(args.categories)

        if args.verbose:
            logger = logging.getLogger(__name__)
            logger.info("\n📋 Detailed Results:")
            for category, result in results.items():
                status_emoji = "✅" if result.get("status") == "passed" else "❌"
                status = result.get("status", "unknown")
                logger.info(f"{status_emoji} {category}: {status}")
                if result.get("duration"):
                    logger.info(f"   ⏱️  Duration: {result['duration']:.2f}s")
                if result.get("test_count"):
                    logger.info(f"   🧪 Tests: {result['test_count']}")
                if result.get("failures"):
                    logger.info(f"   ❌ Failures: {len(result['failures'])}")

        # Generate comprehensive report
        report = runner.generate_report()

        # Display summary
        logger = logging.getLogger(__name__)
        logger.info("\n📊 QA Validation Summary:")
        summary = report["summary"]

        status_emoji = {
            "passed": "✅",
            "partial_success": "⚠️",
            "failed": "❌",
            "critical_failure": "🚨",
        }.get(summary["overall_status"], "❓")

        logger.info(f"Status: {status_emoji} {summary['overall_status'].upper()}")
        logger.info(f"Duration: {summary['total_duration']:.2f}s")
        logger.info(f"Categories: {summary['categories_tested']}")

        test_summary = summary["test_summary"]
        logger.info(
            f"Tests: {test_summary['total_tests']} total, "
            f"{test_summary['total_failures']} failed"
        )
        logger.info(f"Success Rate: {test_summary['success_rate']:.1f}%")

        # Performance metrics
        perf_metrics = summary["performance_metrics"]
        logger.info(f"Performance: {perf_metrics['total_execution_time']:.2f}s total")
        logger.info(f"Slowest Category: {perf_metrics['slowest_category']}")

        # Recommendations
        logger.info("\n💡 Recommendations:")
        for i, rec in enumerate(summary["recommendations"], 1):
            logger.info(f"{i}. {rec}")

        # Save report
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"\n📄 Full report saved to: {args.output}")

        # Exit with appropriate code
        if summary["overall_status"] in ["passed", "partial_success"]:
            if args.verbose:
                logger.info("\n🎉 QA Validation completed successfully!")
            return 0
        else:
            if args.verbose:
                logger.error("\n💥 QA Validation failed - see report for details")
            return 1

    except Exception as e:
        if args.verbose:
            logger.error(f"\n💥 QA Validation failed with error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
