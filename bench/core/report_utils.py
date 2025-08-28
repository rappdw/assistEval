"""Report utility functions for formatting and data processing.

This module provides helper functions for report generation, including
text formatting, table creation, and data transformation utilities.
"""

from __future__ import annotations

from typing import Any


def format_score_table(scores: dict[str, float], max_width: int = 80) -> str:
    """Format scores into aligned table."""
    if not scores:
        return "No scores available"

    # Calculate column widths
    name_width = max(len(name) for name in scores.keys())
    score_width = max(len(f"{score:.1f}") for score in scores.values())

    # Ensure minimum widths
    name_width = max(name_width, 8)  # "Provider"
    score_width = max(score_width, 5)  # "Score"

    # Check if table fits in max_width
    total_width = name_width + score_width + 3  # 3 for " | "
    if total_width > max_width:
        # Truncate provider names if necessary
        name_width = max_width - score_width - 3

    # Create table
    lines = []
    lines.append(f"{'Provider':<{name_width}} | {'Score':>{score_width}}")
    lines.append("-" * name_width + "-|-" + "-" * score_width)

    for name, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        truncated_name = truncate_text(name, name_width)
        lines.append(f"{truncated_name:<{name_width}} | {score:>{score_width}.1f}")

    return "\n".join(lines)


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format percentage with proper rounding."""
    return f"{value:.{decimals}f}%"


def format_duration(seconds: float) -> str:
    """Format execution duration in human-readable form."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.0f}s"
    else:
        hours = int(seconds // 3600)
        remaining_minutes = int((seconds % 3600) // 60)
        return f"{hours}h {remaining_minutes}m"


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text with ellipsis."""
    if len(text) <= max_length:
        return text

    if max_length <= len(suffix):
        return text[:max_length]

    return text[: max_length - len(suffix)] + suffix


def format_score_difference(
    score_a: float, score_b: float, show_sign: bool = True
) -> str:
    """Format score difference with appropriate sign and precision."""
    diff = score_a - score_b

    if abs(diff) < 0.05:
        return "±0.0"

    sign = "+" if diff > 0 and show_sign else ""
    return f"{sign}{diff:.1f}"


def create_markdown_table(
    headers: list[str], rows: list[list[Any]], alignments: list[str] | None = None
) -> str:
    """Create a markdown table from headers and rows.

    Args:
        headers: Column headers
        rows: Table data rows
        alignments: Column alignments ('left', 'center', 'right')
    """
    if not headers or not rows:
        return ""

    if alignments is None:
        alignments = ["left"] * len(headers)

    # Ensure all rows have same number of columns
    max_cols = len(headers)
    normalized_rows = []
    for row in rows:
        normalized_row = [str(cell) for cell in row[:max_cols]]
        while len(normalized_row) < max_cols:
            normalized_row.append("")
        normalized_rows.append(normalized_row)

    # Calculate column widths
    col_widths = [len(header) for header in headers]
    for row in normalized_rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    # Create table
    lines = []

    # Header row
    header_line = "|"
    for _, (header, width) in enumerate(zip(headers, col_widths, strict=False)):
        header_line += f" {header:<{width}} |"
    lines.append(header_line)

    # Separator row
    separator_line = "|"
    for _, (alignment, width) in enumerate(zip(alignments, col_widths, strict=False)):
        if alignment == "center":
            sep = ":" + "-" * (width - 2) + ":"
        elif alignment == "right":
            sep = "-" * (width - 1) + ":"
        else:  # left
            sep = "-" * width
        separator_line += f" {sep} |"
    lines.append(separator_line)

    # Data rows
    for row in normalized_rows:
        data_line = "|"
        for _, (cell, width, alignment) in enumerate(
            zip(row, col_widths, alignments, strict=False)
        ):
            if alignment == "center":
                formatted_cell = f"{cell:^{width}}"
            elif alignment == "right":
                formatted_cell = f"{cell:>{width}}"
            else:  # left
                formatted_cell = f"{cell:<{width}}"
            data_line += f" {formatted_cell} |"
        lines.append(data_line)

    return "\n".join(lines)


def extract_task_name(task_id: str) -> str:
    """Extract human-readable task name from task ID."""
    # Map common task IDs to readable names
    task_names = {
        "offline.task1.metrics_csv": "Metrics from CSV",
        "offline.task2.ssn_regex": "SSN Regex Validation",
        "offline.task3.exec_summary": "Executive Summary",
        "online.deep_research.agentic_ai": "Deep Research - Agentic AI",
    }

    return task_names.get(task_id, task_id)


def calculate_statistical_significance(
    scores_a: list[float], scores_b: list[float], alpha: float = 0.05
) -> dict[str, Any]:
    """Calculate statistical significance between two score sets.

    Note: This is a simplified implementation. For production use,
    consider using scipy.stats for more robust statistical tests.
    """
    if len(scores_a) < 2 or len(scores_b) < 2:
        return {"significant": False, "reason": "Insufficient data"}

    # Calculate basic statistics
    mean_a = sum(scores_a) / len(scores_a)
    mean_b = sum(scores_b) / len(scores_b)

    # Simple variance calculation
    var_a = sum((x - mean_a) ** 2 for x in scores_a) / (len(scores_a) - 1)
    var_b = sum((x - mean_b) ** 2 for x in scores_b) / (len(scores_b) - 1)

    # Simplified t-test approximation
    pooled_se = ((var_a / len(scores_a)) + (var_b / len(scores_b))) ** 0.5

    if pooled_se == 0:
        return {"significant": False, "reason": "No variance"}

    t_stat = abs(mean_a - mean_b) / pooled_se

    # Very simplified significance check (assumes normal distribution)
    # For a more accurate test, use scipy.stats.ttest_ind
    critical_value = 2.0  # Approximate for small samples

    return {
        "significant": t_stat > critical_value,
        "t_statistic": t_stat,
        "mean_difference": mean_a - mean_b,
        "effect_size": abs(mean_a - mean_b) / ((var_a + var_b) / 2) ** 0.5
        if var_a + var_b > 0
        else 0,
    }


def format_error_summary(errors: list[str], max_errors: int = 5) -> str:
    """Format error list for display, limiting to most common errors."""
    if not errors:
        return "No errors"

    # Count error frequencies
    error_counts: dict[str, int] = {}
    for error in errors:
        error_counts[error] = error_counts.get(error, 0) + 1

    # Sort by frequency, then alphabetically
    sorted_errors = sorted(error_counts.items(), key=lambda x: (-x[1], x[0]))

    # Format top errors
    formatted_errors = []
    for error, count in sorted_errors[:max_errors]:
        if count > 1:
            formatted_errors.append(f"{error} ({count}x)")
        else:
            formatted_errors.append(error)

    result = "; ".join(formatted_errors)

    # Add truncation notice if needed
    if len(sorted_errors) > max_errors:
        remaining = len(sorted_errors) - max_errors
        result += f"; ... and {remaining} more"

    return result


def normalize_provider_name(provider_name: str) -> str:
    """Normalize provider name for consistent display."""
    # Common provider name mappings
    name_mappings = {
        "chatgpt": "ChatGPT",
        "gpt-4": "GPT-4",
        "gpt-3.5-turbo": "GPT-3.5 Turbo",
        "copilot": "Microsoft Copilot",
        "copilot_manual": "Microsoft Copilot (Manual)",
        "claude": "Claude",
        "gemini": "Gemini",
    }

    return name_mappings.get(provider_name.lower(), provider_name.title())


def create_progress_bar(current: int, total: int, width: int = 20) -> str:
    """Create a simple ASCII progress bar."""
    if total == 0:
        return "[" + " " * width + "] 0%"

    progress = min(current / total, 1.0)
    filled = int(progress * width)
    bar = "█" * filled + "░" * (width - filled)
    percentage = int(progress * 100)

    return f"[{bar}] {percentage}%"
