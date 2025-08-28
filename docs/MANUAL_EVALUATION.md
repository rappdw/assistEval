# Manual Copilot Evaluation Guide

## Overview
This guide explains how to conduct manual evaluations with Microsoft Copilot when API access is not available.

## Process

### 1. Trigger Manual Evaluation
- Go to Actions → Manual Copilot Evaluation
- Select test set (offline/online/all)
- Specify number of repetitions (default: 1)
- Click "Run workflow"

### 2. Download Evaluation Package
- Wait for workflow completion
- Download "manual-copilot-evaluation-[run-number]" artifact
- Extract the package containing:
  - `manual_prompts.txt` - Raw prompts for copy/paste
  - `evaluation_template.md` - Formatted template with response sections

### 3. Evaluate with Microsoft Copilot

#### For Each Task:
1. **Copy System Prompt**: Copy the system prompt to Microsoft Copilot
2. **Copy User Prompt**: Copy the user prompt to Microsoft Copilot
3. **Capture Response**: Copy Copilot's complete response
4. **Record in Template**: Paste the response in the corresponding section

#### Example Workflow:
```
TASK: offline.task1.metrics_csv

SYSTEM PROMPT:
You are an enterprise assistant. Follow instructions exactly. Do not browse the web.
Do not fabricate sources. Respond only in valid JSON format.

USER PROMPT:
[Task-specific prompt content]

COPILOT RESPONSE:
[Paste Copilot's response here]
```

### 4. Complete Evaluation Template

Fill out the `evaluation_template.md` file with all responses:

```markdown
# Manual Copilot Evaluation - 2025-01-15

## Task 1: Metrics from CSV
**Task ID**: offline.task1.metrics_csv

**System Prompt**:
You are an enterprise assistant...

**User Prompt**:
Analyze the following CSV data...

**Copilot Response**:
{
  "task1_data_metrics": {
    "precision": 0.7500,
    "recall": 0.6000,
    ...
  }
}

---

## Task 2: SSN Regex Validation
**Task ID**: offline.task2.ssn_regex

[Continue for all tasks...]
```

### 5. Upload Completed Results

#### Option A: GitHub Issue Comment
1. Go to the evaluation issue created by the workflow
2. Attach the completed `evaluation_template.md` file
3. Comment: `/process-evaluation` to trigger automated processing

#### Option B: Manual Processing
1. Use the CLI tool to process results locally:
   ```bash
   python scripts/run_bench.py process-manual \
     --responses evaluation_template.md \
     --output results/manual_copilot_$(date +%Y%m%d_%H%M%S)
   ```

### 6. Review Results
- Automated processing generates standard evaluation reports
- Results include scores, comparisons, and detailed breakdowns
- Reports are uploaded as workflow artifacts

## Best Practices

### Consistency Guidelines
- **Use Fresh Sessions**: Start each task in a new Copilot conversation
- **Copy Exactly**: Copy prompts exactly as provided, including formatting
- **Complete Responses**: Capture Copilot's full response, including any explanations
- **No Editing**: Don't modify Copilot's responses for formatting or correctness

### Quality Assurance
- **Verify JSON**: Ensure JSON responses are complete and properly formatted
- **Check Task IDs**: Confirm each response is matched to the correct task
- **Document Issues**: Note any Copilot errors or unexpected behaviors
- **Multiple Runs**: For stability analysis, ensure each repetition uses a fresh session

### Common Issues

#### Copilot Doesn't Follow JSON Format
- **Retry**: Try the prompt again with emphasis on JSON requirement
- **Extract JSON**: If response contains JSON within text, extract the JSON portion
- **Document**: Note the formatting issue in your evaluation notes

#### Copilot Refuses Task
- **Rephrase**: Try slight variations of the prompt
- **Context**: Provide additional context about the evaluation purpose
- **Document**: Record refusal and any error messages

#### Incomplete Responses
- **Continue**: Ask Copilot to continue or complete the response
- **Retry**: Start fresh if response seems truncated
- **Document**: Note any completion issues

## Evaluation Standards

### Response Validation
- **JSON Compliance**: All responses must be valid JSON where required
- **Field Completeness**: All required fields must be present
- **Format Adherence**: Responses should match expected schema

### Documentation Requirements
- **Timestamp**: Record when evaluation was conducted
- **Version**: Note Copilot version/interface used
- **Environment**: Document any relevant context (browser, settings, etc.)
- **Issues**: Log any problems encountered during evaluation

## Automated Processing

### What Gets Processed
- **Response Extraction**: JSON and structured data extraction
- **Validation**: Schema and format validation
- **Scoring**: Objective scoring using standard evaluators
- **Comparison**: Side-by-side comparison with ChatGPT results

### Processing Output
- **Scores**: Individual task scores and overall performance
- **Reports**: Markdown and JSON formatted reports
- **Artifacts**: Raw responses, processed data, and evaluation metadata
- **Comparisons**: Performance comparison charts and analysis

## Troubleshooting

### Workflow Issues
- **Missing Artifacts**: Check workflow logs for generation errors
- **Processing Failures**: Verify response format matches expected schema
- **Permission Errors**: Ensure proper repository access for issue creation

### Evaluation Issues
- **Copilot Access**: Verify Microsoft Copilot subscription and access
- **Response Quality**: Some tasks may require multiple attempts
- **Format Problems**: Use provided templates to ensure consistency

### Technical Support
- **GitHub Issues**: Report technical problems via repository issues
- **Documentation**: Check repository README for additional guidance
- **Workflow Logs**: Review GitHub Actions logs for detailed error information

## Integration with CI/CD

### Automated Triggers
- **Scheduled**: Manual evaluations can be triggered on schedule
- **PR Integration**: Evaluation results can be compared in pull requests
- **Release Validation**: Manual evaluation as part of release process

### Result Integration
- **Benchmark Comparison**: Results automatically compared with ChatGPT benchmarks
- **Trend Analysis**: Historical performance tracking across evaluations
- **Quality Gates**: Integration with CI/CD quality requirements

---

## Quick Reference

### Essential Commands
```bash
# Trigger manual evaluation
gh workflow run manual-copilot.yml -f test_set=offline

# Process completed evaluation
python scripts/run_bench.py process-manual --responses results.md

# Generate comparison report
python scripts/make_report.py --results results/ --compare-providers
```

### File Formats
- **Input**: `evaluation_template.md` with structured responses
- **Output**: Standard benchmark result format
- **Reports**: Markdown and JSON formats available

### Quality Checklist
- [ ] All prompts copied exactly as provided
- [ ] All responses captured completely
- [ ] JSON responses validated for syntax
- [ ] Task IDs correctly matched
- [ ] Evaluation template fully completed
- [ ] Results uploaded and processed successfully
