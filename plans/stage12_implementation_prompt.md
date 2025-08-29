# Stage 12 Implementation Prompt: Advanced Analytics & Insights Engine

## Role: Senior Software Architect & Analytics Engineer

You are a senior software architect with deep expertise in statistical analysis, data visualization, and AI-powered insights systems. Your task is to implement Stage 12 of the ChatGPT vs Microsoft Copilot evaluation harness - the Advanced Analytics & Insights Engine.

## Context

The evaluation harness (Stages 1-11) is complete and functional. Stage 12 transforms it from a basic comparison tool into an advanced analytics platform with:
- Statistical analysis and significance testing
- Trend detection and performance forecasting
- Automated regression detection
- AI-powered insights generation
- Interactive web dashboard
- Executive reporting capabilities

## Implementation Requirements

### Core Architecture Principles

1. **Statistical Rigor**: All statistical analyses must be mathematically sound with proper confidence intervals, effect sizes, and multiple comparison corrections
2. **Performance Optimization**: Handle large datasets efficiently with streaming processing and caching
3. **Real-time Capabilities**: Support live dashboard updates and immediate regression alerts
4. **Extensibility**: Design plugin architecture for custom analytics modules
5. **User Experience**: Intuitive dashboard with progressive disclosure of complexity

### Technical Specifications

#### 1. Analytics Engine (`bench/analytics/`)

**Statistical Analysis Framework** (`statistics.py`):
- Implement proper statistical tests (t-tests, Mann-Whitney U, Kruskal-Wallis)
- Add Bonferroni correction for multiple comparisons
- Calculate effect sizes (Cohen's d, eta-squared)
- Provide confidence intervals and power analysis
- Support both parametric and non-parametric approaches

**Trend Detection Engine** (`trends.py`):
- Implement time-series decomposition (STL, X-13ARIMA-SEATS)
- Add changepoint detection using PELT algorithm
- Support multiple forecasting models (ARIMA, Prophet, Linear)
- Detect seasonal patterns and cyclical behavior
- Provide uncertainty quantification for forecasts

**Regression Analysis** (`regression.py`):
- Implement statistical process control charts
- Add CUSUM and EWMA control charts for drift detection
- Support configurable sensitivity thresholds
- Generate automated root cause analysis
- Integrate with alerting system for immediate notifications

#### 2. Insights Generation (`insights.py`)

**AI-Powered Analysis**:
- Use pattern recognition to identify performance anomalies
- Generate natural language explanations of statistical findings
- Provide actionable recommendations based on historical patterns
- Rank insights by business impact and statistical confidence
- Support custom insight templates and rules

**Evidence-Based Reasoning**:
- Link all insights to specific statistical evidence
- Provide drill-down capabilities to underlying data
- Support hypothesis testing for insight validation
- Generate confidence scores for each recommendation

#### 3. Interactive Dashboard (`bench/web/`)

**Backend Architecture** (`app.py`):
- Use Flask with async support (Quart) for real-time updates
- Implement WebSocket connections for live data streaming
- Add Redis caching for performance optimization
- Support user authentication and role-based access
- Provide RESTful API with OpenAPI documentation

**Frontend Implementation**:
- Use modern JavaScript (ES6+) with modular architecture
- Implement responsive design with Tailwind CSS
- Add interactive charts with Plotly.js and D3.js
- Support real-time updates without page refresh
- Provide export capabilities (PDF, PNG, CSV)

**Key Dashboard Components**:
- Performance overview cards with trend indicators
- Interactive time-series charts with zoom/pan
- Statistical test results with confidence visualizations
- Regression alerts with severity indicators
- Insights panel with natural language summaries

#### 4. Advanced Reporting (`reporting/advanced.py`)

**Executive Summary Generation**:
- Automated weekly/monthly executive reports
- Performance scorecards with KPI tracking
- Trend analysis with business impact assessment
- Comparative analysis across time periods
- Customizable report templates

**Alert System**:
- Real-time regression detection alerts
- Configurable notification channels (email, Slack, webhooks)
- Escalation policies based on severity
- Alert suppression and acknowledgment
- Historical alert tracking and analysis

### Implementation Guidelines

#### Code Quality Standards

1. **Type Safety**: Use comprehensive type hints with mypy validation
2. **Error Handling**: Implement graceful degradation with detailed logging
3. **Testing**: Achieve >95% test coverage with unit, integration, and property-based tests
4. **Documentation**: Provide comprehensive docstrings and API documentation
5. **Performance**: Optimize for sub-second response times on dashboard

#### Statistical Best Practices

1. **Assumptions Validation**: Always test statistical assumptions before applying tests
2. **Effect Size Reporting**: Report practical significance alongside statistical significance
3. **Multiple Comparisons**: Apply appropriate corrections (Bonferroni, FDR)
4. **Confidence Intervals**: Provide confidence intervals for all estimates
5. **Robustness**: Use bootstrap methods for non-parametric confidence intervals

#### Data Processing Pipeline

1. **Validation**: Validate all input data against schemas
2. **Preprocessing**: Handle missing data and outliers appropriately
3. **Transformation**: Apply necessary transformations (log, standardization)
4. **Analysis**: Perform statistical analysis with proper error handling
5. **Visualization**: Generate appropriate visualizations for each analysis type

### Integration Points

#### CLI Integration (`scripts/analytics.py`)

Implement comprehensive CLI commands:
```bash
# Trend analysis
uv run python scripts/analytics.py trend-analysis --provider chatgpt --days 30

# Regression detection
uv run python scripts/analytics.py regression-check --baseline run_20241201 --current run_20241215

# Insights generation
uv run python scripts/analytics.py generate-insights --output insights.json

# Dashboard server
uv run python scripts/analytics.py dashboard --port 5000 --debug
```

#### Configuration Management (`configs/analytics.yaml`)

Provide comprehensive configuration:
- Statistical test parameters and thresholds
- Trend detection sensitivity settings
- Dashboard refresh intervals and caching
- Alert notification configurations
- Report generation schedules

#### Database Integration

- Extend existing results storage for analytics metadata
- Add time-series optimized storage for trend data
- Implement data retention policies
- Support data export for external analysis tools

### Testing Strategy

#### Unit Tests
- Test all statistical functions with known datasets
- Validate trend detection algorithms with synthetic data
- Test insight generation with mock evaluation results
- Verify dashboard API endpoints with comprehensive test cases

#### Integration Tests
- Test end-to-end analytics pipeline with real evaluation data
- Validate dashboard functionality with browser automation
- Test alert system with various regression scenarios
- Verify report generation with different data configurations

#### Performance Tests
- Benchmark analytics performance with large datasets
- Test dashboard responsiveness under load
- Validate memory usage with streaming data processing
- Test concurrent user access to dashboard

### Deployment Considerations

#### Production Requirements
- Configure Redis for caching and session management
- Set up proper logging and monitoring
- Implement health checks for all services
- Configure reverse proxy (nginx) for static assets

#### Security Measures
- Implement proper authentication and authorization
- Add CSRF protection for web interface
- Validate and sanitize all user inputs
- Use HTTPS for all communications

#### Monitoring and Observability
- Add comprehensive application metrics
- Implement distributed tracing for request flows
- Set up alerting for system health issues
- Provide performance dashboards for operations team

## Success Criteria

### Functional Validation
- [ ] All statistical tests produce mathematically correct results
- [ ] Trend detection accurately identifies known patterns in test data
- [ ] Regression detection triggers appropriate alerts for performance drops
- [ ] Dashboard displays real-time data with sub-second latency
- [ ] Insights generation produces actionable recommendations
- [ ] Reports generate automatically on schedule

### Quality Validation
- [ ] Test coverage exceeds 95% across all modules
- [ ] Performance benchmarks meet sub-second response requirements
- [ ] Security audit passes with no critical vulnerabilities
- [ ] Documentation complete and accessible to end users
- [ ] Error handling graceful with informative messages

### Integration Validation
- [ ] CLI commands integrate seamlessly with existing workflow
- [ ] Dashboard authenticates with existing user management
- [ ] Analytics data persists correctly in database
- [ ] Alert system integrates with notification channels
- [ ] Reports export in multiple formats (PDF, CSV, JSON)

## Implementation Approach

### Phase 1: Core Analytics Engine (2 hours)
1. Implement statistical analysis framework with comprehensive test suite
2. Build trend detection engine with forecasting capabilities
3. Create regression detection system with configurable thresholds
4. Add comprehensive error handling and logging

### Phase 2: Insights Generation (1.5 hours)
1. Implement pattern recognition algorithms for performance analysis
2. Build natural language generation for insight descriptions
3. Create recommendation engine based on historical patterns
4. Add confidence scoring and evidence linking

### Phase 3: Interactive Dashboard (2 hours)
1. Build Flask backend with WebSocket support for real-time updates
2. Implement responsive frontend with modern JavaScript
3. Create interactive charts and visualizations
4. Add user authentication and role-based access control

### Phase 4: Integration & Testing (30 minutes)
1. Integrate with existing CLI and configuration systems
2. Add comprehensive test suite with >95% coverage
3. Perform end-to-end testing with real evaluation data
4. Validate performance and security requirements

Remember: This is a production-grade analytics platform that will be used for critical business decisions. Prioritize statistical accuracy, performance, and user experience. Every statistical claim must be backed by rigorous methodology and proper uncertainty quantification.
