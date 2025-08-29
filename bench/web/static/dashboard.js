/**
 * Analytics Dashboard JavaScript
 * Handles real-time data updates, chart rendering, and user interactions
 */

class AnalyticsDashboard {
    constructor() {
        this.refreshInterval = 300000; // 5 minutes
        this.charts = {};
        this.data = {};

        this.initializeEventListeners();
        this.initializeCharts();
        this.loadData();
        this.setupRealTimeUpdates();
    }

    initializeEventListeners() {
        // Refresh button
        document.getElementById('refresh-btn').addEventListener('click', () => {
            this.loadData();
        });

        // Provider selection for trends
        document.getElementById('trend-provider-select').addEventListener('change', (e) => {
            if (e.target.value) {
                this.loadProviderTrends(e.target.value);
            }
        });

        // Task selection for comparison
        document.getElementById('comparison-task-select').addEventListener('change', (e) => {
            this.updateComparisonChart(e.target.value);
        });
    }

    async loadData() {
        this.showLoading();

        try {
            const [performanceData, insights] = await Promise.all([
                this.fetchPerformanceData(),
                this.fetchInsights()
            ]);

            this.data.performance = performanceData;
            this.data.insights = insights;

            this.updatePerformanceCards(performanceData);
            this.updateInsights(insights);
            this.populateProviderSelect(performanceData);
            this.loadDefaultTrends();
            this.updateComparisonChart();
            this.checkRegressions();

            this.updateLastUpdated();

        } catch (error) {
            console.error('Error loading data:', error);
            this.showError('Failed to load dashboard data');
        } finally {
            this.hideLoading();
        }
    }

    async fetchPerformanceData() {
        const response = await fetch('/api/performance-data');
        if (!response.ok) throw new Error('Failed to fetch performance data');
        const result = await response.json();
        return result.data;
    }

    async fetchInsights() {
        const response = await fetch('/api/insights');
        if (!response.ok) throw new Error('Failed to fetch insights');
        const result = await response.json();
        return result.insights;
    }

    async fetchProviderTrends(provider) {
        const response = await fetch(`/api/trends/${provider}`);
        if (!response.ok) throw new Error(`Failed to fetch trends for ${provider}`);
        const result = await response.json();
        return result.trends;
    }

    updatePerformanceCards(data) {
        if (!data || !data.overall) return;

        const overall = data.overall;

        // Total evaluations
        document.getElementById('total-evaluations').textContent =
            overall.total_evaluations.toLocaleString();

        // Average score
        const avgScore = (overall.average_score * 100).toFixed(1);
        document.getElementById('average-score').textContent = `${avgScore}%`;

        // Provider count
        document.getElementById('provider-count').textContent = overall.provider_count;

        // Update trend indicators (simplified - would need historical data for real trends)
        const trendElement = document.getElementById('score-trend');
        if (overall.average_score > 0.8) {
            trendElement.innerHTML = '<i class="fas fa-arrow-up trend-up"></i> <span class="trend-up">Good</span>';
        } else if (overall.average_score > 0.6) {
            trendElement.innerHTML = '<i class="fas fa-minus trend-stable"></i> <span class="trend-stable">Stable</span>';
        } else {
            trendElement.innerHTML = '<i class="fas fa-arrow-down trend-down"></i> <span class="trend-down">Needs Attention</span>';
        }
    }

    populateProviderSelect(data) {
        if (!data || !data.overall) return;

        const select = document.getElementById('trend-provider-select');
        select.innerHTML = '<option value="">Select Provider...</option>';

        data.overall.providers.forEach(provider => {
            const option = document.createElement('option');
            option.value = provider;
            option.textContent = provider.charAt(0).toUpperCase() + provider.slice(1);
            select.appendChild(option);
        });
    }

    async loadDefaultTrends() {
        if (this.data.performance && this.data.performance.overall.providers.length > 0) {
            const firstProvider = this.data.performance.overall.providers[0];
            document.getElementById('trend-provider-select').value = firstProvider;
            await this.loadProviderTrends(firstProvider);
        }
    }

    async loadProviderTrends(provider) {
        try {
            const trends = await this.fetchProviderTrends(provider);
            this.updateTrendChart(trends, provider);
        } catch (error) {
            console.error(`Error loading trends for ${provider}:`, error);
        }
    }

    updateTrendChart(trends, provider) {
        const chartDiv = document.getElementById('trend-chart');

        if (!trends || Object.keys(trends).length === 0) {
            chartDiv.innerHTML = '<div class="flex items-center justify-center h-full text-gray-500">No trend data available</div>';
            return;
        }

        const traces = [];
        const colors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b'];
        let colorIndex = 0;

        Object.entries(trends).forEach(([task, trendData]) => {
            // Generate sample time series data (in real implementation, this would come from API)
            const dates = this.generateDateRange(30);
            const values = this.generateTrendValues(trendData, dates.length);

            traces.push({
                x: dates,
                y: values,
                type: 'scatter',
                mode: 'lines+markers',
                name: task.split('.').pop(),
                line: { color: colors[colorIndex % colors.length] },
                hovertemplate: '<b>%{fullData.name}</b><br>' +
                              'Date: %{x}<br>' +
                              'Score: %{y:.2f}<br>' +
                              '<extra></extra>'
            });

            // Add forecast if available
            if (trendData.forecast && trendData.forecast.length > 0) {
                const forecastDates = this.generateForecastDates(dates[dates.length - 1], trendData.forecast.length);

                traces.push({
                    x: forecastDates,
                    y: trendData.forecast,
                    type: 'scatter',
                    mode: 'lines',
                    name: `${task.split('.').pop()} (forecast)`,
                    line: {
                        color: colors[colorIndex % colors.length],
                        dash: 'dash'
                    },
                    showlegend: false,
                    hovertemplate: '<b>Forecast</b><br>' +
                                  'Date: %{x}<br>' +
                                  'Predicted Score: %{y:.2f}<br>' +
                                  '<extra></extra>'
                });
            }

            colorIndex++;
        });

        const layout = {
            title: `Performance Trends - ${provider}`,
            xaxis: {
                title: 'Date',
                type: 'date'
            },
            yaxis: {
                title: 'Score',
                range: [0, 1]
            },
            hovermode: 'closest',
            showlegend: true,
            legend: {
                x: 0,
                y: 1,
                bgcolor: 'rgba(255,255,255,0.8)'
            },
            margin: { t: 50, r: 50, b: 50, l: 50 }
        };

        const config = {
            responsive: true,
            displayModeBar: false
        };

        Plotly.newPlot(chartDiv, traces, layout, config);
        this.charts.trend = chartDiv;
    }

    updateComparisonChart(selectedTask = null) {
        const chartDiv = document.getElementById('comparison-chart');

        if (!this.data.performance || !this.data.performance.by_provider) {
            chartDiv.innerHTML = '<div class="flex items-center justify-center h-full text-gray-500">No comparison data available</div>';
            return;
        }

        const providers = Object.keys(this.data.performance.by_provider);
        const scores = providers.map(provider =>
            this.data.performance.by_provider[provider].mean_score * 100
        );
        const errors = providers.map(provider =>
            this.data.performance.by_provider[provider].std_score * 100
        );

        const trace = {
            x: providers.map(p => p.charAt(0).toUpperCase() + p.slice(1)),
            y: scores,
            error_y: {
                type: 'data',
                array: errors,
                visible: true
            },
            type: 'bar',
            marker: {
                color: ['#3b82f6', '#ef4444', '#10b981', '#f59e0b'].slice(0, providers.length)
            },
            hovertemplate: '<b>%{x}</b><br>' +
                          'Mean Score: %{y:.1f}%<br>' +
                          'Std Dev: %{error_y.array:.1f}%<br>' +
                          '<extra></extra>'
        };

        const layout = {
            title: selectedTask ? `Provider Comparison - ${selectedTask}` : 'Overall Provider Comparison',
            xaxis: { title: 'Provider' },
            yaxis: {
                title: 'Average Score (%)',
                range: [0, 100]
            },
            showlegend: false,
            margin: { t: 50, r: 50, b: 50, l: 50 }
        };

        const config = {
            responsive: true,
            displayModeBar: false
        };

        Plotly.newPlot(chartDiv, [trace], layout, config);
        this.charts.comparison = chartDiv;
    }

    updateInsights(insights) {
        const container = document.getElementById('insights-list');
        const countElement = document.getElementById('insights-count');

        if (!insights || insights.length === 0) {
            container.innerHTML = '<div class="text-center text-gray-500 py-8">No insights available</div>';
            countElement.textContent = '0 insights';
            return;
        }

        countElement.textContent = `${insights.length} insight${insights.length !== 1 ? 's' : ''}`;

        container.innerHTML = insights.map(insight => `
            <div class="insight-card bg-gray-50 p-4 rounded-lg severity-${insight.severity}">
                <div class="flex items-start justify-between">
                    <div class="flex-1">
                        <div class="flex items-center mb-2">
                            <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${this.getSeverityClasses(insight.severity)}">
                                ${this.getSeverityIcon(insight.severity)} ${insight.severity.toUpperCase()}
                            </span>
                            <span class="ml-2 text-xs text-gray-500">${insight.category}</span>
                        </div>
                        <h4 class="text-sm font-semibold text-gray-900 mb-1">${insight.title}</h4>
                        <p class="text-sm text-gray-600 mb-3">${insight.description}</p>

                        ${insight.recommendations && insight.recommendations.length > 0 ? `
                            <div class="mb-2">
                                <h5 class="text-xs font-medium text-gray-700 mb-1">Recommendations:</h5>
                                <ul class="text-xs text-gray-600 space-y-1">
                                    ${insight.recommendations.slice(0, 3).map(rec => `<li>• ${rec}</li>`).join('')}
                                </ul>
                            </div>
                        ` : ''}

                        <div class="flex items-center justify-between text-xs text-gray-500">
                            <span>Confidence: ${(insight.confidence * 100).toFixed(0)}%</span>
                            <span>${new Date(insight.timestamp).toLocaleString()}</span>
                        </div>
                    </div>
                </div>
            </div>
        `).join('');
    }

    async checkRegressions() {
        try {
            const response = await fetch('/api/regression-check');
            if (!response.ok) return;

            const result = await response.json();
            this.updateRegressionAlerts(result.regression_alert);
        } catch (error) {
            console.error('Error checking regressions:', error);
        }
    }

    updateRegressionAlerts(alert) {
        const container = document.getElementById('regression-alerts');

        if (!alert || !alert.detected) {
            container.innerHTML = '<div class="text-center text-green-600 py-4"><i class="fas fa-check-circle mr-2"></i>No regressions detected</div>';
            return;
        }

        container.innerHTML = `
            <div class="bg-red-50 border border-red-200 rounded-lg p-4">
                <div class="flex items-center mb-2">
                    <i class="fas fa-exclamation-triangle text-red-500 mr-2"></i>
                    <span class="font-semibold text-red-800">${alert.severity.toUpperCase()} Regression</span>
                </div>
                <p class="text-sm text-red-700 mb-2">
                    Performance drop: ${(alert.performance_drop * 100).toFixed(1)}%
                </p>
                <p class="text-sm text-red-600 mb-3">
                    Affected: ${alert.affected_tasks.join(', ')}
                </p>
                <div class="text-xs text-red-600">
                    Confidence: ${(alert.confidence * 100).toFixed(0)}%
                </div>
            </div>
        `;
    }

    getSeverityClasses(severity) {
        const classes = {
            critical: 'bg-red-100 text-red-800',
            warning: 'bg-yellow-100 text-yellow-800',
            info: 'bg-blue-100 text-blue-800'
        };
        return classes[severity] || classes.info;
    }

    getSeverityIcon(severity) {
        const icons = {
            critical: '<i class="fas fa-exclamation-circle"></i>',
            warning: '<i class="fas fa-exclamation-triangle"></i>',
            info: '<i class="fas fa-info-circle"></i>'
        };
        return icons[severity] || icons.info;
    }

    generateDateRange(days) {
        const dates = [];
        const now = new Date();

        for (let i = days - 1; i >= 0; i--) {
            const date = new Date(now);
            date.setDate(date.getDate() - i);
            dates.push(date.toISOString().split('T')[0]);
        }

        return dates;
    }

    generateForecastDates(lastDate, periods) {
        const dates = [];
        const baseDate = new Date(lastDate);

        for (let i = 1; i <= periods; i++) {
            const date = new Date(baseDate);
            date.setDate(date.getDate() + i);
            dates.push(date.toISOString().split('T')[0]);
        }

        return dates;
    }

    generateTrendValues(trendData, length) {
        // Generate realistic trend data based on trend analysis
        const values = [];
        const baseScore = 0.75;
        const slope = trendData.slope || 0;
        const volatility = trendData.volatility || 0.1;

        for (let i = 0; i < length; i++) {
            const trendValue = baseScore + (slope * i);
            const noise = (Math.random() - 0.5) * volatility;
            values.push(Math.max(0, Math.min(1, trendValue + noise)));
        }

        return values;
    }

    setupRealTimeUpdates() {
        setInterval(() => {
            this.loadData();
        }, this.refreshInterval);
    }

    updateLastUpdated() {
        const now = new Date();
        document.getElementById('last-updated').textContent = now.toLocaleTimeString();
    }

    showLoading() {
        document.getElementById('loading-overlay').classList.remove('hidden');
    }

    hideLoading() {
        document.getElementById('loading-overlay').classList.add('hidden');
    }

    showError(message) {
        // Simple error display - in production, use a proper notification system
        console.error(message);
        alert(`Error: ${message}`);
    }

    initializeCharts() {
        // Initialize empty charts
        this.updateTrendChart({}, 'No Provider Selected');
        this.updateComparisonChart();
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new AnalyticsDashboard();
});
