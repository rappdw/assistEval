# Stage 13 QA Prompt: Production Optimization & Enterprise Features

## Role: Senior QA Engineer & Enterprise Systems Validator

You are a senior QA engineer with specialized expertise in enterprise software validation, performance testing, security auditing, and production system quality assurance. Your mission is to comprehensively validate Stage 13 of the ChatGPT vs Microsoft Copilot evaluation harness - Production Optimization & Enterprise Features.

## QA Scope & Objectives

### Primary Validation Areas
1. **Performance Optimization**: Validate caching, async processing, and database optimization
2. **Multi-Tenancy**: Ensure complete data isolation and tenant management
3. **Enterprise Security**: Validate SAML SSO, audit logging, and compliance features
4. **Scalability**: Test horizontal scaling and microservices architecture
5. **Production Readiness**: Validate monitoring, alerting, and operational capabilities
6. **Compliance**: Ensure GDPR, SOC2, and enterprise audit requirements

## Critical Test Categories

### 1. Performance Optimization Testing

#### Caching Validation
```python
class TestCachingPerformance:
    """Validate caching layer performance and correctness."""

    def test_cache_hit_performance(self):
        """Validate cache hits are 10x faster than database queries."""
        # Cold cache - measure database query time
        start = time.time()
        result1 = get_evaluation_data(provider="chatgpt", days=30)
        db_time = time.time() - start

        # Warm cache - measure cached query time
        start = time.time()
        result2 = get_evaluation_data(provider="chatgpt", days=30)
        cache_time = time.time() - start

        assert cache_time < db_time / 10
        assert result1 == result2

    def test_cache_invalidation_correctness(self):
        """Validate cache invalidation maintains data consistency."""
        # Cache initial data
        cached_data = get_evaluation_data(evaluation_id="test_123")

        # Update underlying data
        update_evaluation_result("test_123", {"score": 95})

        # Verify cache is invalidated and fresh data returned
        fresh_data = get_evaluation_data(evaluation_id="test_123")
        assert fresh_data["score"] == 95
        assert fresh_data != cached_data

    def test_cache_memory_management(self):
        """Validate cache doesn't cause memory leaks."""
        initial_memory = get_memory_usage()

        # Generate large amount of cached data
        for i in range(10000):
            get_evaluation_data(evaluation_id=f"test_{i}")

        # Force cache eviction
        trigger_cache_cleanup()

        final_memory = get_memory_usage()
        assert final_memory < initial_memory * 1.1  # Max 10% increase
```

#### Async Processing Validation
```python
class TestAsyncProcessing:
    """Validate async processing performance and reliability."""

    def test_async_task_throughput(self):
        """Validate system processes 10,000+ evaluations per hour."""
        start_time = time.time()
        task_ids = []

        # Submit 1000 evaluation tasks
        for i in range(1000):
            task_id = submit_evaluation_task({
                "provider": "chatgpt",
                "test_case": f"test_{i}"
            })
            task_ids.append(task_id)

        # Wait for completion
        completed_tasks = wait_for_task_completion(task_ids, timeout=360)  # 6 minutes
        total_time = time.time() - start_time

        assert len(completed_tasks) == 1000
        assert total_time < 360  # All tasks complete within 6 minutes

        # Calculate throughput (tasks per hour)
        throughput = (1000 / total_time) * 3600
        assert throughput > 10000

    def test_task_retry_mechanism(self):
        """Validate failed tasks are retried appropriately."""
        # Submit task that will fail initially
        task_id = submit_evaluation_task({
            "provider": "failing_provider",
            "test_case": "retry_test",
            "max_retries": 3
        })

        # Monitor task retries
        retry_count = monitor_task_retries(task_id, timeout=300)
        assert retry_count == 3

        # Verify final failure is recorded
        task_result = get_task_result(task_id)
        assert task_result["status"] == "failed"
        assert task_result["retry_count"] == 3
```

### 2. Multi-Tenancy Validation

#### Data Isolation Testing
```python
class TestTenantIsolation:
    """Validate complete tenant data isolation."""

    def test_row_level_security_enforcement(self):
        """Validate RLS prevents cross-tenant data access."""
        # Create data for tenant A
        with tenant_context("tenant_a"):
            eval_a = create_evaluation_result({
                "provider": "chatgpt",
                "score": 85,
                "tenant_id": "tenant_a"
            })

        # Try to access from tenant B context
        with tenant_context("tenant_b"):
            results = query_evaluation_results()
            tenant_b_ids = [r.id for r in results]
            assert eval_a.id not in tenant_b_ids

        # Verify tenant A can still access their data
        with tenant_context("tenant_a"):
            results = query_evaluation_results()
            tenant_a_ids = [r.id for r in results]
            assert eval_a.id in tenant_a_ids

    def test_api_endpoint_isolation(self):
        """Validate API endpoints enforce tenant isolation."""
        # Create evaluation for tenant A
        tenant_a_token = get_tenant_api_token("tenant_a")
        response_a = requests.post("/api/evaluations",
            headers={"Authorization": f"Bearer {tenant_a_token}"},
            json={"provider": "chatgpt", "test_case": "isolation_test"}
        )
        eval_id = response_a.json()["id"]

        # Try to access with tenant B token
        tenant_b_token = get_tenant_api_token("tenant_b")
        response_b = requests.get(f"/api/evaluations/{eval_id}",
            headers={"Authorization": f"Bearer {tenant_b_token}"}
        )
        assert response_b.status_code == 404  # Not found for tenant B

    def test_resource_quota_enforcement(self):
        """Validate tenant resource quotas are enforced."""
        # Set tenant quota to 100 evaluations per month
        set_tenant_quota("tenant_test", {"evaluations_per_month": 100})

        # Submit 100 evaluations (should succeed)
        for i in range(100):
            response = submit_evaluation("tenant_test", f"test_{i}")
            assert response.status_code == 200

        # Submit 101st evaluation (should fail)
        response = submit_evaluation("tenant_test", "test_101")
        assert response.status_code == 429  # Rate limited
        assert "quota exceeded" in response.json()["error"]
```

### 3. Enterprise Security Testing

#### SAML SSO Validation
```python
class TestSAMLSSO:
    """Validate SAML SSO integration."""

    def test_saml_authentication_flow(self):
        """Test complete SAML authentication flow."""
        # Initiate SSO
        sso_url = initiate_saml_sso(return_url="/dashboard")
        assert "saml" in sso_url.lower()

        # Mock SAML response from IdP
        saml_response = create_mock_saml_response(
            user_id="test.user@company.com",
            attributes={
                "role": "admin",
                "tenant": "enterprise_tenant",
                "email": "test.user@company.com"
            }
        )

        # Process SAML response
        auth_result = process_saml_response(saml_response)
        assert auth_result["authenticated"] == True
        assert auth_result["user_id"] == "test.user@company.com"
        assert auth_result["role"] == "admin"

    def test_saml_attribute_mapping(self):
        """Validate SAML attribute mapping to user roles."""
        test_cases = [
            {"saml_role": "admin", "expected_role": "administrator"},
            {"saml_role": "user", "expected_role": "standard_user"},
            {"saml_role": "viewer", "expected_role": "read_only"}
        ]

        for case in test_cases:
            saml_response = create_mock_saml_response(
                user_id="test@company.com",
                attributes={"role": case["saml_role"]}
            )

            user_info = process_saml_response(saml_response)
            assert user_info["role"] == case["expected_role"]
```

#### Audit Logging Validation
```python
class TestAuditLogging:
    """Validate comprehensive audit logging."""

    def test_evaluation_access_logging(self):
        """Validate evaluation access is logged."""
        # Access evaluation data
        user_token = get_user_token("test_user", "tenant_a")
        response = requests.get("/api/evaluations/eval_123",
            headers={"Authorization": f"Bearer {user_token}"}
        )

        # Verify audit log entry
        audit_logs = get_audit_logs(
            user_id="test_user",
            action="evaluation_access",
            timeframe="last_5_minutes"
        )

        assert len(audit_logs) == 1
        assert audit_logs[0]["resource"] == "eval_123"
        assert audit_logs[0]["action"] == "evaluation_access"
        assert audit_logs[0]["user_id"] == "test_user"

    def test_configuration_change_logging(self):
        """Validate configuration changes are logged."""
        # Make configuration change
        admin_token = get_admin_token("admin_user")
        response = requests.put("/api/config/analytics",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={"cache_ttl": 7200}
        )

        # Verify audit log entry
        audit_logs = get_audit_logs(
            action="configuration_change",
            timeframe="last_5_minutes"
        )

        assert len(audit_logs) == 1
        assert audit_logs[0]["details"]["field"] == "cache_ttl"
        assert audit_logs[0]["details"]["old_value"] == 3600
        assert audit_logs[0]["details"]["new_value"] == 7200
```

### 4. Scalability & Microservices Testing

#### Horizontal Scaling Validation
```python
class TestHorizontalScaling:
    """Validate horizontal scaling capabilities."""

    def test_auto_scaling_response(self):
        """Validate auto-scaling responds to load increases."""
        # Get initial pod count
        initial_pods = get_pod_count("evaluation-service")

        # Generate high load
        load_generator = LoadGenerator(
            target_url="/api/evaluations",
            concurrent_users=500,
            duration=300  # 5 minutes
        )

        with load_generator:
            # Wait for auto-scaling to trigger
            time.sleep(120)  # 2 minutes

            # Verify pods have scaled up
            scaled_pods = get_pod_count("evaluation-service")
            assert scaled_pods > initial_pods

            # Verify response times remain acceptable
            avg_response_time = load_generator.get_avg_response_time()
            assert avg_response_time < 2.0  # Under 2 seconds

    def test_load_balancing_distribution(self):
        """Validate load balancer distributes traffic evenly."""
        # Send 1000 requests
        service_instances = get_service_instances("evaluation-service")
        request_counts = {instance: 0 for instance in service_instances}

        for _ in range(1000):
            response = requests.get("/api/health")
            instance_id = response.headers.get("X-Instance-ID")
            if instance_id in request_counts:
                request_counts[instance_id] += 1

        # Verify even distribution (within 10% variance)
        expected_per_instance = 1000 / len(service_instances)
        for instance, count in request_counts.items():
            variance = abs(count - expected_per_instance) / expected_per_instance
            assert variance < 0.1  # Within 10%
```

#### Circuit Breaker Testing
```python
class TestCircuitBreaker:
    """Validate circuit breaker prevents cascade failures."""

    def test_circuit_breaker_opens_on_failures(self):
        """Validate circuit breaker opens after failure threshold."""
        # Configure circuit breaker with low threshold for testing
        configure_circuit_breaker("analytics-service", {
            "failure_threshold": 3,
            "timeout": 60
        })

        # Cause service failures
        with mock_service_failure("analytics-service"):
            # Make requests that will fail
            for i in range(5):
                try:
                    call_analytics_service("trend_analysis", {"provider": "chatgpt"})
                except Exception:
                    pass

        # Verify circuit breaker is open
        circuit_state = get_circuit_breaker_state("analytics-service")
        assert circuit_state == "OPEN"

        # Verify subsequent calls fail fast
        start_time = time.time()
        try:
            call_analytics_service("trend_analysis", {"provider": "chatgpt"})
        except CircuitBreakerOpenError:
            pass

        call_time = time.time() - start_time
        assert call_time < 0.1  # Fails fast (under 100ms)
```

### 5. Production Monitoring Testing

#### Metrics Collection Validation
```python
class TestMetricsCollection:
    """Validate comprehensive metrics collection."""

    def test_prometheus_metrics_accuracy(self):
        """Validate Prometheus metrics match actual system behavior."""
        # Reset metrics
        reset_prometheus_metrics()

        # Perform known operations
        for i in range(10):
            submit_evaluation_task({"provider": "chatgpt", "test_case": f"test_{i}"})

        # Wait for metrics to be collected
        time.sleep(30)

        # Verify metrics accuracy
        metrics = get_prometheus_metrics()
        assert metrics["evaluations_total"]["chatgpt"] == 10
        assert metrics["active_tasks_gauge"] >= 0

    def test_custom_business_metrics(self):
        """Validate custom business metrics are collected."""
        # Perform business operations
        create_tenant("new_tenant")
        submit_evaluation_task({"provider": "chatgpt", "tenant": "new_tenant"})

        # Verify business metrics
        metrics = get_business_metrics()
        assert "tenant_count" in metrics
        assert "evaluations_per_tenant" in metrics
        assert metrics["evaluations_per_tenant"]["new_tenant"] == 1
```

#### Alerting System Validation
```python
class TestAlertingSystem:
    """Validate alerting system functionality."""

    def test_performance_degradation_alerts(self):
        """Validate alerts trigger on performance degradation."""
        # Configure alert rule
        configure_alert_rule({
            "name": "high_response_time",
            "condition": "avg_response_time > 5.0",
            "severity": "warning",
            "notification_channels": ["test_channel"]
        })

        # Simulate performance degradation
        with mock_slow_responses(delay=6.0):
            # Generate requests to trigger alert
            for _ in range(20):
                requests.get("/api/evaluations")
                time.sleep(1)

        # Verify alert was triggered
        alerts = get_triggered_alerts(timeframe="last_5_minutes")
        assert any(alert["name"] == "high_response_time" for alert in alerts)

    def test_alert_escalation_policies(self):
        """Validate alert escalation based on severity."""
        # Configure escalation policy
        configure_escalation_policy({
            "critical": ["pagerduty", "sms"],
            "warning": ["slack", "email"],
            "info": ["email"]
        })

        # Trigger critical alert
        trigger_test_alert("critical", "System down")

        # Verify escalation
        notifications = get_sent_notifications(timeframe="last_1_minute")
        assert any(n["channel"] == "pagerduty" for n in notifications)
        assert any(n["channel"] == "sms" for n in notifications)
```

### 6. Compliance Testing

#### GDPR Compliance Validation
```python
class TestGDPRCompliance:
    """Validate GDPR compliance features."""

    def test_data_retention_policies(self):
        """Validate data retention policies are enforced."""
        # Create old evaluation data
        old_evaluation = create_evaluation_result({
            "provider": "chatgpt",
            "created_at": datetime.now() - timedelta(days=2556)  # Over 7 years
        })

        # Run data retention cleanup
        run_data_retention_cleanup()

        # Verify old data is deleted
        with pytest.raises(NotFoundError):
            get_evaluation_result(old_evaluation.id)

    def test_right_to_be_forgotten(self):
        """Validate user data deletion requests."""
        # Create user data
        user_id = "gdpr_test_user"
        create_user_evaluation_data(user_id, {
            "evaluations": [{"provider": "chatgpt", "score": 85}],
            "preferences": {"theme": "dark"}
        })

        # Submit deletion request
        deletion_request_id = submit_data_deletion_request(user_id)

        # Process deletion
        process_deletion_request(deletion_request_id)

        # Verify all user data is deleted
        user_data = get_user_data(user_id)
        assert user_data is None

    def test_data_export_functionality(self):
        """Validate user data export for GDPR compliance."""
        user_id = "export_test_user"

        # Create user data
        create_user_evaluation_data(user_id, {
            "evaluations": [{"provider": "chatgpt", "score": 90}],
            "analytics": [{"trend": "improving"}]
        })

        # Request data export
        export_data = export_user_data(user_id)

        # Verify export completeness
        assert "evaluations" in export_data
        assert "analytics" in export_data
        assert len(export_data["evaluations"]) == 1
        assert export_data["evaluations"][0]["score"] == 90
```

## Load Testing Strategy

### Comprehensive Load Testing
```python
class TestProductionLoad:
    """Comprehensive production load testing."""

    def test_concurrent_user_capacity(self):
        """Test system capacity with 1000+ concurrent users."""
        load_test = LoadTest(
            target_url="https://evaluation-harness.com",
            scenarios=[
                {"name": "dashboard_access", "weight": 40, "users": 400},
                {"name": "evaluation_submission", "weight": 30, "users": 300},
                {"name": "analytics_queries", "weight": 20, "users": 200},
                {"name": "report_generation", "weight": 10, "users": 100}
            ],
            duration=1800  # 30 minutes
        )

        results = load_test.run()

        # Validate performance requirements
        assert results["avg_response_time"] < 2.0
        assert results["95th_percentile"] < 5.0
        assert results["error_rate"] < 0.01  # Less than 1%

    def test_database_performance_under_load(self):
        """Test database performance under high load."""
        # Generate concurrent database operations
        async def db_operation():
            return await query_evaluation_results(limit=100)

        # Run 500 concurrent queries
        start_time = time.time()
        tasks = [db_operation() for _ in range(500)]
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time

        # Validate performance
        assert total_time < 30  # All queries complete within 30 seconds
        assert len(results) == 500
        assert all(len(result) <= 100 for result in results)
```

## Security Testing

### Penetration Testing
```python
class TestSecurityVulnerabilities:
    """Security vulnerability testing."""

    def test_sql_injection_protection(self):
        """Test protection against SQL injection attacks."""
        malicious_inputs = [
            "'; DROP TABLE evaluations; --",
            "1' OR '1'='1",
            "1; SELECT * FROM users; --"
        ]

        for malicious_input in malicious_inputs:
            response = requests.get(f"/api/evaluations?id={malicious_input}")
            # Should return error, not execute malicious SQL
            assert response.status_code in [400, 404]

        # Verify database integrity
        table_count = get_database_table_count()
        assert table_count > 0  # Tables still exist

    def test_xss_protection(self):
        """Test protection against XSS attacks."""
        xss_payloads = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>"
        ]

        for payload in xss_payloads:
            response = requests.post("/api/evaluations", json={
                "provider": "chatgpt",
                "notes": payload
            })

            # Verify payload is sanitized
            if response.status_code == 200:
                evaluation = response.json()
                assert "<script>" not in evaluation["notes"]
                assert "javascript:" not in evaluation["notes"]
```

## Acceptance Criteria

### Performance Requirements
- [ ] 10x response time improvement with caching
- [ ] Support 1000+ concurrent users with <2s response times
- [ ] Process 10,000+ evaluations per hour
- [ ] Database queries average <100ms

### Enterprise Requirements
- [ ] Complete tenant data isolation verified
- [ ] SAML SSO integration functional
- [ ] Comprehensive audit logging operational
- [ ] GDPR compliance features working

### Scalability Requirements
- [ ] Horizontal scaling verified up to 20 instances
- [ ] Load balancing distributes traffic evenly
- [ ] Circuit breakers prevent cascade failures
- [ ] Auto-scaling responds within 2 minutes

### Security Requirements
- [ ] Zero-trust architecture implemented
- [ ] All injection attacks prevented
- [ ] API rate limiting enforces quotas
- [ ] Security headers prevent common attacks

### Monitoring Requirements
- [ ] Prometheus metrics accurate and comprehensive
- [ ] Alerting system triggers appropriately
- [ ] Performance monitoring detects degradation
- [ ] Business metrics track key indicators

Remember: This is an enterprise-grade production system handling sensitive data for multiple organizations. Every security control must be thoroughly validated, and performance requirements must be met under realistic production loads. Prioritize security and data integrity above all else.
