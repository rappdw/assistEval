# Stage 13 Implementation Prompt: Production Optimization & Enterprise Features

## Role: Principal Software Engineer & Enterprise Architect

You are a principal software engineer with extensive experience in enterprise software architecture, performance optimization, and production systems. Your task is to implement Stage 13 of the ChatGPT vs Microsoft Copilot evaluation harness - Production Optimization & Enterprise Features.

## Context

The evaluation harness with advanced analytics (Stages 1-12) is complete. Stage 13 transforms it into an enterprise-grade production system with:
- Performance optimization (caching, async processing, database tuning)
- Multi-tenancy with complete data isolation
- Enterprise security (SAML SSO, MFA, audit logging)
- Horizontal scalability with microservices
- Comprehensive monitoring and alerting
- Compliance features (GDPR, SOC2)

## Implementation Requirements

### Core Architecture Principles

1. **Performance First**: Sub-second response times under production load
2. **Security by Design**: Zero-trust architecture with defense in depth
3. **Scalability**: Horizontal scaling to support 1000+ concurrent users
4. **Observability**: Complete visibility into system behavior
5. **Compliance**: Built-in GDPR and SOC2 compliance features

### Technical Specifications

#### 1. Performance Optimization (`bench/optimization/`)

**Caching Strategy**:
- Implement Redis-based distributed caching with intelligent cache invalidation
- Add cache warming strategies for frequently accessed data
- Implement cache-aside pattern with automatic fallback
- Support cache clustering for high availability

**Async Processing**:
- Use Celery with Redis broker for background task processing
- Implement task prioritization and retry mechanisms
- Add progress tracking for long-running evaluations
- Support distributed task execution across multiple workers

**Database Optimization**:
- Implement connection pooling with pgbouncer
- Add read replica routing for analytics queries
- Create composite indexes for common query patterns
- Implement query result caching with automatic invalidation

#### 2. Multi-Tenancy (`bench/enterprise/tenancy.py`)

**Tenant Isolation**:
- Implement row-level security (RLS) in PostgreSQL
- Add tenant-aware query filters at ORM level
- Support tenant-specific configuration overrides
- Implement resource quotas and usage tracking

**Tenant Management**:
- Support tenant onboarding with automated provisioning
- Implement tenant tier management (Basic, Professional, Enterprise)
- Add tenant-specific branding and customization
- Support tenant data export and migration

#### 3. Enterprise Security (`bench/enterprise/security.py`)

**Authentication & Authorization**:
- Implement SAML 2.0 SSO with multiple identity providers
- Add OAuth2/OIDC support for API access
- Implement role-based access control (RBAC) with fine-grained permissions
- Support multi-factor authentication (TOTP, SMS, hardware tokens)

**Security Hardening**:
- Implement API rate limiting with tenant-specific limits
- Add comprehensive input validation and sanitization
- Support API key management with automatic rotation
- Implement security headers and CSRF protection

#### 4. Microservices Architecture (`bench/microservices/`)

**Service Decomposition**:
- Evaluation Service: Handle evaluation processing
- Analytics Service: Compute trends and insights
- User Service: Manage authentication and authorization
- Notification Service: Handle alerts and notifications

**Service Communication**:
- Use async messaging with Redis Streams
- Implement service discovery with health checks
- Add circuit breakers to prevent cascade failures
- Support distributed tracing with OpenTelemetry

#### 5. Monitoring & Observability (`bench/monitoring/`)

**Metrics Collection**:
- Implement Prometheus metrics for all services
- Add custom business metrics (evaluations/hour, user activity)
- Support distributed tracing across microservices
- Implement structured logging with correlation IDs

**Alerting System**:
- Configure alerting rules for performance degradation
- Implement escalation policies based on severity
- Support multiple notification channels (Slack, email, PagerDuty)
- Add alert suppression and acknowledgment

### Implementation Guidelines

#### Performance Optimization Strategy

1. **Caching Implementation**:
```python
class IntelligentCache:
    """Multi-layer caching with automatic invalidation."""

    def __init__(self, redis_client, local_cache_size=1000):
        self.redis = redis_client
        self.local_cache = LRUCache(local_cache_size)

    async def get_with_fallback(self, key: str, fetch_func: Callable):
        """Get from cache with automatic fallback to source."""
        # L1: Local cache
        if value := self.local_cache.get(key):
            return value

        # L2: Redis cache
        if value := await self.redis.get(key):
            self.local_cache[key] = value
            return value

        # L3: Source data with cache population
        value = await fetch_func()
        await self.set_multi_layer(key, value)
        return value
```

2. **Async Processing Architecture**:
```python
@celery_app.task(bind=True, max_retries=3)
def process_evaluation_async(self, evaluation_config: Dict) -> str:
    """Process evaluation with progress tracking."""
    try:
        # Update progress
        self.update_state(state='PROGRESS', meta={'current': 0, 'total': 100})

        # Process evaluation with progress updates
        result = process_evaluation_with_progress(
            evaluation_config,
            progress_callback=lambda p: self.update_state(
                state='PROGRESS',
                meta={'current': p, 'total': 100}
            )
        )

        return result
    except Exception as exc:
        self.retry(countdown=60, exc=exc)
```

#### Multi-Tenancy Implementation

1. **Row-Level Security**:
```sql
-- Enable RLS on evaluation results
ALTER TABLE evaluation_results ENABLE ROW LEVEL SECURITY;

-- Create policy for tenant isolation
CREATE POLICY tenant_isolation ON evaluation_results
    FOR ALL TO application_role
    USING (tenant_id = current_setting('app.current_tenant_id'));
```

2. **Tenant-Aware ORM**:
```python
class TenantAwareQuery:
    """Automatically filter queries by tenant."""

    def __init__(self, session, tenant_id: str):
        self.session = session
        self.tenant_id = tenant_id

    def filter_by_tenant(self, query):
        """Add tenant filter to all queries."""
        return query.filter(self.model.tenant_id == self.tenant_id)
```

#### Enterprise Security Implementation

1. **SAML SSO Integration**:
```python
class SAMLAuthProvider:
    """SAML 2.0 authentication provider."""

    def __init__(self, saml_settings: Dict):
        self.auth = OneLogin_Saml2_Auth(saml_settings)

    def initiate_sso(self, return_url: str) -> str:
        """Initiate SAML SSO flow."""
        return self.auth.login(return_to=return_url)

    def process_sso_response(self, saml_response: str) -> Dict:
        """Process SAML response and extract user info."""
        self.auth.process_response()
        if self.auth.is_authenticated():
            return {
                'user_id': self.auth.get_nameid(),
                'attributes': self.auth.get_attributes(),
                'session_index': self.auth.get_session_index()
            }
        raise AuthenticationError("SAML authentication failed")
```

2. **API Rate Limiting**:
```python
class TenantAwareRateLimiter:
    """Rate limiting with tenant-specific limits."""

    def __init__(self, redis_client):
        self.redis = redis_client

    async def check_rate_limit(self, tenant_id: str, endpoint: str) -> bool:
        """Check if request is within rate limits."""
        limits = await self.get_tenant_limits(tenant_id)
        key = f"rate_limit:{tenant_id}:{endpoint}"

        current = await self.redis.incr(key)
        if current == 1:
            await self.redis.expire(key, 3600)  # 1 hour window

        return current <= limits.get(endpoint, 1000)
```

#### Microservices Implementation

1. **Service Base Class**:
```python
class BaseService:
    """Base class for all microservices."""

    def __init__(self, service_name: str):
        self.service_name = service_name
        self.app = FastAPI(title=service_name)
        self.setup_middleware()
        self.setup_health_checks()
        self.setup_metrics()

    def setup_middleware(self):
        """Setup common middleware."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"]
        )
        self.app.add_middleware(PrometheusMiddleware)
        self.app.add_middleware(TracingMiddleware)

    def setup_health_checks(self):
        """Setup health check endpoints."""
        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "service": self.service_name}
```

2. **Circuit Breaker Pattern**:
```python
class CircuitBreaker:
    """Circuit breaker for service resilience."""

    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
            else:
                raise CircuitBreakerOpenError()

        try:
            result = await func(*args, **kwargs)
            self.on_success()
            return result
        except Exception as e:
            self.on_failure()
            raise e
```

### Deployment Architecture

#### Kubernetes Deployment

1. **Service Deployment**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: evaluation-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: evaluation-service
  template:
    metadata:
      labels:
        app: evaluation-service
    spec:
      containers:
      - name: evaluation-service
        image: evaluation-harness:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: url
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
```

2. **Horizontal Pod Autoscaler**:
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: evaluation-service-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: evaluation-service
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Testing Strategy

#### Performance Testing
```python
class TestPerformanceOptimization:
    """Comprehensive performance testing."""

    async def test_cache_performance_improvement(self):
        """Verify caching improves response times by 10x."""
        # Measure without cache
        start = time.time()
        result1 = await get_analytics_data(provider="chatgpt", days=30)
        uncached_time = time.time() - start

        # Measure with cache
        start = time.time()
        result2 = await get_analytics_data(provider="chatgpt", days=30)
        cached_time = time.time() - start

        assert cached_time < uncached_time / 10
        assert result1 == result2

    async def test_concurrent_user_performance(self):
        """Test system handles 1000 concurrent users."""
        async def simulate_user():
            async with aiohttp.ClientSession() as session:
                async with session.get("/api/dashboard") as resp:
                    assert resp.status == 200
                    assert (await resp.json())["status"] == "success"

        # Simulate 1000 concurrent users
        tasks = [simulate_user() for _ in range(1000)]
        start = time.time()
        await asyncio.gather(*tasks)
        total_time = time.time() - start

        assert total_time < 10  # All requests complete within 10 seconds
```

#### Security Testing
```python
class TestEnterpriseSecurity:
    """Enterprise security validation."""

    def test_tenant_data_isolation(self):
        """Verify complete tenant data isolation."""
        # Create data for tenant A
        tenant_a_data = create_evaluation_result(tenant_id="tenant_a")

        # Try to access from tenant B context
        with tenant_context("tenant_b"):
            results = get_evaluation_results()
            assert tenant_a_data.id not in [r.id for r in results]

    def test_saml_sso_integration(self):
        """Test SAML SSO authentication flow."""
        # Mock SAML response
        saml_response = create_mock_saml_response(
            user_id="test@company.com",
            attributes={"role": "admin", "tenant": "enterprise"}
        )

        # Process SSO
        user_info = process_saml_response(saml_response)
        assert user_info["user_id"] == "test@company.com"
        assert user_info["role"] == "admin"
```

## Success Criteria

### Performance Requirements
- [ ] 10x response time improvement with caching enabled
- [ ] Support 1000+ concurrent users with <2s response times
- [ ] Process 10,000+ evaluations per hour
- [ ] Database queries average <100ms

### Enterprise Requirements
- [ ] Complete tenant data isolation verified
- [ ] SAML SSO integration functional with major providers
- [ ] Comprehensive audit logging captures all actions
- [ ] GDPR compliance features operational

### Scalability Requirements
- [ ] Horizontal scaling verified up to 20 service instances
- [ ] Load balancing distributes traffic evenly
- [ ] Circuit breakers prevent cascade failures
- [ ] Auto-scaling responds to load within 2 minutes

### Security Requirements
- [ ] Zero-trust architecture implemented
- [ ] API rate limiting enforces tenant quotas
- [ ] All communications encrypted (TLS 1.3)
- [ ] Security headers prevent common attacks

## Implementation Phases

### Phase 1: Performance Optimization (2 hours)
1. Implement Redis caching layer with intelligent invalidation
2. Add Celery async processing with progress tracking
3. Optimize database queries and add connection pooling
4. Implement comprehensive performance monitoring

### Phase 2: Multi-Tenancy (2 hours)
1. Implement row-level security and tenant isolation
2. Add tenant management and provisioning
3. Implement tenant-specific resource quotas
4. Add tenant-aware query filtering

### Phase 3: Enterprise Security (2 hours)
1. Implement SAML SSO with multiple identity providers
2. Add comprehensive audit logging
3. Implement API rate limiting and security hardening
4. Add GDPR compliance features

### Phase 4: Microservices & Scalability (2 hours)
1. Decompose monolith into microservices
2. Implement service discovery and health checks
3. Add circuit breakers and resilience patterns
4. Configure Kubernetes deployment with auto-scaling

Remember: This is an enterprise-grade production system that will handle sensitive evaluation data for multiple organizations. Prioritize security, performance, and reliability. Every component must be production-ready with comprehensive error handling, monitoring, and documentation.
