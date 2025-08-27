# Stage 4 Implementation Plan: Provider Abstraction & Adapters

**Project**: ChatGPT vs Microsoft Copilot Evaluation Harness
**Stage**: 4 of 11
**Created**: 2025-08-27
**Status**: Ready for Implementation

## Overview
**Objective**: Implement provider interface abstraction and build ChatGPT API integration with manual Copilot adapter
**Priority**: High
**Estimated Effort**: 3-4 hours
**Dependencies**: Stage 3 (Completed)

## Detailed Task Breakdown

### 1. Provider Base Class (`bench/adapters/base.py`)

#### 1.1 Abstract Provider Interface
**Purpose**: Define the contract for all provider implementations

**Implementation Requirements**:
- **Abstract Base Class**: `Provider` using `abc.ABC`
- **Core Method**: `invoke(system: str, user: str, *, options: dict, capabilities: dict) -> dict`
- **Return Format**: `{"raw_text": str}` for consistent response handling
- **Capability Enforcement**: Interface for constraint validation
- **Error Handling**: Base exception classes for provider failures

**Provider Interface Specification**:
```python
from abc import ABC, abstractmethod
from typing import Any

class Provider(ABC):
    """Abstract base class for AI provider implementations."""

    def __init__(self, name: str, **kwargs: Any) -> None:
        """Initialize provider with name and configuration."""
        self.name = name
        self.config = kwargs

    @abstractmethod
    def invoke(
        self,
        system: str,
        user: str,
        *,
        options: dict[str, Any],
        capabilities: dict[str, Any]
    ) -> dict[str, str]:
        """Invoke the provider with prompts and constraints.

        Returns:
            Dictionary with 'raw_text' key containing response
        """
        pass

    def validate_capabilities(self, capabilities: dict[str, Any]) -> None:
        """Validate and enforce capability constraints."""
        pass
```

#### 1.2 Capability Enforcement System
**Capabilities Schema**:
- **`web`**: `"forbidden"` | `"allowed"` | `"required"`
- **`tools`**: List of allowed tools/capabilities
- **`json_required`**: Boolean for JSON response format requirement
- **`timeout_seconds`**: Request timeout configuration

### 2. ChatGPT Provider (`bench/adapters/chatgpt.py`)

#### 2.1 OpenAI API Integration
**Purpose**: Deterministic ChatGPT integration with reproducible settings

**Implementation Requirements**:
- **OpenAI SDK**: Use `openai` library for API calls
- **Deterministic Settings**: temperature=0, seed=42, consistent model
- **JSON Response**: Enforce `response_format="json_object"` when required
- **Error Handling**: Retry logic with exponential backoff
- **Rate Limiting**: Respect OpenAI API limits

**Configuration from `configs/providers.yaml`**:
```yaml
chatgpt:
  adapter: "chatgpt"
  options:
    model: "gpt-4"
    temperature: 0
    max_tokens: 1200
    seed: 42
    response_format: "json_object"
    timeout_seconds: 60
```

#### 2.2 Capability Enforcement
**Implementation Details**:
- **Web Access**: Add system prompt constraints for `web: "forbidden"`
- **JSON Requirement**: Set `response_format` and validate response format
- **Tool Restrictions**: Disable function calling when `tools: []`
- **Prompt Modification**: Inject capability constraints into system prompt

#### 2.3 Error Handling & Retries
**Retry Strategy**:
- **Rate Limits**: Exponential backoff (1s, 2s, 4s, 8s)
- **Network Errors**: 3 retry attempts with jitter
- **API Errors**: Distinguish between retryable and fatal errors
- **Timeout Handling**: Respect configured timeout limits

### 3. Copilot Manual Provider (`bench/adapters/copilot_manual.py`)

#### 3.1 Interactive Prompt Display
**Purpose**: Manual paste mode for Copilot when API is unavailable

**Implementation Requirements**:
- **Prompt Formatting**: Display system and user prompts clearly
- **Capability Display**: Show constraints and requirements
- **Input Collection**: Wait for manual paste of response
- **Response Storage**: Save raw text for evaluation

#### 3.2 User Interface Design
**Display Format**:
```
=== COPILOT EVALUATION PROMPT ===
Provider: Microsoft Copilot (Manual Mode)
Test: {test_name}
Constraints: {capability_summary}

SYSTEM PROMPT:
{system_prompt}

USER PROMPT:
{user_prompt}

=== INSTRUCTIONS ===
1. Copy the prompts above to Microsoft Copilot
2. Paste the complete response below
3. Press Enter twice to submit

Response:
```

#### 3.3 Response Processing
**Features**:
- **Input Validation**: Check for empty responses
- **Format Preservation**: Maintain original formatting
- **Timeout Handling**: Configurable timeout for manual input
- **Artifact Storage**: Save response with metadata

### 4. Capability Constraint System

#### 4.1 Constraint Validation
**Implementation Requirements**:
- **Web Access Validation**: Ensure compliance with `web` setting
- **JSON Format Checking**: Validate response format when required
- **Tool Usage Monitoring**: Track and restrict tool usage
- **Prompt Injection**: Add constraint language to system prompts

#### 4.2 Constraint Enforcement Strategies
**Web Access Control**:
- **Forbidden**: Add "Do not browse the web" to system prompt
- **Required**: Add "Browsing required; provide sources with dates"
- **Allowed**: No additional constraints

**JSON Response Enforcement**:
- **API Level**: Set `response_format="json_object"` for ChatGPT
- **Prompt Level**: Add JSON schema requirements to user prompt
- **Validation**: Parse and validate response format post-processing

### 5. Provider Factory & Registry

#### 5.1 Provider Factory Pattern
**Purpose**: Dynamic provider instantiation from configuration

**Implementation**:
```python
class ProviderFactory:
    """Factory for creating provider instances from configuration."""

    _providers = {
        "chatgpt": ChatGPTProvider,
        "copilot_manual": CopilotManualProvider,
    }

    @classmethod
    def create_provider(cls, config: dict[str, Any]) -> Provider:
        """Create provider instance from configuration."""
        adapter_type = config["adapter"]
        provider_class = cls._providers.get(adapter_type)
        if not provider_class:
            raise ValueError(f"Unknown provider adapter: {adapter_type}")
        return provider_class(**config.get("options", {}))
```

#### 5.2 Configuration Loading
**Features**:
- **YAML Loading**: Parse `configs/providers.yaml`
- **Validation**: Validate against provider schemas
- **Environment Variables**: Support API key injection
- **Error Handling**: Clear error messages for misconfigurations

## Implementation Steps

### Step 1: Base Provider Interface (45 minutes)
1. **Create `bench/adapters/base.py`**:
   - Implement abstract `Provider` class
   - Define `invoke()` method signature
   - Add capability validation interface
   - Create provider exception classes

2. **Add capability enforcement utilities**:
   - Constraint validation functions
   - Prompt modification helpers
   - Response format checkers

### Step 2: ChatGPT Provider Implementation (90 minutes)
1. **Create `bench/adapters/chatgpt.py`**:
   - Implement OpenAI API integration
   - Add deterministic settings (temperature=0, seed=42)
   - Implement JSON response format enforcement
   - Add comprehensive error handling

2. **Add retry logic and rate limiting**:
   - Exponential backoff for rate limits
   - Network error retry with jitter
   - Timeout handling and cancellation

3. **Implement capability enforcement**:
   - Web access constraint injection
   - Tool restriction implementation
   - JSON format validation

### Step 3: Copilot Manual Provider (60 minutes)
1. **Create `bench/adapters/copilot_manual.py`**:
   - Implement interactive prompt display
   - Add manual response collection
   - Create user-friendly interface

2. **Add response processing**:
   - Input validation and formatting
   - Timeout handling for manual input
   - Artifact storage with metadata

### Step 4: Provider Factory & Integration (45 minutes)
1. **Create provider factory**:
   - Dynamic provider instantiation
   - Configuration loading and validation
   - Error handling for unknown adapters

2. **Integration testing**:
   - Test provider creation from config
   - Validate capability enforcement
   - Test error handling scenarios

## Success Criteria

- [ ] **Provider Interface**: Abstract base class with clear contract
- [ ] **ChatGPT Integration**: Working API integration with deterministic settings
- [ ] **Manual Copilot**: Interactive paste mode with user-friendly interface
- [ ] **Capability Enforcement**: Web access, JSON format, and tool restrictions working
- [ ] **Configuration Loading**: Providers created from YAML configuration
- [ ] **Error Handling**: Comprehensive error handling with retries
- [ ] **Testing**: Unit tests for all provider implementations
- [ ] **Documentation**: Clear usage examples and API documentation

## Deliverables

### Core Implementation
- `bench/adapters/base.py` - Abstract provider interface
- `bench/adapters/chatgpt.py` - ChatGPT API integration
- `bench/adapters/copilot_manual.py` - Manual Copilot adapter
- Provider factory and registry system

### Configuration Integration
- Provider loading from `configs/providers.yaml`
- Capability constraint enforcement
- Error handling and validation

### Testing & Documentation
- Unit tests for all provider implementations
- Integration tests with mock APIs
- Usage documentation and examples

## Dependencies

**Required**: Stage 3 (Schema & Configuration Foundation) - Completed
**Enables**: Stage 5 (Core Runner & CLI)

## Technical Requirements

### External Dependencies
- `openai` - OpenAI API client library
- `pydantic` - Data validation and settings management
- `tenacity` - Retry logic with exponential backoff
- `rich` - Enhanced terminal output for manual mode

### Environment Variables
- `OPENAI_API_KEY` - Required for ChatGPT provider
- `OPENAI_ORG_ID` - Optional organization ID

### Error Handling
- **Provider Errors**: Base exception class with specific subtypes
- **API Errors**: Rate limiting, authentication, network failures
- **Configuration Errors**: Invalid settings, missing keys
- **Capability Violations**: Constraint enforcement failures

## Notes

- Provider implementations must be thread-safe for future parallel execution
- All API calls should be logged for debugging and audit purposes
- Manual mode should gracefully handle user interruption (Ctrl+C)
- Configuration validation should happen at startup, not runtime
- Provider responses must preserve original formatting for evaluation

---

**Next Action**: Begin implementation of provider base class and interfaces
**Following Stage**: Stage 5 - Core Runner & CLI
