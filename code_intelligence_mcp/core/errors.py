"""Error definitions for Code Intelligence MCP."""

from typing import Optional, Dict, Any


class ServiceError(Exception):
    """Base exception for service errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.error_code = error_code or "SERVICE_ERROR"
        self.details = details or {}


class ConfigurationError(ServiceError):
    """Configuration related errors."""
    
    def __init__(self, message: str, config_key: Optional[str] = None):
        super().__init__(
            message,
            error_code="CONFIGURATION_ERROR",
            details={"config_key": config_key} if config_key else {}
        )


class InitializationError(ServiceError):
    """Service initialization errors."""
    
    def __init__(self, service_name: str, reason: str):
        super().__init__(
            f"Failed to initialize {service_name}: {reason}",
            error_code="INITIALIZATION_ERROR",
            details={"service": service_name, "reason": reason}
        )


class NotInitializedError(ServiceError):
    """Service not initialized error."""
    
    def __init__(self, service_name: str):
        super().__init__(
            f"{service_name} is not initialized",
            error_code="NOT_INITIALIZED",
            details={"service": service_name}
        )


class DependencyError(ServiceError):
    """Dependency resolution errors."""
    
    def __init__(self, message: str, dependency: Optional[str] = None):
        super().__init__(
            message,
            error_code="DEPENDENCY_ERROR",
            details={"dependency": dependency} if dependency else {}
        )


class ValidationError(ServiceError):
    """Input validation errors."""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Any = None):
        super().__init__(
            message,
            error_code="VALIDATION_ERROR",
            details={"field": field, "value": value} if field else {}
        )
