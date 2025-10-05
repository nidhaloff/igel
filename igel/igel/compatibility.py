"""
Backward compatibility module for igel.
Handles version-specific compatibility issues and provides fallbacks.
"""

import sys
import warnings
from typing import Any, Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)

class CompatibilityManager:
    """Manages backward compatibility across different igel versions."""
    
    def __init__(self):
        self.version_info = self._get_version_info()
        self.compatibility_mode = self._determine_compatibility_mode()
    
    def _get_version_info(self) -> Dict[str, Any]:
        """Get version information for compatibility checks."""
        try:
            import igel
            return {
                "igel_version": getattr(igel, '__version__', 'unknown'),
                "python_version": sys.version_info,
                "platform": sys.platform
            }
        except ImportError:
            return {
                "igel_version": "unknown",
                "python_version": sys.version_info,
                "platform": sys.platform
            }
    
    def _determine_compatibility_mode(self) -> str:
        """Determine which compatibility mode to use."""
        version = self.version_info.get("igel_version", "unknown")
        
        if version == "unknown":
            return "legacy"
        
        try:
            # Parse version string (e.g., "0.4.0" -> (0, 4, 0))
            version_parts = version.split('.')
            major = int(version_parts[0])
            minor = int(version_parts[1]) if len(version_parts) > 1 else 0
            patch = int(version_parts[2]) if len(version_parts) > 2 else 0
            
            if major < 0 or (major == 0 and minor < 4):
                return "legacy"
            elif major == 0 and minor == 4:
                return "modern"
            else:
                return "latest"
        except (ValueError, IndexError):
            return "legacy"
    
    def get_import_strategy(self, module_name: str) -> str:
        """Get the appropriate import strategy for a module."""
        if self.compatibility_mode == "legacy":
            return "relative"
        elif self.compatibility_mode == "modern":
            return "absolute"
        else:
            return "absolute"
    
    def handle_deprecated_feature(self, feature_name: str, replacement: str = None) -> None:
        """Handle deprecated features with appropriate warnings."""
        if self.compatibility_mode == "legacy":
            # Don't show deprecation warnings in legacy mode
            return
        
        warning_msg = f"Feature '{feature_name}' is deprecated"
        if replacement:
            warning_msg += f". Use '{replacement}' instead."
        
        warnings.warn(warning_msg, DeprecationWarning, stacklevel=3)
    
    def get_model_registry_class(self):
        """Get the appropriate ModelRegistry class based on version."""
        try:
            if self.compatibility_mode == "legacy":
                # Try relative import first
                from .model_registry import ModelRegistry
            else:
                # Try absolute import
                from igel.model_registry import ModelRegistry
            return ModelRegistry
        except ImportError:
            try:
                # Fallback to relative import
                from .model_registry import ModelRegistry
                return ModelRegistry
            except ImportError:
                # Final fallback - create a minimal registry
                return self._create_minimal_registry()
    
    def _create_minimal_registry(self):
        """Create a minimal registry for backward compatibility."""
        class MinimalRegistry:
            def __init__(self):
                self.models = {}
            
            def register_model(self, model_id: str, model_data: Dict) -> bool:
                self.models[model_id] = model_data
                return True
            
            def get_model(self, model_id: str) -> Optional[Dict]:
                return self.models.get(model_id)
            
            def list_models(self, name_filter: str = None) -> list:
                if name_filter:
                    return [m for m in self.models.values() if name_filter in m.get('name', '')]
                return list(self.models.values())
        
        return MinimalRegistry
    
    def get_ab_testing_class(self):
        """Get the appropriate A/B testing class based on version."""
        try:
            if self.compatibility_mode == "legacy":
                from .ab_testing import ModelComparison
            else:
                from igel.ab_testing import ModelComparison
            return ModelComparison
        except ImportError:
            # Fallback to relative import
            from .ab_testing import ModelComparison
            return ModelComparison
    
    def get_data_loading_strategy(self) -> str:
        """Get the appropriate data loading strategy."""
        if self.compatibility_mode == "legacy":
            return "pandas_basic"
        else:
            return "pandas_enhanced"
    
    def should_use_enhanced_features(self) -> bool:
        """Determine if enhanced features should be available."""
        return self.compatibility_mode in ["modern", "latest"]
    
    def get_cli_options(self) -> Dict[str, Any]:
        """Get CLI options based on compatibility mode."""
        if self.compatibility_mode == "legacy":
            return {
                "use_legacy_ab_testing": True,
                "enable_visualizations": False,
                "enable_export": False
            }
        else:
            return {
                "use_legacy_ab_testing": False,
                "enable_visualizations": True,
                "enable_export": True
            }

# Global compatibility manager instance
compatibility_manager = CompatibilityManager()

def get_compatibility_manager() -> CompatibilityManager:
    """Get the global compatibility manager instance."""
    return compatibility_manager

def ensure_backward_compatibility(func):
    """Decorator to ensure backward compatibility for functions."""
    def wrapper(*args, **kwargs):
        # Check if we're in legacy mode
        if compatibility_manager.compatibility_mode == "legacy":
            # Strip out new parameters that might not be supported
            legacy_kwargs = {k: v for k, v in kwargs.items() 
                           if not k.startswith('enhanced_') and k not in ['visualize', 'export_results']}
            return func(*args, **legacy_kwargs)
        else:
            return func(*args, **kwargs)
    return wrapper

def log_compatibility_info():
    """Log compatibility information for debugging."""
    manager = get_compatibility_manager()
    logger.info(f"Compatibility mode: {manager.compatibility_mode}")
    logger.info(f"Version info: {manager.version_info}")
    logger.info(f"CLI options: {manager.get_cli_options()}")
