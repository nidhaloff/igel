"""
Data Validation Framework for Igel.

This module provides comprehensive data validation capabilities including
schema validation, data quality checks, and automated data profiling.
Addresses GitHub issue #297 - Create Data Validation Framework.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Callable
import logging
from dataclasses import dataclass
from datetime import datetime
import json

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a data validation check."""
    check_name: str
    passed: bool
    message: str
    details: Dict[str, Any]
    timestamp: datetime


class DataValidator:
    """Comprehensive data validation framework."""
    
    def __init__(self):
        self.validation_results: List[ValidationResult] = []
        self.schema: Optional[Dict] = None
    
    def set_schema(self, schema: Dict[str, Any]) -> None:
        """
        Set the expected data schema.
        
        Args:
            schema: Dictionary defining expected column types and constraints
        """
        self.schema = schema
        logger.info(f"Schema set with {len(schema)} columns")
    
    def validate_schema(self, data: pd.DataFrame) -> ValidationResult:
        """Validate data against the defined schema."""
        if self.schema is None:
            return ValidationResult(
                check_name="schema_validation",
                passed=False,
                message="No schema defined",
                details={},
                timestamp=datetime.now()
            )
        
        issues = []
        passed = True
        
        # Check column existence
        expected_cols = set(self.schema.keys())
        actual_cols = set(data.columns)
        
        missing_cols = expected_cols - actual_cols
        extra_cols = actual_cols - expected_cols
        
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")
            passed = False
        
        if extra_cols:
            issues.append(f"Extra columns: {extra_cols}")
        
        # Check data types
        for col, expected_type in self.schema.items():
            if col in data.columns:
                actual_type = str(data[col].dtype)
                if expected_type != actual_type:
                    issues.append(f"Column {col}: expected {expected_type}, got {actual_type}")
                    passed = False
        
        return ValidationResult(
            check_name="schema_validation",
            passed=passed,
            message="Schema validation completed",
            details={"issues": issues, "missing_cols": list(missing_cols), "extra_cols": list(extra_cols)},
            timestamp=datetime.now()
        )
    
    def check_missing_values(self, data: pd.DataFrame, threshold: float = 0.1) -> ValidationResult:
        """Check for missing values in the dataset."""
        missing_counts = data.isnull().sum()
        missing_percentages = (missing_counts / len(data)) * 100
        
        high_missing_cols = missing_percentages[missing_percentages > threshold * 100]
        
        passed = len(high_missing_cols) == 0
        message = f"Missing value check: {len(high_missing_cols)} columns exceed {threshold*100}% missing"
        
        return ValidationResult(
            check_name="missing_values",
            passed=passed,
            message=message,
            details={
                "missing_counts": missing_counts.to_dict(),
                "missing_percentages": missing_percentages.to_dict(),
                "high_missing_cols": high_missing_cols.to_dict()
            },
            timestamp=datetime.now()
        )
    
    def check_duplicates(self, data: pd.DataFrame) -> ValidationResult:
        """Check for duplicate rows."""
        duplicate_count = data.duplicated().sum()
        passed = duplicate_count == 0
        message = f"Duplicate check: {duplicate_count} duplicate rows found"
        
        return ValidationResult(
            check_name="duplicates",
            passed=passed,
            message=message,
            details={"duplicate_count": int(duplicate_count)},
            timestamp=datetime.now()
        )
    
    def check_outliers(self, data: pd.DataFrame, method: str = "iqr") -> ValidationResult:
        """Check for outliers using IQR method."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        outlier_info = {}
        
        for col in numeric_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
            outlier_info[col] = {
                "count": len(outliers),
                "percentage": (len(outliers) / len(data)) * 100,
                "bounds": {"lower": lower_bound, "upper": upper_bound}
            }
        
        total_outliers = sum(info["count"] for info in outlier_info.values())
        passed = total_outliers < len(data) * 0.05  # Less than 5% outliers
        
        return ValidationResult(
            check_name="outliers",
            passed=passed,
            message=f"Outlier check: {total_outliers} outliers found across {len(numeric_cols)} numeric columns",
            details={"outlier_info": outlier_info, "total_outliers": total_outliers},
            timestamp=datetime.now()
        )
    
    def profile_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive data profile."""
        profile = {
            "shape": data.shape,
            "columns": list(data.columns),
            "dtypes": data.dtypes.to_dict(),
            "missing_values": data.isnull().sum().to_dict(),
            "memory_usage": data.memory_usage(deep=True).to_dict(),
            "numeric_summary": {},
            "categorical_summary": {}
        }
        
        # Numeric columns summary
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            profile["numeric_summary"] = data[numeric_cols].describe().to_dict()
        
        # Categorical columns summary
        categorical_cols = data.select_dtypes(include=["object", "category"]).columns
        for col in categorical_cols:
            profile["categorical_summary"][col] = {
                "unique_count": data[col].nunique(),
                "most_common": data[col].value_counts().head().to_dict()
            }
        
        return profile
    
    def validate_all(self, data: pd.DataFrame, checks: List[str] = None) -> List[ValidationResult]:
        """
        Run all validation checks.
        
        Args:
            data: DataFrame to validate
            checks: List of check names to run (default: all)
        
        Returns:
            List of validation results
        """
        if checks is None:
            checks = ["schema", "missing_values", "duplicates", "outliers"]
        
        results = []
        
        for check in checks:
            if check == "schema":
                result = self.validate_schema(data)
            elif check == "missing_values":
                result = self.check_missing_values(data)
            elif check == "duplicates":
                result = self.check_duplicates(data)
            elif check == "outliers":
                result = self.check_outliers(data)
            else:
                logger.warning(f"Unknown check: {check}")
                continue
            
            results.append(result)
            self.validation_results.append(result)
        
        return results
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validation results."""
        if not self.validation_results:
            return {"message": "No validation results available"}
        
        total_checks = len(self.validation_results)
        passed_checks = sum(1 for r in self.validation_results if r.passed)
        failed_checks = total_checks - passed_checks
        
        return {
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "failed_checks": failed_checks,
            "success_rate": passed_checks / total_checks if total_checks > 0 else 0,
            "results": [
                {
                    "check_name": r.check_name,
                    "passed": r.passed,
                    "message": r.message,
                    "timestamp": r.timestamp.isoformat()
                }
                for r in self.validation_results
            ]
        }


def validate_dataframe(data: pd.DataFrame, schema: Dict = None) -> Dict[str, Any]:
    """
    Quick function to validate a DataFrame.
    
    Args:
        data: DataFrame to validate
        schema: Optional schema definition
    
    Returns:
        Validation summary
    """
    validator = DataValidator()
    
    if schema:
        validator.set_schema(schema)
    
    validator.validate_all(data)
    return validator.get_validation_summary()
