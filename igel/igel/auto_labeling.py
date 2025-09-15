"""
Automated Data Labeling Tools for Igel.

This module provides automated data labeling capabilities.
Addresses GitHub issue #286 - Implement Automated Data Labeling Tools.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Callable
import logging

logger = logging.getLogger(__name__)


class AutoLabeler:
    """Automated data labeling using rule-based and ML approaches."""
    
    def __init__(self):
        self.rules = []
        self.labeled_data = None
    
    def add_rule(self, rule_func: Callable, label: str, description: str = ""):
        """
        Add a labeling rule.
        
        Args:
            rule_func: Function that returns True/False for labeling
            label: Label to assign when rule is True
            description: Description of the rule
        """
        self.rules.append({
            "function": rule_func,
            "label": label,
            "description": description
        })
        logger.info(f"Added rule: {description}")
    
    def apply_rules(self, data: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
        """
        Apply all rules to the data.
        
        Args:
            data: DataFrame to label
            text_column: Column containing text to analyze
            
        Returns:
            DataFrame with labels
        """
        result = data.copy()
        result["auto_label"] = "unlabeled"
        
        for rule in self.rules:
            mask = rule["function"](result[text_column])
            result.loc[mask, "auto_label"] = rule["label"]
        
        self.labeled_data = result
        logger.info(f"Applied {len(self.rules)} rules to {len(data)} samples")
        return result
    
    def get_label_distribution(self) -> Dict[str, int]:
        """Get distribution of auto-generated labels."""
        if self.labeled_data is None:
            raise ValueError("No labeled data. Run apply_rules() first.")
        
        return self.labeled_data["auto_label"].value_counts().to_dict()


def create_sentiment_rules() -> List[Dict]:
    """Create common sentiment analysis rules."""
    rules = []
    
    # Positive sentiment rules
    positive_words = ["good", "great", "excellent", "amazing", "love", "best", "perfect"]
    rules.append({
        "function": lambda text: text.str.lower().str.contains("|".join(positive_words), na=False),
        "label": "positive",
        "description": "Contains positive words"
    })
    
    # Negative sentiment rules
    negative_words = ["bad", "terrible", "awful", "hate", "worst", "horrible", "disappointed"]
    rules.append({
        "function": lambda text: text.str.lower().str.contains("|".join(negative_words), na=False),
        "label": "negative", 
        "description": "Contains negative words"
    })
    
    return rules


def auto_label_sentiment(data: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
    """
    Quick function to auto-label sentiment data.
    
    Args:
        data: DataFrame with text data
        text_column: Column containing text
        
    Returns:
        DataFrame with sentiment labels
    """
    labeler = AutoLabeler()
    
    # Add sentiment rules
    sentiment_rules = create_sentiment_rules()
    for rule in sentiment_rules:
        labeler.add_rule(rule["function"], rule["label"], rule["description"])
    
    return labeler.apply_rules(data, text_column)
