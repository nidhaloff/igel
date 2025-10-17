"""
Memory optimization utilities for igel.
Optimize memory usage for large datasets.
"""

import numpy as np
import pandas as pd
import psutil
import gc
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class MemoryOptimizer:
    """
    Memory optimization utilities for large datasets.
    """
    
    def __init__(self):
        """Initialize memory optimizer."""
        self.original_dtypes = {}
        self.memory_usage = {}
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': process.memory_percent()
        }
    
    def optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage.
        
        Args:
            df: DataFrame to optimize
            
        Returns:
            Optimized DataFrame
        """
        original_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
        
        # Store original dtypes
        self.original_dtypes = df.dtypes.to_dict()
        
        # Optimize numeric columns
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
            if df[col].dtype == 'int64':
                df[col] = pd.to_numeric(df[col], downcast='integer')
            elif df[col].dtype == 'float64':
                df[col] = pd.to_numeric(df[col], downcast='float')
        
        # Optimize categorical columns
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() / len(df) < 0.5:  # If less than 50% unique values
                df[col] = df[col].astype('category')
        
        optimized_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
        
        self.memory_usage = {
            'original_mb': original_memory,
            'optimized_mb': optimized_memory,
            'savings_mb': original_memory - optimized_memory,
            'savings_percent': (original_memory - optimized_memory) / original_memory * 100
        }
        
        logger.info(f"Memory optimization: {original_memory:.2f}MB -> {optimized_memory:.2f}MB "
                   f"({self.memory_usage['savings_percent']:.1f}% reduction)")
        
        return df
    
    def optimize_array(self, arr: np.ndarray) -> np.ndarray:
        """Optimize numpy array memory usage."""
        original_dtype = arr.dtype
        
        # Downcast if possible
        if arr.dtype == np.float64:
            arr = arr.astype(np.float32)
        elif arr.dtype == np.int64:
            arr = arr.astype(np.int32)
        
        logger.info(f"Array optimized: {original_dtype} -> {arr.dtype}")
        return arr
    
    def chunk_dataframe(self, df: pd.DataFrame, chunk_size: int = 10000) -> list:
        """
        Split DataFrame into chunks for memory-efficient processing.
        
        Args:
            df: DataFrame to chunk
            chunk_size: Size of each chunk
            
        Returns:
            List of DataFrame chunks
        """
        chunks = []
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i + chunk_size].copy()
            chunks.append(chunk)
        
        logger.info(f"DataFrame split into {len(chunks)} chunks of max size {chunk_size}")
        return chunks
    
    def clear_memory(self):
        """Clear memory by forcing garbage collection."""
        gc.collect()
        logger.info("Memory cleared via garbage collection")
    
    def get_optimization_report(self) -> str:
        """Generate memory optimization report."""
        if not self.memory_usage:
            return "No optimization performed yet."
        
        report = []
        report.append("Memory Optimization Report")
        report.append("=" * 25)
        report.append(f"Original Memory: {self.memory_usage['original_mb']:.2f} MB")
        report.append(f"Optimized Memory: {self.memory_usage['optimized_mb']:.2f} MB")
        report.append(f"Memory Saved: {self.memory_usage['savings_mb']:.2f} MB")
        report.append(f"Savings: {self.memory_usage['savings_percent']:.1f}%")
        
        return "\n".join(report)
