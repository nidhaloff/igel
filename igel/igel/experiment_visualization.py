"""
Experiment Visualization and Analysis for Igel.

This module provides comprehensive visualization capabilities for:
- Experiment comparison and analysis
- Model performance tracking
- Metric visualization
- Model lineage visualization
- Deployment monitoring
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# Set style for matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ExperimentVisualizer:
    """
    Comprehensive visualization system for experiments and models.
    
    Provides functionality for:
    - Experiment comparison charts
    - Metric tracking over time
    - Model performance analysis
    - Lineage visualization
    - Deployment monitoring
    """
    
    def __init__(self, experiment_tracker=None, model_versioning=None):
        """
        Initialize the experiment visualizer.
        
        Args:
            experiment_tracker: ExperimentTracker instance
            model_versioning: ModelVersioning instance
        """
        self.experiment_tracker = experiment_tracker
        self.model_versioning = model_versioning
    
    def plot_experiment_comparison(self, 
                                 experiment_ids: List[str],
                                 metrics: List[str],
                                 figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Create a comprehensive experiment comparison plot.
        
        Args:
            experiment_ids (List[str]): List of experiment IDs to compare
            metrics (List[str]): List of metrics to plot
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        if not self.experiment_tracker:
            raise ValueError("Experiment tracker not provided")
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        # Get all runs from experiments
        all_runs = []
        experiment_names = []
        
        for exp_id in experiment_ids:
            experiment = self.experiment_tracker.get_experiment(exp_id)
            if experiment:
                runs = self.experiment_tracker.list_runs(exp_id)
                all_runs.extend(runs)
                experiment_names.append(experiment.name)
        
        if not all_runs:
            logger.warning("No runs found for comparison")
            return fig
        
        # 1. Metric comparison (top left)
        ax1 = axes[0]
        self._plot_metric_comparison(all_runs, metrics, ax1)
        
        # 2. Timeline of experiments (top right)
        ax2 = axes[1]
        self._plot_experiment_timeline(all_runs, experiment_names, ax2)
        
        # 3. Parameter comparison (bottom left)
        ax3 = axes[2]
        self._plot_parameter_comparison(all_runs, ax3)
        
        # 4. Performance distribution (bottom right)
        ax4 = axes[3]
        self._plot_performance_distribution(all_runs, metrics, ax4)
        
        plt.tight_layout()
        return fig
    
    def _plot_metric_comparison(self, runs: List, metrics: List[str], ax):
        """Plot metric comparison across runs."""
        data = []
        labels = []
        
        for run in runs:
            for metric in metrics:
                if metric in run.metrics:
                    data.append(run.metrics[metric])
                    labels.append(f"{run.name}_{metric}")
        
        if data:
            bars = ax.bar(range(len(data)), data, alpha=0.7)
            ax.set_title("Metric Comparison Across Runs")
            ax.set_ylabel("Metric Value")
            ax.set_xticks(range(len(data)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, value in zip(bars, data):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{value:.3f}', ha='center', va='bottom')
    
    def _plot_experiment_timeline(self, runs: List, experiment_names: List[str], ax):
        """Plot experiment timeline."""
        dates = []
        names = []
        
        for run in runs:
            try:
                date = datetime.fromisoformat(run.start_time.replace('Z', '+00:00'))
                dates.append(date)
                names.append(run.name or run.run_id[:8])
            except:
                continue
        
        if dates:
            ax.scatter(dates, names, alpha=0.7, s=100)
            ax.set_title("Experiment Timeline")
            ax.set_xlabel("Date")
            ax.set_ylabel("Run Name")
            ax.tick_params(axis='x', rotation=45)
    
    def _plot_parameter_comparison(self, runs: List, ax):
        """Plot parameter comparison across runs."""
        param_data = {}
        
        for run in runs:
            for param, value in run.parameters.items():
                if param not in param_data:
                    param_data[param] = []
                param_data[param].append(value)
        
        if param_data:
            # Create box plot for numerical parameters
            numerical_params = []
            numerical_values = []
            
            for param, values in param_data.items():
                if all(isinstance(v, (int, float)) for v in values):
                    numerical_params.append(param)
                    numerical_values.append(values)
            
            if numerical_values:
                ax.boxplot(numerical_values, labels=numerical_params)
                ax.set_title("Parameter Distribution")
                ax.set_ylabel("Parameter Value")
                ax.tick_params(axis='x', rotation=45)
    
    def _plot_performance_distribution(self, runs: List, metrics: List[str], ax):
        """Plot performance distribution."""
        metric_data = {}
        
        for run in runs:
            for metric in metrics:
                if metric in run.metrics:
                    if metric not in metric_data:
                        metric_data[metric] = []
                    metric_data[metric].append(run.metrics[metric])
        
        if metric_data:
            for metric, values in metric_data.items():
                ax.hist(values, alpha=0.7, label=metric, bins=10)
            
            ax.set_title("Performance Distribution")
            ax.set_xlabel("Metric Value")
            ax.set_ylabel("Frequency")
            ax.legend()
    
    def create_interactive_dashboard(self, experiment_ids: List[str]) -> go.Figure:
        """
        Create an interactive Plotly dashboard for experiment analysis.
        
        Args:
            experiment_ids (List[str]): List of experiment IDs
            
        Returns:
            go.Figure: Interactive Plotly figure
        """
        if not self.experiment_tracker:
            raise ValueError("Experiment tracker not provided")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Metric Comparison', 'Timeline', 'Parameter Analysis', 'Performance Trends'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "box"}, {"type": "scatter"}]]
        )
        
        # Get all runs
        all_runs = []
        for exp_id in experiment_ids:
            runs = self.experiment_tracker.list_runs(exp_id)
            all_runs.extend(runs)
        
        if not all_runs:
            return fig
        
        # 1. Metric comparison (bar chart)
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        for metric in metrics:
            values = []
            names = []
            for run in all_runs:
                if metric in run.metrics:
                    values.append(run.metrics[metric])
                    names.append(run.name or run.run_id[:8])
            
            if values:
                fig.add_trace(
                    go.Bar(x=names, y=values, name=metric),
                    row=1, col=1
                )
        
        # 2. Timeline (scatter plot)
        dates = []
        names = []
        for run in all_runs:
            try:
                date = datetime.fromisoformat(run.start_time.replace('Z', '+00:00'))
                dates.append(date)
                names.append(run.name or run.run_id[:8])
            except:
                continue
        
        if dates:
            fig.add_trace(
                go.Scatter(x=dates, y=names, mode='markers', name='Runs'),
                row=1, col=2
            )
        
        # 3. Parameter analysis (box plot)
        param_data = {}
        for run in all_runs:
            for param, value in run.parameters.items():
                if param not in param_data:
                    param_data[param] = []
                if isinstance(value, (int, float)):
                    param_data[param].append(value)
        
        for param, values in param_data.items():
            if values:
                fig.add_trace(
                    go.Box(y=values, name=param),
                    row=2, col=1
                )
        
        # 4. Performance trends (line chart)
        for run in all_runs:
            if 'accuracy' in run.metrics:
                fig.add_trace(
                    go.Scatter(x=[run.start_time], y=[run.metrics['accuracy']], 
                              mode='markers', name=run.name or run.run_id[:8]),
                    row=2, col=2
                )
        
        fig.update_layout(height=800, title_text="Experiment Analysis Dashboard")
        return fig
    
    def plot_model_lineage(self, version_id: str, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Create a model lineage visualization.
        
        Args:
            version_id (str): Model version ID
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        if not self.model_versioning:
            raise ValueError("Model versioning not provided")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add nodes and edges
        self._build_lineage_graph(G, version_id)
        
        if G.nodes():
            # Layout
            pos = nx.spring_layout(G, k=1, iterations=50)
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                                 node_size=2000, ax=ax)
            
            # Draw edges
            nx.draw_networkx_edges(G, pos, edge_color='gray', 
                                 arrows=True, arrowsize=20, ax=ax)
            
            # Draw labels
            labels = {node: G.nodes[node].get('label', node[:8]) for node in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
            
            ax.set_title("Model Lineage")
            ax.axis('off')
        
        return fig
    
    def _build_lineage_graph(self, G: nx.DiGraph, version_id: str, visited: set = None):
        """Recursively build lineage graph."""
        if visited is None:
            visited = set()
        
        if version_id in visited:
            return
        
        visited.add(version_id)
        
        version = self.model_versioning.get_version(version_id)
        if not version:
            return
        
        # Add current node
        G.add_node(version_id, label=f"{version.model_name} v{version.version}")
        
        # Add parent node
        lineage = self.model_versioning.get_lineage(version_id)
        if lineage.get('parent_version_id'):
            parent_id = lineage['parent_version_id']
            G.add_node(parent_id, label=f"Parent v{self.model_versioning.get_version(parent_id).version if self.model_versioning.get_version(parent_id) else 'Unknown'}")
            G.add_edge(parent_id, version_id)
            
            # Recursively add parent lineage
            self._build_lineage_graph(G, parent_id, visited)
    
    def plot_deployment_monitoring(self, 
                                 deployment_ids: List[str],
                                 figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Create deployment monitoring visualization.
        
        Args:
            deployment_ids (List[str]): List of deployment IDs
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        if not self.model_versioning:
            raise ValueError("Model versioning not provided")
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        deployments = []
        for dep_id in deployment_ids:
            # This would need to be implemented in ModelVersioning
            # For now, we'll create mock data
            pass
        
        # Mock deployment data for demonstration
        deployment_data = [
            {'environment': 'staging', 'status': 'active', 'performance': 0.95},
            {'environment': 'production', 'status': 'active', 'performance': 0.92},
            {'environment': 'testing', 'status': 'inactive', 'performance': 0.89}
        ]
        
        # 1. Deployment status (top left)
        ax1 = axes[0]
        status_counts = {}
        for dep in deployment_data:
            status = dep['status']
            status_counts[status] = status_counts.get(status, 0) + 1
        
        if status_counts:
            ax1.pie(status_counts.values(), labels=status_counts.keys(), autopct='%1.1f%%')
            ax1.set_title("Deployment Status Distribution")
        
        # 2. Performance by environment (top right)
        ax2 = axes[1]
        env_perf = {}
        for dep in deployment_data:
            env = dep['environment']
            perf = dep['performance']
            if env not in env_perf:
                env_perf[env] = []
            env_perf[env].append(perf)
        
        if env_perf:
            ax2.bar(env_perf.keys(), [np.mean(v) for v in env_perf.values()])
            ax2.set_title("Average Performance by Environment")
            ax2.set_ylabel("Performance")
        
        # 3. Deployment timeline (bottom left)
        ax3 = axes[2]
        # Mock timeline data
        dates = [datetime.now() - timedelta(days=i) for i in range(10)]
        deployments_count = [1, 2, 1, 3, 2, 1, 2, 3, 1, 2]
        ax3.plot(dates, deployments_count, marker='o')
        ax3.set_title("Deployment Timeline")
        ax3.set_ylabel("Number of Deployments")
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Performance trends (bottom right)
        ax4 = axes[3]
        # Mock performance data
        performance_trend = [0.85, 0.87, 0.89, 0.91, 0.93, 0.92, 0.94, 0.93, 0.95, 0.94]
        ax4.plot(range(len(performance_trend)), performance_trend, marker='o')
        ax4.set_title("Performance Trend")
        ax4.set_ylabel("Performance")
        ax4.set_xlabel("Time")
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_metric_tracking_plot(self, 
                                  run_ids: List[str],
                                  metrics: List[str],
                                  figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Create metric tracking plot over time.
        
        Args:
            run_ids (List[str]): List of run IDs
            metrics (List[str]): List of metrics to track
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        if not self.experiment_tracker:
            raise ValueError("Experiment tracker not provided")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        for run_id in run_ids:
            run = self.experiment_tracker.get_run(run_id)
            if not run:
                continue
            
            # Get metric history from database
            # This would need to be implemented in ExperimentTracker
            # For now, we'll use the final metrics
            
            for metric in metrics:
                if metric in run.metrics:
                    # Mock time series data
                    time_points = range(10)
                    metric_values = [run.metrics[metric] * (0.95 + 0.1 * np.random.random()) for _ in time_points]
                    
                    ax.plot(time_points, metric_values, 
                           label=f"{run.name}_{metric}", marker='o')
        
        ax.set_title("Metric Tracking Over Time")
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Metric Value")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def export_visualization_report(self, 
                                  experiment_ids: List[str],
                                  output_path: str,
                                  format: str = 'html'):
        """
        Export a comprehensive visualization report.
        
        Args:
            experiment_ids (List[str]): List of experiment IDs
            output_path (str): Output file path
            format (str): Output format ('html', 'pdf', 'png')
        """
        if format == 'html':
            # Create interactive dashboard
            fig = self.create_interactive_dashboard(experiment_ids)
            fig.write_html(output_path)
        elif format == 'pdf':
            # Create static plots
            fig = self.plot_experiment_comparison(experiment_ids, ['accuracy', 'precision', 'recall'])
            fig.savefig(output_path, format='pdf', bbox_inches='tight')
        elif format == 'png':
            # Create static plots
            fig = self.plot_experiment_comparison(experiment_ids, ['accuracy', 'precision', 'recall'])
            fig.savefig(output_path, format='png', dpi=300, bbox_inches='tight')
        
        logger.info(f"Exported visualization report to {output_path}")


class ModelAnalyzer:
    """
    Model analysis and comparison utilities.
    """
    
    def __init__(self, model_versioning=None):
        """
        Initialize the model analyzer.
        
        Args:
            model_versioning: ModelVersioning instance
        """
        self.model_versioning = model_versioning
    
    def compare_model_performance(self, version_ids: List[str]) -> pd.DataFrame:
        """
        Compare performance of multiple model versions.
        
        Args:
            version_ids (List[str]): List of version IDs to compare
            
        Returns:
            pd.DataFrame: Comparison table
        """
        if not self.model_versioning:
            raise ValueError("Model versioning not provided")
        
        return self.model_versioning.compare_versions(version_ids)
    
    def plot_model_comparison(self, version_ids: List[str], figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Create a comprehensive model comparison plot.
        
        Args:
            version_ids (List[str]): List of version IDs to compare
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        versions = []
        for version_id in version_ids:
            version = self.model_versioning.get_version(version_id)
            if version:
                versions.append(version)
        
        if not versions:
            return fig
        
        # 1. Performance metrics comparison (top left)
        ax1 = axes[0]
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        for metric in metrics:
            values = []
            labels = []
            for version in versions:
                if metric in version.metrics:
                    values.append(version.metrics[metric])
                    labels.append(f"{version.model_name} v{version.version}")
            
            if values:
                ax1.bar(labels, values, alpha=0.7, label=metric)
        
        ax1.set_title("Performance Metrics Comparison")
        ax1.set_ylabel("Metric Value")
        ax1.tick_params(axis='x', rotation=45)
        ax1.legend()
        
        # 2. Model parameters comparison (top right)
        ax2 = axes[1]
        # This would need to be implemented based on specific parameter types
        
        # 3. Creation timeline (bottom left)
        ax3 = axes[2]
        dates = []
        names = []
        for version in versions:
            try:
                date = datetime.fromisoformat(version.created_at.replace('Z', '+00:00'))
                dates.append(date)
                names.append(f"{version.model_name} v{version.version}")
            except:
                continue
        
        if dates:
            ax3.scatter(dates, names, alpha=0.7, s=100)
            ax3.set_title("Model Creation Timeline")
            ax3.set_xlabel("Date")
            ax3.tick_params(axis='x', rotation=45)
        
        # 4. Status distribution (bottom right)
        ax4 = axes[3]
        status_counts = {}
        for version in versions:
            status = version.status
            status_counts[status] = status_counts.get(status, 0) + 1
        
        if status_counts:
            ax4.pie(status_counts.values(), labels=status_counts.keys(), autopct='%1.1f%%')
            ax4.set_title("Model Status Distribution")
        
        plt.tight_layout()
        return fig 