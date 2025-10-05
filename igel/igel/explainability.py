"""
Model Explainability Dashboard for igel.
Provides comprehensive model interpretation and explainability features.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)

class ModelExplainer:
    """
    Comprehensive model explainability framework.
    """
    
    def __init__(self, model: BaseEstimator, feature_names: List[str] = None):
        """
        Initialize the model explainer.
        
        Args:
            model: The trained model to explain
            feature_names: Names of the features
        """
        self.model = model
        self.feature_names = feature_names
        self.explanations = {}
        self.interpretation_results = {}
        
    def explain_model(self, X: np.ndarray, y: np.ndarray = None, 
                     explanation_types: List[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive model explanations.
        
        Args:
            X: Feature matrix
            y: Target values (optional)
            explanation_types: Types of explanations to generate
            
        Returns:
            Dictionary containing all explanations
        """
        if explanation_types is None:
            explanation_types = ['feature_importance', 'partial_dependence', 'shap_values', 'lime_explanation']
        
        explanations = {}
        
        for exp_type in explanation_types:
            try:
                if exp_type == 'feature_importance':
                    explanations[exp_type] = self._get_feature_importance()
                elif exp_type == 'partial_dependence':
                    explanations[exp_type] = self._get_partial_dependence(X)
                elif exp_type == 'shap_values':
                    explanations[exp_type] = self._get_shap_values(X)
                elif exp_type == 'lime_explanation':
                    explanations[exp_type] = self._get_lime_explanation(X)
                elif exp_type == 'permutation_importance':
                    explanations[exp_type] = self._get_permutation_importance(X, y)
                elif exp_type == 'model_summary':
                    explanations[exp_type] = self._get_model_summary()
                else:
                    logger.warning(f"Unknown explanation type: {exp_type}")
            except Exception as e:
                logger.warning(f"Failed to generate {exp_type}: {e}")
                explanations[exp_type] = {'error': str(e)}
        
        self.explanations = explanations
        return explanations
    
    def _get_feature_importance(self) -> Dict[str, Any]:
        """Get feature importance from the model."""
        importance_dict = {}
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            if self.feature_names:
                importance_dict = dict(zip(self.feature_names, importances))
            else:
                importance_dict = {f'feature_{i}': imp for i, imp in enumerate(importances)}
        elif hasattr(self.model, 'coef_'):
            # For linear models, use absolute coefficients
            coefs = np.abs(self.model.coef_)
            if len(coefs.shape) > 1:
                coefs = coefs[0]  # Take first class for multi-class
            if self.feature_names:
                importance_dict = dict(zip(self.feature_names, coefs))
            else:
                importance_dict = {f'feature_{i}': coef for i, coef in enumerate(coefs)}
        else:
            importance_dict = {'message': 'Feature importance not available for this model type'}
        
        return importance_dict
    
    def _get_partial_dependence(self, X: np.ndarray, features: List[int] = None) -> Dict[str, Any]:
        """Calculate partial dependence plots."""
        try:
            from sklearn.inspection import partial_dependence
        except ImportError:
            return {'error': 'sklearn.inspection.partial_dependence not available'}
        
        if features is None:
            features = list(range(min(5, X.shape[1])))  # Top 5 features
        
        pd_results = {}
        
        for feature_idx in features:
            try:
                pd_result = partial_dependence(
                    self.model, X, [feature_idx], 
                    grid_resolution=20
                )
                pd_results[f'feature_{feature_idx}'] = {
                    'values': pd_result['values'][0].tolist(),
                    'partial_dependence': pd_result['partial_dependence'][0].tolist()
                }
            except Exception as e:
                pd_results[f'feature_{feature_idx}'] = {'error': str(e)}
        
        return pd_results
    
    def _get_shap_values(self, X: np.ndarray) -> Dict[str, Any]:
        """Calculate SHAP values for model explanation."""
        try:
            import shap
        except ImportError:
            return {'error': 'SHAP library not available. Install with: pip install shap'}
        
        try:
            # Create SHAP explainer
            if hasattr(self.model, 'predict_proba'):
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(X[:100])  # Limit to 100 samples for performance
            else:
                explainer = shap.Explainer(self.model)
                shap_values = explainer(X[:100])
            
            # Calculate summary statistics
            if isinstance(shap_values, list):
                # Multi-class case
                shap_values_mean = np.mean([np.abs(sv) for sv in shap_values], axis=0)
            else:
                shap_values_mean = np.abs(shap_values.values)
            
            # Get feature importance from SHAP
            feature_importance = np.mean(shap_values_mean, axis=0)
            
            if self.feature_names:
                shap_importance = dict(zip(self.feature_names, feature_importance))
            else:
                shap_importance = {f'feature_{i}': imp for i, imp in enumerate(feature_importance)}
            
            return {
                'feature_importance': shap_importance,
                'shap_values': shap_values_mean.tolist() if hasattr(shap_values_mean, 'tolist') else str(shap_values_mean),
                'summary': 'SHAP values calculated successfully'
            }
        except Exception as e:
            return {'error': f'SHAP calculation failed: {str(e)}'}
    
    def _get_lime_explanation(self, X: np.ndarray) -> Dict[str, Any]:
        """Generate LIME explanations for individual predictions."""
        try:
            from lime.lime_tabular import LimeTabularExplainer
        except ImportError:
            return {'error': 'LIME library not available. Install with: pip install lime'}
        
        try:
            # Create LIME explainer
            explainer = LimeTabularExplainer(
                X, 
                feature_names=self.feature_names,
                mode='classification' if isinstance(self.model, ClassifierMixin) else 'regression'
            )
            
            # Explain a few samples
            explanations = []
            for i in range(min(3, len(X))):
                exp = explainer.explain_instance(X[i], self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict)
                explanations.append({
                    'sample_index': i,
                    'explanation': exp.as_list(),
                    'score': exp.score
                })
            
            return {
                'sample_explanations': explanations,
                'summary': f'LIME explanations generated for {len(explanations)} samples'
            }
        except Exception as e:
            return {'error': f'LIME explanation failed: {str(e)}'}
    
    def _get_permutation_importance(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Calculate permutation importance."""
        try:
            from sklearn.inspection import permutation_importance
        except ImportError:
            return {'error': 'sklearn.inspection.permutation_importance not available'}
        
        try:
            # Calculate permutation importance
            perm_importance = permutation_importance(
                self.model, X, y, n_repeats=10, random_state=42
            )
            
            if self.feature_names:
                importance_dict = dict(zip(self.feature_names, perm_importance.importances_mean))
            else:
                importance_dict = {f'feature_{i}': imp for i, imp in enumerate(perm_importance.importances_mean)}
            
            return {
                'feature_importance': importance_dict,
                'importances_std': perm_importance.importances_std.tolist(),
                'summary': 'Permutation importance calculated successfully'
            }
        except Exception as e:
            return {'error': f'Permutation importance calculation failed: {str(e)}'}
    
    def _get_model_summary(self) -> Dict[str, Any]:
        """Get a summary of the model."""
        summary = {
            'model_type': self.model.__class__.__name__,
            'parameters': self.model.get_params(),
            'n_features': getattr(self.model, 'n_features_in_', 'Unknown'),
            'training_samples': getattr(self.model, 'n_samples_', 'Unknown')
        }
        
        # Add model-specific information
        if hasattr(self.model, 'feature_importances_'):
            summary['has_feature_importance'] = True
        if hasattr(self.model, 'coef_'):
            summary['has_coefficients'] = True
        if hasattr(self.model, 'predict_proba'):
            summary['has_probability_predictions'] = True
        
        return summary
    
    def create_explanation_dashboard(self, X: np.ndarray, y: np.ndarray = None, 
                                   save_path: str = None) -> None:
        """
        Create a comprehensive explanation dashboard.
        
        Args:
            X: Feature matrix
            y: Target values (optional)
            save_path: Path to save the dashboard
        """
        # Generate all explanations
        explanations = self.explain_model(X, y)
        
        # Create dashboard
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Explainability Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Feature Importance
        self._plot_feature_importance(axes[0, 0], explanations.get('feature_importance', {}))
        
        # 2. Partial Dependence
        self._plot_partial_dependence(axes[0, 1], explanations.get('partial_dependence', {}))
        
        # 3. SHAP Summary
        self._plot_shap_summary(axes[0, 2], explanations.get('shap_values', {}))
        
        # 4. Model Performance
        self._plot_model_performance(axes[1, 0], X, y)
        
        # 5. Feature Correlation
        self._plot_feature_correlation(axes[1, 1], X)
        
        # 6. Model Summary
        self._plot_model_summary(axes[1, 2], explanations.get('model_summary', {}))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Explanation dashboard saved to {save_path}")
        else:
            plt.show()
    
    def _plot_feature_importance(self, ax, importance_data):
        """Plot feature importance."""
        if 'error' in importance_data or not importance_data:
            ax.text(0.5, 0.5, 'Feature importance\nnot available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Feature Importance')
            return
        
        # Sort features by importance
        sorted_features = sorted(importance_data.items(), key=lambda x: x[1], reverse=True)
        features, importances = zip(*sorted_features[:10])  # Top 10 features
        
        ax.barh(range(len(features)), importances)
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.set_xlabel('Importance')
        ax.set_title('Top 10 Feature Importance')
        ax.invert_yaxis()
    
    def _plot_partial_dependence(self, ax, pd_data):
        """Plot partial dependence."""
        if 'error' in pd_data or not pd_data:
            ax.text(0.5, 0.5, 'Partial dependence\nnot available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Partial Dependence')
            return
        
        # Plot first available partial dependence
        for feature_name, pd_info in pd_data.items():
            if 'error' not in pd_info:
                ax.plot(pd_info['values'], pd_info['partial_dependence'], 'o-')
                ax.set_xlabel(feature_name)
                ax.set_ylabel('Partial Dependence')
                ax.set_title(f'Partial Dependence: {feature_name}')
                break
        else:
            ax.text(0.5, 0.5, 'No partial dependence\ndata available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Partial Dependence')
    
    def _plot_shap_summary(self, ax, shap_data):
        """Plot SHAP summary."""
        if 'error' in shap_data or 'feature_importance' not in shap_data:
            ax.text(0.5, 0.5, 'SHAP values\nnot available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('SHAP Summary')
            return
        
        # Plot SHAP feature importance
        importance_data = shap_data['feature_importance']
        sorted_features = sorted(importance_data.items(), key=lambda x: x[1], reverse=True)
        features, importances = zip(*sorted_features[:10])
        
        ax.barh(range(len(features)), importances)
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.set_xlabel('SHAP Importance')
        ax.set_title('SHAP Feature Importance')
        ax.invert_yaxis()
    
    def _plot_model_performance(self, ax, X, y):
        """Plot model performance metrics."""
        if y is None:
            ax.text(0.5, 0.5, 'No target data\nfor performance metrics', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Model Performance')
            return
        
        # Calculate predictions
        predictions = self.model.predict(X)
        
        if isinstance(self.model, ClassifierMixin):
            accuracy = accuracy_score(y, predictions)
            ax.text(0.5, 0.5, f'Accuracy: {accuracy:.4f}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
        else:
            mse = mean_squared_error(y, predictions)
            r2 = r2_score(y, predictions)
            ax.text(0.5, 0.5, f'MSE: {mse:.4f}\nR²: {r2:.4f}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
        
        ax.set_title('Model Performance')
        ax.axis('off')
    
    def _plot_feature_correlation(self, ax, X):
        """Plot feature correlation matrix."""
        if X.shape[1] > 20:
            # For high-dimensional data, show correlation with target
            if hasattr(self.model, 'predict'):
                y_pred = self.model.predict(X)
                correlations = [np.corrcoef(X[:, i], y_pred)[0, 1] for i in range(X.shape[1])]
                ax.bar(range(len(correlations)), correlations)
                ax.set_title('Feature-Target Correlation')
                ax.set_xlabel('Feature Index')
                ax.set_ylabel('Correlation')
            else:
                ax.text(0.5, 0.5, 'Too many features\nfor correlation plot', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Feature Correlation')
        else:
            # For low-dimensional data, show full correlation matrix
            corr_matrix = np.corrcoef(X.T)
            im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            ax.set_title('Feature Correlation Matrix')
            plt.colorbar(im, ax=ax)
    
    def _plot_model_summary(self, ax, summary_data):
        """Plot model summary information."""
        if 'error' in summary_data or not summary_data:
            ax.text(0.5, 0.5, 'Model summary\nnot available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Model Summary')
            return
        
        # Create summary text
        summary_text = f"Model: {summary_data.get('model_type', 'Unknown')}\n"
        summary_text += f"Features: {summary_data.get('n_features', 'Unknown')}\n"
        summary_text += f"Samples: {summary_data.get('training_samples', 'Unknown')}\n"
        
        if summary_data.get('has_feature_importance'):
            summary_text += "✓ Feature Importance\n"
        if summary_data.get('has_coefficients'):
            summary_text += "✓ Coefficients\n"
        if summary_data.get('has_probability_predictions'):
            summary_text += "✓ Probability Predictions\n"
        
        ax.text(0.5, 0.5, summary_text, ha='center', va='center', 
               transform=ax.transAxes, fontsize=10)
        ax.set_title('Model Summary')
        ax.axis('off')
    
    def generate_explanation_report(self) -> str:
        """Generate a comprehensive explanation report."""
        report = []
        report.append("Model Explainability Report")
        report.append("=" * 30)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Model Type: {self.model.__class__.__name__}")
        
        for exp_type, exp_data in self.explanations.items():
            report.append(f"\n{exp_type.replace('_', ' ').title()}:")
            report.append("-" * 20)
            
            if 'error' in exp_data:
                report.append(f"Error: {exp_data['error']}")
            elif exp_type == 'feature_importance':
                if isinstance(exp_data, dict) and exp_data:
                    sorted_features = sorted(exp_data.items(), key=lambda x: x[1], reverse=True)
                    for feature, importance in sorted_features[:10]:
                        report.append(f"{feature}: {importance:.4f}")
                else:
                    report.append("No feature importance data available")
            elif exp_type == 'model_summary':
                for key, value in exp_data.items():
                    report.append(f"{key}: {value}")
            else:
                report.append(f"Data available: {len(exp_data)} items")
        
        return "\n".join(report)
    
    def save_explanations(self, filepath: str):
        """Save explanations to file."""
        explanation_data = {
            'model_type': self.model.__class__.__name__,
            'feature_names': self.feature_names,
            'explanations': self.explanations,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(f"{filepath}_explanations.json", 'w') as f:
            json.dump(explanation_data, f, indent=2, default=str)
        
        logger.info(f"Explanations saved to {filepath}_explanations.json")


class ExplainabilityDashboard:
    """
    Interactive dashboard for model explainability.
    """
    
    def __init__(self, explainer: ModelExplainer):
        """
        Initialize the dashboard.
        
        Args:
            explainer: ModelExplainer instance
        """
        self.explainer = explainer
    
    def launch_dashboard(self, X: np.ndarray, y: np.ndarray = None, 
                        port: int = 8050, debug: bool = False):
        """
        Launch an interactive dashboard.
        
        Args:
            X: Feature matrix
            y: Target values
            port: Port for the dashboard
            debug: Enable debug mode
        """
        try:
            import dash
            from dash import dcc, html, Input, Output
            import plotly.graph_objs as go
            import plotly.express as px
        except ImportError:
            logger.error("Dash and Plotly required for interactive dashboard. Install with: pip install dash plotly")
            return
        
        # Generate explanations
        explanations = self.explainer.explain_model(X, y)
        
        # Create Dash app
        app = dash.Dash(__name__)
        
        app.layout = html.Div([
            html.H1("Model Explainability Dashboard", style={'textAlign': 'center'}),
            
            html.Div([
                dcc.Tabs(id='tabs', value='overview', children=[
                    dcc.Tab(label='Overview', value='overview'),
                    dcc.Tab(label='Feature Importance', value='importance'),
                    dcc.Tab(label='Partial Dependence', value='partial'),
                    dcc.Tab(label='SHAP Values', value='shap'),
                    dcc.Tab(label='Model Performance', value='performance')
                ])
            ]),
            
            html.Div(id='tab-content')
        ])
        
        @app.callback(Output('tab-content', 'children'), [Input('tabs', 'value')])
        def render_tab_content(tab):
            if tab == 'overview':
                return self._create_overview_tab(explanations)
            elif tab == 'importance':
                return self._create_importance_tab(explanations)
            elif tab == 'partial':
                return self._create_partial_tab(explanations)
            elif tab == 'shap':
                return self._create_shap_tab(explanations)
            elif tab == 'performance':
                return self._create_performance_tab(X, y)
        
        # Launch dashboard
        app.run_server(debug=debug, port=port)
    
    def _create_overview_tab(self, explanations):
        """Create overview tab content."""
        return html.Div([
            html.H3("Model Overview"),
            html.P(f"Model Type: {self.explainer.model.__class__.__name__}"),
            html.P(f"Number of Features: {len(self.explainer.feature_names) if self.explainer.feature_names else 'Unknown'}"),
            html.H3("Available Explanations"),
            html.Ul([
                html.Li(f"{exp_type}: {'✓' if 'error' not in exp_data else '✗'}")
                for exp_type, exp_data in explanations.items()
            ])
        ])
    
    def _create_importance_tab(self, explanations):
        """Create feature importance tab content."""
        importance_data = explanations.get('feature_importance', {})
        
        if 'error' in importance_data or not importance_data:
            return html.Div([
                html.H3("Feature Importance"),
                html.P("Feature importance not available for this model.")
            ])
        
        # Create bar chart
        sorted_features = sorted(importance_data.items(), key=lambda x: x[1], reverse=True)
        features, importances = zip(*sorted_features[:15])
        
        fig = go.Figure(data=[
            go.Bar(x=list(features), y=list(importances))
        ])
        fig.update_layout(
            title="Feature Importance",
            xaxis_title="Features",
            yaxis_title="Importance"
        )
        
        return html.Div([
            html.H3("Feature Importance"),
            dcc.Graph(figure=fig)
        ])
    
    def _create_partial_tab(self, explanations):
        """Create partial dependence tab content."""
        pd_data = explanations.get('partial_dependence', {})
        
        if 'error' in pd_data or not pd_data:
            return html.Div([
                html.H3("Partial Dependence"),
                html.P("Partial dependence data not available.")
            ])
        
        # Create plots for each feature
        plots = []
        for feature_name, pd_info in pd_data.items():
            if 'error' not in pd_info:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=pd_info['values'],
                    y=pd_info['partial_dependence'],
                    mode='lines+markers',
                    name=feature_name
                ))
                fig.update_layout(
                    title=f"Partial Dependence: {feature_name}",
                    xaxis_title=feature_name,
                    yaxis_title="Partial Dependence"
                )
                plots.append(dcc.Graph(figure=fig))
        
        return html.Div([
            html.H3("Partial Dependence Plots")
        ] + plots)
    
    def _create_shap_tab(self, explanations):
        """Create SHAP values tab content."""
        shap_data = explanations.get('shap_values', {})
        
        if 'error' in shap_data or 'feature_importance' not in shap_data:
            return html.Div([
                html.H3("SHAP Values"),
                html.P("SHAP values not available.")
            ])
        
        # Create SHAP importance plot
        importance_data = shap_data['feature_importance']
        sorted_features = sorted(importance_data.items(), key=lambda x: x[1], reverse=True)
        features, importances = zip(*sorted_features[:15])
        
        fig = go.Figure(data=[
            go.Bar(x=list(features), y=list(importances))
        ])
        fig.update_layout(
            title="SHAP Feature Importance",
            xaxis_title="Features",
            yaxis_title="SHAP Importance"
        )
        
        return html.Div([
            html.H3("SHAP Values"),
            dcc.Graph(figure=fig)
        ])
    
    def _create_performance_tab(self, X, y):
        """Create model performance tab content."""
        if y is None:
            return html.Div([
                html.H3("Model Performance"),
                html.P("No target data available for performance metrics.")
            ])
        
        predictions = self.explainer.model.predict(X)
        
        if isinstance(self.explainer.model, ClassifierMixin):
            accuracy = accuracy_score(y, predictions)
            metrics_text = f"Accuracy: {accuracy:.4f}"
        else:
            mse = mean_squared_error(y, predictions)
            r2 = r2_score(y, predictions)
            metrics_text = f"MSE: {mse:.4f}, R²: {r2:.4f}"
        
        return html.Div([
            html.H3("Model Performance"),
            html.P(metrics_text)
        ])
