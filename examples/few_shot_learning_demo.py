#!/usr/bin/env python3
"""
Few-shot Learning Demo for igel

This script demonstrates the few-shot learning capabilities of igel, including:
1. Model-Agnostic Meta-Learning (MAML)
2. Prototypical Networks
3. Domain Adaptation
4. Transfer Learning

Usage:
    python examples/few_shot_learning_demo.py
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import os

# Import igel few-shot learning modules
from igel.few_shot_learning import (
    MAMLClassifier, 
    PrototypicalNetwork, 
    DomainAdaptation, 
    TransferLearning,
    create_few_shot_dataset,
    evaluate_few_shot_model
)

def create_sample_data(n_samples=1000, n_features=20, n_classes=5, random_state=42):
    """Create sample classification data for demonstration."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_clusters_per_class=1,
        n_redundant=5,
        n_informative=10,
        random_state=random_state
    )
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    return df

def create_domain_data(base_data, domain_shift=0.5, random_state=42):
    """Create domain-shifted data for domain adaptation demo."""
    np.random.seed(random_state)
    
    # Add domain shift by modifying features
    X = base_data.drop(columns=['target']).values
    y = base_data['target'].values
    
    # Add noise and shift to create domain shift
    X_shifted = X + np.random.normal(0, domain_shift, X.shape)
    
    # Create new DataFrame
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    df_shifted = pd.DataFrame(X_shifted, columns=feature_names)
    df_shifted['target'] = y
    
    return df_shifted

def demo_maml():
    """Demonstrate MAML (Model-Agnostic Meta-Learning)."""
    print("\n" + "="*50)
    print("DEMO: Model-Agnostic Meta-Learning (MAML)")
    print("="*50)
    
    # Create sample data
    data = create_sample_data(n_samples=500, n_classes=4)
    X = data.drop(columns=['target']).values
    y = data['target'].values
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    # Initialize MAML classifier
    maml = MAMLClassifier(
        inner_lr=0.01,
        outer_lr=0.001,
        num_tasks=5,
        shots_per_task=3,
        inner_steps=3,
        meta_epochs=20  # Reduced for demo
    )
    
    print("\nTraining MAML model...")
    maml.fit(X, y)
    
    # Create few-shot tasks for evaluation
    tasks = create_few_shot_dataset(X, y, n_way=2, k_shot=3, n_query=3)
    
    # Evaluate on few-shot tasks
    results = evaluate_few_shot_model(maml, tasks)
    
    print(f"\nMAML Results:")
    print(f"Mean accuracy: {results['mean_accuracy']:.4f}")
    print(f"Std accuracy: {results['std_accuracy']:.4f}")
    
    # Save model
    joblib.dump(maml, "maml_model.joblib")
    print("MAML model saved as 'maml_model.joblib'")
    
    return maml

def demo_prototypical_networks():
    """Demonstrate Prototypical Networks."""
    print("\n" + "="*50)
    print("DEMO: Prototypical Networks")
    print("="*50)
    
    # Create sample data
    data = create_sample_data(n_samples=500, n_classes=4)
    X = data.drop(columns=['target']).values
    y = data['target'].values
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    # Initialize Prototypical Network
    proto_net = PrototypicalNetwork(
        embedding_dim=32,
        num_tasks=5,
        shots_per_task=3,
        meta_epochs=20  # Reduced for demo
    )
    
    print("\nTraining Prototypical Network...")
    proto_net.fit(X, y)
    
    # Create few-shot tasks for evaluation
    tasks = create_few_shot_dataset(X, y, n_way=2, k_shot=3, n_query=3)
    
    # Evaluate on few-shot tasks
    results = evaluate_few_shot_model(proto_net, tasks)
    
    print(f"\nPrototypical Network Results:")
    print(f"Mean accuracy: {results['mean_accuracy']:.4f}")
    print(f"Std accuracy: {results['std_accuracy']:.4f}")
    
    # Save model
    joblib.dump(proto_net, "prototypical_network.joblib")
    print("Prototypical Network model saved as 'prototypical_network.joblib'")
    
    return proto_net

def demo_domain_adaptation():
    """Demonstrate Domain Adaptation."""
    print("\n" + "="*50)
    print("DEMO: Domain Adaptation")
    print("="*50)
    
    # Create source and target domain data
    source_data = create_sample_data(n_samples=300, n_classes=3)
    target_data = create_domain_data(source_data, domain_shift=0.8)
    
    X_source = source_data.drop(columns=['target']).values
    y_source = source_data['target'].values
    X_target = target_data.drop(columns=['target']).values
    y_target = target_data['target'].values
    
    print(f"Source domain shape: {X_source.shape}")
    print(f"Target domain shape: {X_target.shape}")
    
    # Create a base model
    from sklearn.ensemble import RandomForestClassifier
    base_model = RandomForestClassifier(n_estimators=50, random_state=42)
    
    # Train on source domain
    print("\nTraining base model on source domain...")
    base_model.fit(X_source, y_source)
    
    # Evaluate on source domain
    source_pred = base_model.predict(X_source)
    source_acc = accuracy_score(y_source, source_pred)
    print(f"Source domain accuracy: {source_acc:.4f}")
    
    # Evaluate on target domain (before adaptation)
    target_pred_before = base_model.predict(X_target)
    target_acc_before = accuracy_score(y_target, target_pred_before)
    print(f"Target domain accuracy (before adaptation): {target_acc_before:.4f}")
    
    # Perform domain adaptation
    print("\nPerforming domain adaptation...")
    adapter = DomainAdaptation(base_model)
    adapted_model = adapter.adapt_model(
        X_source, y_source, X_target, y_target, 
        adaptation_method='fine_tune'
    )
    
    # Evaluate on target domain (after adaptation)
    target_pred_after = adapted_model.predict(X_target)
    target_acc_after = accuracy_score(y_target, target_pred_after)
    print(f"Target domain accuracy (after adaptation): {target_acc_after:.4f}")
    
    print(f"Improvement: {target_acc_after - target_acc_before:.4f}")
    
    # Save adapted model
    joblib.dump(adapted_model, "adapted_model.joblib")
    print("Adapted model saved as 'adapted_model.joblib'")
    
    return adapted_model

def demo_transfer_learning():
    """Demonstrate Transfer Learning."""
    print("\n" + "="*50)
    print("DEMO: Transfer Learning")
    print("="*50)
    
    # Create source and target data
    source_data = create_sample_data(n_samples=500, n_classes=4)
    target_data = create_sample_data(n_samples=100, n_classes=3, random_state=123)
    
    X_source = source_data.drop(columns=['target']).values
    y_source = source_data['target'].values
    X_target = target_data.drop(columns=['target']).values
    y_target = target_data['target'].values
    
    print(f"Source data shape: {X_source.shape}")
    print(f"Target data shape: {X_target.shape}")
    
    # Train source model
    from sklearn.ensemble import RandomForestClassifier
    source_model = RandomForestClassifier(n_estimators=100, random_state=42)
    source_model.fit(X_source, y_source)
    
    print(f"Source model accuracy: {source_model.score(X_source, y_source):.4f}")
    
    # Perform transfer learning
    print("\nPerforming transfer learning...")
    transfer = TransferLearning(source_model)
    
    # Method 1: Feature extraction
    print("\n1. Feature extraction method:")
    transfer_model_fe = transfer.create_transfer_model(
        source_model, X_target, y_target, method='feature_extraction'
    )
    fe_acc = transfer_model_fe.score(X_target, y_target)
    print(f"Feature extraction accuracy: {fe_acc:.4f}")
    
    # Method 2: Fine-tuning
    print("\n2. Fine-tuning method:")
    transfer_model_ft = transfer.create_transfer_model(
        source_model, X_target, y_target, method='fine_tuning'
    )
    ft_acc = transfer_model_ft.score(X_target, y_target)
    print(f"Fine-tuning accuracy: {ft_acc:.4f}")
    
    # Save transfer models
    joblib.dump(transfer_model_fe, "transfer_model_fe.joblib")
    joblib.dump(transfer_model_ft, "transfer_model_ft.joblib")
    print("\nTransfer learning models saved")
    
    return transfer_model_fe, transfer_model_ft

def demo_few_shot_tasks():
    """Demonstrate few-shot task creation and evaluation."""
    print("\n" + "="*50)
    print("DEMO: Few-shot Task Creation")
    print("="*50)
    
    # Create sample data
    data = create_sample_data(n_samples=400, n_classes=6)
    X = data.drop(columns=['target']).values
    y = data['target'].values
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    # Create few-shot tasks
    tasks = create_few_shot_dataset(
        X, y, 
        n_way=3,      # 3 classes per task
        k_shot=5,     # 5 examples per class for support
        n_query=5     # 5 examples per class for query
    )
    
    print(f"Created {len(tasks)} few-shot tasks")
    
    # Analyze task distribution
    for i, (support_X, support_y, query_X, query_y) in enumerate(tasks[:3]):
        print(f"\nTask {i+1}:")
        print(f"  Support set: {support_X.shape}, classes: {np.unique(support_y)}")
        print(f"  Query set: {query_X.shape}, classes: {np.unique(query_y)}")
    
    return tasks

def main():
    """Run all few-shot learning demonstrations."""
    print("IGEL Few-shot Learning Demo")
    print("="*60)
    print("This demo showcases the few-shot learning capabilities of igel")
    print("including MAML, Prototypical Networks, Domain Adaptation, and Transfer Learning.")
    
    # Create output directory
    os.makedirs("few_shot_demo_output", exist_ok=True)
    
    try:
        # Demo 1: MAML
        maml_model = demo_maml()
        
        # Demo 2: Prototypical Networks
        proto_model = demo_prototypical_networks()
        
        # Demo 3: Domain Adaptation
        adapted_model = demo_domain_adaptation()
        
        # Demo 4: Transfer Learning
        transfer_models = demo_transfer_learning()
        
        # Demo 5: Few-shot Task Creation
        tasks = demo_few_shot_tasks()
        
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nGenerated files:")
        print("- maml_model.joblib")
        print("- prototypical_network.joblib")
        print("- adapted_model.joblib")
        print("- transfer_model_fe.joblib")
        print("- transfer_model_ft.joblib")
        
        print("\nNext steps:")
        print("1. Use these models for predictions on new data")
        print("2. Experiment with different hyperparameters")
        print("3. Try the CLI commands:")
        print("   - igel few-shot-learn --data_path=your_data.csv --yaml_path=config.yaml")
        print("   - igel domain-adapt --source_data=source.csv --target_data=target.csv")
        print("   - igel transfer-learn --source_model=model.joblib --target_data=new_data.csv")
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 