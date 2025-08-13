# Few-Shot Learning in igel

This document describes the few-shot learning capabilities added to igel, addressing GitHub issue #237 "Add Support for Few-Shot Learning".

## Overview

Few-shot learning enables machine learning models to learn from very few examples, making it particularly useful when:
- Limited labeled data is available
- New classes need to be learned quickly
- Domain adaptation is required
- Transfer learning is needed

## Features Implemented

### 1. Model-Agnostic Meta-Learning (MAML)

MAML is a meta-learning algorithm that learns to quickly adapt to new tasks with few examples by learning good initial parameters.

**Key Features:**
- Inner loop adaptation for task-specific learning
- Outer loop meta-update for learning good initial parameters
- Configurable learning rates and training parameters
- Support for both classification and regression tasks

**Usage:**
```python
from igel.few_shot_learning import MAMLClassifier

# Initialize MAML
maml = MAMLClassifier(
    inner_lr=0.01,        # Inner loop learning rate
    outer_lr=0.001,       # Outer loop learning rate
    num_tasks=10,         # Tasks per meta-epoch
    shots_per_task=5,     # Examples per class
    inner_steps=5,        # Gradient steps for adaptation
    meta_epochs=100       # Meta-training epochs
)

# Train the model
maml.fit(X, y)

# Make predictions
predictions = maml.predict(X_new)
```

### 2. Prototypical Networks

Prototypical Networks learn a metric space where classification can be performed by computing distances to prototype representations of each class.

**Key Features:**
- Embedding-based approach for few-shot learning
- Computes class prototypes from support examples
- Uses Euclidean distance for classification
- Efficient for multi-class few-shot problems

**Usage:**
```python
from igel.few_shot_learning import PrototypicalNetwork

# Initialize Prototypical Network
proto_net = PrototypicalNetwork(
    embedding_dim=64,     # Embedding dimension
    num_tasks=10,         # Tasks per meta-epoch
    shots_per_task=5,     # Examples per class
    meta_epochs=100       # Meta-training epochs
)

# Train the model
proto_net.fit(X, y)

# Make predictions
predictions = proto_net.predict(X_new)
```

### 3. Domain Adaptation

Domain adaptation utilities help adapt models trained on a source domain to perform well on a target domain.

**Supported Methods:**
- **Fine-tuning**: Adapt the entire model to the target domain
- **Domain Adversarial**: Use adversarial training for domain adaptation
- **MAML-based**: Use MAML for domain adaptation

**Usage:**
```python
from igel.few_shot_learning import DomainAdaptation
from sklearn.ensemble import RandomForestClassifier

# Create base model
base_model = RandomForestClassifier()

# Initialize domain adaptation
adapter = DomainAdaptation(base_model)

# Adapt model to target domain
adapted_model = adapter.adapt_model(
    source_X, source_y,    # Source domain data
    target_X, target_y,    # Target domain data
    adaptation_method='fine_tune'
)
```

### 4. Transfer Learning

Transfer learning utilities leverage pre-trained models for new tasks.

**Supported Methods:**
- **Feature Extraction**: Extract features from pre-trained model
- **Fine-tuning**: Fine-tune the entire pre-trained model

**Usage:**
```python
from igel.few_shot_learning import TransferLearning
import joblib

# Load pre-trained model
source_model = joblib.load('pretrained_model.joblib')

# Initialize transfer learning
transfer = TransferLearning(source_model)

# Create transfer model
transfer_model = transfer.create_transfer_model(
    source_model, X_target, y_target,
    method='feature_extraction'
)
```

## CLI Commands

### Few-Shot Learning Training

```bash
igel few-shot-learn \
    --data_path=data/train.csv \
    --yaml_path=config.yaml \
    --n_way=3 \
    --k_shot=5 \
    --n_query=5
```

### Domain Adaptation

```bash
igel domain-adapt \
    --source_data=source.csv \
    --target_data=target.csv \
    --method=fine_tune \
    --output_model=adapted_model.joblib
```

### Transfer Learning

```bash
igel transfer-learn \
    --source_model=pretrained.joblib \
    --target_data=new_data.csv \
    --method=feature_extraction \
    --output_model=transfer_model.joblib
```

## Configuration Files

### Few-Shot Learning Configuration

```yaml
# few_shot_config.yaml
dataset:
  missing_values: "drop"
  categorical_encoding: "label"
  scaling: "standard"
  random_numbers:
    generate_reproducible: true
    seed: 42

model:
  type: "few_shot_learning"
  algorithm: "MAML"  # or "PrototypicalNetwork"
  
  arguments:
    # MAML parameters
    inner_lr: 0.01
    outer_lr: 0.001
    num_tasks: 10
    shots_per_task: 5
    inner_steps: 5
    meta_epochs: 100
    
    # Prototypical Network parameters
    # embedding_dim: 64

target: ["target"]
```

## Utility Functions

### Creating Few-Shot Tasks

```python
from igel.few_shot_learning import create_few_shot_dataset

# Create few-shot tasks
tasks = create_few_shot_dataset(
    X, y,
    n_way=3,      # Number of classes per task
    k_shot=5,     # Examples per class for support
    n_query=5     # Examples per class for query
)
```

### Evaluating Few-Shot Models

```python
from igel.few_shot_learning import evaluate_few_shot_model

# Evaluate model on few-shot tasks
results = evaluate_few_shot_model(model, tasks)

print(f"Mean accuracy: {results['mean_accuracy']:.4f}")
print(f"Std accuracy: {results['std_accuracy']:.4f}")
```

## Examples

### Complete Example: MAML Training

```python
import pandas as pd
from igel.few_shot_learning import MAMLClassifier, create_few_shot_dataset, evaluate_few_shot_model

# Load data
data = pd.read_csv('data.csv')
X = data.drop(columns=['target']).values
y = data['target'].values

# Initialize MAML
maml = MAMLClassifier(
    inner_lr=0.01,
    outer_lr=0.001,
    num_tasks=10,
    shots_per_task=5,
    inner_steps=5,
    meta_epochs=100
)

# Train model
maml.fit(X, y)

# Create evaluation tasks
tasks = create_few_shot_dataset(X, y, n_way=2, k_shot=5, n_query=5)

# Evaluate
results = evaluate_few_shot_model(maml, tasks)
print(f"Mean accuracy: {results['mean_accuracy']:.4f}")

# Save model
import joblib
joblib.dump(maml, 'maml_model.joblib')
```

### Domain Adaptation Example

```python
import pandas as pd
from igel.few_shot_learning import DomainAdaptation
from sklearn.ensemble import RandomForestClassifier

# Load source and target data
source_data = pd.read_csv('source.csv')
target_data = pd.read_csv('target.csv')

X_source = source_data.drop(columns=['target']).values
y_source = source_data['target'].values
X_target = target_data.drop(columns=['target']).values
y_target = target_data['target'].values

# Create and train base model
base_model = RandomForestClassifier()
base_model.fit(X_source, y_source)

# Perform domain adaptation
adapter = DomainAdaptation(base_model)
adapted_model = adapter.adapt_model(
    X_source, y_source, X_target, y_target,
    adaptation_method='fine_tune'
)

# Save adapted model
import joblib
joblib.dump(adapted_model, 'adapted_model.joblib')
```

## Best Practices

### 1. Data Preparation
- Ensure sufficient class diversity in your dataset
- Balance the number of examples per class
- Preprocess data consistently across domains

### 2. Hyperparameter Tuning
- Start with default parameters and adjust based on results
- Use cross-validation for hyperparameter selection
- Monitor meta-loss during training

### 3. Evaluation
- Use multiple few-shot tasks for evaluation
- Report mean and standard deviation of accuracy
- Compare against baseline methods

### 4. Model Selection
- Use MAML for complex adaptation scenarios
- Use Prototypical Networks for metric-based learning
- Use domain adaptation when source and target domains differ

## Limitations and Considerations

1. **Computational Cost**: Meta-learning can be computationally expensive
2. **Data Requirements**: Still requires sufficient data for meta-training
3. **Hyperparameter Sensitivity**: Performance can be sensitive to hyperparameters
4. **Task Similarity**: Assumes tasks are related during meta-training

## Future Enhancements

Potential improvements for future versions:
- Support for regression tasks in MAML
- More sophisticated domain adaptation methods
- Integration with deep learning frameworks
- Automated hyperparameter optimization
- Support for multi-modal few-shot learning

## References

1. Finn, C., Abbeel, P., & Levine, S. (2017). Model-agnostic meta-learning for fast adaptation of deep networks.
2. Snell, J., Swersky, K., & Zemel, R. (2017). Prototypical networks for few-shot learning.
3. Ganin, Y., & Lempitsky, V. (2015). Unsupervised domain adaptation by backpropagation.

## Support

For issues and questions related to few-shot learning in igel:
- Check the examples in `examples/few_shot_learning_demo.py`
- Review the configuration examples
- Open an issue on the GitHub repository 