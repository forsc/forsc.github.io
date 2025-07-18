---
title: "Distributed Model Training with Ray on Databricks: 80% Efficiency Gains"
date: "2024-11-10"
category: "MLOps"
tags: ["Ray", "Databricks", "Distributed Computing", "MLOps"]
excerpt: "How we achieved massive performance improvements in model training using Ray's distributed computing framework. Includes code examples and performance benchmarks from production deployments."
---

# Distributed Model Training with Ray on Databricks: 80% Efficiency Gains

## The Scale Challenge

Training machine learning models on massive datasets requires more than just better algorithms—it demands efficient distributed computing. At AB-InBev, we process terabytes of transaction data to train recommendation models, and our traditional single-node training was becoming a bottleneck.

Enter Ray: a unified framework for scaling Python and AI applications. By implementing Ray on Databricks, we achieved an 80% improvement in training efficiency while maintaining model quality.

## Why Ray + Databricks?

### The Perfect Combination
- **Ray**: Handles distributed computing complexities
- **Databricks**: Provides managed Spark infrastructure and MLflow integration
- **Combined**: Seamless scaling from laptop to cluster

```python
import ray
from ray import train
from ray.train import Trainer
from ray.train.integrations.mlflow import MLflowLoggerCallback

# Initialize Ray on Databricks cluster
ray.init(address="ray://head-node:10001")
print(f"Ray cluster: {ray.cluster_resources()}")
```

## Distributed XGBoost Training

### Traditional Single-Node Approach
```python
# Old approach - single node training
import xgboost as xgb
import time

start_time = time.time()
model = xgb.XGBRegressor(n_estimators=1000, max_depth=6)
model.fit(X_train, y_train)
training_time = time.time() - start_time
print(f"Single-node training time: {training_time:.2f} seconds")
```

### Ray Distributed Approach
```python
from ray.train.xgboost import XGBoostTrainer
from ray.air.config import ScalingConfig

def train_distributed_xgboost():
    trainer = XGBoostTrainer(
        params={
            "objective": "reg:squarederror",
            "n_estimators": 1000,
            "max_depth": 6,
            "learning_rate": 0.1,
            "tree_method": "hist",  # Optimized for distributed training
        },
        datasets={"train": ray_train_dataset, "validation": ray_val_dataset},
        scaling_config=ScalingConfig(
            num_workers=8,  # Use 8 workers
            resources_per_worker={"CPU": 4, "memory": 8000000000}
        ),
        run_config=train.RunConfig(
            callbacks=[MLflowLoggerCallback(experiment_name="distributed_xgb")]
        )
    )
    
    result = trainer.fit()
    return result

# Run distributed training
start_time = time.time()
result = train_distributed_xgboost()
distributed_time = time.time() - start_time
print(f"Distributed training time: {distributed_time:.2f} seconds")
```

## Advanced Ray Tune Hyperparameter Optimization

### Parallel Hyperparameter Search
```python
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.bayesopt import BayesOptSearch

def objective(config):
    # Training function with hyperparameters
    model = xgb.XGBRegressor(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        learning_rate=config["learning_rate"],
        subsample=config["subsample"],
        colsample_bytree=config["colsample_bytree"]
    )
    
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, predictions))
    
    return {"rmse": rmse}

# Define search space
search_space = {
    "n_estimators": tune.randint(100, 1000),
    "max_depth": tune.randint(3, 10),
    "learning_rate": tune.loguniform(0.01, 0.3),
    "subsample": tune.uniform(0.6, 1.0),
    "colsample_bytree": tune.uniform(0.6, 1.0)
}

# Configure search algorithm and scheduler
search_alg = BayesOptSearch(metric="rmse", mode="min")
scheduler = ASHAScheduler(
    time_attr="training_iteration",
    metric="rmse",
    mode="min",
    max_t=100,
    grace_period=10
)

# Run hyperparameter optimization
tuner = tune.Tuner(
    objective,
    param_space=search_space,
    tune_config=tune.TuneConfig(
        search_alg=search_alg,
        scheduler=scheduler,
        num_samples=100,  # Try 100 different combinations
        max_concurrent_trials=20  # Run 20 trials in parallel
    ),
    run_config=train.RunConfig(
        name="xgboost_hyperopt",
        callbacks=[MLflowLoggerCallback()]
    )
)

results = tuner.fit()
best_config = results.get_best_result().config
print(f"Best hyperparameters: {best_config}")
```

## Custom Ensemble Training Pipeline

### Distributed Ensemble Strategy
```python
from ray.data import Dataset
import ray.data as rd

class DistributedEnsembleTrainer:
    def __init__(self, models_config, scaling_config):
        self.models_config = models_config
        self.scaling_config = scaling_config
        self.trained_models = {}
    
    def prepare_data(self, data_path):
        # Load and preprocess data using Ray Data
        dataset = rd.read_parquet(data_path)
        
        # Distributed preprocessing
        processed_dataset = dataset.map_batches(
            self.preprocess_batch,
            batch_format="pandas",
            num_cpus=2
        )
        
        # Split data
        train_dataset, val_dataset = processed_dataset.train_test_split(test_size=0.2)
        return train_dataset, val_dataset
    
    def preprocess_batch(self, batch):
        # Feature engineering pipeline
        batch['temporal_features'] = self.extract_temporal_features(batch)
        batch['interaction_features'] = self.create_interaction_features(batch)
        return batch
    
    def train_model(self, model_name, model_config, train_data, val_data):
        if model_config['type'] == 'xgboost':
            trainer = XGBoostTrainer(
                params=model_config['params'],
                datasets={"train": train_data, "validation": val_data},
                scaling_config=self.scaling_config
            )
        elif model_config['type'] == 'lightgbm':
            trainer = LightGBMTrainer(
                params=model_config['params'],
                datasets={"train": train_data, "validation": val_data},
                scaling_config=self.scaling_config
            )
        
        result = trainer.fit()
        return result.checkpoint
    
    def train_ensemble(self, train_data, val_data):
        # Train multiple models in parallel
        @ray.remote
        def train_single_model(model_name, model_config):
            return self.train_model(model_name, model_config, train_data, val_data)
        
        # Submit all training jobs
        futures = []
        for model_name, model_config in self.models_config.items():
            future = train_single_model.remote(model_name, model_config)
            futures.append((model_name, future))
        
        # Collect results
        for model_name, future in futures:
            checkpoint = ray.get(future)
            self.trained_models[model_name] = checkpoint
            print(f"Completed training: {model_name}")
        
        return self.trained_models

# Configuration for ensemble
models_config = {
    "xgb_model_1": {
        "type": "xgboost",
        "params": {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.1}
    },
    "xgb_model_2": {
        "type": "xgboost", 
        "params": {"n_estimators": 800, "max_depth": 4, "learning_rate": 0.05}
    },
    "lgbm_model": {
        "type": "lightgbm",
        "params": {"n_estimators": 600, "max_depth": 8, "learning_rate": 0.08}
    }
}

scaling_config = ScalingConfig(num_workers=6, resources_per_worker={"CPU": 4})

# Train ensemble
ensemble_trainer = DistributedEnsembleTrainer(models_config, scaling_config)
train_data, val_data = ensemble_trainer.prepare_data("s3://data-bucket/training-data/")
trained_models = ensemble_trainer.train_ensemble(train_data, val_data)
```

## Advanced Ray Data Pipeline

### Efficient Data Processing
```python
import ray.data as rd
from ray.data.preprocessors import StandardScaler, LabelEncoder

class RayDataPipeline:
    def __init__(self):
        self.preprocessors = {}
    
    def create_pipeline(self, data_path, target_column):
        # Read data from various sources
        if data_path.endswith('.parquet'):
            dataset = rd.read_parquet(data_path)
        elif data_path.endswith('.csv'):
            dataset = rd.read_csv(data_path)
        else:
            dataset = rd.read_datasource(data_path)
        
        # Distributed feature engineering
        dataset = dataset.map_batches(
            self.feature_engineering,
            batch_format="pandas",
            num_cpus=2
        )
        
        # Handle categorical variables
        categorical_columns = self.get_categorical_columns(dataset)
        for col in categorical_columns:
            encoder = LabelEncoder(columns=[col])
            dataset = encoder.fit_transform(dataset)
            self.preprocessors[f"{col}_encoder"] = encoder
        
        # Scale numerical features
        numerical_columns = self.get_numerical_columns(dataset)
        scaler = StandardScaler(columns=numerical_columns)
        dataset = scaler.fit_transform(dataset)
        self.preprocessors['scaler'] = scaler
        
        # Split features and target
        feature_columns = [col for col in dataset.columns() if col != target_column]
        
        return dataset.select_columns(feature_columns), dataset.select_columns([target_column])
    
    def feature_engineering(self, batch):
        # Custom feature engineering logic
        batch['interaction_term'] = batch['feature_1'] * batch['feature_2']
        batch['log_feature'] = np.log1p(batch['numerical_feature'])
        batch['date_features'] = pd.to_datetime(batch['timestamp']).dt.day_of_week
        return batch
    
    def get_categorical_columns(self, dataset):
        # Identify categorical columns
        sample = dataset.take(1)[0]
        return [col for col, val in sample.items() if isinstance(val, str)]
    
    def get_numerical_columns(self, dataset):
        # Identify numerical columns
        sample = dataset.take(1)[0]
        return [col for col, val in sample.items() if isinstance(val, (int, float))]

# Usage
pipeline = RayDataPipeline()
X, y = pipeline.create_pipeline("s3://bucket/data.parquet", "target")
```

## Performance Monitoring and Optimization

### Ray Dashboard Integration
```python
import ray.util.state as state

def monitor_cluster_performance():
    # Get cluster status
    cluster_status = state.summarize_actors()
    print(f"Active actors: {len(cluster_status)}")
    
    # Monitor resource utilization
    resources = ray.cluster_resources()
    print(f"Available CPUs: {resources.get('CPU', 0)}")
    print(f"Available Memory: {resources.get('memory', 0) / 1e9:.2f} GB")
    
    # Check for failed tasks
    failed_tasks = state.list_tasks(filters=[("state", "=", "FAILED")])
    if failed_tasks:
        print(f"Warning: {len(failed_tasks)} failed tasks detected")

# Custom metrics collection
class TrainingMetricsCollector:
    def __init__(self):
        self.metrics = []
    
    def log_training_metrics(self, epoch, metrics):
        timestamp = time.time()
        self.metrics.append({
            'timestamp': timestamp,
            'epoch': epoch,
            'train_loss': metrics['train_loss'],
            'val_loss': metrics['val_loss'],
            'memory_usage': self.get_memory_usage(),
            'cpu_utilization': self.get_cpu_utilization()
        })
    
    def get_memory_usage(self):
        resources = ray.cluster_resources()
        used_memory = ray.available_resources().get('memory', 0)
        total_memory = resources.get('memory', 0)
        return (total_memory - used_memory) / total_memory
    
    def export_metrics(self, path):
        pd.DataFrame(self.metrics).to_csv(path, index=False)
```

## Production Deployment on Databricks

### Databricks Integration
```python
# Databricks notebook integration
import mlflow
from mlflow.tracking import MlflowClient

def deploy_ray_model_pipeline():
    # Start MLflow run
    with mlflow.start_run(run_name="ray_distributed_training"):
        # Initialize Ray cluster
        ray.init(address="ray://databricks-cluster:10001")
        
        # Log cluster configuration
        mlflow.log_params({
            "num_workers": 8,
            "worker_cpus": 4,
            "worker_memory": "8GB",
            "ray_version": ray.__version__
        })
        
        # Run distributed training
        result = train_distributed_xgboost()
        
        # Log results
        mlflow.log_metrics({
            "training_time": result.metrics["training_time"],
            "best_score": result.metrics["validation_score"],
            "num_iterations": result.metrics["iterations"]
        })
        
        # Save model
        model_uri = mlflow.sklearn.log_model(
            result.model,
            "distributed_model",
            registered_model_name="ProductionRecommender"
        )
        
        ray.shutdown()
        return model_uri

# Automated deployment pipeline
def create_deployment_pipeline():
    from databricks.feature_store import FeatureStoreClient
    
    fs = FeatureStoreClient()
    
    # Create feature table
    feature_table = fs.create_table(
        name="recommendation_features",
        primary_keys=["user_id", "product_id"],
        df=feature_df,
        description="Features for recommendation model"
    )
    
    # Deploy model with Ray serving
    model_uri = deploy_ray_model_pipeline()
    
    # Create model serving endpoint
    client = MlflowClient()
    client.create_model_version(
        name="ProductionRecommender",
        source=model_uri,
        description="Ray-trained ensemble model"
    )
```

## Performance Results

### Benchmark Comparison

| Metric | Single Node | Ray Distributed | Improvement |
|--------|-------------|-----------------|-------------|
| Training Time | 45 minutes | 9 minutes | **80% faster** |
| Memory Usage | 32 GB | 8 GB per node | **75% reduction** |
| Hyperparameter Trials | 10/hour | 120/hour | **12x throughput** |
| Model Quality (RMSE) | 0.245 | 0.242 | **1.2% better** |
| Cost per Training Run | $15 | $8 | **47% savings** |

### Scaling Characteristics
```python
# Performance analysis
training_times = {
    1: 45.2,   # 1 worker
    2: 24.1,   # 2 workers  
    4: 13.8,   # 4 workers
    8: 9.2,    # 8 workers
    16: 7.1,   # 16 workers
    32: 6.8    # 32 workers (diminishing returns)
}

# Calculate efficiency
base_time = training_times[1]
for workers, time in training_times.items():
    efficiency = (base_time / (workers * time)) * 100
    print(f"{workers} workers: {efficiency:.1f}% efficiency")
```

## Key Learnings

### 1. Right-Size Your Cluster
- **Sweet spot**: 8-16 workers for our workload
- **Beyond 16**: Diminishing returns due to communication overhead
- **Cost optimization**: Scale down during development, up for production

### 2. Data Pipeline Optimization
- **Bottleneck**: I/O often more critical than compute
- **Solution**: Use Ray Data for efficient data loading and preprocessing
- **Impact**: 3x faster data pipeline

### 3. Memory Management
- **Challenge**: Large models can cause OOM errors
- **Solution**: Partition data and use Ray's object store
- **Result**: Stable training on 100GB+ datasets

## Best Practices

### 1. Cluster Configuration
```python
# Optimal Ray cluster setup for ML workloads
ray_config = {
    "num_cpus": 4,
    "num_gpus": 0,  # CPU-only for XGBoost
    "memory": 8 * 1024**3,  # 8GB per worker
    "object_store_memory": 2 * 1024**3  # 2GB object store
}
```

### 2. Error Handling
```python
@ray.remote(max_retries=3, retry_exceptions=True)
def robust_training_task(config):
    try:
        return train_model(config)
    except Exception as e:
        print(f"Training failed: {e}")
        raise
```

### 3. Resource Management
```python
# Prevent resource exhaustion
@ray.remote(num_cpus=2, memory=4*1024**3)
def memory_intensive_task():
    # Task with explicit resource requirements
    pass
```

## Conclusion

Ray on Databricks has transformed our ML training pipeline, delivering:
- **80% faster training** through efficient parallelization
- **Cost savings** through better resource utilization  
- **Better models** via extensive hyperparameter search
- **Simplified operations** with managed infrastructure

The combination of Ray's distributed computing capabilities and Databricks' managed platform provides a powerful foundation for scaling ML workloads. Start small, measure everything, and scale incrementally based on your specific bottlenecks.

---

*Want to learn more about distributed ML? Connect with me on [LinkedIn](https://linkedin.com/in/rahulbhow) for insights on scaling ML systems.* 