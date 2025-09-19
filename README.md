# Biological Feature Reduction Pipeline

A feature reduction pipeline designed specifically for biological datasets where interpretability and feature traceability are critical. Unlike standard dimensionality reduction techniques that transform or eliminate features, this pipeline maintains interpretability through intelligent feature grouping and configurable filtering.

## Key Features

- **Variance-based filtering**: Removes zero-variance and low-variance features with configurable thresholds
- **Abundance filtering**: Eliminates core genes (highly prevalent across samples) and cloud genes (rare occurrences) based on presence thresholds
- **Identical feature clustering**: Groups features with identical values across all samples while maintaining traceability
- **Correlation-based grouping**: Combines highly correlated features using network analysis and connected components, preserving relationship information
- **Feature provenance tracking**: Maintains detailed logs of which features were combined, using intuitive naming conventions (e.g., `gene_A~gene_B~gene_C`)
- **Visualization support**: Optional correlation network visualization to understand feature relationships
- **Flexible output**: Returns processed data with optional correlation graph for further analysis

## Use Cases

- **Pangenome analysis**: Keep only shell genes, remove core genes and cloud genes by customizable thresholds
- **Microbiome studies**: Filter rare species, combine co-occurring taxa
- **Genomic variant analysis**: Group correlated SNPs, remove monomorphic sites
- **General biological ML**: Any biological dataset requiring interpretable dimensionality reduction

## Technical Features

- NetworkX-based correlation clustering
- Configurable thresholds for all filtering steps
- Optional logging and file output
- Compatible with pandas DataFrames and standard ML pipelines

## Installation

```bash
pip install git+https://github.com/sun-qibo/biological_feature_reduction_pipeline.git
```

## Quick Start


```python
# Example 1: Basic fit/transform workflow
reducer = BiologicalFeatureReducer(
    variance_threshold=0.01,
    correlation_threshold=0.8,
    presence_low_threshold=0.1,
    presence_high_threshold=0.9
)

# Fit and transform data
data_reduced = reducer.fit_transform(data)

# Save the fitted model
reducer.save_model('my_reducer.pkl')

# Get detailed report
report = reducer.get_reduction_report()
print(f"Original features: {report.original_shape[1]}")
print(f"Final features: {report.final_shape[1]}")
reducer.save_report('reduction_report.txt')   

# Later: load and use the saved model
loaded_reducer = BiologicalFeatureReducer.load_model('my_reducer.pkl')
new_data_reduced = loaded_reducer.transform(new_data)

# Example 2: Get feature mappings
feature_mapping = reducer.get_feature_mapping()
print(f"Original feature 'gene1' maps to: {feature_mapping.get('gene1', 'removed')}")

# Example 3: Access fitted parameters
params = reducer.get_fitted_parameters()
print(f"Features removed due to low variance: {params.low_variance_features}")
print(f"Correlation clusters: {params.correlation_feature_clusters}")

```



```python
# previous version , can only fit and transform at a time
from biological_feature_reducer import BiologicalFeatureReducer

# New class-based interface
reducer = BiologicalFeatureReducer(
    variance_threshold=0.01,
    correlation_threshold=0.8,
    core_gene_threshold=0.95,
    cloud_gene_threshold=0.05
)

# Perform reduction
df_reduced = reducer.fit_transform(df_data, visualize_clusters=True)



# Save results
reducer.save_results(df_reduced, "reduced_features.csv")
```

```python

# Deprecated version for backward compatibility 
from biological_feature_reducer import feature_reduction_pipeline
reduced_data = feature_reduction_pipeline(
    df_data, 
    variance_threshold=0.01,
    correlation_threshold=0.8,
    filter_core_genes=0.95,
    filter_cloud_genes=0.05
)
```
Perfect for researchers who need to reduce feature dimensionality while maintaining biological interpretability and understanding which original features contribute to model predictions.
