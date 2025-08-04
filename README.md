# biological_feature_reduction_pipeline

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

from biological_feature_reduction_pipeline import feature_reduction_pipeline

# Basic usage
reduced_data = feature_reduction_pipeline(
    df_data, 
    variance_threshold=0.01,
    correlation_threshold=0.8,
    filter_core_genes=0.95,
    filter_cloud_genes=0.05
)

Perfect for researchers who need to reduce feature dimensionality while maintaining biological interpretability and understanding which original features contribute to model predictions.
