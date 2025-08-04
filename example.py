"""
Example usage of the Biological Feature Reduction Pipeline

This script demonstrates how to use the BiologicalFeatureReducer with toy datasets
that simulate real biological data scenarios.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from biological_feature_reducer import BiologicalFeatureReducer


def create_pangenome_toy_data(n_samples=50, n_features=200, random_seed=42):
    """
    Create toy pangenome data with different gene categories.
    
    Returns DataFrame with:
    - Core genes (present in >95% of samples)
    - Shell genes (present in 15-95% of samples) 
    - Cloud genes (present in <15% of samples)
    - Correlated gene clusters
    - Zero variance genes
    """
    np.random.seed(random_seed)
    
    data = {}
    feature_names = []
    
    # 1. Core genes (10 genes, present in >95% samples)
    n_core = 10
    for i in range(n_core):
        # Core genes present in 95-100% of samples
        presence_rate = np.random.uniform(0.95, 1.0)
        gene_data = np.random.binomial(1, presence_rate, n_samples)
        data[f'core_gene_{i+1}'] = gene_data
        feature_names.append(f'core_gene_{i+1}')
    
    # 2. Cloud genes (20 genes, present in <15% samples)
    n_cloud = 20
    for i in range(n_cloud):
        # Cloud genes present in 1-15% of samples
        presence_rate = np.random.uniform(0.01, 0.15)
        gene_data = np.random.binomial(1, presence_rate, n_samples)
        data[f'cloud_gene_{i+1}'] = gene_data
        feature_names.append(f'cloud_gene_{i+1}')
    
    # 3. Shell genes - create some correlated clusters
    n_shell_clusters = 8
    genes_per_cluster = 4
    
    for cluster in range(n_shell_clusters):
        # Base pattern for this cluster
        base_presence_rate = np.random.uniform(0.3, 0.8)
        base_pattern = np.random.binomial(1, base_presence_rate, n_samples)
        
        for gene in range(genes_per_cluster):
            if gene == 0:
                # First gene in cluster - use base pattern
                gene_data = base_pattern.copy()
            else:
                # Other genes - highly correlated with base pattern
                correlation_strength = np.random.uniform(0.8, 0.95)
                noise = np.random.binomial(1, 1-correlation_strength, n_samples)
                gene_data = np.where(noise, 1-base_pattern, base_pattern)
            
            data[f'shell_gene_cluster{cluster+1}_gene{gene+1}'] = gene_data
            feature_names.append(f'shell_gene_cluster{cluster+1}_gene{gene+1}')
    
    # 4. Identical genes (duplicates)
    n_identical_sets = 5
    for i in range(n_identical_sets):
        # Create base pattern
        base_pattern = np.random.binomial(1, 0.5, n_samples)
        # Create 2-3 identical copies
        n_copies = np.random.randint(2, 4)
        for copy in range(n_copies):
            data[f'identical_set{i+1}_copy{copy+1}'] = base_pattern.copy()
            feature_names.append(f'identical_set{i+1}_copy{copy+1}')
    
    # 5. Zero variance genes (all 0 or all 1)
    n_zero_var = 8
    for i in range(n_zero_var):
        if i < 4:
            gene_data = np.zeros(n_samples, dtype=int)  # All absent
        else:
            gene_data = np.ones(n_samples, dtype=int)   # All present
        data[f'zero_var_gene_{i+1}'] = gene_data
        feature_names.append(f'zero_var_gene_{i+1}')
    
    # 6. Low variance genes
    n_low_var = 10
    for i in range(n_low_var):
        # Very low presence rate
        presence_rate = np.random.uniform(0.001, 0.01)
        gene_data = np.random.binomial(1, presence_rate, n_samples)
        data[f'low_var_gene_{i+1}'] = gene_data
        feature_names.append(f'low_var_gene_{i+1}')
    
    # Create DataFrame
    df = pd.DataFrame(data)
    df.index = [f'sample_{i+1}' for i in range(n_samples)]
    
    return df


def create_microbiome_toy_data(n_samples=40, n_species=150, random_seed=42):
    """
    Create toy microbiome abundance data.
    
    Returns DataFrame with:
    - Highly abundant species (core microbiome)
    - Rare species
    - Co-occurring species clusters
    - Log-normal abundance distributions
    """
    np.random.seed(random_seed)
    
    data = {}
    
    # 1. Core microbiome species (high abundance, present in most samples)
    n_core_species = 8
    for i in range(n_core_species):
        # Core species with log-normal distribution
        base_abundance = np.random.uniform(100, 1000)
        abundances = np.random.lognormal(
            mean=np.log(base_abundance), 
            sigma=0.5, 
            size=n_samples
        )
        # Some samples might not have this species
        presence_mask = np.random.binomial(1, 0.9, n_samples)
        abundances = abundances * presence_mask
        data[f'core_species_{i+1}'] = abundances
    
    # 2. Co-occurring species clusters (correlated abundances)
    n_clusters = 6
    species_per_cluster = 3
    
    for cluster in range(n_clusters):
        # Base abundance pattern for cluster
        base_abundance = np.random.uniform(10, 200)
        base_pattern = np.random.lognormal(
            mean=np.log(base_abundance), 
            sigma=0.7, 
            size=n_samples
        )
        presence_rate = np.random.uniform(0.4, 0.8)
        cluster_presence = np.random.binomial(1, presence_rate, n_samples)
        
        for species in range(species_per_cluster):
            # Add some variation but keep correlation
            noise_factor = np.random.uniform(0.5, 1.5, n_samples)
            species_abundance = base_pattern * noise_factor * cluster_presence
            # Add some independent presence/absence
            individual_presence = np.random.binomial(1, 0.9, n_samples)
            species_abundance = species_abundance * individual_presence
            
            data[f'cluster{cluster+1}_species_{species+1}'] = species_abundance
    
    # 3. Rare species (low abundance, present in few samples)
    n_rare = 30
    for i in range(n_rare):
        base_abundance = np.random.uniform(1, 20)
        abundances = np.random.lognormal(
            mean=np.log(base_abundance), 
            sigma=0.8, 
            size=n_samples
        )
        # Very low presence rate
        presence_rate = np.random.uniform(0.05, 0.2)
        presence_mask = np.random.binomial(1, presence_rate, n_samples)
        abundances = abundances * presence_mask
        data[f'rare_species_{i+1}'] = abundances
    
    # 4. Intermediate abundance species
    n_intermediate = n_species - n_core_species - (n_clusters * species_per_cluster) - n_rare
    for i in range(n_intermediate):
        base_abundance = np.random.uniform(20, 100)
        abundances = np.random.lognormal(
            mean=np.log(base_abundance), 
            sigma=0.6, 
            size=n_samples
        )
        presence_rate = np.random.uniform(0.3, 0.7)
        presence_mask = np.random.binomial(1, presence_rate, n_samples)
        abundances = abundances * presence_mask
        data[f'intermediate_species_{i+1}'] = abundances
    
    # Create DataFrame
    df = pd.DataFrame(data)
    df.index = [f'sample_{i+1}' for i in range(n_samples)]
    
    # Round to reasonable precision
    df = df.round(2)
    
    return df


def example_pangenome_analysis():
    """Example: Pangenome analysis workflow"""
    print("=" * 60)
    print("EXAMPLE 1: PANGENOME ANALYSIS")
    print("=" * 60)
    
    # Create toy pangenome data
    print("Creating toy pangenome dataset...")
    df_pangenome = create_pangenome_toy_data(n_samples=100, n_features=200)
    print(f"Created dataset with shape: {df_pangenome.shape}")
    print(f"Data type: Binary presence/absence (0/1)")
    print()
    
    # Initialize reducer for pangenome analysis
    reducer = BiologicalFeatureReducer(
        variance_threshold=0.01,        # Remove very low variance genes
        correlation_threshold=0.9,      # High threshold for gene co-occurrence
        core_gene_threshold=0.95,       # Remove genes present in >95% samples
        cloud_gene_threshold=0.05       # Remove genes present in <5% samples
    )
    
    # Apply reduction pipeline
    print("Applying feature reduction pipeline...")
    df_reduced = reducer.fit_transform(
        df_pangenome,
        visualize_clusters=False  # Set to True to see correlation networks
    )
    
    # Show results
    print(f"\nReduction Results:")
    print(f"Original features: {df_pangenome.shape[1]}")
    print(f"Reduced features: {df_reduced.shape[1]}")
    print(f"Reduction ratio: {df_reduced.shape[1]/df_pangenome.shape[1]:.1%}")
    
    # Get detailed report
    report = reducer.get_reduction_report()
    print(f"\nDetailed Report:")
    for step, count in report.reduction_summary.items():
        print(f"  {step}: {count} features affected")
    
    print(f"\nFeature clusters found: {len(report.feature_clusters)}")
    for cluster_name, features in list(report.feature_clusters.items())[:3]:
        print(f"  {cluster_name}: {features}")
    
    return df_pangenome, df_reduced, reducer


def example_microbiome_analysis():
    """Example: Microbiome abundance analysis workflow"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: MICROBIOME ANALYSIS")
    print("=" * 60)
    
    # Create toy microbiome data
    print("Creating toy microbiome dataset...")
    df_microbiome = create_microbiome_toy_data(n_samples=80, n_species=120)
    print(f"Created dataset with shape: {df_microbiome.shape}")
    print(f"Data type: Species abundance counts")
    print()
    
    # Convert to presence/absence for this example
    # (In practice, you might work with relative abundances)
    df_binary = (df_microbiome > 0).astype(int)
    
    # Initialize reducer for microbiome analysis
    reducer = BiologicalFeatureReducer(
        variance_threshold=0.005,       # Lower threshold for abundance data
        correlation_threshold=0.85,     # Co-occurring species
        core_gene_threshold=0.9,        # Highly prevalent species
        cloud_gene_threshold=0.1        # Very rare species
    )
    
    # Apply reduction pipeline
    print("Applying feature reduction pipeline...")
    df_reduced = reducer.fit_transform(
        df_binary,
        combination_method='mean',  # Average correlated species abundances
        visualize_clusters=False
    )
    
    # Show results
    print(f"\nReduction Results:")
    print(f"Original species: {df_binary.shape[1]}")
    print(f"Reduced features: {df_reduced.shape[1]}")
    print(f"Reduction ratio: {df_reduced.shape[1]/df_binary.shape[1]:.1%}")
    
    # Show some example reduced feature names
    print(f"\nExample combined features:")
    combined_features = [col for col in df_reduced.columns if '~' in col][:5]
    for feature in combined_features:
        print(f"  {feature}")
    
    return df_microbiome, df_reduced, reducer


def example_custom_pipeline():
    """Example: Custom pipeline with specific steps"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: CUSTOM PIPELINE")
    print("=" * 60)
    
    # Create data
    df_data = create_pangenome_toy_data(n_samples=60, n_features=100)
    
    # Initialize reducer
    reducer = BiologicalFeatureReducer(
        variance_threshold=0.02,
        correlation_threshold=0.8
    )
    
    # Run only specific steps
    print("Running custom pipeline with specific steps...")
    df_reduced = reducer.fit_transform(
        df_data,
        steps_to_perform=['zero_variance', 'identical', 'correlated'],  # Skip abundance filtering
        combination_method='median',  # Use median instead of mean
        visualize_clusters=True  # Show the correlation network
    )
    
    print(f"\nCustom pipeline results:")
    print(f"Original: {df_data.shape[1]} features")
    print(f"Reduced: {df_reduced.shape[1]} features")
    
    # Show correlation graph info
    if reducer.correlation_graph:
        print(f"Correlation network: {len(reducer.correlation_graph.nodes)} nodes, "
              f"{len(reducer.correlation_graph.edges)} edges")
    
    return df_data, df_reduced, reducer


def main():
    """Run all examples"""
    print("Biological Feature Reduction Pipeline - Examples")
    print("Using toy datasets that simulate real biological data\n")
    
    # Example 1: Pangenome analysis
    df_pan, df_pan_reduced, reducer_pan = example_pangenome_analysis()
    
    # Example 2: Microbiome analysis  
    df_micro, df_micro_reduced, reducer_micro = example_microbiome_analysis()
    
    # Example 3: Custom pipeline
    df_custom, df_custom_reduced, reducer_custom = example_custom_pipeline()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("All examples completed successfully!")
    print("The pipeline effectively reduced feature dimensionality while")
    print("preserving interpretability through feature grouping and tracking.")
    print("\nKey benefits demonstrated:")
    print("- Removes uninformative features (zero/low variance)")
    print("- Handles abundance-based filtering (core/cloud genes)")
    print("- Groups correlated features while maintaining traceability")
    print("- Provides detailed reports of all reduction steps")
    print("- Offers flexible pipeline configuration")


if __name__ == "__main__":
    main()
