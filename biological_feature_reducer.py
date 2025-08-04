import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ReductionReport:
    """Container for tracking feature reduction steps and results."""
    original_shape: Tuple[int, int] = (0, 0)
    final_shape: Tuple[int, int] = (0, 0)
    steps_performed: List[str] = field(default_factory=list)
    features_removed: Dict[str, List[str]] = field(default_factory=dict)
    feature_clusters: Dict[str, List[str]] = field(default_factory=dict)
    reduction_summary: Dict[str, int] = field(default_factory=dict)


class BiologicalFeatureReducer:
    """
    A feature reduction pipeline for biological datasets that maintains interpretability
    through intelligent feature grouping and abundance-based filtering.
    
    Parameters
    ----------
    variance_threshold : float, default=0.01
        Threshold below which features are considered low variance
    correlation_threshold : float, default=0.8
        Threshold above which features are considered highly correlated
    core_gene_threshold : float, optional
        Threshold above which features are considered core genes (highly prevalent)
    cloud_gene_threshold : float, optional
        Threshold below which features are considered cloud genes (rare)
    """
    
    def __init__(
        self, 
        variance_threshold: float = 0.01,
        correlation_threshold: float = 0.8,
        core_gene_threshold: Optional[float] = None,
        cloud_gene_threshold: Optional[float] = None
    ):
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.core_gene_threshold = core_gene_threshold
        self.cloud_gene_threshold = cloud_gene_threshold
        
        # Initialize tracking attributes
        self.report = ReductionReport()
        self.correlation_graph: Optional[nx.Graph] = None
        
    def _validate_input(self, df_data: pd.DataFrame) -> None:
        """Validate input DataFrame."""
        if not isinstance(df_data, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        if df_data.empty:
            raise ValueError("Input DataFrame is empty")
        if df_data.shape[1] == 0:
            raise ValueError("DataFrame has no columns")
        
    def _log_step(self, step_name: str, features_removed: List[str], current_shape: Tuple[int, int]) -> None:
        """Log a reduction step."""
        self.report.steps_performed.append(step_name)
        self.report.features_removed[step_name] = features_removed
        self.report.reduction_summary[step_name] = len(features_removed)
        logger.info(f"{step_name}: Removed {len(features_removed)} features, shape: {current_shape}")
        
    def remove_zero_variance_features(self, df_data: pd.DataFrame) -> pd.DataFrame:
        """Remove features that have the same value across all samples."""
        to_drop = list(df_data.columns[df_data.nunique() == 1])
        self._log_step("zero_variance_removal", to_drop, df_data.shape)
        
        if to_drop:
            logger.debug(f"Zero variance features: {to_drop}")
            
        return df_data.drop(columns=to_drop)

    def remove_low_variance_features(self, df_data: pd.DataFrame) -> pd.DataFrame:
        """Remove features with variance below threshold."""
        to_drop = list(df_data.columns[df_data.std() < self.variance_threshold])
        self._log_step("low_variance_removal", to_drop, df_data.shape)
        
        if to_drop:
            logger.debug(f"Low variance features (< {self.variance_threshold}): {to_drop}")
            
        return df_data.drop(columns=to_drop)

    def remove_abundance_features(
        self, 
        df_data: pd.DataFrame, 
        min_threshold: Optional[float] = None, 
        max_threshold: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Remove features based on abundance thresholds.
        
        Parameters
        ----------
        df_data : pd.DataFrame
            Input data
        min_threshold : float, optional
            Remove features with presence below this threshold (cloud genes)
        max_threshold : float, optional  
            Remove features with presence above this threshold (core genes)
        """
        to_drop = []
        
        if max_threshold is not None:
            core_genes = list(df_data.columns[df_data.sum() / df_data.shape[0] > max_threshold])
            to_drop.extend(core_genes)
            logger.info(f"Found {len(core_genes)} core genes (> {max_threshold} presence)")
            
        if min_threshold is not None:
            cloud_genes = list(df_data.columns[df_data.sum() / df_data.shape[0] < min_threshold])
            to_drop.extend(cloud_genes)
            logger.info(f"Found {len(cloud_genes)} cloud genes (< {min_threshold} presence)")
            
        # Remove duplicates while preserving order
        to_drop = list(dict.fromkeys(to_drop))
        
        step_name = "abundance_filtering"
        self._log_step(step_name, to_drop, df_data.shape)
        
        return df_data.drop(columns=to_drop)

    def combine_identical_features(self, df_data: pd.DataFrame) -> pd.DataFrame:
        """Group features that have identical values across all samples."""
        features = list(df_data.columns)
        clusters = []
        
        while features:
            current_feature = features[0]
            # Find all features identical to current feature
            cluster = [col for col in features if df_data[col].equals(df_data[current_feature])]
            clusters.append(cluster)
            
            # Remove clustered features from remaining features
            features = [f for f in features if f not in cluster]
        
        # Track clusters with more than one feature
        multi_feature_clusters = [cluster for cluster in clusters if len(cluster) > 1]
        
        logger.info(f"Found {len(multi_feature_clusters)} clusters of identical features")
        for i, cluster in enumerate(multi_feature_clusters):
            cluster_name = f"identical_cluster_{i+1}"
            self.report.feature_clusters[cluster_name] = cluster
            logger.debug(f"Identical cluster: {cluster}")
        
        # Create new dataframe with one representative per cluster
        df_reduced = df_data[[cluster[0] for cluster in clusters]].copy()
        
        # Rename columns to show cluster membership
        new_columns = []
        for cluster in clusters:
            if len(cluster) > 1:
                new_columns.append('~'.join(cluster))
            else:
                new_columns.append(cluster[0])
        
        df_reduced.columns = new_columns
        
        # Log the step
        features_combined = [f for cluster in multi_feature_clusters for f in cluster[1:]]
        self._log_step("identical_feature_combination", features_combined, df_reduced.shape)
        
        return df_reduced

    def combine_correlated_features(
        self, 
        df_data: pd.DataFrame, 
        method: str = 'mean'
    ) -> Tuple[pd.DataFrame, nx.Graph]:
        """
        Combine highly correlated features using network analysis.
        
        Parameters
        ----------
        df_data : pd.DataFrame
            Input data
        method : str, default='mean'
            Method to combine correlated features ('mean', 'first', 'median')
        """
        if df_data.shape[1] < 2:
            logger.warning("Less than 2 features remaining, skipping correlation analysis")
            return df_data, nx.Graph()
            
        # Calculate correlation matrix
        corr_matrix = df_data.corr()
        
        # Create graph for correlation relationships
        G = nx.Graph()
        G.add_nodes_from(corr_matrix.index)
        
        # Add edges for correlations above threshold
        rows, cols = np.where(
            (np.abs(corr_matrix.values) >= self.correlation_threshold) & 
            (np.triu(np.ones_like(corr_matrix.values, dtype=bool), k=1))
        )
        
        for i, j in zip(rows, cols):
            correlation_value = corr_matrix.values[i, j]
            G.add_edge(
                corr_matrix.index[i], 
                corr_matrix.index[j], 
                weight=round(correlation_value, 3)
            )
        
        # Find connected components (correlation clusters)
        clusters = [list(component) for component in nx.connected_components(G)]
        
        # Log cluster information
        multi_feature_clusters = [cluster for cluster in clusters if len(cluster) > 1]
        logger.info(f"Found {len(multi_feature_clusters)} correlation clusters (>= {self.correlation_threshold})")
        logger.info(f"Found {len([c for c in clusters if len(c) == 1])} unclustered features")
        
        for i, cluster in enumerate(multi_feature_clusters):
            cluster_name = f"correlation_cluster_{i+1}"
            self.report.feature_clusters[cluster_name] = cluster
            logger.debug(f"Correlation cluster: {cluster}")
        
        # Combine features within each cluster
        combined_features = []
        new_column_names = []
        
        for cluster in clusters:
            if len(cluster) == 1:
                # Single feature, keep as is
                combined_features.append(df_data[cluster[0]])
                new_column_names.append(cluster[0])
            else:
                # Multiple features, combine them
                cluster_data = df_data[cluster]
                if method == 'mean':
                    combined_feature = cluster_data.mean(axis=1)
                elif method == 'median':
                    combined_feature = cluster_data.median(axis=1)
                elif method == 'first':
                    combined_feature = cluster_data.iloc[:, 0]
                else:
                    raise ValueError(f"Unknown combination method: {method}")
                
                combined_features.append(combined_feature)
                new_column_names.append('~'.join(sorted(cluster)))
        
        # Create new dataframe
        df_reduced = pd.concat(combined_features, axis=1)
        df_reduced.columns = new_column_names
        
        # Clean up graph (remove isolated nodes)
        G.remove_nodes_from([node for node in G.nodes if G.degree(node) == 0])
        self.correlation_graph = G
        
        # Log the step
        features_combined = [f for cluster in multi_feature_clusters for f in cluster[1:]]
        self._log_step("correlation_based_combination", features_combined, df_reduced.shape)
        
        return df_reduced, G

    def visualize_correlation_clusters(self, figsize: Tuple[int, int] = (12, 8)) -> None:
        """Visualize the correlation network graph."""
        if self.correlation_graph is None or len(self.correlation_graph.nodes) == 0:
            logger.warning("No correlation graph available for visualization")
            return
            
        plt.figure(figsize=figsize)
        pos = nx.spring_layout(self.correlation_graph, k=1, iterations=50)
        
        # Draw nodes and edges
        nx.draw_networkx_nodes(
            self.correlation_graph, pos, 
            node_size=300, node_color='lightblue', alpha=0.7
        )
        nx.draw_networkx_edges(
            self.correlation_graph, pos, 
            edge_color='gray', alpha=0.6, width=1.5
        )
        nx.draw_networkx_labels(
            self.correlation_graph, pos, 
            font_size=8, font_color='black'
        )
        
        # Add edge labels with correlation values
        edge_labels = nx.get_edge_attributes(self.correlation_graph, 'weight')
        nx.draw_networkx_edge_labels(
            self.correlation_graph, pos, edge_labels, 
            font_size=6, alpha=0.8
        )
        
        plt.title(f"Feature Correlation Network (threshold >= {self.correlation_threshold})")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def fit_transform(
        self, 
        df_data: pd.DataFrame,
        steps_to_perform: Optional[List[str]] = None,
        combination_method: str = 'mean',
        visualize_clusters: bool = False
    ) -> pd.DataFrame:
        """
        Main pipeline method to perform feature reduction.
        
        Parameters
        ----------
        df_data : pd.DataFrame
            Input biological data
        steps_to_perform : list of str, optional
            Specific steps to perform. If None, performs all applicable steps.
            Options: ['abundance_filter', 'zero_variance', 'low_variance', 
                     'identical', 'correlated']
        combination_method : str, default='mean'
            Method to combine correlated features
        visualize_clusters : bool, default=False
            Whether to show correlation cluster visualization
            
        Returns
        -------
        pd.DataFrame
            Reduced feature dataset
        """
        # Validate input
        self._validate_input(df_data)
        
        # Initialize report
        self.report = ReductionReport()
        self.report.original_shape = df_data.shape
        
        logger.info(f"Starting feature reduction pipeline. Original shape: {df_data.shape}")
        
        # Create a copy to avoid modifying original data
        df_processed = df_data.copy()
        
        # Define default steps
        if steps_to_perform is None:
            steps_to_perform = ['abundance_filter', 'zero_variance', 'low_variance', 'identical', 'correlated']
        
        # Step 1: Abundance filtering (core/cloud genes)
        if 'abundance_filter' in steps_to_perform:
            if self.core_gene_threshold is not None or self.cloud_gene_threshold is not None:
                df_processed = self.remove_abundance_features(
                    df_processed, 
                    min_threshold=self.cloud_gene_threshold,
                    max_threshold=self.core_gene_threshold
                )
        
        # Step 2: Zero variance features
        if 'zero_variance' in steps_to_perform:
            df_processed = self.remove_zero_variance_features(df_processed)
        
        # Step 3: Low variance features  
        if 'low_variance' in steps_to_perform:
            df_processed = self.remove_low_variance_features(df_processed)
        
        # Step 4: Identical features
        if 'identical' in steps_to_perform:
            df_processed = self.combine_identical_features(df_processed)
        
        # Step 5: Correlated features
        if 'correlated' in steps_to_perform:
            df_processed, _ = self.combine_correlated_features(df_processed, method=combination_method)
        
        # Finalize report
        self.report.final_shape = df_processed.shape
        
        # Show visualization if requested
        if visualize_clusters:
            self.visualize_correlation_clusters()
        
        # Log final summary
        logger.info(f"Feature reduction complete. Final shape: {df_processed.shape}")
        logger.info(f"Reduction ratio: {df_processed.shape[1]/df_data.shape[1]:.2%}")
        
        return df_processed
    
    def get_reduction_report(self) -> ReductionReport:
        """Get detailed report of the reduction process."""
        return self.report
    
    def save_results(self, df_data: pd.DataFrame, filepath: str, sep: str = '\t') -> None:
        """Save reduced data to file."""
        df_data.to_csv(filepath, sep=sep, header=True, index=True)
        logger.info(f"Results saved to: {filepath}")


# Convenience function for backward compatibility
def feature_reduction_pipeline(
    df_data: pd.DataFrame,
    variance_threshold: float = 0.01,
    correlation_threshold: float = 0.8,
    filter_core_genes: Optional[float] = None,
    filter_cloud_genes: Optional[float] = None,
    viz_corr_clusters: bool = False,
    fname: Optional[str] = None,
    return_graph: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, nx.Graph]]:
    """
    Legacy function interface for backward compatibility.
    
    For new code, prefer using BiologicalFeatureReducer class directly.
    """
    warnings.warn(
        "This function interface is deprecated. Use BiologicalFeatureReducer class instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    reducer = BiologicalFeatureReducer(
        variance_threshold=variance_threshold,
        correlation_threshold=correlation_threshold,
        core_gene_threshold=filter_core_genes,
        cloud_gene_threshold=filter_cloud_genes
    )
    
    df_reduced = reducer.fit_transform(
        df_data, 
        visualize_clusters=viz_corr_clusters
    )
    
    if fname is not None:
        reducer.save_results(df_reduced, f'../data/{fname}.csv')
    
    if return_graph:
        return df_reduced, reducer.correlation_graph
    else:
        return df_reduced
