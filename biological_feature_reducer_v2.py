import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import logging
import pickle
from typing import Dict, List, Optional, Tuple, Union, Set
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

@dataclass
class FittedTransformation:
    """Container for storing fitted transformation parameters."""
    # Features to remove
    zero_variance_features: Set[str] = field(default_factory=set)
    low_variance_features: Set[str] = field(default_factory=set)
    abundance_filtered_features: Set[str] = field(default_factory=set)
    
    # Feature combinations/mappings
    identical_feature_clusters: Dict[str, List[str]] = field(default_factory=dict)
    correlation_feature_clusters: Dict[str, List[str]] = field(default_factory=dict)
    
    # Thresholds used during fitting
    variance_threshold: float = 0.01
    correlation_threshold: Optional[float] = None
    presence_high_threshold: Optional[float] = None
    presence_low_threshold: Optional[float] = None
    
    # Processing parameters
    combination_method: str = 'mean'
    steps_performed: List[str] = field(default_factory=list)
    
    # Final feature names after all transformations
    final_features: List[str] = field(default_factory=list)
    original_features: List[str] = field(default_factory=list)


class BiologicalFeatureReducer:
    """
    A feature reduction pipeline for biological datasets that maintains interpretability
    through intelligent feature grouping and abundance-based filtering.
    
    This class supports scikit-learn style fit/transform pattern:
    - fit(): Learn reduction parameters from training data
    - transform(): Apply learned transformations to new data
    - fit_transform(): Fit and transform in one step
    
    Parameters
    ----------
    variance_threshold : float, default=0.01
        Threshold below which features are considered low variance
    correlation_threshold : float, default=None
        Threshold above which features are considered highly correlated
    presence_high_threshold : float, optional
        Threshold above which features are considered omnipresent (core genes)
    presence_low_threshold : float, optional
        Threshold below which features are considered too rare (cloud genes, rare taxa)
    """
    
    def __init__(
        self, 
        variance_threshold: float = 0.01,
        correlation_threshold: Optional[float] = None,
        presence_high_threshold: Optional[float] = None,
        presence_low_threshold: Optional[float] = None
    ):
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.presence_high_threshold = presence_high_threshold
        self.presence_low_threshold = presence_low_threshold
        
        # Initialize tracking attributes
        self.report = ReductionReport()
        self.correlation_graph: Optional[nx.Graph] = None
        
        # Fitted transformation parameters
        self.fitted_transformation_: Optional[FittedTransformation] = None
        self.is_fitted_: bool = False
        
    def _validate_input(self, df_data: pd.DataFrame) -> None:
        """Validate input DataFrame."""
        if not isinstance(df_data, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        if df_data.empty:
            raise ValueError("Input DataFrame is empty")
        if df_data.shape[1] == 0:
            raise ValueError("DataFrame has no columns")
            
    def _check_is_fitted(self) -> None:
        """Check if the reducer has been fitted."""
        if not self.is_fitted_:
            raise ValueError("This BiologicalFeatureReducer instance is not fitted yet. "
                           "Call 'fit' with appropriate arguments before using this estimator.")
        
    def _log_step(self, step_name: str, features_removed: List[str], current_shape: Tuple[int, int]) -> None:
        """Log a reduction step."""
        self.report.steps_performed.append(step_name)
        self.report.features_removed[step_name] = features_removed
        self.report.reduction_summary[step_name] = len(features_removed)
        logger.info(f"{step_name}: Removed {len(features_removed)} features, shape: {current_shape}")

    def _find_zero_variance_features(self, df_data: pd.DataFrame) -> Set[str]:
        """Identify features that have the same value across all samples."""
        return set(df_data.columns[df_data.nunique() == 1])

    def _find_low_variance_features(self, df_data: pd.DataFrame) -> Set[str]:
        """Identify features with variance below threshold."""
        return set(df_data.columns[df_data.std() < self.variance_threshold])

    def _find_abundance_features(self, df_data: pd.DataFrame) -> Set[str]:
        """Identify features to remove based on abundance thresholds."""
        to_remove = set()
        
        if self.presence_high_threshold is not None:
            core_genes = set(df_data.columns[df_data.sum() / df_data.shape[0] > self.presence_high_threshold])
            to_remove.update(core_genes)
            logger.info(f"Found {len(core_genes)} core genes (> {self.presence_high_threshold} presence)")
            
        if self.presence_low_threshold is not None:
            cloud_genes = set(df_data.columns[df_data.sum() / df_data.shape[0] < self.presence_low_threshold])
            to_remove.update(cloud_genes)
            logger.info(f"Found {len(cloud_genes)} cloud genes (< {self.presence_low_threshold} presence)")
            
        return to_remove

    def _find_identical_features(self, df_data: pd.DataFrame) -> Dict[str, List[str]]:
        """Identify clusters of identical features."""
        features = list(df_data.columns)
        clusters = []
        
        while features:
            current_feature = features[0]
            # Find all features identical to current feature
            cluster = [col for col in features if df_data[col].equals(df_data[current_feature])]
            clusters.append(cluster)
            
            # Remove clustered features from remaining features
            features = [f for f in features if f not in cluster]
        
        # Return only clusters with more than one feature
        identical_clusters = {}
        cluster_idx = 1
        for cluster in clusters:
            if len(cluster) > 1:
                cluster_name = f"identical_cluster_{cluster_idx}"
                identical_clusters[cluster_name] = cluster
                cluster_idx += 1
                
        return identical_clusters

    def _find_correlated_features(self, df_data: pd.DataFrame) -> Dict[str, List[str]]:
        """Identify clusters of correlated features."""
        if df_data.shape[1] < 2 or self.correlation_threshold is None:
            return {}
            
        # Calculate correlation matrix
        corr_matrix = df_data.corr()
        
        # Create graph for correlation relationships
        G = nx.Graph()
        G.add_nodes_from(corr_matrix.index)
        
        # Add edges for correlations above threshold
        rows, cols = np.where(
            (corr_matrix.values >= self.correlation_threshold) & 
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
        
        # Store graph for visualization
        G.remove_nodes_from([node for node in G.nodes if G.degree(node) == 0])
        self.correlation_graph = G
        
        # Return only clusters with more than one feature
        correlation_clusters = {}
        cluster_idx = 1
        for cluster in clusters:
            if len(cluster) > 1:
                cluster_name = f"correlation_cluster_{cluster_idx}"
                correlation_clusters[cluster_name] = cluster
                cluster_idx += 1
                
        return correlation_clusters

    def _apply_feature_removal(self, df_data: pd.DataFrame, features_to_remove: Set[str], step_name: str) -> pd.DataFrame:
        """Apply feature removal and log the step."""
        available_features = set(df_data.columns)
        features_to_remove = features_to_remove.intersection(available_features)
        
        if features_to_remove:
            df_reduced = df_data.drop(columns=list(features_to_remove))
            self._log_step(step_name, list(features_to_remove), df_reduced.shape)
            return df_reduced
        else:
            self._log_step(step_name, [], df_data.shape)
            return df_data

    def _apply_identical_clustering(self, df_data: pd.DataFrame, clusters: Dict[str, List[str]]) -> pd.DataFrame:
        """Apply identical feature clustering."""
        if not clusters:
            self._log_step("identical_feature_combination", [], df_data.shape)
            return df_data
        
        # Get all features that will be combined (excluding representatives)
        features_combined = []
        for cluster_features in clusters.values():
            features_combined.extend(cluster_features[1:])  # Skip representative (first feature)
        
        # Create mapping for renaming
        rename_mapping = {}
        available_features = set(df_data.columns)
        
        for cluster_name, cluster_features in clusters.items():
            # Filter to only include features present in current data
            present_features = [f for f in cluster_features if f in available_features]
            if len(present_features) > 1:
                representative = present_features[0]
                new_name = '~'.join(present_features)
                rename_mapping[representative] = new_name
        
        # Apply transformations
        df_reduced = df_data.drop(columns=[f for f in features_combined if f in df_data.columns])
        df_reduced = df_reduced.rename(columns=rename_mapping)
        
        self._log_step("identical_feature_combination", 
                      [f for f in features_combined if f in available_features], 
                      df_reduced.shape)
        return df_reduced

    def _apply_correlation_clustering(self, df_data: pd.DataFrame, clusters: Dict[str, List[str]], method: str = 'mean') -> pd.DataFrame:
        """Apply correlation-based feature clustering."""
        if not clusters:
            self._log_step("correlation_based_combination", [], df_data.shape)
            return df_data
            
        available_features = set(df_data.columns)
        combined_features = []
        new_column_names = []
        features_to_combine = set()
        
        # Track which features are part of clusters
        clustered_features = set()
        for cluster_features in clusters.values():
            clustered_features.update(cluster_features)
        
        # Process clustered features
        for cluster_name, cluster_features in clusters.items():
            present_features = [f for f in cluster_features if f in available_features]
            if len(present_features) > 1:
                cluster_data = df_data[present_features]
                if method == 'mean':
                    combined_feature = cluster_data.mean(axis=1)
                elif method == 'median':
                    combined_feature = cluster_data.median(axis=1)
                elif method == 'first':
                    combined_feature = cluster_data.iloc[:, 0]
                else:
                    raise ValueError(f"Unknown combination method: {method}")
                
                combined_features.append(combined_feature)
                new_column_names.append('~'.join(sorted(present_features)))
                features_to_combine.update(present_features)
        
        # Add unclustered features
        unclustered_features = [f for f in df_data.columns if f not in features_to_combine]
        if unclustered_features:
            unclustered_data = df_data[unclustered_features]
            for col in unclustered_data.columns:
                combined_features.append(unclustered_data[col])
                new_column_names.append(col)
        
        # Create new dataframe
        if combined_features:
            df_reduced = pd.concat(combined_features, axis=1)
            df_reduced.columns = new_column_names
        else:
            df_reduced = pd.DataFrame(index=df_data.index)
        
        features_combined = [f for cluster_features in clusters.values() 
                           for f in cluster_features[1:] if f in available_features]
        self._log_step("correlation_based_combination", features_combined, df_reduced.shape)
        
        return df_reduced

    def fit(
        self, 
        df_data: pd.DataFrame,
        steps_to_perform: Optional[List[str]] = None,
        combination_method: str = 'mean'
    ) -> 'BiologicalFeatureReducer':
        """
        Fit the feature reducer on training data to learn transformation parameters.
        
        Parameters
        ----------
        df_data : pd.DataFrame
            Training biological data to learn transformations from
        steps_to_perform : list of str, optional
            Specific steps to perform. If None, performs all applicable steps.
            Options: ['abundance_filter', 'zero_variance', 'low_variance', 
                     'identical', 'correlated']
        combination_method : str, default='mean'
            Method to combine correlated features
            
        Returns
        -------
        self : BiologicalFeatureReducer
            Returns self for method chaining
        """
        # Validate input
        self._validate_input(df_data)
        
        logger.info(f"Fitting feature reducer on data with shape: {df_data.shape}")
        
        # Initialize fitted transformation object
        self.fitted_transformation_ = FittedTransformation(
            variance_threshold=self.variance_threshold,
            correlation_threshold=self.correlation_threshold,
            presence_high_threshold=self.presence_high_threshold,
            presence_low_threshold=self.presence_low_threshold,
            combination_method=combination_method,
            original_features=list(df_data.columns)
        )
        
        # Define default steps
        if steps_to_perform is None:
            steps_to_perform = ['abundance_filter', 'zero_variance', 'low_variance', 'identical']
            if self.correlation_threshold is not None:
                steps_to_perform.append('correlated')
        
        self.fitted_transformation_.steps_performed = steps_to_perform.copy()
        
        # Create a working copy for analysis
        df_working = df_data.copy()
        
        # Learn transformation parameters for each step
        if 'abundance_filter' in steps_to_perform:
            if self.presence_high_threshold is not None or self.presence_low_threshold is not None:
                abundance_features = self._find_abundance_features(df_working)
                self.fitted_transformation_.abundance_filtered_features = abundance_features
                df_working = df_working.drop(columns=list(abundance_features.intersection(df_working.columns)))
        
        if 'zero_variance' in steps_to_perform:
            zero_var_features = self._find_zero_variance_features(df_working)
            self.fitted_transformation_.zero_variance_features = zero_var_features
            df_working = df_working.drop(columns=list(zero_var_features.intersection(df_working.columns)))
        
        if 'low_variance' in steps_to_perform:
            low_var_features = self._find_low_variance_features(df_working)
            self.fitted_transformation_.low_variance_features = low_var_features
            df_working = df_working.drop(columns=list(low_var_features.intersection(df_working.columns)))
        
        if 'identical' in steps_to_perform:
            identical_clusters = self._find_identical_features(df_working)
            self.fitted_transformation_.identical_feature_clusters = identical_clusters
            # Apply identical clustering to working data for next steps
            df_working = self._apply_identical_clustering(df_working, identical_clusters)
        
        if 'correlated' in steps_to_perform:
            correlation_clusters = self._find_correlated_features(df_working)
            self.fitted_transformation_.correlation_feature_clusters = correlation_clusters
            # Apply correlation clustering to get final feature names
            df_working = self._apply_correlation_clustering(df_working, correlation_clusters, combination_method)
        
        # Store final feature names
        self.fitted_transformation_.final_features = list(df_working.columns)
        
        # Mark as fitted
        self.is_fitted_ = True
        
        logger.info(f"Feature reducer fitted. Final feature count: {len(self.fitted_transformation_.final_features)}")
        logger.info(f"Reduction ratio: {len(self.fitted_transformation_.final_features)/len(df_data.columns):.2%}")
        
        return self

    def transform(self, df_data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the fitted transformations to new data.
        
        Parameters
        ----------
        df_data : pd.DataFrame
            New biological data to transform using fitted parameters
            
        Returns
        -------
        pd.DataFrame
            Transformed dataset with reduced features
        """
        # Check if fitted
        self._check_is_fitted()
        
        # Validate input
        self._validate_input(df_data)
        
        # Initialize report for this transformation
        self.report = ReductionReport()
        self.report.original_shape = df_data.shape
        
        logger.info(f"Transforming data with shape: {df_data.shape}")
        
        # Create working copy
        df_processed = df_data.copy()
        
        # Apply transformations in the same order as fitting
        for step in self.fitted_transformation_.steps_performed:
            if step == 'abundance_filter':
                df_processed = self._apply_feature_removal(
                    df_processed, 
                    self.fitted_transformation_.abundance_filtered_features,
                    "abundance_filtering"
                )
            elif step == 'zero_variance':
                df_processed = self._apply_feature_removal(
                    df_processed,
                    self.fitted_transformation_.zero_variance_features,
                    "zero_variance_removal"
                )
            elif step == 'low_variance':
                df_processed = self._apply_feature_removal(
                    df_processed,
                    self.fitted_transformation_.low_variance_features,
                    "low_variance_removal"
                )
            elif step == 'identical':
                df_processed = self._apply_identical_clustering(
                    df_processed,
                    self.fitted_transformation_.identical_feature_clusters
                )
            elif step == 'correlated':
                df_processed = self._apply_correlation_clustering(
                    df_processed,
                    self.fitted_transformation_.correlation_feature_clusters,
                    self.fitted_transformation_.combination_method
                )
        
        # Finalize report
        self.report.final_shape = df_processed.shape
        
        logger.info(f"Transformation complete. Final shape: {df_processed.shape}")
        logger.info(f"Reduction ratio: {df_processed.shape[1]/df_data.shape[1]:.2%}")
        
        return df_processed

    def fit_transform(
        self, 
        df_data: pd.DataFrame,
        steps_to_perform: Optional[List[str]] = None,
        combination_method: str = 'mean',
        visualize_clusters: bool = False
    ) -> pd.DataFrame:
        """
        Fit the reducer and transform the data in one step.
        
        Parameters
        ----------
        df_data : pd.DataFrame
            Input biological data
        steps_to_perform : list of str, optional
            Specific steps to perform. If None, performs all applicable steps.
        combination_method : str, default='mean'
            Method to combine correlated features
        visualize_clusters : bool, default=False
            Whether to show correlation cluster visualization
            
        Returns
        -------
        pd.DataFrame
            Reduced feature dataset
        """
        return self.fit(df_data, steps_to_perform, combination_method).transform(df_data)

    def save_model(self, filepath: str) -> None:
        """
        Save the fitted reducer to disk using pickle.
        
        Parameters
        ----------
        filepath : str
            Path where to save the fitted model
        """
        self._check_is_fitted()
        
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Fitted reducer saved to: {filepath}")

    @classmethod
    def load_model(cls, filepath: str) -> 'BiologicalFeatureReducer':
        """
        Load a fitted reducer from disk.
        
        Parameters
        ----------
        filepath : str
            Path to the saved model file
            
        Returns
        -------
        BiologicalFeatureReducer
            Loaded fitted reducer
        """
        with open(filepath, 'rb') as f:
            reducer = pickle.load(f)
        
        if not isinstance(reducer, cls):
            raise TypeError(f"Loaded object is not a {cls.__name__}")
        
        logger.info(f"Fitted reducer loaded from: {filepath}")
        return reducer

    def get_feature_mapping(self) -> Dict[str, str]:
        """
        Get mapping from original features to final combined features.
        
        Returns
        -------
        dict
            Mapping from original feature names to final feature names
        """
        self._check_is_fitted()
        
        mapping = {}
        
        # Start with all original features mapping to themselves
        for feature in self.fitted_transformation_.original_features:
            mapping[feature] = feature
        
        # Apply identical clustering mappings
        for cluster_name, cluster_features in self.fitted_transformation_.identical_feature_clusters.items():
            combined_name = '~'.join(cluster_features)
            for feature in cluster_features:
                mapping[feature] = combined_name
        
        # Apply correlation clustering mappings
        for cluster_name, cluster_features in self.fitted_transformation_.correlation_feature_clusters.items():
            combined_name = '~'.join(sorted(cluster_features))
            for feature in cluster_features:
                if feature in mapping:  # Feature might have been removed in previous steps
                    current_name = mapping[feature]
                    mapping[feature] = combined_name
        
        # Remove features that were filtered out
        removed_features = (
            self.fitted_transformation_.abundance_filtered_features |
            self.fitted_transformation_.zero_variance_features |
            self.fitted_transformation_.low_variance_features
        )
        
        for feature in removed_features:
            if feature in mapping:
                del mapping[feature]
        
        return mapping

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
        
        threshold = self.correlation_threshold or "N/A"
        plt.title(f"Feature Correlation Network (threshold >= {threshold})")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def get_reduction_report(self) -> ReductionReport:
        """Get detailed report of the reduction process."""
        return self.report
    
    def save_results(self, df_data: pd.DataFrame, filepath: str, sep: str = '\t') -> None:
        """Save reduced data to file."""
        df_data.to_csv(filepath, sep=sep, header=True, index=True)
        logger.info(f"Results saved to: {filepath}")

    def get_fitted_parameters(self) -> Optional[FittedTransformation]:
        """Get the fitted transformation parameters."""
        return self.fitted_transformation_


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
        presence_high_threshold=filter_core_genes,
        presence_low_threshold=filter_cloud_genes
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