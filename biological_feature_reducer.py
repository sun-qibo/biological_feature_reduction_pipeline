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
        self.report = ReductionReport()
        self.correlation_graph: Optional[nx.Graph] = None
        # For fit/transform API
        self.fitted_ = False
        self._features_to_drop = {}
        self._identical_clusters = []
        self._correlation_clusters = []
        self._final_columns = None
        self._combination_method = None

    def fit(
        self,
        df_data: pd.DataFrame,
        steps_to_perform: Optional[List[str]] = None,
        combination_method: str = 'mean',
    ) -> 'BiologicalFeatureReducer':
        """
        Fit the reducer on a dataset, learning which features to drop/cluster.
        """
        self._validate_input(df_data)
        self.report = ReductionReport()
        self.report.original_shape = df_data.shape
        self._features_to_drop = {}
        self._identical_clusters = []
        self._correlation_clusters = []
        self._final_columns = None
        self._combination_method = combination_method
        logger.info(f"Fitting feature reduction pipeline. Original shape: {df_data.shape}")
        df_processed = df_data.copy()
        if steps_to_perform is None:
            steps_to_perform = ['abundance_filter', 'zero_variance', 'low_variance', 'identical']
        if self.correlation_threshold is not None:
            steps_to_perform.append('correlated')
        # Step 1: Abundance filtering
        if 'abundance_filter' in steps_to_perform:
            to_drop = []
            if self.presence_high_threshold is not None:
                core_genes = list(df_processed.columns[df_processed.sum() / df_processed.shape[0] > self.presence_high_threshold])
                to_drop.extend(core_genes)
            if self.presence_low_threshold is not None:
                cloud_genes = list(df_processed.columns[df_processed.sum() / df_processed.shape[0] < self.presence_low_threshold])
                to_drop.extend(cloud_genes)
            to_drop = list(dict.fromkeys(to_drop))
            self._features_to_drop['abundance_filtering'] = to_drop
            df_processed = df_processed.drop(columns=to_drop)
        # Step 2: Zero variance
        if 'zero_variance' in steps_to_perform:
            to_drop = list(df_processed.columns[df_processed.nunique() == 1])
            self._features_to_drop['zero_variance_removal'] = to_drop
            df_processed = df_processed.drop(columns=to_drop)
        # Step 3: Low variance
        if 'low_variance' in steps_to_perform:
            to_drop = list(df_processed.columns[df_processed.std() < self.variance_threshold])
            self._features_to_drop['low_variance_removal'] = to_drop
            df_processed = df_processed.drop(columns=to_drop)
        # Step 4: Identical features
        if 'identical' in steps_to_perform:
            features = list(df_processed.columns)
            clusters = []
            while features:
                current_feature = features[0]
                cluster = [col for col in features if df_processed[col].equals(df_processed[current_feature])]
                clusters.append(cluster)
                features = [f for f in features if f not in cluster]
            self._identical_clusters = clusters
            df_processed = df_processed[[cluster[0] for cluster in clusters]].copy()
            new_columns = []
            for cluster in clusters:
                if len(cluster) > 1:
                    new_columns.append('~'.join(cluster))
                else:
                    new_columns.append(cluster[0])
            df_processed.columns = new_columns
        # Step 5: Correlated features
        if 'correlated' in steps_to_perform:
            if df_processed.shape[1] < 2:
                self._correlation_clusters = [[col] for col in df_processed.columns]
            else:
                corr_matrix = df_processed.corr()
                G = nx.Graph()
                G.add_nodes_from(corr_matrix.index)
                rows, cols = np.where(
                    (corr_matrix.values >= self.correlation_threshold) & 
                    (np.triu(np.ones_like(corr_matrix.values, dtype=bool), k=1))
                )
                for i, j in zip(rows, cols):
                    G.add_edge(corr_matrix.index[i], corr_matrix.index[j])
                clusters = [list(component) for component in nx.connected_components(G)]
                self._correlation_clusters = clusters
                # Save the graph for visualization
                G.remove_nodes_from([node for node in G.nodes if G.degree(node) == 0])
                self.correlation_graph = G
            # Save the final columns after correlation combination
            final_columns = []
            for cluster in self._correlation_clusters:
                if len(cluster) > 1:
                    final_columns.append('~'.join(sorted(cluster)))
                else:
                    final_columns.append(cluster[0])
            self._final_columns = final_columns
        else:
            # If no correlation step, final columns are whatever is left
            self._final_columns = list(df_processed.columns)
        self.fitted_ = True
        return self

    def transform(self, df_data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform a new dataset using the reduction plan learned in fit.
        """
        if not self.fitted_:
            raise RuntimeError("You must call fit before transform.")
        self._validate_input(df_data)
        df_processed = df_data.copy()
        # Apply abundance, zero variance, low variance drops
        for step in ['abundance_filtering', 'zero_variance_removal', 'low_variance_removal']:
            to_drop = self._features_to_drop.get(step, [])
            df_processed = df_processed.drop(columns=[col for col in to_drop if col in df_processed.columns], errors='ignore')
        # Combine identical features
        if self._identical_clusters:
            new_df = pd.DataFrame(index=df_processed.index)
            for cluster in self._identical_clusters:
                cols_in_df = [col for col in cluster if col in df_processed.columns]
                if cols_in_df:
                    new_col = df_processed[cols_in_df[0]]
                    col_name = '~'.join(cluster) if len(cluster) > 1 else cluster[0]
                    new_df[col_name] = new_col
            df_processed = new_df
        # Combine correlated features
        if self._correlation_clusters:
            combined_features = []
            new_column_names = []
            for cluster in self._correlation_clusters:
                cols_in_df = [col for col in cluster if col in df_processed.columns]
                if not cols_in_df:
                    continue
                if len(cols_in_df) == 1:
                    combined_features.append(df_processed[cols_in_df[0]])
                    new_column_names.append(cols_in_df[0])
                else:
                    cluster_data = df_processed[cols_in_df]
                    method = self._combination_method or 'mean'
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
            if combined_features:
                df_processed = pd.concat(combined_features, axis=1)
                df_processed.columns = new_column_names
        # Reorder columns to match fit
        if self._final_columns is not None:
            cols_in_df = [col for col in self._final_columns if col in df_processed.columns]
            df_processed = df_processed[cols_in_df]
        return df_processed

    def fit_transform(
        self, 
        df_data: pd.DataFrame,
        steps_to_perform: Optional[List[str]] = None,
        combination_method: str = 'mean',
        visualize_clusters: bool = False
    ) -> pd.DataFrame:
        """
        Fit to data, then transform it.
        """
        self.fit(df_data, steps_to_perform=steps_to_perform, combination_method=combination_method)
        df_processed = self.transform(df_data)
        self.report.final_shape = df_processed.shape
        if visualize_clusters:
            self.visualize_correlation_clusters()
        logger.info(f"Feature reduction complete. Final shape: {df_processed.shape}")
        logger.info(f"Reduction ratio: {df_processed.shape[1]/df_data.shape[1]:.2%}")
        return df_processed
    