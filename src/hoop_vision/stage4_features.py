"""Stage 4: Feature Extraction - Extract engineered features from graphs."""

import json
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import networkx as nx

from .config import Config


class FeatureExtractor:
    """Extract engineered features from temporal event graphs."""

    def __init__(self, config: Config):
        """
        Initialize feature extractor.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.output_dir = config.features_output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_node_features(self, G: nx.DiGraph, node: str) -> Dict[str, Any]:
        """
        Extract features for a single node.

        Args:
            G: Graph
            node: Node ID

        Returns:
            Dictionary of features
        """
        node_data = G.nodes[node]

        # Basic node features
        features = {
            "frame_idx": node_data["frame_idx"],
            "timestamp": node_data["timestamp"],
            "num_detections": node_data["num_detections"]
        }

        # Count detections by class
        detection_counts = {}
        for detection in node_data["detections"]:
            class_name = detection["class_name"]
            detection_counts[class_name] = detection_counts.get(class_name, 0) + 1

        # Add detection counts as features
        for class_name, count in detection_counts.items():
            features[f"count_{class_name}"] = count

        # Graph structure features
        features["in_degree"] = G.in_degree(node)
        features["out_degree"] = G.out_degree(node)

        # Temporal features - look back at previous N frames
        lookback_detections = []
        for predecessor in G.predecessors(node):
            pred_data = G.nodes[predecessor]
            lookback_detections.append(pred_data["num_detections"])

        if lookback_detections:
            features["lookback_avg_detections"] = sum(lookback_detections) / len(lookback_detections)
            features["lookback_max_detections"] = max(lookback_detections)
        else:
            features["lookback_avg_detections"] = 0
            features["lookback_max_detections"] = 0

        return features

    def extract_graph_features(self, G: nx.DiGraph) -> pd.DataFrame:
        """
        Extract features for all nodes in graph.

        Args:
            G: Graph

        Returns:
            DataFrame with features for each node
        """
        features_list = []
        for node in G.nodes():
            node_features = self.extract_node_features(G, node)
            features_list.append(node_features)

        df = pd.DataFrame(features_list)
        # Sort by frame index
        df = df.sort_values("frame_idx").reset_index(drop=True)

        return df

    def process_clip(self, clip_id: str) -> Path:
        """
        Extract features for a single clip.

        Args:
            clip_id: Unique identifier for clip

        Returns:
            Path to directory containing features
        """
        print(f"Extracting features for clip: {clip_id}")

        # Load graph
        graphs_dir = self.config.graphs_output_dir / clip_id
        graph_path = graphs_dir / "graph.gpickle"
        if not graph_path.exists():
            raise ValueError(f"Graph file not found: {graph_path}")

        G = nx.read_gpickle(graph_path)
        print(f"  Loaded graph: {G.number_of_nodes()} nodes")

        # Extract features
        features_df = self.extract_graph_features(G)
        print(f"  Extracted {len(features_df.columns)} features for {len(features_df)} nodes")

        # Save features
        clip_output_dir = self.output_dir / clip_id
        clip_output_dir.mkdir(parents=True, exist_ok=True)

        features_path = clip_output_dir / "features.csv"
        features_df.to_csv(features_path, index=False)

        # Save feature metadata
        metadata = {
            "clip_id": clip_id,
            "num_samples": len(features_df),
            "num_features": len(features_df.columns),
            "feature_names": list(features_df.columns)
        }
        metadata_path = clip_output_dir / "features_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"  Saved features to: {clip_output_dir}")
        return clip_output_dir

    def process_all_clips(self) -> None:
        """Process all clips with graphs."""
        graphs_dir = self.config.graphs_output_dir
        if not graphs_dir.exists():
            print(f"Graphs directory not found: {graphs_dir}")
            return

        clip_dirs = [d for d in graphs_dir.iterdir() if d.is_dir()]
        print(f"Found {len(clip_dirs)} clips to process")

        for clip_dir in clip_dirs:
            clip_id = clip_dir.name
            self.process_clip(clip_id)


def main():
    """Run feature extraction stage."""
    config = Config()
    extractor = FeatureExtractor(config)
    extractor.process_all_clips()


if __name__ == "__main__":
    main()
