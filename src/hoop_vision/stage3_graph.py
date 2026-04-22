"""Stage 3: Graph Construction - Build temporal event graph from detections."""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Any
import networkx as nx

from .config import Config


class GraphBuilder:
    """Build temporal graph from detected events."""

    def __init__(self, config: Config):
        """
        Initialize graph builder.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.output_dir = config.graphs_output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_graph(self, events: List[Dict[str, Any]]) -> nx.DiGraph:
        """
        Build temporal directed graph from events.

        Args:
            events: List of event dictionaries

        Returns:
            NetworkX directed graph
        """
        G = nx.DiGraph()

        # Add nodes for each frame event
        for event in events:
            node_id = f"frame_{event['frame_idx']}"
            G.add_node(
                node_id,
                frame_idx=event["frame_idx"],
                timestamp=event["timestamp"],
                num_detections=len(event["detections"]),
                detections=event["detections"]
            )

        # Add temporal edges (each frame -> next frame within temporal window)
        sorted_events = sorted(events, key=lambda e: e["frame_idx"])
        for i, event in enumerate(sorted_events):
            current_node = f"frame_{event['frame_idx']}"
            current_time = event["timestamp"]

            # Connect to subsequent frames within temporal window
            for j in range(i + 1, len(sorted_events)):
                next_event = sorted_events[j]
                next_node = f"frame_{next_event['frame_idx']}"
                next_time = next_event["timestamp"]

                time_diff = next_time - current_time
                if time_diff <= self.config.max_temporal_distance:
                    G.add_edge(
                        current_node,
                        next_node,
                        time_diff=time_diff,
                        edge_type="temporal"
                    )
                else:
                    break  # Beyond temporal window

        return G

    def process_clip(self, clip_id: str) -> Path:
        """
        Build graph for a single clip.

        Args:
            clip_id: Unique identifier for clip

        Returns:
            Path to directory containing graph
        """
        print(f"Building graph for clip: {clip_id}")

        # Load events
        events_dir = self.config.events_output_dir / clip_id
        events_path = events_dir / "events.json"
        if not events_path.exists():
            raise ValueError(f"Events file not found: {events_path}")

        with open(events_path) as f:
            events = json.load(f)

        print(f"  Loaded {len(events)} events")

        # Build graph
        G = self.build_graph(events)
        print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        # Save graph
        clip_output_dir = self.output_dir / clip_id
        clip_output_dir.mkdir(parents=True, exist_ok=True)

        graph_path = clip_output_dir / "graph.gpickle"
        nx.write_gpickle(G, graph_path)

        # Save graph metadata
        metadata = {
            "clip_id": clip_id,
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
            "num_events": len(events)
        }
        metadata_path = clip_output_dir / "graph_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"  Saved graph to: {clip_output_dir}")
        return clip_output_dir

    def process_all_clips(self) -> None:
        """Process all clips with detected events."""
        events_dir = self.config.events_output_dir
        if not events_dir.exists():
            print(f"Events directory not found: {events_dir}")
            return

        clip_dirs = [d for d in events_dir.iterdir() if d.is_dir()]
        print(f"Found {len(clip_dirs)} clips to process")

        for clip_dir in clip_dirs:
            clip_id = clip_dir.name
            self.process_clip(clip_id)


def main():
    """Run graph construction stage."""
    config = Config()
    builder = GraphBuilder(config)
    builder.process_all_clips()


if __name__ == "__main__":
    main()
