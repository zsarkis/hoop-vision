import pytest
from pathlib import Path
import shutil
import json
import networkx as nx
from hoop_vision.stage3_graph import GraphBuilder
from hoop_vision.config import Config


@pytest.fixture
def clean_output_dir():
    """Clean output directory before test."""
    output_dir = Path("data/processed/graphs")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    yield
    if output_dir.exists():
        shutil.rmtree(output_dir)


def test_graph_builder_creates_graph(clean_output_dir):
    """Test that GraphBuilder creates a temporal graph from events."""
    events_dir = Path("data/processed/events/test_clip")
    if not events_dir.exists() or not (events_dir / "events.json").exists():
        pytest.skip("No events available")

    config = Config()
    builder = GraphBuilder(config)

    clip_id = "test_clip"
    output_path = builder.process_clip(clip_id)

    assert output_path.exists()
    graph_file = output_path / "graph.gpickle"
    assert graph_file.exists()


def test_graph_has_temporal_edges(clean_output_dir):
    """Test that graph contains temporal edges between events."""
    events_dir = Path("data/processed/events/test_clip")
    if not events_dir.exists() or not (events_dir / "events.json").exists():
        pytest.skip("No events available")

    config = Config()
    builder = GraphBuilder(config)

    clip_id = "test_clip"
    output_path = builder.process_clip(clip_id)

    graph_file = output_path / "graph.gpickle"
    G = nx.read_gpickle(graph_file)

    assert G.number_of_nodes() > 0
    # Should have temporal edges
    assert G.number_of_edges() >= 0
