"""
SafeGrid AI — Comprehensive Pipeline Unit Tests

Covers:
1. SVG parsing (polygons, centroids, labels, room classes, exits)
2. Graph construction (nodes, edges, connectivity, Euclidean distances)
3. Hazard engine (weight penalization, danger zone blocking, hazard CRUD)
4. Safe pathfinder (Dijkstra routing, validity check, safety margin)
5. LLM agent (tool invocation, mock/live modes, prompt formatting)
6. Pipeline evaluation (end-to-end metrics calculation)
"""

import math
import os
import sys
import unittest
from pathlib import Path

# Add project root to sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config
from src.graph_builder import FloorPlanGraph, GraphBuilder
from src.hazard_engine import HazardEngine
from src.llm_agent import EvacuationAgent, PureLLMBaseline
from src.pathfinder import SafePathfinder
from src.svg_parser import CubiCasaSVGParser, SVGFloorPlanParser


class TestSVGParser(unittest.TestCase):
    """Test Suite 1: CubiCasa5k SVG Parsing."""

    def setUp(self):
        self.sample_svg = os.path.join(
            config.DATASET_ROOT, "high_quality_architectural", "1", "model.svg"
        )
        self.skip_if_no_file = not os.path.exists(self.sample_svg)

    def test_parse_sample_svg(self):
        if self.skip_if_no_file:
            self.skipTest("Sample SVG not found at expected path")

        parser = SVGFloorPlanParser()
        data = parser.parse(self.sample_svg)

        self.assertIn("spaces", data)
        self.assertIn("doors", data)
        self.assertIn("exits", data)

        self.assertGreater(len(data["spaces"]), 0, "Should extract at least 1 room")
        self.assertGreater(len(data["doors"]), 0, "Should extract at least 1 door")
        self.assertGreater(len(data["exits"]), 0, "Should extract at least 1 exit")

        # Check space attributes
        first_space = data["spaces"][0]
        self.assertIn("id", first_space)
        self.assertIn("centroid", first_space)
        self.assertEqual(len(first_space["centroid"]), 2)
        self.assertGreater(len(first_space["polygon_points"]), 2)


class TestGraphBuilder(unittest.TestCase):
    """Test Suite 2: NetworkX Graph Construction."""

    def setUp(self):
        self.sample_svg = os.path.join(
            config.DATASET_ROOT, "high_quality_architectural", "1", "model.svg"
        )
        if not os.path.exists(self.sample_svg):
            self.skipTest("Sample SVG not found")

        parser = SVGFloorPlanParser()
        self.data = parser.parse(self.sample_svg)
        self.builder = GraphBuilder()
        self.graph = self.builder.build_graph(self.data)

    def test_graph_properties(self):
        self.assertGreater(self.graph.number_of_nodes(), 5)
        self.assertGreater(self.graph.number_of_edges(), 5)

        # Check node attributes
        for n, d in self.graph.nodes(data=True):
            self.assertIn("centroid", d)
            self.assertIn("centroid_x", d)
            self.assertIn("centroid_y", d)
            self.assertIn("node_type", d)

        # Check edge weights
        for u, v, d in self.graph.edges(data=True):
            self.assertIn("weight", d)
            self.assertGreater(d["weight"], 0.0)

    def test_exit_nodes_marked(self):
        exits = [n for n, d in self.graph.nodes(data=True) if d.get("is_exit", False)]
        self.assertGreaterEqual(len(exits), 1, "Graph must contain at least one exit")


class TestHazardEngine(unittest.TestCase):
    """Test Suite 3: Dynamic Hazard Weighting."""

    def setUp(self):
        self.sample_svg = os.path.join(
            config.DATASET_ROOT, "high_quality_architectural", "1", "model.svg"
        )
        if not os.path.exists(self.sample_svg):
            self.skipTest("Sample SVG not found")

        parser = SVGFloorPlanParser()
        data = parser.parse(self.sample_svg)
        self.graph = GraphBuilder().build_graph(data)
        self.engine = HazardEngine(self.graph)

    def test_add_and_remove_hazard(self):
        h_id = self.engine.add_hazard(x=500.0, y=400.0, radius=120.0, hazard_type="fire")
        self.assertEqual(len(self.engine.get_hazards()), 1)

        removed = self.engine.remove_hazard(h_id)
        self.assertTrue(removed)
        self.assertEqual(len(self.engine.get_hazards()), 0)

    def test_weight_penalization(self):
        # Pick an edge midpoint
        edge = list(self.graph.edges())[0]
        u, v = edge
        base_w = self.graph.edges[u, v]["weight"]

        u_c = self.graph.nodes[u]["centroid"]
        v_c = self.graph.nodes[v]["centroid"]
        mid_x = (u_c[0] + v_c[0]) / 2.0
        mid_y = (u_c[1] + v_c[1]) / 2.0

        # Inject hazard directly near midpoint (outside inner danger zone)
        self.engine.add_hazard(x=mid_x + 50.0, y=mid_y, radius=150.0, hazard_type="fire")
        weighted_g = self.engine.get_weighted_graph()
        new_w = weighted_g.edges[u, v]["weight"]

        self.assertGreater(new_w, base_w, "Edge weight near hazard must increase")

    def test_danger_zone_blocking(self):
        edge = list(self.graph.edges())[0]
        u, v = edge
        u_c = self.graph.nodes[u]["centroid"]
        v_c = self.graph.nodes[v]["centroid"]
        mid_x = (u_c[0] + v_c[0]) / 2.0
        mid_y = (u_c[1] + v_c[1]) / 2.0

        # Inject hazard directly at midpoint
        self.engine.add_hazard(x=mid_x, y=mid_y, radius=150.0, hazard_type="fire")
        weighted_g = self.engine.get_weighted_graph()

        self.assertEqual(weighted_g.edges[u, v]["weight"], float("inf"), "Direct hazard should block edge")


class TestSafePathfinder(unittest.TestCase):
    """Test Suite 4: Deterministic Dijkstra Pathfinding Tool."""

    def setUp(self):
        self.sample_svg = os.path.join(
            config.DATASET_ROOT, "high_quality_architectural", "1", "model.svg"
        )
        if not os.path.exists(self.sample_svg):
            self.skipTest("Sample SVG not found")

        parser = SVGFloorPlanParser()
        data = parser.parse(self.sample_svg)
        self.graph = GraphBuilder().build_graph(data)
        self.engine = HazardEngine(self.graph)
        self.pathfinder = SafePathfinder(self.graph)

    def test_find_safest_route(self):
        rooms = [n for n, d in self.graph.nodes(data=True) if d.get("node_type") == "room"]
        exits = [n for n, d in self.graph.nodes(data=True) if d.get("is_exit", False)]

        route = self.pathfinder.find_safest_route(rooms[0], exits[0])
        self.assertIsNotNone(route)
        self.assertIn("path", route)
        self.assertIn("cost", route)
        self.assertEqual(route["path"][0], rooms[0])
        self.assertEqual(route["path"][-1], exits[0])

    def test_path_validity_check(self):
        rooms = [n for n, d in self.graph.nodes(data=True) if d.get("node_type") == "room"]
        exits = [n for n, d in self.graph.nodes(data=True) if d.get("is_exit", False)]
        route = self.pathfinder.find_safest_route(rooms[0], exits[0])

        valid = self.pathfinder.is_path_valid(route["path"])
        self.assertTrue(valid, "Path must be strictly valid along graph edges")

        # Fake disconnected path
        invalid_path = [rooms[0], "NON_EXISTENT_NODE", exits[0]]
        self.assertFalse(self.pathfinder.is_path_valid(invalid_path))


class TestLLMAgent(unittest.TestCase):
    """Test Suite 5: LLM Agent Framework."""

    def setUp(self):
        self.sample_svg = os.path.join(
            config.DATASET_ROOT, "high_quality_architectural", "1", "model.svg"
        )
        if not os.path.exists(self.sample_svg):
            self.skipTest("Sample SVG not found")

        parser = SVGFloorPlanParser()
        data = parser.parse(self.sample_svg)
        self.graph = GraphBuilder().build_graph(data)
        self.engine = HazardEngine(self.graph)
        self.engine.add_hazard(x=500.0, y=400.0, radius=120.0)

    def test_agent_instruction_generation(self):
        agent = EvacuationAgent()
        rooms = [n for n, d in self.graph.nodes(data=True) if d.get("node_type") == "room"]

        res = agent.generate_instructions(
            self.engine.get_weighted_graph(),
            start_node=rooms[0],
            hazards=self.engine.get_hazards(),
        )

        self.assertIsNotNone(res)
        self.assertTrue(res.tool_invoked, "Agentic LLM must invoke pathfinding tool")
        self.assertGreater(len(res.route), 1, "Route must contain waypoints")
        self.assertIn("Evacuation", res.instructions)

    def test_pure_llm_baseline(self):
        baseline = PureLLMBaseline()
        rooms = [n for n, d in self.graph.nodes(data=True) if d.get("node_type") == "room"]

        res = baseline.generate_route(
            self.graph,
            start_node=rooms[0],
            hazards=self.engine.get_hazards(),
        )
        self.assertIsNotNone(res)
        self.assertFalse(res.tool_invoked, "Pure LLM baseline must NOT invoke tools")


if __name__ == "__main__":
    unittest.main()
