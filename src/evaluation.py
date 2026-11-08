"""
SafeGrid AI — Pipeline Evaluation & Benchmarking Module

Executes batch evaluations across CubiCasa5k blueprints and emergency scenarios
to validate routing validity, safety margins, execution accuracy, and performance.
"""

from __future__ import annotations

import glob
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config
from src.graph_builder import GraphBuilder
from src.hazard_engine import HazardEngine
from src.llm_agent import EvacuationAgent, PureLLMBaseline
from src.pathfinder import SafePathfinder
from src.svg_parser import SVGFloorPlanParser

logger = logging.getLogger(__name__)


@dataclass
class ScenarioResult:
    """Metrics recorded for a single evaluation scenario."""

    blueprint_id: str
    scenario_id: str
    num_nodes: int
    num_edges: int
    num_hazards: int
    start_node: str
    exit_node: str
    route_found: bool
    path_length: int
    validity: bool  # Path edges all exist in graph
    path_cost: float
    baseline_cost: float
    optimality_ratio: float  # path_cost / baseline_cost
    safety_margin: float  # Min distance from path to any hazard center
    cumulative_hazard_exposure: float
    tool_invoked: bool
    processing_time_ms: float
    method: str = "Agentic_Dijkstra"


class PipelineEvaluator:
    """Batch evaluator for testing evacuation routing and LLM agent performance."""

    def __init__(
        self,
        dataset_root: Optional[str] = None,
        random_seed: int = config.RANDOM_SEED,
    ) -> None:
        self.dataset_root = dataset_root or config.DATASET_ROOT
        self.random_seed = random_seed
        random.seed(random_seed)
        self.results: List[ScenarioResult] = []
        self.parser = SVGFloorPlanParser()
        self.builder = GraphBuilder()

    def discover_blueprints(self, limit: int = config.NUM_EVAL_BLUEPRINTS) -> List[str]:
        """Find available CubiCasa5k model.svg files."""
        patterns = [
            os.path.join(self.dataset_root, "high_quality_architectural", "*", "model.svg"),
            os.path.join(self.dataset_root, "high_quality", "*", "model.svg"),
            os.path.join(self.dataset_root, "colorful", "*", "model.svg"),
        ]
        svg_files: List[str] = []
        for pat in patterns:
            svg_files.extend(glob.glob(pat))
            if len(svg_files) >= limit:
                break
        return svg_files[:limit]

    def evaluate_blueprint(
        self,
        svg_path: str,
        num_scenarios: int = 5,
    ) -> List[ScenarioResult]:
        """Run multiple emergency scenarios on a single blueprint."""
        scenario_results: List[ScenarioResult] = []
        blueprint_id = Path(svg_path).parent.name

        try:
            parsed = self.parser.parse(svg_path)
            graph = self.builder.build_graph(parsed)
        except Exception as e:
            logger.warning("Failed to parse blueprint %s: %e", svg_path, e)
            return scenario_results

        if graph.number_of_nodes() < 2:
            return scenario_results

        exit_nodes = [n for n, d in graph.nodes(data=True) if d.get("is_exit", False)]
        room_nodes = [n for n, d in graph.nodes(data=True) if d.get("node_type") == "room"]

        if not exit_nodes or not room_nodes:
            return scenario_results

        # Get baseline shortest pathfinder (without hazards)
        baseline_pf = SafePathfinder(graph)

        # Bounding box of floor plan for hazard sampling
        xs = [d.get("centroid_x", d.get("centroid", (0, 0))[0]) for _, d in graph.nodes(data=True)]
        ys = [d.get("centroid_y", d.get("centroid", (0, 0))[1]) for _, d in graph.nodes(data=True)]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        for s_idx in range(num_scenarios):
            start_room = random.choice(room_nodes)
            target_exit = random.choice(exit_nodes)

            # Baseline route
            baseline_route = baseline_pf.find_safest_route(start_room, target_exit)
            baseline_cost = baseline_route["cost"] if baseline_route else 1.0

            # Dynamic hazard injection
            hazard_engine = HazardEngine(graph)
            num_hazards = random.randint(1, 3)
            for _ in range(num_hazards):
                hx = random.uniform(min_x, max_x)
                hy = random.uniform(min_y, max_y)
                hazard_engine.add_hazard(
                    x=hx,
                    y=hy,
                    radius=random.uniform(80.0, 160.0),
                    hazard_type=random.choice(["fire", "smoke"]),
                    severity=random.uniform(0.8, 1.5),
                )

            t0 = time.perf_counter()
            weighted_graph = hazard_engine.get_weighted_graph()
            pathfinder = SafePathfinder(weighted_graph)
            route_dict = pathfinder.find_safest_route(start_room, target_exit)
            t_elapsed_ms = (time.perf_counter() - t0) * 1000.0

            if route_dict:
                path = route_dict["path"]
                valid = pathfinder.is_path_valid(path)
                cost = route_dict["cost"]
                safety_margin = route_dict["safety_margin"]
                exposure = pathfinder.get_path_hazard_exposure(path)
                opt_ratio = cost / baseline_cost if baseline_cost > 0 else 1.0
                route_found = True
                path_len = len(path)
            else:
                valid = False
                cost = float("inf")
                safety_margin = 0.0
                exposure = float("inf")
                opt_ratio = float("inf")
                route_found = False
                path_len = 0

            res = ScenarioResult(
                blueprint_id=blueprint_id,
                scenario_id=f"{blueprint_id}_s{s_idx}",
                num_nodes=graph.number_of_nodes(),
                num_edges=graph.number_of_edges(),
                num_hazards=num_hazards,
                start_node=start_room,
                exit_node=target_exit,
                route_found=route_found,
                path_length=path_len,
                validity=valid,
                path_cost=cost,
                baseline_cost=baseline_cost,
                optimality_ratio=opt_ratio,
                safety_margin=safety_margin,
                cumulative_hazard_exposure=exposure,
                tool_invoked=True,
                processing_time_ms=t_elapsed_ms,
                method="Agentic_Dijkstra",
            )
            scenario_results.append(res)
            self.results.append(res)

        return scenario_results

    def run_batch_evaluation(
        self,
        num_blueprints: int = 10,
        num_scenarios_per_bp: int = 5,
    ) -> pd.DataFrame:
        """Run batch evaluation across discovered blueprints."""
        blueprints = self.discover_blueprints(limit=num_blueprints)
        logger.info("Starting batch evaluation across %d blueprints...", len(blueprints))

        for idx, bp_path in enumerate(blueprints):
            logger.info("[%d/%d] Evaluating %s...", idx + 1, len(blueprints), bp_path)
            self.evaluate_blueprint(bp_path, num_scenarios=num_scenarios_per_bp)

        return self.generate_report()

    def compare_pure_llm_vs_agentic(
        self,
        sample_size: int = 20,
    ) -> pd.DataFrame:
        """Head-to-head comparison between Pure LLM (reasoning over graph text)
        and SafeGrid Agentic Dijkstra Tool routing."""
        blueprints = self.discover_blueprints(limit=sample_size)
        agentic_agent = EvacuationAgent()
        pure_llm = PureLLMBaseline()

        comparison_records: List[Dict[str, Any]] = []

        for bp_path in blueprints:
            try:
                parsed = self.parser.parse(bp_path)
                graph = self.builder.build_graph(parsed)
            except Exception:
                continue

            rooms = [n for n, d in graph.nodes(data=True) if d.get("node_type") == "room"]
            exits = [n for n, d in graph.nodes(data=True) if d.get("is_exit", False)]
            if not rooms or not exits:
                continue

            start = random.choice(rooms)
            target = random.choice(exits)

            he = HazardEngine(graph)
            he.add_hazard(x=400, y=400, radius=120, hazard_type="fire")
            wg = he.get_weighted_graph()

            # 1. Agentic Tool Route
            t0 = time.perf_counter()
            agentic_res = agentic_agent.generate_instructions(wg, start, he.get_hazards(), target)
            t_agentic = time.perf_counter() - t0

            # 2. Pure LLM Route
            t0 = time.perf_counter()
            pure_res = pure_llm.generate_route(graph, start, he.get_hazards(), target)
            t_pure = time.perf_counter() - t0

            comparison_records.append({
                "blueprint": Path(bp_path).parent.name,
                "agentic_valid": SafePathfinder(graph).is_path_valid(agentic_res.route),
                "agentic_safety_margin": agentic_res.safety_margin,
                "agentic_tool_used": agentic_res.tool_invoked,
                "agentic_latency": t_agentic,
                "pure_llm_valid": SafePathfinder(graph).is_path_valid(pure_res.route),
                "pure_llm_safety_margin": pure_res.safety_margin,
                "pure_llm_tool_used": False,
                "pure_llm_latency": t_pure,
            })

        return pd.DataFrame(comparison_records)

    def generate_report(self) -> pd.DataFrame:
        """Aggregate results into a summary DataFrame."""
        if not self.results:
            return pd.DataFrame()

        data = [asdict(r) for r in self.results]
        df = pd.DataFrame(data)

        # Print quick summary
        valid_rate = (df["validity"].sum() / len(df)) * 100.0 if len(df) > 0 else 0.0
        avg_safety = df[df["safety_margin"] < float("inf")]["safety_margin"].mean()
        avg_time = df["processing_time_ms"].mean()

        print("\n================ SafeGrid AI Evaluation Summary ================")
        print(f"Total Scenarios Evaluated: {len(df)}")
        print(f"Path Validity Rate:        {valid_rate:.2f}%")
        print(f"Average Safety Margin:     {avg_safety:.2f} units")
        print(f"Average Processing Time:   {avg_time:.2f} ms")
        print("=================================================================\n")

        return df

    def export_results(self, output_path: str) -> None:
        """Export scenario results to CSV or JSON."""
        if not self.results:
            logger.warning("No results to export.")
            return

        df = pd.DataFrame([asdict(r) for r in self.results])
        if output_path.endswith(".csv"):
            df.to_csv(output_path, index=False)
        elif output_path.endswith(".json"):
            df.to_json(output_path, orient="records", indent=2)
        logger.info("Exported evaluation results to %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    evaluator = PipelineEvaluator()
    df_results = evaluator.run_batch_evaluation(num_blueprints=5, num_scenarios_per_bp=3)
    print(df_results.head())
