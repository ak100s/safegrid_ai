"""
SafeGrid AI — LLM Agent Framework with Tool Binding

Provides an LLM-powered evacuation coordinator using Google Gemini API
with LangChain tool binding, plus a pure-LLM baseline for comparison.

Classes:
    EvacuationAgent  — Agentic coordinator that binds pathfinding tools
                       and translates deterministic routes into natural
                       language evacuation instructions.
    PureLLMBaseline  — Baseline that asks the LLM to reason over raw
                       graph data without deterministic tool access,
                       useful for measuring hallucination rates.

When GOOGLE_API_KEY is not set, both classes fall back to a mock /
simulation mode that produces reasonable synthetic outputs so the
rest of the pipeline (benchmarking, evaluation) can run without
network access.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx

# ---------------------------------------------------------------------------
# Lazy / optional imports — guarded so the module loads even when
# LangChain or the Google GenAI SDK are not installed.
# ---------------------------------------------------------------------------
try:
    from langchain_core.tools import tool
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
import sys
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config import GOOGLE_API_KEY, GEMINI_MODEL

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class EvacuationResult:
    """Structured output from the evacuation agent."""

    route: List[str]
    instructions: str
    safety_margin: float
    tool_invoked: bool
    processing_time: float = 0.0
    model_used: str = ""
    is_mock: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary."""
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════════════
# Helper — determine whether we should use mock mode
# ═══════════════════════════════════════════════════════════════════════════

def _should_use_mock() -> bool:
    """Return True when no usable API key is available or LangChain is missing."""
    api_key = os.environ.get("GOOGLE_API_KEY", "") or GOOGLE_API_KEY
    if not api_key:
        logger.info("GOOGLE_API_KEY not set — falling back to mock mode.")
        return True
    if not _LANGCHAIN_AVAILABLE:
        logger.info("LangChain / langchain-google-genai not installed — falling back to mock mode.")
        return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
# Pathfinding tool wrappers — created as module-level LangChain @tool funcs
# ═══════════════════════════════════════════════════════════════════════════
# These closures are re-bound at runtime to the concrete graph by the agent
# before each invocation so the LLM can call them.

# We keep a module-level mutable container that the tools read from.
_tool_context: Dict[str, Any] = {
    "graph": None,
    "hazards": None,
}


def _hazard_penalty(graph: nx.Graph, path: List[str], hazards: List[Dict[str, Any]]) -> float:
    """Compute the minimum distance from *any* node on *path* to the
    nearest active hazard centroid.  Returns ``float('inf')`` when
    *hazards* is empty.
    """
    if not hazards or not path:
        return float("inf")

    min_dist = float("inf")
    for node in path:
        node_data = graph.nodes[node]
        nx_pos = (node_data.get("x", 0.0), node_data.get("y", 0.0))
        for h in hazards:
            hx, hy = h.get("x", 0.0), h.get("y", 0.0)
            dist = ((nx_pos[0] - hx) ** 2 + (nx_pos[1] - hy) ** 2) ** 0.5
            min_dist = min(min_dist, dist)
    return min_dist


def _find_exit_nodes(graph: nx.Graph) -> List[str]:
    """Return nodes flagged as exits in the graph."""
    exits: List[str] = []
    for node, data in graph.nodes(data=True):
        if data.get("is_exit", False):
            exits.append(node)
    # If no nodes are explicitly flagged, look for nodes whose label
    # contains common exit-related keywords.
    if not exits:
        for node, data in graph.nodes(data=True):
            label = str(data.get("label", node)).lower()
            if any(kw in label for kw in ("exit", "entrance", "balcony", "terrace", "garage")):
                exits.append(node)
    return exits


def _dijkstra_safest_route(
    graph: nx.Graph,
    start: str,
    end: str,
    hazards: List[Dict[str, Any]],
) -> Optional[List[str]]:
    """Run Dijkstra on the hazard-weighted graph.

    Edge weights are expected to already incorporate hazard penalties
    (via the hazard engine), but if a ``weight`` attribute is missing
    we fall back to unit weights.
    """
    if start not in graph or end not in graph:
        return None
    try:
        path = nx.dijkstra_path(graph, start, end, weight="weight")
        return path
    except nx.NetworkXNoPath:
        return None


# ---- LangChain tool definitions (only when LangChain is available) ------

if _LANGCHAIN_AVAILABLE:

    @tool
    def find_safest_route(start_node: str, exit_node: str) -> str:
        """Find the safest evacuation route from *start_node* to *exit_node*
        through the building's navigation graph, avoiding hazardous zones.

        Args:
            start_node: The identifier of the occupant's current room/node.
            exit_node:  The identifier of the target exit node.

        Returns:
            A JSON string with keys ``route`` (list of node ids) and
            ``safety_margin`` (minimum distance to nearest hazard).
        """
        graph: nx.Graph = _tool_context["graph"]
        hazards: List[Dict[str, Any]] = _tool_context["hazards"] or []

        if graph is None:
            return json.dumps({"error": "Graph not loaded."})

        path = _dijkstra_safest_route(graph, start_node, exit_node, hazards)
        if path is None:
            return json.dumps({
                "error": f"No path found from {start_node} to {exit_node}.",
                "route": [],
                "safety_margin": 0.0,
            })

        margin = _hazard_penalty(graph, path, hazards)
        return json.dumps({
            "route": path,
            "safety_margin": round(margin, 2),
        })

    @tool
    def find_nearest_exit(start_node: str) -> str:
        """Identify the nearest safe exit from *start_node* in the
        building's navigation graph.

        Args:
            start_node: The identifier of the occupant's current room/node.

        Returns:
            A JSON string with the nearest exit node id, the route, and
            its safety margin.
        """
        graph: nx.Graph = _tool_context["graph"]
        hazards: List[Dict[str, Any]] = _tool_context["hazards"] or []

        if graph is None:
            return json.dumps({"error": "Graph not loaded."})

        exits = _find_exit_nodes(graph)
        if not exits:
            return json.dumps({
                "error": "No exit nodes found in graph.",
                "exit_node": None,
                "route": [],
                "safety_margin": 0.0,
            })

        best_path: Optional[List[str]] = None
        best_exit: Optional[str] = None
        best_cost = float("inf")

        for ex in exits:
            try:
                cost = nx.dijkstra_path_length(graph, start_node, ex, weight="weight")
                if cost < best_cost:
                    best_cost = cost
                    best_exit = ex
                    best_path = nx.dijkstra_path(graph, start_node, ex, weight="weight")
            except nx.NetworkXNoPath:
                continue

        if best_path is None:
            return json.dumps({
                "error": f"No reachable exit from {start_node}.",
                "exit_node": None,
                "route": [],
                "safety_margin": 0.0,
            })

        margin = _hazard_penalty(graph, best_path, hazards)
        return json.dumps({
            "exit_node": best_exit,
            "route": best_path,
            "safety_margin": round(margin, 2),
        })


# ═══════════════════════════════════════════════════════════════════════════
# System Prompt
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "You are an **Emergency Evacuation Coordinator** embedded in the SafeGrid AI "
    "system.  Your mission is to guide building occupants to safety during an "
    "active emergency.\n\n"
    "## Capabilities\n"
    "You have access to two deterministic pathfinding tools:\n"
    "1. `find_safest_route(start_node, exit_node)` — returns the optimal "
    "hazard-avoiding route between two nodes in the building's navigation graph.\n"
    "2. `find_nearest_exit(start_node)` — identifies the closest reachable exit "
    "from a given location.\n\n"
    "## Instructions\n"
    "- **Always** invoke a pathfinding tool before providing directions.  Never "
    "guess or fabricate a route.\n"
    "- Translate the node-sequence returned by the tool into clear, step-by-step "
    "natural-language directions a panicked occupant can follow.\n"
    "- Mention safety margins and any hazards the route avoids.\n"
    "- Be concise, calm, and authoritative.\n"
    "- If no safe route exists, say so clearly and advise the occupant to shelter "
    "in place and call emergency services.\n"
)


# ═══════════════════════════════════════════════════════════════════════════
# EvacuationAgent — Agentic LLM with tool binding
# ═══════════════════════════════════════════════════════════════════════════

class EvacuationAgent:
    """LLM-powered evacuation coordinator with deterministic tool binding.

    When a valid ``GOOGLE_API_KEY`` is available and LangChain is installed
    the agent calls Google Gemini.  Otherwise it transparently falls back to
    a mock mode that produces realistic synthetic outputs so the evaluation
    pipeline can run without network access.

    Parameters
    ----------
    api_key : str, optional
        Override for ``GOOGLE_API_KEY``.  Falls back to env / config.
    model : str, optional
        Gemini model identifier.  Defaults to ``config.GEMINI_MODEL``.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("GOOGLE_API_KEY", "") or GOOGLE_API_KEY
        self._model_name = model or GEMINI_MODEL
        self._mock_mode = not self._api_key or not _LANGCHAIN_AVAILABLE

        if self._mock_mode:
            logger.warning(
                "EvacuationAgent initialised in MOCK mode "
                "(no API key or LangChain unavailable)."
            )
            self._llm = None
            self._agent = None
        else:
            self._llm = ChatGoogleGenerativeAI(
                model=self._model_name,
                google_api_key=self._api_key,
                temperature=0.2,
                convert_system_message_to_human=True,
            )
            # Bind the pathfinding tools
            self._tools = [find_safest_route, find_nearest_exit]
            self._llm_with_tools = self._llm.bind_tools(self._tools)
            logger.info("EvacuationAgent initialised with Gemini model %s.", self._model_name)

    # ── public API ────────────────────────────────────────────────────────

    def generate_instructions(
        self,
        graph: nx.Graph,
        start_node: str,
        hazards: List[Dict[str, Any]],
        exit_node: Optional[str] = None,
    ) -> EvacuationResult:
        """Generate natural-language evacuation instructions for one occupant.

        Parameters
        ----------
        graph : nx.Graph
            The building's navigation graph (nodes with ``x``, ``y``,
            ``label``, ``is_exit``; edges with ``weight``).
        start_node : str
            Node where the occupant is located.
        hazards : list of dict
            Active hazards, each with keys ``x``, ``y``, ``radius``,
            ``hazard_type``.
        exit_node : str, optional
            If provided, route to this specific exit.  Otherwise the
            agent will find the nearest exit.

        Returns
        -------
        EvacuationResult
        """
        t0 = time.time()

        if self._mock_mode:
            result = self._mock_generate(graph, start_node, hazards, exit_node)
        else:
            result = self._live_generate(graph, start_node, hazards, exit_node)

        result.processing_time = round(time.time() - t0, 4)
        result.model_used = self._model_name
        return result

    # Convenience alias for benchmark / evaluation scripts
    generate_route = generate_instructions

    def batch_evacuate(
        self,
        graph: nx.Graph,
        occupant_locations: List[str],
        hazards: List[Dict[str, Any]],
    ) -> List[EvacuationResult]:
        """Generate evacuation instructions for multiple occupants.

        Occupants are prioritised by their proximity to the nearest
        hazard (closest first) so the most at-risk people receive
        instructions first.

        Parameters
        ----------
        graph : nx.Graph
            The building's navigation graph.
        occupant_locations : list of str
            Node ids where occupants are located.
        hazards : list of dict
            Active hazards.

        Returns
        -------
        list of EvacuationResult
        """
        # --- Prioritise by proximity to hazard (ascending distance) ------
        def _min_hazard_distance(node: str) -> float:
            if not hazards or node not in graph:
                return float("inf")
            nd = graph.nodes[node]
            nx_pos = (nd.get("x", 0.0), nd.get("y", 0.0))
            return min(
                ((nx_pos[0] - h["x"]) ** 2 + (nx_pos[1] - h["y"]) ** 2) ** 0.5
                for h in hazards
            )

        sorted_occupants = sorted(occupant_locations, key=_min_hazard_distance)
        logger.info(
            "Batch evacuating %d occupants (priority order: %s).",
            len(sorted_occupants),
            sorted_occupants[:5],
        )

        results: List[EvacuationResult] = []
        for loc in sorted_occupants:
            try:
                result = self.generate_instructions(graph, loc, hazards)
                results.append(result)
            except Exception as exc:
                logger.error("Failed to generate instructions for %s: %s", loc, exc)
                results.append(EvacuationResult(
                    route=[],
                    instructions=f"ERROR: Could not generate instructions for {loc}: {exc}",
                    safety_margin=0.0,
                    tool_invoked=False,
                    is_mock=self._mock_mode,
                ))
        return results

    # ── private: live LLM invocation ──────────────────────────────────────

    def _live_generate(
        self,
        graph: nx.Graph,
        start_node: str,
        hazards: List[Dict[str, Any]],
        exit_node: Optional[str] = None,
    ) -> EvacuationResult:
        """Call Gemini with tool binding and parse the response."""

        # Inject the graph & hazards into the tool context so the
        # @tool functions can access them.
        _tool_context["graph"] = graph
        _tool_context["hazards"] = hazards

        # Build the user message with situational context
        exits = _find_exit_nodes(graph)
        hazard_desc = "; ".join(
            f"{h.get('hazard_type', 'unknown')} at ({h['x']:.0f}, {h['y']:.0f}), "
            f"radius {h.get('radius', 150):.0f}"
            for h in hazards
        ) or "No active hazards."

        target_info = (
            f"Route the occupant to exit node **{exit_node}**."
            if exit_node
            else "Find the nearest safe exit for the occupant."
        )

        user_msg = (
            f"## Current Situation\n"
            f"- **Occupant location**: node `{start_node}`\n"
            f"- **Active hazards**: {hazard_desc}\n"
            f"- **Available exits**: {', '.join(exits) if exits else 'none identified'}\n\n"
            f"{target_info}\n\n"
            f"Use the pathfinding tools to compute the safest route, then provide "
            f"clear step-by-step evacuation instructions."
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ]

        tool_was_invoked = False
        route: List[str] = []
        safety_margin: float = 0.0

        try:
            # Invoke the LLM — it may request tool calls
            response = self._llm_with_tools.invoke(messages)

            # Process tool calls if any
            if hasattr(response, "tool_calls") and response.tool_calls:
                tool_was_invoked = True
                for tc in response.tool_calls:
                    tool_name = tc["name"]
                    tool_args = tc["args"]

                    # Execute the tool
                    if tool_name == "find_safest_route":
                        tool_result = find_safest_route.invoke(tool_args)
                    elif tool_name == "find_nearest_exit":
                        tool_result = find_nearest_exit.invoke(tool_args)
                    else:
                        tool_result = json.dumps({"error": f"Unknown tool: {tool_name}"})

                    parsed = json.loads(tool_result)
                    if "route" in parsed:
                        route = parsed["route"]
                    if "safety_margin" in parsed:
                        safety_margin = parsed["safety_margin"]

                    # Feed tool result back for the LLM to produce NL instructions
                    from langchain_core.messages import ToolMessage
                    messages.append(response)
                    messages.append(ToolMessage(
                        content=tool_result,
                        tool_call_id=tc["id"],
                    ))

                # Second LLM call to get natural language instructions
                final_response = self._llm.invoke(messages)
                instructions = final_response.content
            else:
                # LLM answered directly (shouldn't happen but handle it)
                instructions = response.content
                logger.warning("LLM did not invoke any tool — response may be hallucinated.")

        except Exception as exc:
            logger.error("Live LLM invocation failed: %s — falling back to mock.", exc)
            return self._mock_generate(graph, start_node, hazards, exit_node)

        return EvacuationResult(
            route=route,
            instructions=instructions,
            safety_margin=safety_margin,
            tool_invoked=tool_was_invoked,
            is_mock=False,
        )

    # ── private: mock / simulation mode ───────────────────────────────────

    def _mock_generate(
        self,
        graph: nx.Graph,
        start_node: str,
        hazards: List[Dict[str, Any]],
        exit_node: Optional[str] = None,
    ) -> EvacuationResult:
        """Produce synthetic but structurally valid output using
        deterministic pathfinding only (no LLM call).
        """
        if start_node not in graph:
            return EvacuationResult(
                route=[],
                instructions=f"[MOCK] Node '{start_node}' not found in graph.",
                safety_margin=0.0,
                tool_invoked=True,
                is_mock=True,
            )

        # Determine target exit
        if exit_node and exit_node in graph:
            target = exit_node
        else:
            exits = _find_exit_nodes(graph)
            if not exits:
                # Pick a random node as pseudo-exit for mock purposes
                candidates = [n for n in graph.nodes if n != start_node]
                target = random.choice(candidates) if candidates else start_node
            else:
                # Pick nearest exit by hop count
                best, best_len = exits[0], float("inf")
                for ex in exits:
                    try:
                        length = nx.dijkstra_path_length(graph, start_node, ex, weight="weight")
                        if length < best_len:
                            best, best_len = ex, length
                    except nx.NetworkXNoPath:
                        continue
                target = best

        route = _dijkstra_safest_route(graph, start_node, target, hazards) or []
        margin = _hazard_penalty(graph, route, hazards) if route else 0.0

        # Synthesise natural-language instructions from the route
        if route:
            steps: List[str] = []
            for i, node in enumerate(route):
                label = graph.nodes[node].get("label", node)
                if i == 0:
                    steps.append(f"1. You are currently in **{label}**.")
                elif i == len(route) - 1:
                    steps.append(f"{i + 1}. Proceed to **{label}** — this is your exit. Go now!")
                else:
                    steps.append(f"{i + 1}. Move through **{label}**.")
            instructions_text = (
                "[MOCK] Evacuation Instructions (generated without LLM):\n"
                + "\n".join(steps)
                + f"\n\nSafety margin from nearest hazard: {margin:.1f} units."
            )
        else:
            instructions_text = (
                "[MOCK] No safe route found. Shelter in place and call emergency services."
            )

        return EvacuationResult(
            route=route,
            instructions=instructions_text,
            safety_margin=round(margin, 2),
            tool_invoked=True,  # mock still uses deterministic pathfinding
            is_mock=True,
        )


# ═══════════════════════════════════════════════════════════════════════════
# PureLLMBaseline — No tool binding, raw graph data only
# ═══════════════════════════════════════════════════════════════════════════

class PureLLMBaseline:
    """Baseline LLM agent that receives raw graph data (node list + edge
    list) and is asked to reason about evacuation without access to
    deterministic pathfinding tools.

    This exists to demonstrate the difference between pure LLM reasoning
    (prone to hallucination) and agentic tool-augmented generation.

    In mock mode the baseline intentionally produces *plausible but
    sometimes invalid* routes to simulate hallucination behaviour.
    """

    BASELINE_PROMPT = (
        "You are an emergency evacuation assistant.  You are given the "
        "floor plan of a building as a graph (nodes = rooms/spaces, "
        "edges = connections).  Given the occupant's location and active "
        "hazards, determine the safest evacuation route.\n\n"
        "**You do NOT have access to any pathfinding algorithm.**  You must "
        "figure out the route yourself by reasoning over the graph data.\n\n"
        "Return your answer as JSON with keys:\n"
        "- `route`: ordered list of node IDs from start to exit\n"
        "- `instructions`: natural-language step-by-step directions\n"
        "- `safety_margin`: your estimated safety margin (number)\n"
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("GOOGLE_API_KEY", "") or GOOGLE_API_KEY
        self._model_name = model or GEMINI_MODEL
        self._mock_mode = not self._api_key or not _LANGCHAIN_AVAILABLE

        if self._mock_mode:
            logger.warning("PureLLMBaseline initialised in MOCK mode.")
            self._llm = None
        else:
            self._llm = ChatGoogleGenerativeAI(
                model=self._model_name,
                google_api_key=self._api_key,
                temperature=0.4,
                convert_system_message_to_human=True,
            )
            logger.info("PureLLMBaseline initialised with Gemini model %s.", self._model_name)

    # ── public API ────────────────────────────────────────────────────────

    def generate_instructions(
        self,
        graph: nx.Graph,
        start_node: str,
        hazards: List[Dict[str, Any]],
        exit_node: Optional[str] = None,
    ) -> EvacuationResult:
        """Ask the LLM to produce evacuation instructions from raw graph
        data — without any deterministic pathfinding tool.

        Parameters match :meth:`EvacuationAgent.generate_instructions`.
        """
        t0 = time.time()

        if self._mock_mode:
            result = self._mock_generate(graph, start_node, hazards, exit_node)
        else:
            result = self._live_generate(graph, start_node, hazards, exit_node)

        result.processing_time = round(time.time() - t0, 4)
        result.model_used = self._model_name
        return result

    # Convenience alias for benchmark / evaluation scripts
    generate_route = generate_instructions

    # ── private: live LLM invocation ──────────────────────────────────────

    def _live_generate(
        self,
        graph: nx.Graph,
        start_node: str,
        hazards: List[Dict[str, Any]],
        exit_node: Optional[str] = None,
    ) -> EvacuationResult:
        """Send raw graph data to the LLM and parse its response."""

        # Serialise graph to text
        node_list = []
        for n, data in graph.nodes(data=True):
            node_list.append({
                "id": n,
                "label": data.get("label", n),
                "x": round(data.get("x", 0), 1),
                "y": round(data.get("y", 0), 1),
                "is_exit": data.get("is_exit", False),
            })

        edge_list = []
        for u, v, data in graph.edges(data=True):
            edge_list.append({
                "from": u, "to": v,
                "weight": round(data.get("weight", 1.0), 2),
            })

        hazard_desc = json.dumps(hazards, default=str)

        target_info = (
            f"Route the occupant to exit **{exit_node}**."
            if exit_node
            else "Find the nearest safe exit."
        )

        user_msg = (
            f"## Building Graph\n"
            f"Nodes ({len(node_list)}):\n```json\n{json.dumps(node_list, indent=1)}\n```\n\n"
            f"Edges ({len(edge_list)}):\n```json\n{json.dumps(edge_list, indent=1)}\n```\n\n"
            f"## Situation\n"
            f"- Occupant location: `{start_node}`\n"
            f"- Active hazards: {hazard_desc}\n\n"
            f"{target_info}"
        )

        messages = [
            SystemMessage(content=self.BASELINE_PROMPT),
            HumanMessage(content=user_msg),
        ]

        try:
            response = self._llm.invoke(messages)
            content = response.content

            # Attempt to parse JSON from the response
            route, instructions, margin = self._parse_llm_json(content)

        except Exception as exc:
            logger.error("PureLLMBaseline invocation failed: %s", exc)
            return self._mock_generate(graph, start_node, hazards, exit_node)

        return EvacuationResult(
            route=route,
            instructions=instructions,
            safety_margin=margin,
            tool_invoked=False,
            is_mock=False,
        )

    @staticmethod
    def _parse_llm_json(content: str) -> Tuple[List[str], str, float]:
        """Best-effort parse of the LLM's JSON response."""
        import re

        # Try to find JSON block in code fence or raw
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        if json_match:
            raw = json_match.group(1)
        else:
            # Try bare JSON
            brace_start = content.find("{")
            brace_end = content.rfind("}")
            if brace_start != -1 and brace_end != -1:
                raw = content[brace_start : brace_end + 1]
            else:
                return [], content, 0.0

        try:
            data = json.loads(raw)
            route = data.get("route", [])
            instructions = data.get("instructions", content)
            margin = float(data.get("safety_margin", 0.0))
            return route, instructions, margin
        except (json.JSONDecodeError, ValueError):
            return [], content, 0.0

    # ── private: mock / simulation mode ───────────────────────────────────

    def _mock_generate(
        self,
        graph: nx.Graph,
        start_node: str,
        hazards: List[Dict[str, Any]],
        exit_node: Optional[str] = None,
    ) -> EvacuationResult:
        """Simulate LLM behaviour *without* tool access.

        To mimic realistic hallucination patterns the mock:
        - Sometimes produces valid routes (LLMs can get simple graphs right).
        - Sometimes produces routes with non-existent edges (hallucination).
        - Sometimes picks sub-optimal paths.
        """
        nodes = list(graph.nodes)
        if start_node not in graph:
            return EvacuationResult(
                route=[],
                instructions="[MOCK-BASELINE] Start node not in graph.",
                safety_margin=0.0,
                tool_invoked=False,
                is_mock=True,
            )

        exits = _find_exit_nodes(graph) if not exit_node else [exit_node]
        target = exits[0] if exits else (nodes[-1] if nodes else start_node)

        # With 60% probability, produce the correct route (LLMs sometimes
        # get it right on small graphs).  With 40% probability, hallucinate.
        rng = random.Random()  # unseeded for variability
        if rng.random() < 0.6 and target in graph:
            try:
                route = nx.dijkstra_path(graph, start_node, target, weight="weight")
            except nx.NetworkXNoPath:
                route = [start_node, target]  # hallucinated direct path
        else:
            # Hallucinate: pick a random subset of nodes as "route"
            intermediate = rng.sample(
                [n for n in nodes if n not in (start_node, target)],
                k=min(rng.randint(1, 3), len(nodes) - 2) if len(nodes) > 2 else 0,
            )
            route = [start_node] + intermediate + [target]

        margin = _hazard_penalty(graph, route, hazards) if route else 0.0

        steps = []
        for i, node in enumerate(route):
            label = graph.nodes.get(node, {}).get("label", node)
            steps.append(f"{i + 1}. Go to {label}.")

        instructions_text = (
            "[MOCK-BASELINE] Evacuation Instructions (pure LLM, no tools):\n"
            + "\n".join(steps)
        )

        return EvacuationResult(
            route=route,
            instructions=instructions_text,
            safety_margin=round(margin, 2),
            tool_invoked=False,
            is_mock=True,
        )
