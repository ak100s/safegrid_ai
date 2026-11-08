"""
SafeGrid AI — Dijkstra‑based Safe Pathfinding

Wraps NetworkX's Dijkstra shortest‑path algorithm to find the safest
evacuation route through a hazard‑weighted navigation graph.

Typical usage
-------------
>>> pathfinder = SafePathfinder(weighted_graph, hazards)
>>> route = pathfinder.find_safest_route("Room_1", "Exit_0")
>>> print(route["path"])
['Room_1', 'Door_3', 'Corridor_A', 'Exit_0']
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

logger = logging.getLogger(__name__)


class SafePathfinder:
    """Deterministic safest‑path finder over a hazard‑weighted NetworkX graph.

    Parameters
    ----------
    graph : nx.Graph
        A NetworkX graph whose edges carry a ``"weight"`` attribute
        (typically produced by :class:`HazardEngine.get_weighted_graph`).
    hazards : list[dict], optional
        Active hazard descriptors (as returned by
        :meth:`HazardEngine.get_hazards`).  Used to compute safety margins
        and hazard exposure along paths.
    """

    def __init__(
        self,
        graph: nx.Graph,
        hazards: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        if graph is None or not isinstance(graph, nx.Graph):
            raise TypeError("A valid NetworkX Graph instance is required.")
        self._graph = graph
        self._hazards: List[Dict[str, Any]] = (
            hazards if hazards is not None else list(self._graph.graph.get("hazards", []))
        )
        logger.info(
            "SafePathfinder initialised — %d nodes, %d edges, %d active hazard(s).",
            self._graph.number_of_nodes(),
            self._graph.number_of_edges(),
            len(self._hazards),
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _node_centroid(self, node: str) -> Tuple[float, float]:
        """Extract (x, y) centroid from node attributes."""
        data = self._graph.nodes[node]
        if "centroid" in data and isinstance(data["centroid"], (tuple, list)) and len(data["centroid"]) >= 2:
            return float(data["centroid"][0]), float(data["centroid"][1])
        x = data.get("centroid_x", data.get("x", 0.0))
        y = data.get("centroid_y", data.get("y", 0.0))
        return float(x), float(y)

    @staticmethod
    def _euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _min_hazard_distance(self, point: Tuple[float, float]) -> float:
        """Return the minimum distance from *point* to any hazard centre.

        If no hazards exist, returns ``float('inf')``.
        """
        if not self._hazards:
            return float("inf")
        return min(
            self._euclidean(point, (h["x"], h["y"])) for h in self._hazards
        )

    def _build_node_detail(self, node: str) -> Dict[str, Any]:
        """Build a detail dict for a single node."""
        data = self._graph.nodes[node]
        cx, cy = self._node_centroid(node)
        return {
            "node_id": node,
            "type": data.get("type", data.get("node_type", "unknown")),
            "label": data.get("label", node),
            "centroid_x": cx,
            "centroid_y": cy,
        }

    # ── Core pathfinding ──────────────────────────────────────────────────────

    def find_safest_route(
        self,
        start_node: str,
        exit_node: str,
    ) -> Optional[Dict[str, Any]]:
        """Find the lowest‑cost Dijkstra path from *start_node* to *exit_node*.

        Parameters
        ----------
        start_node : str
            Source node ID (e.g. ``"Room_1"``).
        exit_node : str
            Target node ID (e.g. ``"Exit_0"``).

        Returns
        -------
        dict or None
            A route dictionary with keys ``path``, ``cost``,
            ``path_details``, and ``safety_margin``.  Returns ``None`` when
            no path exists (typically because hazards have disconnected the
            graph).
        """
        if start_node not in self._graph:
            logger.error("Start node '%s' not found in graph.", start_node)
            return None
        if exit_node not in self._graph:
            logger.error("Exit node '%s' not found in graph.", exit_node)
            return None

        try:
            path: List[str] = nx.dijkstra_path(
                self._graph, start_node, exit_node, weight="weight"
            )
            cost: float = nx.dijkstra_path_length(
                self._graph, start_node, exit_node, weight="weight"
            )
        except nx.NetworkXNoPath:
            logger.warning(
                "No path from '%s' to '%s' — exit may be blocked by hazards.",
                start_node,
                exit_node,
            )
            return None

        if math.isinf(cost):
            logger.warning(
                "Path from '%s' to '%s' passes through a blocked edge.",
                start_node,
                exit_node,
            )
            return None

        path_details = [self._build_node_detail(n) for n in path]
        safety_margin = self._compute_safety_margin(path)

        logger.info(
            "Route found: %s → %s | cost=%.2f | safety_margin=%.2f | hops=%d",
            start_node,
            exit_node,
            cost,
            safety_margin,
            len(path) - 1,
        )

        return {
            "path": path,
            "cost": cost,
            "path_details": path_details,
            "safety_margin": safety_margin,
        }

    # ── Exit discovery ────────────────────────────────────────────────────────

    def find_all_exits(self) -> List[str]:
        """Return a sorted list of node IDs identified as exits.

        A node is considered an exit when its ``type`` / ``node_type``
        attribute contains one of the common exit keywords (case‑insensitive),
        or its ID starts with ``"Exit"``.
        """
        exit_keywords = {"exit", "external", "entrance", "balcony", "terrace", "garage"}
        exits: List[str] = []
        for node, data in self._graph.nodes(data=True):
            node_type = str(
                data.get("type", data.get("node_type", ""))
            ).lower()
            node_label = str(data.get("label", "")).lower()
            node_id_lower = node.lower()

            if (
                any(kw in node_type for kw in exit_keywords)
                or any(kw in node_label for kw in exit_keywords)
                or node_id_lower.startswith("exit")
            ):
                exits.append(node)

        exits.sort()
        logger.info("Found %d exit node(s): %s", len(exits), exits)
        return exits

    def find_nearest_exit(
        self,
        start_node: str,
    ) -> Optional[Dict[str, Any]]:
        """Find the safest route from *start_node* to the nearest reachable exit.

        'Nearest' is defined by lowest Dijkstra cost (not Euclidean distance).

        Returns
        -------
        dict or None
            The route dict for the best exit, or ``None`` if every exit is
            unreachable.
        """
        exits = self.find_all_exits()
        if not exits:
            logger.warning("No exit nodes found in the graph.")
            return None

        routes = self.find_routes_to_all_exits(start_node)
        if not routes:
            logger.warning(
                "All exits are unreachable from '%s'.", start_node
            )
            return None

        best = routes[0]  # already sorted by cost
        logger.info(
            "Nearest reachable exit from '%s': %s (cost=%.2f)",
            start_node,
            best["path"][-1],
            best["cost"],
        )
        return best

    def find_routes_to_all_exits(
        self,
        start_node: str,
    ) -> List[Dict[str, Any]]:
        """Compute routes from *start_node* to every reachable exit.

        Returns a list of route dicts sorted by ascending cost.  Unreachable
        exits are silently skipped.
        """
        exits = self.find_all_exits()
        routes: List[Dict[str, Any]] = []
        for exit_node in exits:
            route = self.find_safest_route(start_node, exit_node)
            if route is not None:
                routes.append(route)

        routes.sort(key=lambda r: r["cost"])
        logger.info(
            "Computed routes to %d/%d reachable exit(s) from '%s'.",
            len(routes),
            len(exits),
            start_node,
        )
        return routes

    # ── Path validation & analysis ────────────────────────────────────────────

    def is_path_valid(self, path: List[str]) -> bool:
        """Verify that *path* is a contiguous walk in the graph.

        Returns ``False`` if any consecutive pair of nodes is not connected
        by an edge (i.e. the path would require "wall‑jumping").
        """
        if not path:
            return False
        for u, v in zip(path[:-1], path[1:]):
            if not self._graph.has_edge(u, v):
                logger.warning(
                    "Invalid path segment: no edge between '%s' and '%s'.",
                    u,
                    v,
                )
                return False
        return True

    def get_path_hazard_exposure(
        self,
        path: List[str],
    ) -> Dict[str, Any]:
        """Calculate cumulative hazard exposure along *path*.

        For every edge in the path, the midpoint‑to‑hazard distance is
        computed.  Exposure is defined as the sum of
        ``severity / distance`` for each hazard within its radius of each
        edge midpoint.

        Returns
        -------
        dict
            ``total_exposure``   — cumulative exposure score,
            ``edge_exposures``   — per‑edge breakdown,
            ``max_edge_exposure`` — worst single edge,
            ``hazard_encounters`` — number of edges inside any hazard radius.
        """
        if not self._hazards or not path:
            return {
                "total_exposure": 0.0,
                "edge_exposures": [],
                "max_edge_exposure": 0.0,
                "hazard_encounters": 0,
            }

        edge_exposures: List[Dict[str, Any]] = []
        total = 0.0
        max_exposure = 0.0
        encounters = 0

        for u, v in zip(path[:-1], path[1:]):
            cu = self._node_centroid(u)
            cv = self._node_centroid(v)
            mid = ((cu[0] + cv[0]) / 2.0, (cu[1] + cv[1]) / 2.0)

            edge_exp = 0.0
            encountered = False
            for h in self._hazards:
                dist = self._euclidean(mid, (h["x"], h["y"]))
                if dist < h["radius"] and dist > 0:
                    contribution = h.get("severity", 1.0) / dist
                    edge_exp += contribution
                    encountered = True

            if encountered:
                encounters += 1

            edge_exposures.append(
                {"edge": (u, v), "exposure": edge_exp}
            )
            total += edge_exp
            max_exposure = max(max_exposure, edge_exp)

        return {
            "total_exposure": round(total, 4),
            "edge_exposures": edge_exposures,
            "max_edge_exposure": round(max_exposure, 4),
            "hazard_encounters": encounters,
        }

    # ── Safety margin ─────────────────────────────────────────────────────────

    def _compute_safety_margin(self, path: List[str]) -> float:
        """Return the minimum distance from any node centroid on the path to
        the nearest hazard centre.

        A larger value means the path keeps a wider berth from danger.
        """
        if not self._hazards:
            return float("inf")

        margin = float("inf")
        for node in path:
            centroid = self._node_centroid(node)
            dist = self._min_hazard_distance(centroid)
            margin = min(margin, dist)
        return round(margin, 2)

    # ── LLM‑friendly text output ──────────────────────────────────────────────

    @staticmethod
    def format_route_for_llm(route: Dict[str, Any]) -> str:
        """Format a route dictionary as a clean text string for LLM
        consumption.

        Example output::

            Route: Room_1 → Door_3 → Corridor_A → Exit_0
            Total cost: 342.50
            Safety margin: 127.30
            Steps:
              1. Room_1 (Room) at (120.0, 340.0)
              2. Door_3 (Door) at (200.0, 340.0)
              3. Corridor_A (Corridor) at (300.0, 340.0)
              4. Exit_0 (Exit) at (480.0, 340.0)
        """
        if route is None:
            return "No viable route found — all paths are blocked by hazards."

        lines: List[str] = []
        lines.append("Route: " + " → ".join(route["path"]))
        lines.append(f"Total cost: {route['cost']:.2f}")
        margin = route.get("safety_margin", float("inf"))
        margin_str = f"{margin:.2f}" if not math.isinf(margin) else "∞ (no hazards)"
        lines.append(f"Safety margin: {margin_str}")
        lines.append("Steps:")
        for i, detail in enumerate(route["path_details"], 1):
            lines.append(
                f"  {i}. {detail['node_id']} ({detail['type']}) "
                f"at ({detail['centroid_x']:.1f}, {detail['centroid_y']:.1f})"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Module-level standalone routing function
# ---------------------------------------------------------------------------

def dijkstra_safest_path(
    graph_state: nx.Graph,
    start_node: str,
    exit_node: str,
) -> Optional[Dict[str, Any]]:
    """Callable pathfinding function using `networkx.dijkstra_path`.

    Takes `(graph_state, start_node, exit_node)` and calculates the path
    minimizing dynamic weights.

    Args:
        graph_state: The NetworkX graph carrying dynamic hazard edge weights.
        start_node: The starting room / occupant location.
        exit_node: The target safe exit node.

    Returns:
        dict:
            - "path": List of node IDs, e.g. ["Room_12", "Door_4", "Corridor_A", "Exit_0"]
            - "cost": Total path cost (weighted traversal distance)
            - "safety_margin": Minimum distance to any active hazard
            - "path_details": Detailed node metadata sequence
    """
    pathfinder = SafePathfinder(graph_state)
    return pathfinder.find_safest_route(start_node, exit_node)
