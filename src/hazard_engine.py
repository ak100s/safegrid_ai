"""
SafeGrid AI — Dynamic Hazard Weighting Engine

Injects hazards into a navigable NetworkX graph and recalculates edge weights
using exponential penalty functions.  Edges whose midpoints fall inside a
danger zone (< 30 % of the hazard radius) are blocked entirely (weight = ∞).

Typical usage
-------------
>>> engine = HazardEngine(graph)
>>> engine.add_hazard(x=500, y=400, radius=150, hazard_type="fire")
>>> weighted = engine.get_weighted_graph()
"""

from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from shapely.geometry import Point, Polygon

try:
    from safegrid_ai import config
except ImportError:
    import config

logger = logging.getLogger(__name__)


# ─── Data classes ─────────────────────────────────────────────────────────────


@dataclass
class Hazard:
    """Representation of a single hazard source."""

    id: str
    x: float
    y: float
    radius: float
    hazard_type: str
    severity: float = 1.0
    _original_radius: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._original_radius = self.radius

    @property
    def center(self) -> Tuple[float, float]:
        """Return (x, y) centre of the hazard."""
        return (self.x, self.y)

    @property
    def danger_zone_radius(self) -> float:
        """Inner danger zone — edges here are completely blocked."""
        return self.radius * 0.3

    def as_shapely_circle(self, resolution: int = 64) -> Polygon:
        """Return a Shapely polygon approximating the hazard circle."""
        return Point(self.x, self.y).buffer(self.radius, resolution=resolution)

    def danger_zone_polygon(self, resolution: int = 64) -> Polygon:
        """Return a Shapely polygon for the inner danger zone."""
        return Point(self.x, self.y).buffer(
            self.danger_zone_radius, resolution=resolution
        )


# ─── Engine ───────────────────────────────────────────────────────────────────


class HazardEngine:
    """Manages hazards and applies dynamic exponential‑decay weight penalties
    to a NetworkX navigation graph.

    Parameters
    ----------
    graph : nx.Graph
        A NetworkX graph produced by :class:`GraphBuilder`.  Nodes are expected
        to carry ``centroid_x`` and ``centroid_y`` attributes.
    alpha : float, optional
        Exponential penalty multiplier (default from ``config``).
    beta : float, optional
        Exponential decay rate (default from ``config``).
    """

    def __init__(
        self,
        graph: nx.Graph,
        alpha: float = config.HAZARD_PENALTY_ALPHA,
        beta: float = config.HAZARD_DECAY_BETA,
    ) -> None:
        if graph is None or not isinstance(graph, nx.Graph):
            raise TypeError("A valid NetworkX Graph instance is required.")
        self._base_graph: nx.Graph = graph.copy()
        self._graph: nx.Graph = graph.copy()
        self._hazards: Dict[str, Hazard] = {}
        self._alpha = alpha
        self._beta = beta
        logger.info(
            "HazardEngine initialised with %d nodes, %d edges (α=%.2f, β=%.4f)",
            self._graph.number_of_nodes(),
            self._graph.number_of_edges(),
            self._alpha,
            self._beta,
        )

    # ── Hazard CRUD ───────────────────────────────────────────────────────────

    def add_hazard(
        self,
        x: float,
        y: float,
        radius: float = config.DEFAULT_HAZARD_RADIUS,
        hazard_type: str = "fire",
        severity: float = 1.0,
    ) -> str:
        """Inject a new hazard and recalculate edge weights.

        Parameters
        ----------
        x, y : float
            Hazard centre coordinates (SVG / floor‑plan units).
        radius : float
            Affected radius around the centre.
        hazard_type : str
            Descriptive label (``"fire"``, ``"smoke"``, ``"flood"`` …).
        severity : float
            Multiplier applied to the penalty (higher → more dangerous).

        Returns
        -------
        str
            A unique hazard ID that can be used with :meth:`remove_hazard`.
        """
        hazard_id = f"hazard_{uuid.uuid4().hex[:8]}"
        hazard = Hazard(
            id=hazard_id,
            x=x,
            y=y,
            radius=radius,
            hazard_type=hazard_type,
            severity=severity,
        )
        self._hazards[hazard_id] = hazard
        logger.info(
            "Added %s '%s' at (%.1f, %.1f) r=%.1f severity=%.2f",
            hazard_type,
            hazard_id,
            x,
            y,
            radius,
            severity,
        )
        self.update_weights()
        return hazard_id

    def add_hazard_at_node(
        self,
        node_id: str,
        radius: float = config.DEFAULT_HAZARD_RADIUS,
        hazard_type: str = "fire",
        severity: float = 1.0,
    ) -> str:
        """Inject a synthetic hazard centered directly at a specific graph node.

        Example:
            >>> engine.add_hazard_at_node("Room_15", radius=150.0, hazard_type="fire")
        """
        if node_id not in self._graph:
            raise KeyError(f"Node '{node_id}' does not exist in graph.")
        cx, cy = self._node_centroid(node_id)
        return self.add_hazard(
            x=cx,
            y=cy,
            radius=radius,
            hazard_type=hazard_type,
            severity=severity,
        )

    def remove_hazard(self, hazard_id: str) -> bool:
        """Remove a hazard by its ID and recalculate weights.

        Returns ``True`` if the hazard existed and was removed.
        """
        if hazard_id not in self._hazards:
            logger.warning("Hazard '%s' not found — nothing removed.", hazard_id)
            return False
        del self._hazards[hazard_id]
        logger.info("Removed hazard '%s'.", hazard_id)
        self.update_weights()
        return True

    def clear_hazards(self) -> int:
        """Remove **all** hazards and reset edge weights.

        Returns the number of hazards that were cleared.
        """
        count = len(self._hazards)
        self._hazards.clear()
        self._graph = self._base_graph.copy()
        logger.info("Cleared %d hazard(s); graph weights reset.", count)
        return count

    def get_hazards(self) -> List[Dict[str, Any]]:
        """Return a list of dictionaries describing every active hazard."""
        return [
            {
                "id": h.id,
                "x": h.x,
                "y": h.y,
                "radius": h.radius,
                "hazard_type": h.hazard_type,
                "severity": h.severity,
                "danger_zone_radius": h.danger_zone_radius,
            }
            for h in self._hazards.values()
        ]

    # ── Weight calculation ────────────────────────────────────────────────────

    def _node_centroid(self, node: str) -> Tuple[float, float]:
        """Extract the centroid coordinates from a node's attributes."""
        data = self._graph.nodes[node]
        if "centroid" in data and isinstance(data["centroid"], (tuple, list)) and len(data["centroid"]) >= 2:
            return float(data["centroid"][0]), float(data["centroid"][1])
        x = data.get("centroid_x", data.get("x", 0.0))
        y = data.get("centroid_y", data.get("y", 0.0))
        return float(x), float(y)

    @staticmethod
    def _euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def update_weights(self) -> None:
        """Recalculate all edge weights based on active hazards.

        Weight formula per edge (u, v)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        1. ``base_weight`` = Euclidean distance between the two node centroids.
        2. For each hazard *h* whose centre is within ``h.radius`` of the edge
           midpoint:
           - If ``distance < h.danger_zone_radius``: edge is **blocked**
             (weight = ``float('inf')``).
           - Otherwise: ``weight *= severity * α · exp(−β · distance)``.
        3. The ``blocked`` flag is stored as an edge attribute for downstream
           consumers.
        """
        # Reset to base graph weights
        self._graph = self._base_graph.copy()

        if not self._hazards:
            return

        blocked_count = 0
        penalised_count = 0

        for u, v in self._graph.edges():
            cu = self._node_centroid(u)
            cv = self._node_centroid(v)

            base_weight = self._euclidean(cu, cv)
            midpoint = ((cu[0] + cv[0]) / 2.0, (cu[1] + cv[1]) / 2.0)

            weight = base_weight
            blocked = False

            for hazard in self._hazards.values():
                dist = self._euclidean(midpoint, hazard.center)

                if dist < hazard.danger_zone_radius:
                    # Inside the inner danger zone → impassable
                    weight = float("inf")
                    blocked = True
                    break
                elif dist < hazard.radius:
                    # Inside the hazard area → exponential penalty
                    penalty = (
                        hazard.severity
                        * self._alpha
                        * math.exp(-self._beta * dist)
                    )
                    weight *= penalty
                    penalised_count += 1

            if blocked:
                blocked_count += 1

            self._graph.edges[u, v]["weight"] = weight
            self._graph.edges[u, v]["blocked"] = blocked

        logger.info(
            "Weights updated: %d edge(s) blocked, %d penalised.",
            blocked_count,
            penalised_count,
        )

    # ── Accessors ─────────────────────────────────────────────────────────────

    def get_weighted_graph(self) -> nx.Graph:
        """Return a copy of the graph with up‑to‑date hazard weights and metadata."""
        g = self._graph.copy()
        g.graph["hazards"] = self.get_hazards()
        return g

    def get_danger_zones(self) -> List[Dict[str, Any]]:
        """Return Shapely polygons for every active hazard.

        Each element is a dict with:
        - ``"hazard_id"``
        - ``"outer"`` — the full hazard polygon
        - ``"inner"`` — the danger‑zone polygon (blocked area)
        """
        zones: List[Dict[str, Any]] = []
        for h in self._hazards.values():
            zones.append(
                {
                    "hazard_id": h.id,
                    "hazard_type": h.hazard_type,
                    "outer": h.as_shapely_circle(),
                    "inner": h.danger_zone_polygon(),
                }
            )
        return zones

    # ── Simulation ────────────────────────────────────────────────────────────

    def simulate_spread(
        self,
        time_steps: int = 5,
        growth_rate: float = 1.15,
    ) -> List[List[Dict[str, Any]]]:
        """Simulate hazards spreading over discrete time steps.

        Each step multiplies the radius of every hazard by *growth_rate*
        (default 15 % increase per step, modelling fire spread).

        Parameters
        ----------
        time_steps : int
            Number of simulation ticks.
        growth_rate : float
            Multiplicative growth factor applied to each hazard's radius
            per step.

        Returns
        -------
        list[list[dict]]
            A list of snapshots; each snapshot is the output of
            :meth:`get_hazards` after that time step.
        """
        if time_steps < 1:
            raise ValueError("time_steps must be ≥ 1.")

        snapshots: List[List[Dict[str, Any]]] = []

        for step in range(1, time_steps + 1):
            for hazard in self._hazards.values():
                hazard.radius *= growth_rate
            self.update_weights()
            snapshot = self.get_hazards()
            snapshots.append(snapshot)
            logger.debug(
                "Spread step %d/%d — hazards: %s",
                step,
                time_steps,
                [(h["id"], h["radius"]) for h in snapshot],
            )

        logger.info(
            "Simulated %d spread step(s) (growth_rate=%.2f).",
            time_steps,
            growth_rate,
        )
        return snapshots

    def reset_hazard_radii(self) -> None:
        """Reset every hazard's radius to its original value."""
        for hazard in self._hazards.values():
            hazard.radius = hazard._original_radius
        self.update_weights()
        logger.info("All hazard radii reset to original values.")
