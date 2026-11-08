"""
SafeGrid AI — NetworkX Graph Construction from Parsed Floor Plans

Transforms the spatial data produced by :mod:`svg_parser` into a weighted,
undirected ``networkx.Graph`` suitable for shortest-path queries, egress
analysis, and hazard-aware evacuation routing.

Graph semantics
---------------
* **Room nodes** — centroid of each extracted Space polygon.
* **Door / Exit nodes** — centroid of each door threshold polygon.
* **Edges** — created when a door polygon overlaps (or nearly overlaps) a
  room polygon, connecting the door node to every adjacent room.  Rooms
  that directly share a boundary (e.g. corridors) also receive edges.
* **Edge weight** — Euclidean distance between connected centroids.
* **Exit attribute** — exit nodes carry ``is_exit=True``.
"""

from __future__ import annotations

import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
from shapely.geometry import Polygon as ShapelyPolygon

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# Import project config — gracefully fallback if running standalone
try:
    _cfg_dir = str(Path(__file__).resolve().parent.parent)
    if _cfg_dir not in sys.path:
        sys.path.insert(0, _cfg_dir)
    from config import ADJACENCY_BUFFER
except ImportError:
    ADJACENCY_BUFFER = 5.0
    logger.debug("config.py not found; using default ADJACENCY_BUFFER=%.1f", ADJACENCY_BUFFER)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Return the Euclidean distance between two 2-D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _make_shapely_polygon(
    points: List[Tuple[float, float]],
) -> Optional[ShapelyPolygon]:
    """Safely construct a Shapely Polygon, repairing if needed."""
    if len(points) < 3:
        return None
    try:
        poly = ShapelyPolygon(points)
        if not poly.is_valid:
            poly = poly.buffer(0)
        return poly
    except Exception as exc:
        logger.debug("Failed to create polygon: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Core graph builder
# ---------------------------------------------------------------------------

class FloorPlanGraph:
    """Build and query a NetworkX graph from parsed SVG spatial data.

    Usage::

        from svg_parser import parse_svg
        data = parse_svg("model.svg")
        fpg = FloorPlanGraph(data)
        fpg.build()
        G = fpg.graph
    """

    def __init__(
        self,
        parsed_data: Dict[str, List[Dict[str, Any]]],
        adjacency_buffer: float = ADJACENCY_BUFFER,
    ) -> None:
        """
        Args:
            parsed_data: Output of :func:`svg_parser.parse_svg`.
            adjacency_buffer: Pixel buffer for boundary-sharing detection.
        """
        self.parsed_data = parsed_data
        self.adjacency_buffer = adjacency_buffer
        self.graph: nx.Graph = nx.Graph()

        # Internal caches
        self._room_polygons: Dict[str, ShapelyPolygon] = {}
        self._room_centroids: Dict[str, Tuple[float, float]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self) -> nx.Graph:
        """Construct the full graph and return it.

        Steps:
            1. Add room (space) nodes.
            2. Add door / exit nodes.
            3. Connect doors to adjacent rooms.
            4. Connect rooms that share boundaries directly.
            5. Assign edge weights (Euclidean distance).

        Returns:
            The constructed ``networkx.Graph``.
        """
        logger.info("Building floor-plan graph …")
        self._add_space_nodes()
        self._add_door_nodes()
        self._add_exit_nodes()
        self._connect_doors_to_rooms()
        self._connect_adjacent_rooms()
        logger.info(
            "Graph built: %d nodes, %d edges",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )
        return self.graph

    def get_exit_nodes(self) -> List[str]:
        """Return node IDs marked as exits."""
        return [
            n for n, d in self.graph.nodes(data=True) if d.get("is_exit", False)
        ]

    def get_room_nodes(self) -> List[str]:
        """Return node IDs for rooms (spaces)."""
        return [
            n
            for n, d in self.graph.nodes(data=True)
            if d.get("node_type") == "room"
        ]

    def get_door_nodes(self) -> List[str]:
        """Return node IDs for internal doors."""
        return [
            n
            for n, d in self.graph.nodes(data=True)
            if d.get("node_type") == "door"
        ]

    def shortest_path_to_exit(
        self,
        source: str,
    ) -> Optional[Tuple[List[str], float]]:
        """Find the shortest weighted path from *source* to the nearest exit.

        Returns:
            ``(path, total_weight)`` or ``None`` if no exit is reachable.
        """
        exits = self.get_exit_nodes()
        if not exits:
            logger.warning("No exit nodes in graph.")
            return None

        best_path: Optional[List[str]] = None
        best_weight = float("inf")

        for exit_node in exits:
            try:
                path = nx.shortest_path(
                    self.graph, source, exit_node, weight="weight"
                )
                weight = nx.shortest_path_length(
                    self.graph, source, exit_node, weight="weight"
                )
                if weight < best_weight:
                    best_weight = weight
                    best_path = path
            except nx.NetworkXNoPath:
                continue

        if best_path is None:
            return None
        return best_path, best_weight

    def stats(self) -> Dict[str, Any]:
        """Return summary statistics about the graph."""
        rooms = self.get_room_nodes()
        doors = self.get_door_nodes()
        exits = self.get_exit_nodes()
        connected = nx.is_connected(self.graph) if self.graph.number_of_nodes() > 0 else False
        components = nx.number_connected_components(self.graph) if self.graph.number_of_nodes() > 0 else 0

        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "num_rooms": len(rooms),
            "num_doors": len(doors),
            "num_exits": len(exits),
            "is_connected": connected,
            "num_components": components,
            "avg_degree": (
                sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes()
                if self.graph.number_of_nodes() > 0
                else 0.0
            ),
        }

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def visualize(
        self,
        output_path: Optional[str | Path] = None,
        figsize: Tuple[int, int] = (14, 10),
        with_labels: bool = True,
    ) -> None:
        """Draw the graph using matplotlib.

        Node colours:
            - **green** → rooms
            - **orange** → doors
            - **red** → exits

        Args:
            output_path: If given, save the figure instead of showing it.
            figsize: Figure dimensions.
            with_labels: Whether to render node labels.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("matplotlib is required for visualisation.")
            return

        fig, ax = plt.subplots(figsize=figsize)

        # Use actual SVG centroid positions for layout
        pos = {}
        for n, d in self.graph.nodes(data=True):
            cx, cy = d.get("centroid", (0, 0))
            # Flip y-axis so the image orientation matches screen coords
            pos[n] = (cx, -cy)

        # Colour map
        node_colors = []
        for n, d in self.graph.nodes(data=True):
            if d.get("is_exit"):
                node_colors.append("#e74c3c")  # red
            elif d.get("node_type") == "door":
                node_colors.append("#f39c12")  # orange
            else:
                node_colors.append("#2ecc71")  # green

        nx.draw_networkx(
            self.graph,
            pos=pos,
            ax=ax,
            with_labels=with_labels,
            node_color=node_colors,
            node_size=350,
            font_size=7,
            font_weight="bold",
            edge_color="#95a5a6",
            width=1.5,
        )

        ax.set_title("SafeGrid AI — Floor Plan Graph", fontsize=14)
        ax.set_aspect("equal")
        plt.tight_layout()

        if output_path:
            fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
            logger.info("Graph saved to %s", output_path)
            plt.close(fig)
        else:
            plt.show()

    # ------------------------------------------------------------------
    # Internal: node creation
    # ------------------------------------------------------------------

    def _add_space_nodes(self) -> None:
        """Add room/space nodes with polygon & centroid attributes."""
        for space in self.parsed_data.get("spaces", []):
            node_id = space["id"]
            poly = _make_shapely_polygon(space["polygon_points"])
            centroid = tuple(space["centroid"])

            self.graph.add_node(
                node_id,
                node_type="room",
                space_type=space["type"],
                label=space["label"],
                centroid=centroid,
                centroid_x=centroid[0],
                centroid_y=centroid[1],
                polygon=poly,
                is_exit=False,
            )

            if poly is not None:
                self._room_polygons[node_id] = poly
            self._room_centroids[node_id] = centroid

    def _add_door_nodes(self) -> None:
        """Add internal door nodes."""
        for door in self.parsed_data.get("doors", []):
            node_id = door["id"]
            poly = _make_shapely_polygon(door["polygon_points"])
            centroid = tuple(door["centroid"])

            self.graph.add_node(
                node_id,
                node_type="door",
                door_type=door["type"],
                label=door["label"],
                centroid=centroid,
                centroid_x=centroid[0],
                centroid_y=centroid[1],
                polygon=poly,
                is_exit=False,
            )

    def _add_exit_nodes(self) -> None:
        """Add exit (external-wall door) nodes with ``is_exit=True``."""
        for exit_ in self.parsed_data.get("exits", []):
            node_id = exit_["id"]
            poly = _make_shapely_polygon(exit_["polygon_points"])
            centroid = tuple(exit_["centroid"])

            self.graph.add_node(
                node_id,
                node_type="exit",
                door_type=exit_["type"],
                label=exit_["label"],
                centroid=centroid,
                centroid_x=centroid[0],
                centroid_y=centroid[1],
                polygon=poly,
                is_exit=True,
            )

    # ------------------------------------------------------------------
    # Internal: edge creation
    # ------------------------------------------------------------------

    def _connect_doors_to_rooms(self) -> None:
        """Create edges between each door/exit and its adjacent rooms.

        A door is considered adjacent to a room when the door polygon
        (optionally buffered by ``adjacency_buffer``) intersects the room
        polygon.
        """
        door_exit_nodes = [
            n
            for n, d in self.graph.nodes(data=True)
            if d.get("node_type") in ("door", "exit")
        ]

        for de_node in door_exit_nodes:
            de_data = self.graph.nodes[de_node]
            de_poly: Optional[ShapelyPolygon] = de_data.get("polygon")
            de_centroid: Tuple[float, float] = de_data["centroid"]

            if de_poly is None:
                continue

            # Buffer the door polygon slightly for robust intersection
            de_buffered = de_poly.buffer(self.adjacency_buffer)

            for room_id, room_poly in self._room_polygons.items():
                try:
                    if de_buffered.intersects(room_poly):
                        dist = _euclidean(de_centroid, self._room_centroids[room_id])
                        self.graph.add_edge(de_node, room_id, weight=dist)
                except Exception as exc:
                    logger.debug(
                        "Intersection check failed (%s ↔ %s): %s",
                        de_node, room_id, exc,
                    )

    def _connect_adjacent_rooms(self) -> None:
        """Connect rooms that directly share a boundary (no door between).

        Two rooms are considered adjacent when their polygons (each buffered
        by ``adjacency_buffer``) intersect and there is no existing door
        node linking them.
        """
        room_ids = list(self._room_polygons.keys())

        for i, rid_a in enumerate(room_ids):
            poly_a = self._room_polygons[rid_a].buffer(self.adjacency_buffer)
            for rid_b in room_ids[i + 1:]:
                # Skip if already connected (via a shared door)
                if self.graph.has_edge(rid_a, rid_b):
                    continue

                poly_b = self._room_polygons[rid_b]
                try:
                    if poly_a.intersects(poly_b):
                        dist = _euclidean(
                            self._room_centroids[rid_a],
                            self._room_centroids[rid_b],
                        )
                        self.graph.add_edge(
                            rid_a, rid_b, weight=dist, connection="shared_boundary"
                        )
                except Exception as exc:
                    logger.debug(
                        "Adjacency check failed (%s ↔ %s): %s",
                        rid_a, rid_b, exc,
                    )


class GraphBuilder:
    """Wrapper class providing a unified interface for graph construction."""

    def __init__(self, adjacency_buffer: float = ADJACENCY_BUFFER) -> None:
        self.adjacency_buffer = adjacency_buffer

    def build_graph(self, parsed_data: Dict[str, List[Dict[str, Any]]]) -> nx.Graph:
        """Build and return a NetworkX graph from parsed SVG data."""
        fpg = FloorPlanGraph(parsed_data, adjacency_buffer=self.adjacency_buffer)
        return fpg.build()


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def build_graph(
    parsed_data: Dict[str, List[Dict[str, Any]]],
    adjacency_buffer: float = ADJACENCY_BUFFER,
) -> nx.Graph:
    """One-shot helper: build and return a floor-plan graph.

    Args:
        parsed_data: Result of :func:`svg_parser.parse_svg`.
        adjacency_buffer: Pixel buffer for adjacency detection.

    Returns:
        A populated ``networkx.Graph``.
    """
    fpg = FloorPlanGraph(parsed_data, adjacency_buffer=adjacency_buffer)
    return fpg.build()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    )

    # Import the sibling parser
    _this_dir = Path(__file__).resolve().parent
    if str(_this_dir) not in sys.path:
        sys.path.insert(0, str(_this_dir))
    from svg_parser import parse_svg

    _project_root = _this_dir.parent.parent
    _default_svg = _project_root / "cubicasa5k" / "high_quality_architectural" / "1" / "model.svg"

    svg_file = sys.argv[1] if len(sys.argv) > 1 else str(_default_svg)

    print(f"\n{'='*60}")
    print(f"  SafeGrid AI — Graph Builder")
    print(f"  SVG: {svg_file}")
    print(f"{'='*60}\n")

    # Step 1 — parse
    data = parse_svg(svg_file)

    # Step 2 — build graph
    fpg = FloorPlanGraph(data)
    G = fpg.build()

    # Step 3 — print stats
    st = fpg.stats()
    print("\n--- Graph Statistics ---")
    for k, v in st.items():
        print(f"  {k:20s}: {v}")

    # Step 4 — shortest path from each room to nearest exit
    exits = fpg.get_exit_nodes()
    print(f"\nExit nodes: {exits}")
    print("\n--- Shortest paths to nearest exit ---")
    for room in fpg.get_room_nodes():
        result = fpg.shortest_path_to_exit(room)
        if result:
            path, weight = result
            print(f"  {room:12s} -> {path[-1]:12s}  dist={weight:.1f}  hops={len(path)-1}  path={' -> '.join(path)}")
        else:
            print(f"  {room:12s} -> NO PATH TO EXIT")

    # Step 5 — node details
    print("\n--- All Nodes ---")
    for n, d in G.nodes(data=True):
        print(
            f"  {n:12s}  type={d.get('node_type','?'):6s}  "
            f"label={d.get('label',''):12s}  "
            f"exit={d.get('is_exit', False)}  "
            f"centroid=({d['centroid'][0]:.1f}, {d['centroid'][1]:.1f})  "
            f"degree={G.degree(n)}"
        )

    print(f"\n--- All Edges ({G.number_of_edges()}) ---")
    for u, v, d in G.edges(data=True):
        print(f"  {u:12s} <-> {v:12s}  weight={d.get('weight', 0):.1f}")
