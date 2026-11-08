"""
SafeGrid AI — CubiCasa5k SVG Floor Plan Parser

Parses `model.svg` files from the CubiCasa5k dataset, extracting spatial
elements (rooms, doors, windows, exits) with their polygonal boundaries and
computed centroids.  The output is a structured dictionary consumed by
`graph_builder.py` for NetworkX graph construction.

Typical SVG hierarchy:
    <svg> → <g id="Model"> → <g class="Floor"> → <g class="Floorplan Floor-1">
        ├── <g class="Space ...">        (rooms / corridors / closets)
        │       └── <polygon points="...">
        ├── <g class="Wall ...">         (walls, may be "Wall External")
        │       ├── <polygon points="...">
        │       ├── <g id="Door" ...>    (doors embedded in walls)
        │       │       └── <g id="Threshold"> → <polygon>
        │       └── <g id="Window" ...>  (windows embedded in walls)
        └── <g class="SpaceDimensionsLabel">
                └── <g id="NameLabel"> → <text>  (room labels)
"""

from __future__ import annotations

import logging
import math
import os
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from shapely.geometry import Polygon as ShapelyPolygon

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SVG namespace
# ---------------------------------------------------------------------------
SVG_NS = "http://www.w3.org/2000/svg"
NS = {"svg": SVG_NS}


# ---------------------------------------------------------------------------
# Data classes for parsed elements
# ---------------------------------------------------------------------------
@dataclass
class ParsedElement:
    """A single spatial element extracted from the SVG."""

    id: str
    type: str
    label: str
    polygon_points: List[Tuple[float, float]]
    centroid: Tuple[float, float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "label": self.label,
            "polygon_points": self.polygon_points,
            "centroid": self.centroid,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def parse_polygon_points(points_str: str) -> List[Tuple[float, float]]:
    """Parse an SVG polygon ``points`` attribute into a list of (x, y) tuples.

    Handles both comma-separated (``"x,y x,y"``) and space-only formats.
    """
    coords: List[Tuple[float, float]] = []
    # Split on whitespace to get individual "x,y" tokens
    for token in points_str.strip().split():
        parts = token.split(",")
        if len(parts) == 2:
            try:
                coords.append((float(parts[0]), float(parts[1])))
            except ValueError:
                logger.warning("Skipping malformed coordinate token: %s", token)
    return coords


def parse_matrix_transform(transform_str: str) -> Optional[Tuple[float, ...]]:
    """Extract (a, b, c, d, e, f) from ``matrix(a,b,c,d,e,f)``."""
    match = re.search(
        r"matrix\(\s*"
        r"([^,]+),\s*([^,]+),\s*([^,]+),\s*([^,]+),\s*([^,]+),\s*([^)]+)\s*\)",
        transform_str,
    )
    if match:
        return tuple(float(g) for g in match.groups())
    return None


def apply_matrix_transform(
    points: List[Tuple[float, float]],
    matrix: Tuple[float, ...],
) -> List[Tuple[float, float]]:
    """Apply an affine matrix transform to a list of points.

    The SVG matrix ``(a, b, c, d, e, f)`` maps::

        x' = a*x + c*y + e
        y' = b*x + d*y + f
    """
    a, b, c, d, e, f = matrix
    return [(a * x + c * y + e, b * x + d * y + f) for x, y in points]


def compute_centroid(points: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Compute the centroid of a polygon using Shapely.

    Falls back to the arithmetic mean if the polygon is degenerate.
    """
    if len(points) < 3:
        # Degenerate — just average
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    try:
        poly = ShapelyPolygon(points)
        if not poly.is_valid:
            poly = poly.buffer(0)  # attempt repair
        c = poly.centroid
        return (c.x, c.y)
    except Exception:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return (sum(xs) / len(xs), sum(ys) / len(ys))


def _get_class(elem: ET.Element) -> str:
    """Return the ``class`` attribute or empty string."""
    return elem.get("class", "")


def _class_contains(elem: ET.Element, token: str) -> bool:
    """Check whether *token* appears in the element's class list."""
    return token in _get_class(elem).split()


def _collect_ancestor_transform(
    elem: ET.Element,
    parent_map: Dict[ET.Element, ET.Element],
) -> Optional[Tuple[float, ...]]:
    """Walk up to the nearest ancestor with a ``matrix(...)`` transform.

    Returns the composed matrix (only the first one found for simplicity,
    which covers CubiCasa5k usage).
    """
    current = elem
    while current is not None:
        t = current.get("transform", "")
        if "matrix" in t:
            m = parse_matrix_transform(t)
            if m:
                return m
        current = parent_map.get(current)
    return None


# ---------------------------------------------------------------------------
# Core parser
# ---------------------------------------------------------------------------

class CubiCasaSVGParser:
    """Parse a CubiCasa5k ``model.svg`` file into structured spatial data.

    Usage::

        parser = CubiCasaSVGParser(path_to_svg)
        result = parser.parse()
        # result["spaces"], result["doors"], result["exits"]
    """

    def __init__(self, svg_path: Optional[str | Path] = None) -> None:
        self.svg_path = Path(svg_path) if svg_path is not None else None
        if self.svg_path is not None and not self.svg_path.exists():
            raise FileNotFoundError(f"SVG file not found: {self.svg_path}")

        self._tree: Optional[ET.ElementTree] = None
        self._root: Optional[ET.Element] = None
        self._parent_map: Dict[ET.Element, ET.Element] = {}

        # Counters for ID generation
        self._room_counter = 0
        self._door_counter = 0
        self._exit_counter = 0
        self._window_counter = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, svg_path: Optional[str | Path] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Parse the SVG and return a structured dict.

        Args:
            svg_path: Optional path to the SVG. If provided, overrides the path
                      supplied to ``__init__``.

        Returns:
            A dict with keys ``"spaces"``, ``"doors"``, ``"exits"``,
            ``"windows"``.  Each value is a list of element dicts.
        """
        if svg_path is not None:
            self.svg_path = Path(svg_path)
        if self.svg_path is None:
            raise ValueError("No SVG path specified to parse.")
        if not self.svg_path.exists():
            raise FileNotFoundError(f"SVG file not found: {self.svg_path}")

        logger.info("Parsing SVG: %s", self.svg_path)
        self._tree = ET.parse(str(self.svg_path))
        self._root = self._tree.getroot()
        self._parent_map = {child: parent for parent in self._root.iter() for child in parent}

        # Find the first visible Floorplan group (Floor-1)
        floorplan = self._find_floorplan()
        if floorplan is None:
            logger.error("No Floorplan group found in SVG.")
            return {"spaces": [], "doors": [], "exits": [], "windows": []}

        spaces = self._extract_spaces(floorplan)
        doors, exits, windows = self._extract_wall_elements(floorplan)

        logger.info(
            "Parsed %d spaces, %d doors, %d exits, %d windows",
            len(spaces), len(doors), len(exits), len(windows),
        )

        return {
            "spaces": [s.to_dict() for s in spaces],
            "doors": [d.to_dict() for d in doors],
            "exits": [e.to_dict() for e in exits],
            "windows": [w.to_dict() for w in windows],
        }

    # ------------------------------------------------------------------
    # Floorplan discovery
    # ------------------------------------------------------------------

    def _find_floorplan(self) -> Optional[ET.Element]:
        """Locate the first *visible* ``Floorplan Floor-*`` group."""
        assert self._root is not None

        # Walk all <g> elements looking for class="Floorplan Floor-*"
        for g in self._root.iter(f"{{{SVG_NS}}}g"):
            cls = _get_class(g)
            if "Floorplan" in cls and "Floor-" in cls:
                # Skip floors that are hidden via style="display: none;"
                parent = self._parent_map.get(g)
                if parent is not None:
                    parent_style = parent.get("style", "")
                    if "display: none" in parent_style or "display:none" in parent_style:
                        continue
                return g
        return None

    # ------------------------------------------------------------------
    # Space extraction
    # ------------------------------------------------------------------

    def _extract_spaces(self, floorplan: ET.Element) -> List[ParsedElement]:
        """Extract all Space elements (rooms, corridors, closets, etc.)."""
        spaces: List[ParsedElement] = []

        for g in floorplan.iter(f"{{{SVG_NS}}}g"):
            cls = _get_class(g)
            if "Space" not in cls:
                continue

            # Parse the space type from class (e.g., "Space Closet WalkIn" → "Closet WalkIn")
            space_type = cls.replace("Space", "").strip()
            if not space_type:
                space_type = "Undefined"

            # Find the primary polygon
            polygon_elem = g.find(f"{{{SVG_NS}}}polygon")
            if polygon_elem is None:
                continue

            points_str = polygon_elem.get("points", "")
            points = parse_polygon_points(points_str)
            if len(points) < 3:
                logger.debug("Skipping space with <3 points: %s", space_type)
                continue

            # Extract room label from NameLabel text
            label = self._extract_label(g)

            centroid = compute_centroid(points)
            elem_id = f"Room_{self._room_counter}"
            self._room_counter += 1

            spaces.append(
                ParsedElement(
                    id=elem_id,
                    type=space_type,
                    label=label,
                    polygon_points=points,
                    centroid=centroid,
                    metadata={"svg_id": g.get("id", ""), "class": cls},
                )
            )

        return spaces

    def _extract_label(self, space_g: ET.Element) -> str:
        """Extract the room name label from a Space group's SpaceDimensionsLabel."""
        for g in space_g.iter(f"{{{SVG_NS}}}g"):
            cls = _get_class(g)
            if "SpaceDimensionsLabel" not in cls:
                continue
            # Look for the NameLabel text
            for inner_g in g.iter(f"{{{SVG_NS}}}g"):
                if "NameLabel" in _get_class(inner_g):
                    text_elem = inner_g.find(f"{{{SVG_NS}}}text")
                    if text_elem is not None and text_elem.text:
                        return text_elem.text.strip()
        return ""

    # ------------------------------------------------------------------
    # Wall / Door / Window / Exit extraction
    # ------------------------------------------------------------------

    def _extract_wall_elements(
        self,
        floorplan: ET.Element,
    ) -> Tuple[List[ParsedElement], List[ParsedElement], List[ParsedElement]]:
        """Extract doors, exits, and windows from Wall groups.

        Doors on External walls are classified as exits.
        """
        doors: List[ParsedElement] = []
        exits: List[ParsedElement] = []
        windows: List[ParsedElement] = []

        for g in floorplan.iter(f"{{{SVG_NS}}}g"):
            cls = _get_class(g)
            if not cls.startswith("Wall"):
                continue
            # Check for id="Wall" as an additional heuristic
            if g.get("id") != "Wall" and "Wall" not in cls.split():
                continue

            is_external = "External" in cls

            # --- Doors inside this wall ---
            for door_g in g.iter(f"{{{SVG_NS}}}g"):
                if door_g.get("id") != "Door":
                    continue
                door_cls = _get_class(door_g)
                if "Door" not in door_cls:
                    continue

                # Get polygon from the Threshold sub-group
                threshold_points = self._get_threshold_points(door_g)
                if not threshold_points or len(threshold_points) < 3:
                    continue

                centroid = compute_centroid(threshold_points)
                door_type = door_cls.replace("Door", "").strip() or "Swing"

                if is_external:
                    elem_id = f"Exit_{self._exit_counter}"
                    self._exit_counter += 1
                    exits.append(
                        ParsedElement(
                            id=elem_id,
                            type=f"Exit ({door_type})",
                            label="Exit",
                            polygon_points=threshold_points,
                            centroid=centroid,
                            metadata={
                                "wall_class": cls,
                                "is_external": True,
                                "door_class": door_cls,
                            },
                        )
                    )
                else:
                    elem_id = f"Door_{self._door_counter}"
                    self._door_counter += 1
                    doors.append(
                        ParsedElement(
                            id=elem_id,
                            type=f"Door ({door_type})",
                            label="Door",
                            polygon_points=threshold_points,
                            centroid=centroid,
                            metadata={
                                "wall_class": cls,
                                "is_external": False,
                                "door_class": door_cls,
                            },
                        )
                    )

            # --- Windows inside this wall ---
            for win_g in g.iter(f"{{{SVG_NS}}}g"):
                if win_g.get("id") != "Window":
                    continue
                win_cls = _get_class(win_g)
                if "Window" not in win_cls:
                    continue

                # The Window group itself contains a polygon
                poly_elem = win_g.find(f"{{{SVG_NS}}}polygon")
                if poly_elem is None:
                    continue
                points = parse_polygon_points(poly_elem.get("points", ""))
                if len(points) < 3:
                    continue

                centroid = compute_centroid(points)
                elem_id = f"Window_{self._window_counter}"
                self._window_counter += 1
                windows.append(
                    ParsedElement(
                        id=elem_id,
                        type=win_cls,
                        label="Window",
                        polygon_points=points,
                        centroid=centroid,
                        metadata={"wall_class": cls, "is_external": is_external},
                    )
                )

        return doors, exits, windows

    def _get_threshold_points(
        self,
        door_g: ET.Element,
    ) -> List[Tuple[float, float]]:
        """Extract polygon points from a Door's Threshold sub-group."""
        for g in door_g.iter(f"{{{SVG_NS}}}g"):
            if g.get("id") == "Threshold" or "Threshold" in _get_class(g):
                poly = g.find(f"{{{SVG_NS}}}polygon")
                if poly is not None:
                    pts = parse_polygon_points(poly.get("points", ""))
                    # Check for a transform on the Threshold group
                    transform = g.get("transform", "")
                    if "matrix" in transform:
                        m = parse_matrix_transform(transform)
                        if m:
                            pts = apply_matrix_transform(pts, m)
                    return pts
        return []


# Alias for alternate naming convention
SVGFloorPlanParser = CubiCasaSVGParser


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def parse_svg(svg_path: str | Path) -> Dict[str, List[Dict[str, Any]]]:
    """One-shot convenience function to parse a CubiCasa5k SVG.

    Args:
        svg_path: Path to the ``model.svg`` file.

    Returns:
        Structured dict with ``"spaces"``, ``"doors"``, ``"exits"``,
        ``"windows"`` lists.
    """
    parser = CubiCasaSVGParser(svg_path)
    return parser.parse()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    )

    # Resolve project paths
    _this_dir = Path(__file__).resolve().parent
    _project_root = _this_dir.parent.parent  # safegrid_ai/../
    _default_svg = _project_root / "cubicasa5k" / "high_quality_architectural" / "1" / "model.svg"

    svg_file = sys.argv[1] if len(sys.argv) > 1 else str(_default_svg)

    print(f"\n{'='*60}")
    print(f"  SafeGrid AI — SVG Parser")
    print(f"  File: {svg_file}")
    print(f"{'='*60}\n")

    result = parse_svg(svg_file)

    for category in ("spaces", "doors", "exits", "windows"):
        items = result[category]
        print(f"\n--- {category.upper()} ({len(items)}) ---")
        for item in items:
            print(
                f"  {item['id']:12s}  type={item['type']:25s}  "
                f"label={item['label']:12s}  centroid=({item['centroid'][0]:.1f}, {item['centroid'][1]:.1f})  "
                f"vertices={len(item['polygon_points'])}"
            )

    total = sum(len(result[k]) for k in result)
    print(f"\nTotal elements extracted: {total}")
