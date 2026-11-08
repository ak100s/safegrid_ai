#!/usr/bin/env python3
"""
SafeGrid AI — End-to-End Evacuation Example

Demonstrates the complete pipeline:
  1. Parse an SVG floor plan from CubiCasa5k
  2. Build a topological navigation graph
  3. Inject a sample hazard (fire)
  4. Find the safest evacuation route via Dijkstra
  5. Print the results
  6. (Optional) Call LLM agent for natural language instructions
  7. Save a visualization of the graph and evacuation path

Usage:
    python examples/run_evacuation.py --svg path/to/model.svg
    python examples/run_evacuation.py --svg path/to/model.svg --llm
    python examples/run_evacuation.py --svg path/to/model.svg --output evac_plan.png

If no SVG is provided, a built-in demo graph is used.
"""

import argparse
import os
import sys
import time
import io

if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

# ─── Project Imports ──────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import (
    DEFAULT_HAZARD_RADIUS,
    HAZARD_PENALTY_ALPHA,
    HAZARD_DECAY_BETA,
    GOOGLE_API_KEY,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Demo Graph (used when no SVG is available)
# ═══════════════════════════════════════════════════════════════════════════════

def build_demo_graph():
    """
    Build a simple demonstration graph that doesn't require any external
    SVG files or the full parsing pipeline.

    Returns:
        tuple: (graph, node_positions, room_types)
    """
    import networkx as nx

    G = nx.Graph()

    # Nodes: (name, type, x, y)
    node_defs = [
        ("Bedroom_1",    "room",  100, 100),
        ("Corridor_1",   "room",  250, 150),
        ("Living_Room",  "room",  400, 100),
        ("Kitchen",      "room",  400, 300),
        ("Bathroom",     "room",  100, 300),
        ("Door_A",       "door",  175, 125),
        ("Door_B",       "door",  325, 125),
        ("Door_C",       "door",  400, 200),
        ("Door_D",       "door",  175, 250),
        ("Door_E",       "door",  250, 300),
        ("Exit_Front",   "exit",  500, 100),
        ("Exit_Back",    "exit",  500, 300),
    ]

    positions = {}
    room_types = {}
    for name, ntype, x, y in node_defs:
        G.add_node(name, type=ntype, x=x, y=y)
        positions[name] = (x, y)
        room_types[name] = ntype

    # Edges with base weights (traversal cost ~ distance / 100)
    edges = [
        ("Bedroom_1", "Door_A", 1.2),
        ("Door_A", "Corridor_1", 0.8),
        ("Corridor_1", "Door_B", 0.9),
        ("Door_B", "Living_Room", 1.0),
        ("Living_Room", "Door_C", 0.6),
        ("Door_C", "Kitchen", 1.1),
        ("Corridor_1", "Door_D", 1.0),
        ("Door_D", "Bathroom", 0.9),
        ("Bathroom", "Door_E", 0.8),
        ("Door_E", "Kitchen", 1.3),
        ("Living_Room", "Exit_Front", 0.4),
        ("Kitchen", "Exit_Back", 0.5),
    ]

    for u, v, w in edges:
        G.add_edge(u, v, weight=w)

    return G, positions, room_types


# ═══════════════════════════════════════════════════════════════════════════════
# Hazard Injection
# ═══════════════════════════════════════════════════════════════════════════════

def inject_hazard(G, hazard_x, hazard_y, radius=None, alpha=None, beta=None):
    """
    Inject a hazard into the graph by modifying edge weights based on
    proximity to the hazard center.

    Uses the exponential penalty model:
        multiplier = alpha * exp(-beta * distance)

    Nodes within the danger zone (radius * 0.3) are marked as blocked.

    Args:
        G: NetworkX graph with node attributes 'x' and 'y'.
        hazard_x: X coordinate of hazard center.
        hazard_y: Y coordinate of hazard center.
        radius: Hazard effect radius (SVG units).
        alpha: Penalty multiplier amplitude.
        beta: Exponential decay rate.

    Returns:
        tuple: (modified_graph, blocked_nodes, hazard_info)
    """
    import math

    radius = radius or DEFAULT_HAZARD_RADIUS
    alpha = alpha or HAZARD_PENALTY_ALPHA
    beta = beta or HAZARD_DECAY_BETA
    danger_zone = radius * 0.3

    blocked_nodes = set()
    hazard_info = {
        "x": hazard_x,
        "y": hazard_y,
        "radius": radius,
        "type": "fire",
        "danger_zone": danger_zone,
    }

    # Check each node's distance to hazard
    for node in G.nodes():
        nx_attr = G.nodes[node]
        node_x = nx_attr.get("x", 0)
        node_y = nx_attr.get("y", 0)
        dist = math.sqrt((node_x - hazard_x) ** 2 + (node_y - hazard_y) ** 2)

        if dist <= danger_zone:
            blocked_nodes.add(node)

    # Update edge weights
    for u, v, data in G.edges(data=True):
        u_attr = G.nodes[u]
        v_attr = G.nodes[v]

        # Use midpoint of edge for distance calculation
        mid_x = (u_attr.get("x", 0) + v_attr.get("x", 0)) / 2
        mid_y = (u_attr.get("y", 0) + v_attr.get("y", 0)) / 2
        dist = math.sqrt((mid_x - hazard_x) ** 2 + (mid_y - hazard_y) ** 2)

        if u in blocked_nodes or v in blocked_nodes:
            data["weight"] = float("inf")
        elif dist <= radius:
            penalty = alpha * math.exp(-beta * dist)
            data["weight"] = data.get("weight", 1.0) * (1.0 + penalty)

    return G, blocked_nodes, hazard_info


# ═══════════════════════════════════════════════════════════════════════════════
# Pathfinding
# ═══════════════════════════════════════════════════════════════════════════════

def find_safest_route(G, start_node, blocked_nodes=None):
    """
    Find the safest evacuation route from start_node to the nearest exit
    using Dijkstra's algorithm.

    Args:
        G: NetworkX graph with 'weight' edge attributes and 'type' node
           attributes.
        start_node: Starting node name.
        blocked_nodes: Set of nodes to exclude from pathfinding.

    Returns:
        dict: {
            "path": list of node names,
            "cost": total path cost,
            "exit_node": name of the exit reached,
            "success": bool,
            "error": error message if failed,
        }
    """
    import networkx as nx

    blocked_nodes = blocked_nodes or set()

    # Find all exit nodes
    exit_nodes = [
        n for n, d in G.nodes(data=True)
        if d.get("type") == "exit" and n not in blocked_nodes
    ]

    if not exit_nodes:
        return {
            "path": [],
            "cost": float("inf"),
            "exit_node": None,
            "success": False,
            "error": "No reachable exit nodes found.",
        }

    if start_node in blocked_nodes:
        return {
            "path": [],
            "cost": float("inf"),
            "exit_node": None,
            "success": False,
            "error": f"Start node '{start_node}' is in the danger zone.",
        }

    # Build subgraph excluding blocked nodes
    safe_nodes = [n for n in G.nodes() if n not in blocked_nodes]
    H = G.subgraph(safe_nodes).copy()

    best_path = None
    best_cost = float("inf")
    best_exit = None

    for exit_node in exit_nodes:
        try:
            path = nx.dijkstra_path(H, start_node, exit_node, weight="weight")
            cost = nx.dijkstra_path_length(H, start_node, exit_node, weight="weight")
            if cost < best_cost:
                best_path = path
                best_cost = cost
                best_exit = exit_node
        except nx.NetworkXNoPath:
            continue

    if best_path is None:
        return {
            "path": [],
            "cost": float("inf"),
            "exit_node": None,
            "success": False,
            "error": "No viable path to any exit.",
        }

    return {
        "path": best_path,
        "cost": best_cost,
        "exit_node": best_exit,
        "success": True,
        "error": None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════════

def visualize_evacuation(G, positions, room_types, route_result, hazard_info,
                         blocked_nodes, output_path):
    """
    Save a visualization of the navigation graph with the evacuation path
    highlighted and the hazard zone shown.

    Args:
        G: NetworkX graph.
        positions: Dict mapping node name -> (x, y).
        room_types: Dict mapping node name -> type string.
        route_result: Result dict from find_safest_route().
        hazard_info: Hazard info dict from inject_hazard().
        blocked_nodes: Set of blocked node names.
        output_path: File path to save the image.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import networkx as nx

    fig, ax = plt.subplots(figsize=(10, 8))

    # Categorize nodes
    rooms = [n for n in G.nodes() if room_types.get(n) == "room"]
    doors = [n for n in G.nodes() if room_types.get(n) == "door"]
    exits = [n for n in G.nodes() if room_types.get(n) == "exit"]

    # Draw hazard zone
    if hazard_info:
        hz = plt.Circle(
            (hazard_info["x"], hazard_info["y"]),
            hazard_info["radius"] * 0.5,  # Scale for display
            color="red", alpha=0.12, zorder=1,
        )
        ax.add_patch(hz)
        hz_border = plt.Circle(
            (hazard_info["x"], hazard_info["y"]),
            hazard_info["radius"] * 0.5,
            color="red", alpha=0.4, fill=False,
            linestyle="--", linewidth=2, zorder=2,
        )
        ax.add_patch(hz_border)
        ax.text(
            hazard_info["x"], hazard_info["y"],
            "🔥", ha="center", va="center", fontsize=20, zorder=10,
        )

    # Draw all edges (gray)
    nx.draw_networkx_edges(
        G, positions, edge_color="#B0BEC5", width=1.5, alpha=0.5, ax=ax,
    )

    # Highlight evacuation path
    if route_result["success"]:
        path = route_result["path"]
        path_edges = list(zip(path[:-1], path[1:]))
        # Match edge direction
        draw_edges = []
        for u, v in path_edges:
            if G.has_edge(u, v):
                draw_edges.append((u, v))
            elif G.has_edge(v, u):
                draw_edges.append((v, u))
        nx.draw_networkx_edges(
            G, positions, edgelist=draw_edges,
            edge_color="#EAB308", width=4, alpha=0.9, ax=ax,
        )

    # Draw nodes
    # Blocked nodes in dark red
    blocked_list = [n for n in G.nodes() if n in blocked_nodes]
    normal_rooms = [n for n in rooms if n not in blocked_nodes]

    nx.draw_networkx_nodes(
        G, positions, nodelist=normal_rooms,
        node_color="#2563EB", node_size=600,
        edgecolors="white", linewidths=2, ax=ax,
    )
    nx.draw_networkx_nodes(
        G, positions, nodelist=doors,
        node_color="#16A34A", node_size=350,
        node_shape="s", edgecolors="white", linewidths=1.5, ax=ax,
    )
    nx.draw_networkx_nodes(
        G, positions, nodelist=exits,
        node_color="#DC2626", node_size=700,
        node_shape="^", edgecolors="white", linewidths=2, ax=ax,
    )
    if blocked_list:
        nx.draw_networkx_nodes(
            G, positions, nodelist=blocked_list,
            node_color="#7F1D1D", node_size=500,
            edgecolors="white", linewidths=2, alpha=0.6, ax=ax,
        )

    # Labels
    nx.draw_networkx_labels(
        G, positions, font_size=7, font_weight="bold",
        font_color="white", ax=ax,
    )

    # Edge weights
    edge_labels = nx.get_edge_attributes(G, "weight")
    # Filter out inf weights for display
    edge_labels = {k: f"{v:.1f}" for k, v in edge_labels.items() if v != float("inf")}
    try:
        nx.draw_networkx_edge_labels(
            G, positions, edge_labels=edge_labels,
            font_size=7, font_color="#334155",
            bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.7),
            ax=ax,
        )
    except Exception:
        pass

    # Legend
    legend_handles = [
        mpatches.Patch(color="#2563EB", label="Room"),
        mpatches.Patch(color="#16A34A", label="Door"),
        mpatches.Patch(color="#DC2626", label="Exit"),
        mpatches.Patch(color="#EAB308", label="Evacuation Path"),
        mpatches.Patch(color="red", alpha=0.3, label="Hazard Zone"),
    ]
    if blocked_list:
        legend_handles.append(
            mpatches.Patch(color="#7F1D1D", alpha=0.6, label="Blocked (Danger Zone)")
        )
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9)

    status = "SAFE ROUTE FOUND" if route_result["success"] else "NO SAFE ROUTE"
    ax.set_title(
        f"SafeGrid AI — Evacuation Plan [{status}]",
        fontsize=14, fontweight="bold",
    )
    ax.axis("off")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [OK] Visualization saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# LLM Agent (Optional)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_llm_instructions(route_result, hazard_info):
    """
    Use a Gemini LLM to generate natural-language evacuation instructions.

    Requires GOOGLE_API_KEY to be set in the environment.

    Args:
        route_result: Result dict from find_safest_route().
        hazard_info: Hazard info dict.

    Returns:
        str: Natural-language evacuation instructions, or a fallback message.
    """
    if not GOOGLE_API_KEY:
        print("  ⚠ GOOGLE_API_KEY not set — skipping LLM instruction generation.")
        return _fallback_instructions(route_result, hazard_info)

    try:
        import google.generativeai as genai
        from config import GEMINI_MODEL

        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL)

        prompt = (
            "You are an emergency evacuation coordinator. Based on the following "
            "data, provide clear step-by-step evacuation instructions for a person "
            "at the start location.\n\n"
            f"Route: {' -> '.join(route_result['path'])}\n"
            f"Total traversal cost: {route_result['cost']:.2f}\n"
            f"Exit: {route_result['exit_node']}\n"
            f"Hazard: {hazard_info['type']} at ({hazard_info['x']}, {hazard_info['y']}) "
            f"with radius {hazard_info['radius']}\n\n"
            "Provide 3-5 concise, actionable instructions. Include directional "
            "guidance and safety warnings about the hazard."
        )

        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        print(f"  ⚠ LLM call failed: {e}")
        return _fallback_instructions(route_result, hazard_info)


def _fallback_instructions(route_result, hazard_info):
    """Generate simple template-based instructions when LLM is unavailable."""
    if not route_result["success"]:
        return "⚠ ALERT: No safe evacuation route found. Shelter in place and await rescue."

    path = route_result["path"]
    lines = [
        "═══ EVACUATION INSTRUCTIONS ═══",
        "",
        f"⚠ HAZARD DETECTED: {hazard_info['type'].upper()} at "
        f"({hazard_info['x']}, {hazard_info['y']})",
        f"  Danger radius: {hazard_info['radius']} units",
        "",
        "ROUTE:",
    ]
    for i, node in enumerate(path):
        prefix = "  START -> " if i == 0 else "  EXIT  -> " if i == len(path) - 1 else "        -> "
        lines.append(f"{prefix}{node}")

    lines.extend([
        "",
        f"Total cost: {route_result['cost']:.2f}",
        f"Number of waypoints: {len(path)}",
        "",
        "SAFETY REMINDERS:",
        "  • Stay low if smoke is present",
        "  • Do not use elevators",
        "  • Proceed calmly but quickly",
        "  • Help others if safe to do so",
    ])
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# SVG Pipeline (when real SVG is provided)
# ═══════════════════════════════════════════════════════════════════════════════

def run_svg_pipeline(svg_path):
    """
    Run the full pipeline using actual SVG parsing modules.

    Falls back to the demo graph if modules are not available.

    Args:
        svg_path: Path to an SVG floor plan file.

    Returns:
        tuple: (graph, positions, room_types) or None if parsing fails.
    """
    try:
        from src.svg_parser import SVGFloorPlanParser
        from src.graph_builder import GraphBuilder

        print(f"\n  Parsing SVG: {svg_path}")
        t0 = time.time()
        parser = SVGFloorPlanParser()
        spatial_data = parser.parse(svg_path)
        parse_time = time.time() - t0
        print(f"  [OK] SVG parsed in {parse_time:.3f}s")

        print("  Building navigation graph...")
        t0 = time.time()
        builder = GraphBuilder()
        graph = builder.build_graph(spatial_data)
        build_time = time.time() - t0
        print(f"  [OK] Graph built in {build_time:.3f}s")
        print(f"    Nodes: {graph.number_of_nodes()}, "
              f"Edges: {graph.number_of_edges()}")

        # Extract positions and types
        positions = {}
        room_types = {}
        for node, data in graph.nodes(data=True):
            cx = data.get("centroid_x", data.get("centroid", (0, 0))[0])
            cy = data.get("centroid_y", data.get("centroid", (0, 0))[1])
            data["x"] = cx
            data["y"] = cy
            ntype = "exit" if data.get("is_exit") else data.get("node_type", "room")
            data["type"] = ntype
            positions[node] = (cx, cy)
            room_types[node] = ntype

        return graph, positions, room_types

    except ImportError as e:
        print(f"\n  ⚠ SVG pipeline modules not available: {e}")
        print("    Falling back to demo graph.")
        return None
    except Exception as e:
        print(f"\n  [FAIL] SVG parsing failed: {e}")
        print("    Falling back to demo graph.")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="SafeGrid AI — Evacuation Route Planner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run_evacuation.py\n"
            "  python run_evacuation.py --svg path/to/model.svg\n"
            "  python run_evacuation.py --svg path/to/model.svg --llm\n"
            "  python run_evacuation.py --start Bedroom_1 --output plan.png\n"
        ),
    )
    parser.add_argument(
        "--svg", type=str, default=None,
        help="Path to a CubiCasa5k model.svg file. If omitted, uses a demo graph.",
    )
    parser.add_argument(
        "--start", type=str, default=None,
        help="Name of the starting node (default: first room node).",
    )
    parser.add_argument(
        "--hazard-x", type=float, default=400.0,
        help="X coordinate of the hazard center (default: 400).",
    )
    parser.add_argument(
        "--hazard-y", type=float, default=300.0,
        help="Y coordinate of the hazard center (default: 300).",
    )
    parser.add_argument(
        "--hazard-radius", type=float, default=None,
        help=f"Hazard effect radius (default: {DEFAULT_HAZARD_RADIUS}).",
    )
    parser.add_argument(
        "--llm", action="store_true",
        help="Use Gemini LLM for natural-language instructions.",
    )
    parser.add_argument(
        "--output", type=str, default="evacuation_plan.png",
        help="Output path for the visualization image (default: evacuation_plan.png).",
    )
    return parser.parse_args()


def main():
    """Run the end-to-end evacuation planning pipeline."""
    args = parse_args()

    print("=" * 60)
    print("  SafeGrid AI — Evacuation Route Planner")
    print("=" * 60)

    # ── Step 1-2: Parse SVG or use demo graph ────────────────────────────
    if args.svg:
        result = run_svg_pipeline(args.svg)
        if result is None:
            print("\n  Using built-in demo graph instead.")
            G, positions, room_types = build_demo_graph()
        else:
            G, positions, room_types = result
    else:
        print("\n  No SVG provided — using built-in demo graph.")
        G, positions, room_types = build_demo_graph()

    print(f"\n  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # ── Step 3: Determine start node ─────────────────────────────────────
    if args.start:
        start_node = args.start
    else:
        # Pick the first room node
        room_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "room"]
        start_node = room_nodes[0] if room_nodes else list(G.nodes())[0]

    print(f"  Start node: {start_node}")

    # ── Step 4: Inject hazard ────────────────────────────────────────────
    print(f"\n  Injecting hazard at ({args.hazard_x}, {args.hazard_y})...")
    G, blocked_nodes, hazard_info = inject_hazard(
        G, args.hazard_x, args.hazard_y, radius=args.hazard_radius,
    )
    print(f"  [OK] Hazard injected: {hazard_info['type']}")
    if blocked_nodes:
        print(f"    Blocked nodes (danger zone): {', '.join(sorted(blocked_nodes))}")
    else:
        print("    No nodes in immediate danger zone.")

    # ── Step 5: Find safest route ────────────────────────────────────────
    print("\n  Finding safest evacuation route (Dijkstra)...")
    t0 = time.time()
    route_result = find_safest_route(G, start_node, blocked_nodes)
    route_time = time.time() - t0

    if route_result["success"]:
        print(f"  [OK] Route found in {route_time:.4f}s")
        print(f"    Path: {' -> '.join(route_result['path'])}")
        print(f"    Total cost: {route_result['cost']:.2f}")
        print(f"    Exit: {route_result['exit_node']}")
    else:
        print(f"  [FAIL] {route_result['error']}")

    # ── Step 6: Print results ────────────────────────────────────────────
    print("\n" + "─" * 60)
    if args.llm or GOOGLE_API_KEY:
        print("\n  Generating LLM-powered instructions...")
        instructions = generate_llm_instructions(route_result, hazard_info)
    else:
        instructions = _fallback_instructions(route_result, hazard_info)

    print()
    print(instructions)
    print()

    # ── Step 7: Save visualization ───────────────────────────────────────
    print("─" * 60)
    print(f"  Saving visualization to: {args.output}")
    visualize_evacuation(
        G, positions, room_types, route_result, hazard_info,
        blocked_nodes, args.output,
    )

    print("\n" + "=" * 60)
    print("  Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
