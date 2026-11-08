"""
SafeGrid AI Configuration
"""
import os

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(os.path.dirname(PROJECT_ROOT), "cubicasa5k")
BENCHMARK_RESULTS_DIR = os.path.join(PROJECT_ROOT, "benchmarks", "results")

# ─── CubiCasa5k Categories ───────────────────────────────────────────────────
# Navigable space categories we extract from SVG
NAVIGABLE_SPACE_CLASSES = [
    "Room", "LivingRoom", "Bedroom", "Kitchen", "Dining",
    "Corridor", "Hallway", "Entrance",
    "Bath", "Shower", "Toilet",
    "Closet", "WalkIn", "Storage",
    "Sauna", "Garage", "Balcony", "Terrace",
    "Office", "Library", "Undefined",
]

# Door/opening types
DOOR_CLASSES = ["Door", "Swing", "Sliding", "Folding", "Revolving"]

# Exit indicators — doors on external walls
EXIT_INDICATORS = ["External", "Entrance", "Balcony", "Terrace", "Garage"]

# Wall types (for boundary detection)
WALL_CLASSES = ["Wall", "External", "Internal"]

# ─── Graph Construction ──────────────────────────────────────────────────────
# Minimum polygon overlap (in SVG units) to consider adjacency
ADJACENCY_BUFFER = 5.0  # pixels buffer for boundary-sharing detection

# ─── Hazard Engine ────────────────────────────────────────────────────────────
DEFAULT_HAZARD_RADIUS = 150.0       # SVG units
HAZARD_PENALTY_ALPHA = 5.0          # Exponential penalty multiplier
HAZARD_DECAY_BETA = 0.02            # Exponential decay rate

# ─── LLM Configuration ───────────────────────────────────────────────────────
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
GEMINI_MODEL = "gemini-2.0-flash"

# ─── Evaluation ───────────────────────────────────────────────────────────────
NUM_EVAL_BLUEPRINTS = 50            # Number of blueprints for batch eval
NUM_HAZARD_SCENARIOS = 10           # Hazard scenarios per blueprint
RANDOM_SEED = 42
