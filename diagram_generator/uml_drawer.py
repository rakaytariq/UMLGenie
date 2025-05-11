import os

# Make sure Graphviz is visible inside Python even if the venv doesn't inherit PATH
graphviz_path = "C:\\Program Files\\Graphviz-12.2.1-win64\\bin"
if graphviz_path not in os.environ["PATH"]:
    os.environ["PATH"] += os.pathsep + graphviz_path
import json
from graphviz import Digraph


def draw_use_case(data, output_path="uml_use_case"):
    dot = Digraph(format="png")
    dot.attr(rankdir="LR")
    dot.attr("node", shape="box")
    


    # Draw actors
    for actor in data.get("actors", []):
        dot.node(actor, actor, shape="box")

    # Draw use cases (with ellipses)
    for uc in data.get("use_cases", []):
        dot.node(uc, uc, shape="ellipse")

    # Draw relationships
    for rel in data.get("relationships", []):
        a = rel["actor"]
        u = rel["use_case"]
        dot.edge(a, u)

    # Output directory
    output_file = f"{output_path}.png"
    try:
        dot.render(output_file, cleanup=True)
        print(f"✅ Diagram saved as {output_file}")
    except Exception as e:
        print(f"❌ Failed to draw diagram: {e}")

