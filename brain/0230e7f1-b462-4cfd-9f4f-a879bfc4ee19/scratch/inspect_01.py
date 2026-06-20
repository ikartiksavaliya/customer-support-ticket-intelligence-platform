import json
from pathlib import Path

nb_path = Path("/home/ikartiksavaliya/Desktop/Portfolio projects/DL/customer-support-ticket-intelligence-platform/notebooks/01_project_planning.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)
cells = nb.get("cells", [])
for idx, cell in enumerate(cells[:5]):
    print(f"Cell {idx}: type={cell.get('cell_type')}")
    source = "".join(cell.get("source", []))
    print(source[:200])
    print("-" * 40)
