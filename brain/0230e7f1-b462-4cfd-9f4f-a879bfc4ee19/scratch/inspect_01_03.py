import json
from pathlib import Path

notebooks_dir = Path("/home/ikartiksavaliya/Desktop/Portfolio projects/DL/customer-support-ticket-intelligence-platform/notebooks")
for name in ["01_project_planning.ipynb", "02_eda.ipynb", "03_text_preprocessing.ipynb"]:
    nb_path = notebooks_dir / name
    print("=" * 60)
    print(f"Notebook: {name}")
    print("=" * 60)
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    cells = nb.get("cells", [])
    code_cell_idx = 0
    for idx, cell in enumerate(cells):
        if cell.get("cell_type") == "code":
            print(f"--- Code Cell {code_cell_idx} (Cell Index {idx}) ---")
            print("".join(cell.get("source", [])))
            code_cell_idx += 1
            if code_cell_idx >= 3:
                break
