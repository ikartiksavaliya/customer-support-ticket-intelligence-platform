import json
import re
from pathlib import Path

notebooks_dir = Path("/home/ikartiksavaliya/Desktop/Portfolio projects/DL/customer-support-ticket-intelligence-platform/notebooks")

bootstrap_code = [
    "# ==============================================================================\n",
    "# ENVIRONMENT SETUP & DEVICE CONFIGURATION BOOTSTRAP CELL\n",
    "# ==============================================================================\n",
    "import os\n",
    "import sys\n",
    "from pathlib import Path\n",
    "\n",
    "# 1. Detect Environment\n",
    'IS_COLAB = "google.colab" in sys.modules\n',
    "\n",
    "if IS_COLAB:\n",
    '    print("Detected Environment: Google Colab")\n',
    "    \n",
    "    # Mount Google Drive if requested (uncomment if needed)\n",
    "    # from google.colab import drive\n",
    "    # drive.mount('/content/drive')\n",
    "    \n",
    "    # Clone the repository if not present\n",
    '    repo_name = "customer-support-ticket-intelligence-platform"\n',
    '    repo_url = f"https://github.com/ikartiksavaliya/{repo_name}.git"\n',
    '    target_dir = f"/content/{repo_name}"\n',
    "    \n",
    "    if not os.path.exists(target_dir):\n",
    '        print(f"Cloning repository {repo_url}...")\n',
    "        import subprocess\n",
    '        subprocess.run(["git", "clone", repo_url, target_dir], check=True)\n',
    "    \n",
    "    # Change working directory to the repository root\n",
    "    os.chdir(target_dir)\n",
    "    \n",
    "    # Add project root to sys.path\n",
    "    if target_dir not in sys.path:\n",
    "        sys.path.insert(0, target_dir)\n",
    "        \n",
    "    # Install requirements\n",
    '    print("Installing requirements.txt and package in editable mode...")\n',
    "    import subprocess\n",
    '    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)\n',
    '    subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)\n',
    "else:\n",
    '    print("Detected Environment: Local Machine / VS Code")\n',
    "    # Add project root to sys.path by climbing up directories\n",
    "    ROOT = Path.cwd()\n",
    '    while ROOT != ROOT.parent and not (ROOT / "src").exists():\n',
    "        ROOT = ROOT.parent\n",
    "    if str(ROOT) not in sys.path:\n",
    "        sys.path.insert(0, str(ROOT))\n",
    "    os.chdir(str(ROOT))\n",
    "\n",
    "# 2. Verify Imports & Configure Device\n",
    "import torch\n",
    "try:\n",
    "    from src.utils import get_device, set_seed\n",
    "    from src.preprocessing import clean_text\n",
    "    from src.vocabulary import Vocabulary\n",
    "    from src.dataset import TicketDataset\n",
    '    print("✅ Project modules successfully imported!")\n',
    "except ImportError as e:\n",
    '    print(f"❌ Failed to import project modules: {e}")\n',
    "    raise e\n",
    "\n",
    "# Initialize seed for reproducibility\n",
    "set_seed(42)\n",
    "\n",
    "# Get compute device\n",
    "device, device_type, device_name = get_device()\n"
]

bootstrap_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": bootstrap_code
}

for nb_path in sorted(notebooks_dir.glob("*.ipynb")):
    print(f"Processing notebook: {nb_path.name}")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    new_cells = []
    
    # Track if we already inserted the bootstrap cell
    bootstrap_inserted = False
    
    # We will locate the first markdown cell and insert after it,
    # or just prepend if there are no markdown cells.
    first_markdown_idx = -1
    for idx, cell in enumerate(cells):
        if cell.get("cell_type") == "markdown":
            first_markdown_idx = idx
            break
            
    # Process cells
    for idx, cell in enumerate(cells):
        cell_type = cell.get("cell_type")
        source = cell.get("source", [])
        source_str = "".join(source)
        
        # 1. Skip Google Drive mount cells completely
        if cell_type == "code" and ("google.colab" in source_str or "drive.mount" in source_str):
            print(f"  -> Skipping Colab mount code cell {idx}")
            continue
            
        # 2. Clean up sys.path manipulation in code cells
        if cell_type == "code":
            lines = []
            for line in source:
                # Remove sys.path.insert / sys.path.append lines
                if "sys.path.insert" in line or "sys.path.append" in line:
                    continue
                # Replace src.datasets with src.dataset
                line = line.replace("src.datasets", "src.dataset")
                # Replace device = get_device() with device, _, _ = get_device()
                if "device = get_device()" in line:
                    line = line.replace("device = get_device()", "device, _, _ = get_device()")
                lines.append(line)
            cell["source"] = lines
            
        new_cells.append(cell)
        
        # If this is the first markdown cell, we insert the bootstrap cell right after it
        if idx == first_markdown_idx and not bootstrap_inserted:
            new_cells.append(bootstrap_cell)
            bootstrap_inserted = True
            print(f"  -> Inserted bootstrap cell after first markdown cell (idx {idx})")

    # If no markdown cell was found, prepend it at the beginning
    if not bootstrap_inserted:
        new_cells.insert(0, bootstrap_cell)
        print("  -> Prepended bootstrap cell at the beginning")
        
    nb["cells"] = new_cells
    
    # Save the updated notebook
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"  -> Saved {nb_path.name}")
