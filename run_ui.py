from modules.ui.app import run_ui
import yaml 

with open("parameters.yaml", "r") as f:
    options = yaml.safe_load(f)
params = options["ui"]
run_ui(params.pop("database_path"))