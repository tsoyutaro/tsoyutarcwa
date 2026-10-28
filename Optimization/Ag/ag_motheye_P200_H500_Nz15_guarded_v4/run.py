"""Run this one condition independently of any parent research project."""
from pathlib import Path
import os
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
if os.environ.get("MOTHEYE_DEPENDENCY_PATH"):
    sys.path.append(os.environ["MOTHEYE_DEPENDENCY_PATH"])
from optimizer.driver import main
if __name__ == "__main__":
    raise SystemExit(main(["--config",str(HERE/"config.json"),"--output-root",str(HERE/"results/hex_trial1_M20_Q512_guarded_v4"),*sys.argv[1:]]))
