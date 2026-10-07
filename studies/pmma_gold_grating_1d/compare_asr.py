"""Run the shared Li/ASR comparison for this coated PMMA grating study."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from studies.asr_1d_comparison.compare import main

if __name__ == "__main__":
    main(default_study="pmma_gold_grating_1d")
