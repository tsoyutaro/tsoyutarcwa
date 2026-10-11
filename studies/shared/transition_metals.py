"""User-supplied optical constants and defaults shared by both study families."""
from pathlib import Path
import json
from studies.shared.tabulated_nk import load_tabulated_nk

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parent / "transition_metals_data"
FILES = {"Fe": "Fe.csv", "Ni": "Ni.csv", "Ti": "Ti.csv",
         "W": "Ordal(W).csv", "V": "Vanadium.csv"}
NAMES = {"Fe": "Iron", "Ni": "Nickel", "Ti": "Titanium",
         "W": "Tungsten", "V": "Vanadium"}


def material_symbol(value):
    aliases = {key.lower(): key for key in FILES}
    aliases.update({name.lower(): key for key, name in NAMES.items()})
    aliases["nickle"] = "Ni"
    try:
        return aliases[value.strip().lower()]
    except KeyError:
        raise ValueError("Material must be Fe, Ni, Ti, W or V.") from None


def select_material(value, override=None):
    symbol = material_symbol(value)
    path = Path(override).resolve() if override is not None else DATA / FILES[symbol]
    return symbol, path, load_tabulated_nk(path)


def default_wavelengths(symbol):
    # The supplied W/Ordal table starts at 667 nm. No visible-band extrapolation.
    return ("667,680,700", "667:700:5") if symbol == "W" else ("400,550,700", "400:700:5")


def provenance(symbol, path):
    saved = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    info = dict(saved["materials"][symbol]) if path.resolve() == (DATA / FILES[symbol]).resolve() else {
        "material": symbol, "source": "user-selected --metal-csv", "filename": path.name}
    # Actual file hash is recorded separately by each controller.
    return info
