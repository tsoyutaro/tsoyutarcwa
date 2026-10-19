"""Integration checks for the shared Au table and spectral material convention."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .. import reproduce
from studies.shared import gold_dispersion


class MaterialTests(unittest.TestCase):
    def run_spectrum(self, output, *extra):
        args = ["--study", "fig10b", "--profile", "quick", "--orders", "2",
                "--section-nm", "10", "--nfft", "128", "--output-dir", str(output), *extra]
        with redirect_stdout(io.StringIO()):
            self.assertEqual(reproduce.main(args), 0)
        return json.loads((output / "settings.json").read_text(encoding="utf-8"))

    def test_default_spectrum_uses_existing_shared_table_and_records_provenance(self):
        model = gold_dispersion.load_gold_csv(gold_dispersion.DEFAULT_GOLD_CSV)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "spectrum"
            metadata = self.run_spectrum(output, "--wavelengths", "500,810")
            material = metadata["gold_data"]
            self.assertEqual(Path(material["path"]), gold_dispersion.DEFAULT_GOLD_CSV)
            self.assertEqual(material["sha256"], hashlib.sha256(gold_dispersion.DEFAULT_GOLD_CSV.read_bytes()).hexdigest())
            self.assertTrue(material["shared_default"])
            self.assertEqual(len(metadata["computations"]), 4)
            for computation in metadata["computations"]:
                row = computation["row"]
                self.assertNotEqual(row["status"], "FAILED")
                nk = computation["geometry"]["n_metal"]
                self.assertAlmostEqual(abs(complex(nk["real"], nk["imag"]) ** 2 - model(row["wavelength_nm"])), 0, delta=1e-12)
            self.assertTrue((output / "fig10b.csv").is_file())
            self.assertTrue((output / "fig10b.png").is_file())

    def test_csv_override_uses_shared_epsilon_interpolation(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            csv_path = directory / "custom.csv"
            csv_path.write_text("wavelength_nm,n,k\n400,1.5,0.2\n800,0.5,3\n", encoding="utf-8")
            metadata = self.run_spectrum(directory / "spectrum", "--gold-csv", str(csv_path), "--wavelengths", "600")
            expected = .5 * (complex(1.5, .2) ** 2 + complex(.5, 3) ** 2)
            nk_interpolation = complex(1, 1.6) ** 2
            self.assertGreater(abs(expected - nk_interpolation), .1)
            self.assertFalse(metadata["gold_data"]["shared_default"])
            for computation in metadata["computations"]:
                nk = computation["geometry"]["n_metal"]
                self.assertAlmostEqual(abs(complex(nk["real"], nk["imag"]) ** 2 - expected), 0, delta=1e-12)

    def test_out_of_range_spectrum_fails_before_creating_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "not_created"
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    reproduce.main(["--study", "all", "--wavelengths", "100", "--output-dir", str(output)])
            self.assertEqual(error.exception.code, 2)
            self.assertFalse(output.exists())

    def test_all_dispatches_spectrum_without_csv_argument(self):
        calls = []

        class Recorder:
            def __init__(self, args):
                self.out, self.failed, self.records = args.output_dir, 0, []
                self.csv = args.gold_data

            def save_metadata(self):
                pass

        for study in ("fig5", "fig6", "fig9a", "fig9b", "fig10a", "fig10b"):
            setattr(Recorder, study, lambda self, name=study: calls.append((name, self.csv)))
        with patch.object(reproduce, "Runner", Recorder), redirect_stdout(io.StringIO()):
            self.assertEqual(reproduce.main(["--study", "all"]), 0)
        self.assertEqual([name for name, _ in calls], ["fig5", "fig6", "fig9a", "fig9b", "fig10a", "fig10b"])
        self.assertTrue(all(path == gold_dispersion.DEFAULT_GOLD_CSV for _, path in calls))


if __name__ == "__main__":
    unittest.main()
