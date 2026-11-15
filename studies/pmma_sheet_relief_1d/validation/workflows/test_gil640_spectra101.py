"""Control-flow tests; no RCWA solves or GPU allocations are performed."""
from pathlib import Path
import contextlib
import csv
import importlib.util
import io
import json
import tempfile
import unittest
from unittest.mock import patch

STUDY = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('gil640_workflow', STUDY/'automation/run_gil640_spectra101.py')
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.conv = self.root/'convergence'
        self.spectra = self.root/'spectra'
        self.out = self.root/'workflow'
        self.args = ['--convergence-dir', str(self.conv), '--spectrum-root', str(self.spectra),
                     '--output-dir', str(self.out)]
        self.calls = []

    def tearDown(self):
        self.temp.cleanup()

    def state(self):
        return json.loads((self.out/'workflow.json').read_text())

    def fake(self, mode='limit', spectrum_failure=None, nonpassive=False):
        def child(command, log):
            self.calls.append(command)
            log.write_text('completed\n', encoding='utf-8')
            if mode == 'interrupt':
                raise KeyboardInterrupt()
            if Path(command[1]).name == 'converge_gil.py':
                if mode == 'oom':
                    return 3
                cases = []
                for gil in workflow.GILS:
                    cases.append(dict(gil_nm=float(gil), gil_interfaces='none' if gil == '0' else 'both',
                        status='converged' if mode == 'converged' else 'limit_reached', max_order=640,
                        last_completed_order=352 if mode == 'converged' else 640))
                if mode == 'phase':
                    cases[0]['status'] = 'phase_unconverged'
                if mode == 'missing':
                    cases.pop()
                workflow.save_json(self.conv/'gil_summary.json', dict(cases=cases))
                return 0 if mode == 'converged' else 2
            gil = command[command.index('--gil-nm')+1]
            if spectrum_failure is not None and gil == spectrum_failure[0]:
                log.write_text(spectrum_failure[2], encoding='utf-8')
                return spectrum_failure[1]
            dest = Path(command[command.index('--output-dir')+1]); dest.mkdir(parents=True, exist_ok=True)
            rows = [dict(void=void, wavelength_nm=400+3*i, order=640, slices=960, polarization='p',
                gil_nm=float(gil), gil_interfaces='both', period_nm=3720, gold_vertical_nm=50,
                coating='gold', R=.701 if nonpassive else .5, T=.3,
                A_total=-.001 if nonpassive else .2, phase_converged=True,
                passivity_valid=not nonpassive) for void in ['air', 'water'] for i in range(101)]
            with (dest/'spectra.csv').open('w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            workflow.save_json(dest/'completion.json', dict(completed_cases=202, phase_converged=True, point_cache_hits=202))
            return 0
        return child

    def run_workflow(self, fake, extra=()):
        with patch.object(workflow.controller, 'run_child', fake), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return workflow.main([*self.args, *extra])

    def test_limit_exit_two_runs_all_four_spectra(self):
        self.assertEqual(self.run_workflow(self.fake()), 0)
        self.assertEqual(len(self.calls), 5)
        self.assertEqual(self.state()['status'], 'completed')
        self.assertEqual(self.state()['spectrum_cases'], 808)

    def test_early_convergence_still_uses_M640_for_spectra(self):
        self.assertEqual(self.run_workflow(self.fake('converged')), 0)
        for command in self.calls[1:]:
            self.assertEqual(command[command.index('--order')+1], '640')

    def test_convergence_oom_does_not_start_spectra(self):
        self.assertEqual(self.run_workflow(self.fake('oom')), 3)
        self.assertEqual(len(self.calls), 1)

    def test_aggregate_exit_two_with_phase_failure_does_not_start_spectra(self):
        self.assertEqual(self.run_workflow(self.fake('phase')), 4)
        self.assertEqual(len(self.calls), 1)

    def test_incomplete_gil_summary_does_not_start_spectra(self):
        self.assertEqual(self.run_workflow(self.fake('missing')), 4)
        self.assertEqual(len(self.calls), 1)

    def test_spectrum_oom_stops_before_remaining_gils(self):
        self.assertEqual(self.run_workflow(self.fake(spectrum_failure=('0.5', 1, 'torch.OutOfMemoryError: CUDA out of memory.'))), 3)
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(len(self.state()['completed_spectra']), 1)
        self.assertEqual(self.state()['status'], 'oom')

    def test_spectrum_phase_failure_stops_the_loop(self):
        self.assertEqual(self.run_workflow(self.fake(spectrum_failure=('0', 2, 'phase unconverged'))), 2)
        self.assertEqual(len(self.calls), 2)

    def test_nonpassive_values_are_retained_and_reported_after_all_gils(self):
        self.assertEqual(self.run_workflow(self.fake(nonpassive=True)), 2)
        self.assertEqual(len(self.calls), 5)
        self.assertEqual(self.state()['status'], 'completed_with_nonpassive_points')

    def test_dry_run_launches_nothing_and_the_spectrum_arguments_parse(self):
        self.assertEqual(self.run_workflow(self.fake(), ['--dry-run']), 0)
        self.assertFalse(self.calls)
        self.assertFalse(self.out.exists())
        from fff1d.run_fff import parser
        from model import parse_wavelengths
        first, following = workflow.commands(workflow.parser().parse_args(self.args))
        self.assertEqual(first[first.index('--max-order')+1], '640')
        for gil, dest, command in following:
            args = parser().parse_args(command[2:])
            self.assertEqual(len(parse_wavelengths(args.wavelengths)), 101)
            self.assertEqual(args.gil_interfaces, 'both')
            self.assertEqual(args.polarization, 'p')
            self.assertEqual(args.device, 'cuda:0')

    def test_interruption_is_recorded(self):
        self.assertEqual(self.run_workflow(self.fake('interrupt')), 130)
        self.assertEqual(self.state()['status'], 'interrupted')

    def test_second_workflow_does_not_overwrite_active_state(self):
        self.conv.mkdir(); self.out.mkdir()
        state_file = self.out/'workflow.json'; state_file.write_text('{"status":"running"}')
        with workflow.controller.campaign_lock(self.conv):
            self.assertEqual(self.run_workflow(self.fake()), 4)
        self.assertEqual(state_file.read_text(), '{"status":"running"}')
        self.assertFalse(self.calls)


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(WorkflowTests))
    output = Path(__file__).with_name('gil640_spectra101_tests.json')
    output.write_text(json.dumps(dict(passed=result.wasSuccessful(), tests=result.testsRun,
        failures=len(result.failures), errors=len(result.errors),
        control_tests_use_simulated_children=True, real_gpu_calculations_performed=False), indent=2)+'\n', encoding='utf-8')
    raise SystemExit(0 if result.wasSuccessful() else 1)
