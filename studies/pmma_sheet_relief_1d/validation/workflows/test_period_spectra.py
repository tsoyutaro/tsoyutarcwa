"""Workflow requirements and failure/resume tests, without expensive RCWA solves."""
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
spec = importlib.util.spec_from_file_location('period_sweep', STUDY/'automation/run_period_spectra.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


class PeriodSweepTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.out = Path(self.temp.name)/'result'
        self.args = ['--periods-um','5','--slices','4','--start','16','--step','16',
                     '--max-order','80','--device','cpu','--output-dir',str(self.out)]
        self.env = dict(source_sha256='same-physics',device='cpu',torch='test',numpy='test')
        self.calls = []

    def tearDown(self):
        self.temp.cleanup()

    def cases(self):
        return json.loads((self.out/'summary.json').read_text())['cases']

    def fake(self, independent=False, ramp=False, interrupt=None, oom=None, spectrum_phase=False, nonpassive_order=None, power_lookup=None):
        def run(command, log):
            def value(flag): return command[command.index(flag)+1]
            order=int(value('--order'));void=value('--voids');period=float(value('--periods'))
            dest=Path(value('--output-dir')); spectrum=dest.name=='spectrum'
            self.calls.append((period,void,order,spectrum))
            log.write_text('starting\n',encoding='utf-8')
            if not spectrum and (void,order)==interrupt:
                raise KeyboardInterrupt()
            if not spectrum and (void,order)==oom:
                log.write_text('torch.OutOfMemoryError: CUDA out of memory.\n',encoding='utf-8')
                return 1
            if ramp:
                r=.1+.001*order
            elif independent and void=='water':
                r={16:.1,32:.12,48:.14,64:.1402,80:.1403}[order]
            else:
                r=.1+order*1e-6
            power = power_lookup[void][order] if power_lookup else (r,.8-r,.2)
            passive = spectrum or order!=nonpassive_order
            phase = not (spectrum and spectrum_phase)
            coating=value('--coating');gold=float(value('--gold-nm'))
            g=m.geometry(period=period,gold=gold if coating=='gold' else 0.)
            settings=dict(order=order,slices=int(value('--slices')),polarization=value('--polarization'),
                coating=coating,gil_nm=0.,gil_interfaces='both',geometry_mode='scaled',
                source_sha256='same-physics',phase_tol=float(value('--phase-tol')),
                phase_max=int(value('--phase-max')),phase_points=int(value('--phase-points')),
                geometries={f'{coating}/P{period:g}':g})
            waves=[float(w) for w in m.parse_wavelengths(value('--wavelengths'))]
            rows=[dict(void=void,wavelength_nm=w,period_nm=period,polarization=value('--polarization'),
                order=order,slices=int(value('--slices')),R=power[0],T=power[1],A_total=power[2],
                phase_error=1e-6,phase_converged=phase,passivity_valid=passive,
                R_0=.2+order/1000, R_le_20deg=.3+order/1000) for w in waves]
            m.save_json(dest/'settings.json',settings)
            m.save_json(dest/'completion.json',dict(completed_cases=len(rows),phase_converged=phase,passivity_valid=passive))
            m.save_csv(dest/'spectra.csv',rows)
            (dest/'diffraction_orders.csv').write_text(f'm,R_m,T_m\n0,{order/1000},0.5\n',encoding='utf-8')
            (dest/'local_single_pass_orders.csv').write_text('m,R,T\n0,0.1,0.9\n',encoding='utf-8')
            log.write_text('done\n',encoding='utf-8')
            return 2 if not phase else 0
        return run

    def run_sweep(self, fake, extra=()):
        with patch.object(m,'environment',return_value=self.env), patch.object(m.controller,'run_child',fake), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return m.main([*self.args,*extra])

    def test_independent_voids_select_third_last_and_all_101_points(self):
        self.assertEqual(self.run_sweep(self.fake(independent=True)),0)
        self.assertEqual([(x['void'],x['selected_order']) for x in self.cases()],[('air',32),('water',48)])
        for case in self.cases():
            with (Path(case['output_dir'])/'spectrum/spectra.csv').open(encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),101)
            self.assertEqual({float(r['wavelength_nm']) for r in rows},set(range(400,701,3)))

    def test_RTA_only_ignores_large_order_and_cone_changes(self):
        self.assertEqual(self.run_sweep(self.fake()),0)
        self.assertEqual([r['selected_order'] for r in self.cases()],[32,32])

    def test_P3720_three_existing_points_select_first_of_three(self):
        points={'air':{304:(.503507,.216829,.279664),320:(.502848,.217486,.279666),336:(.502451,.217402,.280147)},
                'water':{304:(.551063,.268707,.180230),320:(.550822,.268640,.180538),336:(.551162,.268427,.180411)}}
        self.assertEqual(self.run_sweep(self.fake(power_lookup=points),
            ['--periods-um','3.72','--start','304','--max-order','336']),0)
        self.assertEqual([r['selected_order'] for r in self.cases()],[304,304])

    def test_start320_selects320_after_two_comparisons_at352(self):
        self.assertEqual(self.run_sweep(self.fake(),['--start','320','--max-order','352']),0)
        for case in self.cases():
            self.assertEqual(case['checked_orders'],[320,336,352])
            self.assertEqual(case['selected_order'],320)
            self.assertEqual(case['convergence_final_order'],352)

    def test_large_period_waits_until_third_last_order_includes_spectrum_orders(self):
        self.assertEqual(self.run_sweep(self.fake(),
            ['--periods-um','100','--start','320','--max-order','416']),0)
        for case in self.cases():
            self.assertEqual(case['selected_order'],384)
            self.assertEqual(case['convergence_final_order'],416)
            self.assertEqual(case['checked_orders'],list(range(320,417,16)))

    def test_two_points_cannot_select_a_third_last_order(self):
        self.assertEqual(self.run_sweep(self.fake(),['--start','320','--max-order','336']),2)
        self.assertFalse(any(spectrum for _,_,_,spectrum in self.calls))

    def test_previous_selection_rule_output_is_rejected(self):
        self.assertEqual(self.run_sweep(self.fake()),0)
        manifest=self.out/'P5000/air/identity.json'
        previous=json.loads(manifest.read_text())
        previous['version']=1
        previous['selection']='penultimate computed order, provided all spectrum propagating orders are included'
        m.save_json(manifest,previous)
        self.calls.clear()
        self.assertEqual(self.run_sweep(self.fake()),4)
        self.assertEqual(self.calls,[])

    def test_extending_maximum_reuses_completed_convergence_points(self):
        self.assertEqual(self.run_sweep(self.fake(independent=True),['--max-order','48']),2)
        self.calls.clear()
        self.assertEqual(self.run_sweep(self.fake(independent=True),['--max-order','80']),0)
        self.assertFalse(any(not spectrum and order<=48 for _,_,order,spectrum in self.calls))

    def test_A_change_counts_even_when_R_and_T_changes_each_small(self):
        a=dict(R=.1,T=.7,A_total=.2);b=dict(R=.1008,T=.7008,A_total=.1984)
        self.assertGreater(m.maximum_rta_change(a,b),.001)

    def test_skip_incomplete_propagating_orders_and_wait_for_spectrum_floor(self):
        self.assertEqual(self.run_sweep(self.fake(),['--periods-um','20','--max-order','112']),0)
        for case in self.cases():
            self.assertEqual(case['checked_orders'],[48,64,80,96,112])
            self.assertEqual([r['order'] for r in case['skipped_orders']],[16,32])
            self.assertEqual(case['selected_order'],80)

    def test_limit_never_starts_unconverged_spectrum(self):
        self.assertEqual(self.run_sweep(self.fake(ramp=True),['--max-order','48']),2)
        self.assertTrue(all(r['status']=='limit_reached' for r in self.cases()))
        self.assertFalse(any(spectrum for _,_,_,spectrum in self.calls))

    def test_pass_below_spectrum_floor_does_not_select_invalid_order(self):
        self.assertEqual(self.run_sweep(self.fake(),['--periods-um','20','--max-order','64']),2)
        self.assertFalse(any(spectrum for _,_,_,spectrum in self.calls))

    def test_completed_resume_runs_no_children(self):
        self.assertEqual(self.run_sweep(self.fake()),0)
        self.calls.clear()
        self.assertEqual(self.run_sweep(self.fake()),0)
        self.assertEqual(self.calls,[])

    def test_interrupt_resume_reuses_completed_order(self):
        self.assertEqual(self.run_sweep(self.fake(interrupt=('air',32))),130)
        self.assertEqual(self.cases()[0]['status'],'interrupted')
        self.calls.clear()
        self.assertEqual(self.run_sweep(self.fake()),0)
        self.assertNotIn((5000.,'air',16,False),self.calls)

    def test_oom_stops_that_case_and_preserves_other_void(self):
        self.assertEqual(self.run_sweep(self.fake(oom=('air',32))),3)
        self.assertEqual([r['status'] for r in self.cases()],['oom','completed'])
        self.assertFalse(any(void=='air' and spectrum for _,void,_,spectrum in self.calls))

    def test_nonpassive_point_breaks_consecutive_passes(self):
        self.assertEqual(self.run_sweep(self.fake(nonpassive_order=32)),0)
        self.assertEqual([r['selected_order'] for r in self.cases()],[48,48])

    def test_phase_failure_is_not_completed_spectrum(self):
        self.assertEqual(self.run_sweep(self.fake(spectrum_phase=True)),4)
        self.assertTrue(all(r['status']=='phase_unconverged' for r in self.cases()))

    def test_changed_resolution_rejected_without_recalculating(self):
        self.run_sweep(self.fake());self.calls.clear()
        self.assertEqual(self.run_sweep(self.fake(),['--slices','8']),4)
        self.assertEqual(self.calls,[])

    def test_default_geometry_and_scan(self):
        args=m.parser().parse_args(['--dry-run'])
        cases,waves=m.plan(args)
        self.assertEqual(len(cases),22);self.assertEqual(len(waves),101)
        self.assertEqual((args.start,args.step,args.max_order,args.slices),(320,16,1600,15000))
        self.assertAlmostEqual(cases[-1]['geometry']['height_nm'],100000*15/37.2)
        self.assertEqual(cases[-1]['geometry']['base_width_nm'],100000)
        self.assertEqual(cases[-1]['geometry']['gold_vertical_nm'],50)
        self.assertEqual(cases[-1]['first_computable_order'],320)
        self.assertEqual(cases[-1]['minimum_spectrum_order'],376)


if __name__=='__main__':
    unittest.main(verbosity=2)
