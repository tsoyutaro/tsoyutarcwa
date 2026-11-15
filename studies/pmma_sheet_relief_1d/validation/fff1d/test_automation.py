from pathlib import Path
import contextlib,csv,hashlib,importlib.util,io,json,sys,tempfile,unittest
from types import SimpleNamespace
from unittest.mock import patch

STUDY=Path(__file__).resolve().parents[2]
ROOT=STUDY
spec=importlib.util.spec_from_file_location('auto_converge',STUDY/'automation/converge_until.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class AutoTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.out=Path(self.temp.name)/'campaign'
        self.calls=[]
        self.args=['--axis','order','--start','2','--step','2','--coating','uncoated',
            '--periods','200','--wavelengths','550','--voids','water','--slices','4',
            '--phase-tol','0.00005','--phase-max','65536','--output-dir',str(self.out)]
        self.source=m.source_signature(m.solver_root())

    def tearDown(self):self.temp.cleanup()

    def read(self,name='convergence.json'):return m.load(self.out/name)

    def child(self,mode='stable',fail_at=None,fail_code=1,fail_text='RuntimeError: example',phase=True,nonfinite=False,order_only=False):
        def run(command,log):
            a=m.calculation_parser().parse_args(command[2:]);v=a.order;self.calls.append(v)
            if fail_at==v:
                if fail_text=='interrupt':raise KeyboardInterrupt()
                log.write_text(fail_text,encoding='utf-8');return fail_code
            dest=a.output_dir;dest.mkdir(parents=True,exist_ok=True)
            log.write_text('completed\n',encoding='utf-8')
            r=.1 if mode=='stable' else .01*v
            labels=dict(coating='uncoated',void='water',period_nm=200.,wavelength_nm=550.)
            summaries=[];orders=[]
            for pol in m.selected_polarizations(a.polarization):
                summaries.append(dict(labels,polarization=pol,R='nan' if nonfinite else r,T=1-r,
                    A_total=0.,R_le_20deg=r,R_gt_20deg=0.,phase_error=1e-6,phase_converged=phase))
                for k in range(-v,v+1):
                    rv=(r if k==0 else 0.) if not order_only else ((.01*v if abs(k)==1 else r-.02*v if k==0 else 0.))
                    orders.append(dict(labels,polarization=pol,m=k,R_m=rv,T_m=1-r if k==0 else 0.))
            m.save_csv(dest/'spectra.csv',summaries);m.save_csv(dest/'diffraction_orders.csv',orders)
            (dest/'local_single_pass_orders.csv').write_text('m,R,T\n0,0.1,0.9\n',encoding='utf-8')
            signature=m.source_signature(m.solver_root(a.solver_root),a.polarization)
            m.save_json(dest/'settings.json',dict(source_sha256=signature,order=v,slices=a.slices))
            m.save_json(dest/'completion.json',dict(completed_cases=len(summaries),phase_converged=phase))
            return 0 if phase else 2
        return run

    def run_auto(self,fake,extra=()):
        with patch.object(m,'run_child',fake),contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            return m.main([*self.args,*extra])

    def test_stops_after_two_adjacent_passes(self):
        self.assertEqual(self.run_auto(self.child()),0)
        self.assertEqual(self.calls,[2,4,6]);self.assertEqual(self.read()['status'],'converged')
        self.assertEqual(self.read()['consecutive_passes'],2)

    def test_successful_resume_launches_no_children(self):
        self.run_auto(self.child());self.calls.clear()
        self.assertEqual(self.run_auto(self.child()),0);self.assertEqual(self.calls,[])

    def test_p_only_campaign_and_resume(self):
        self.assertEqual(self.run_auto(self.child(),['--polarization','p']),0)
        self.assertEqual(self.calls,[2,4,6]);self.calls.clear()
        self.assertEqual({r['polarization'] for r in m.read(self.out/'order_6/spectra.csv')},{'p'})
        self.assertEqual(self.run_auto(self.child(),['--polarization','p']),0)
        self.assertEqual(self.calls,[])

    def test_s_only_campaign(self):
        self.assertEqual(self.run_auto(self.child(),['--polarization','s']),0)
        self.assertEqual({r['polarization'] for r in m.read(self.out/'order_6/spectra.csv')},{'s'})

    def test_polarization_change_requires_new_campaign(self):
        self.run_auto(self.child());before=(self.out/'convergence.json').read_bytes()
        self.assertEqual(self.run_auto(self.child(),['--polarization','p']),4)
        self.assertEqual((self.out/'convergence.json').read_bytes(),before)

    def test_oom_keeps_completed_cases_and_does_not_continue(self):
        self.assertEqual(self.run_auto(self.child('ramp',8,1,'torch.OutOfMemoryError: CUDA out of memory.')),3)
        s=self.read();self.assertEqual(s['status'],'oom');self.assertEqual(s['checked_values'],[2,4,6])
        self.assertEqual(s['attempted_value'],8);self.assertEqual(self.calls,[2,4,6,8])
        self.assertTrue((self.out/'logs/order_8.log').exists())

    def test_resume_after_oom_retries_only_failed_resolution(self):
        fake=self.child('ramp',8,1,'torch.cuda.OutOfMemoryError: CUDA out of memory.')
        self.run_auto(fake);self.calls.clear();self.assertEqual(self.run_auto(fake),3)
        self.assertEqual(self.calls,[8]);self.assertEqual(self.read()['checked_values'],[2,4,6])

    def test_other_error_is_not_oom(self):
        self.assertEqual(self.run_auto(self.child(fail_at=2,fail_text='RuntimeError: Passivity failure')),4)
        self.assertEqual(self.read()['status'],'child_error');self.assertEqual(self.calls,[2])

    def test_sigkill_is_not_claimed_as_confirmed_oom(self):
        self.assertEqual(self.run_auto(self.child(fail_at=2,fail_code=-9,fail_text='')),137)
        self.assertEqual(self.read()['status'],'killed_possible_oom')

    def test_phase_failure_stops_instead_of_increasing_order(self):
        self.assertEqual(self.run_auto(self.child(phase=False)),5)
        self.assertEqual(self.read()['status'],'phase_unconverged');self.assertEqual(self.calls,[2])

    def test_explicit_limit_is_not_convergence(self):
        self.assertEqual(self.run_auto(self.child('ramp'),['--max-value','6']),2)
        self.assertEqual(self.calls,[2,4,6]);self.assertEqual(self.read()['status'],'limit_reached')

    def test_limit_can_be_extended_without_recalculating(self):
        self.run_auto(self.child('ramp'),['--max-value','6']);self.calls.clear()
        self.assertEqual(self.run_auto(self.child('ramp'),['--max-value','8']),2)
        self.assertEqual(self.calls,[8]);self.assertEqual(self.read()['checked_values'],[2,4,6,8])

    def test_setting_change_rejected_without_modifying_results(self):
        self.run_auto(self.child());before=(self.out/'convergence.json').read_bytes()
        self.assertEqual(self.run_auto(self.child(),['--tolerance','.002']),4)
        self.assertEqual((self.out/'convergence.json').read_bytes(),before)

    def test_tampered_checkpoint_is_recomputed(self):
        self.run_auto(self.child());self.calls.clear()
        p=self.out/'order_4/spectra.csv';p.write_text(p.read_text(encoding='utf-8-sig')+'\n',encoding='utf-8-sig')
        self.assertEqual(self.run_auto(self.child()),0);self.assertEqual(self.calls,[4])

    def test_nonfinite_power_rejected(self):
        self.assertEqual(self.run_auto(self.child(nonfinite=True)),4)
        self.assertEqual(self.read()['status'],'error');self.assertFalse(self.read()['converged'])

    def test_orders_are_compared_even_if_totals_stay_fixed(self):
        self.assertEqual(self.run_auto(self.child(order_only=True),['--max-value','6']),2)
        self.assertGreater(self.read()['last_comparison']['maximum_change_all_orders_and_totals'],.001)

    def test_interrupt_checkpoint(self):
        self.assertEqual(self.run_auto(self.child(fail_at=4,fail_text='interrupt')),130)
        self.assertEqual(self.read()['status'],'interrupted');self.assertEqual(self.read()['checked_values'],[2])

    def test_parent_memory_error_is_reported_as_oom(self):
        with patch.object(m,'compare',side_effect=MemoryError()):
            self.assertEqual(self.run_auto(self.child()),3)
        self.assertEqual(self.read()['status'],'oom')

    def test_nonpassive_points_cannot_pass_fff_convergence(self):
        original=m.context
        def extra_context(args,extra):
            ctx=original(args,extra);ctx['passivity_column']='passivity_valid';return ctx
        # The synthetic results omit the validity flag; they must not be certified.
        with patch.object(m,'context',extra_context):
            self.assertEqual(self.run_auto(self.child(),['--max-value','6']),2)
        self.assertFalse(self.read()['converged'])
        self.assertFalse(self.read()['last_comparison']['passivity_valid'])

    def test_fff_tests_the_requested_maximum_even_when_step_does_not_land_on_it(self):
        original=m.context
        def extra_context(args,extra):
            ctx=original(args,extra);ctx['include_max']=True;return ctx
        with patch.object(m,'context',extra_context):
            self.assertEqual(self.run_auto(self.child('ramp'),['--max-value','7']),2)
        self.assertEqual(self.calls,[2,4,6,7])

    def test_oom_signatures_and_false_positives(self):
        for message in ['torch.cuda.OutOfMemoryError: x','CUDA out of memory.','MemoryError',
            'numpy._core._exceptions._ArrayMemoryError: Unable to allocate 12 GiB',
            'DefaultCPUAllocator: not enough memory: you tried to allocate 100 bytes.','CUBLAS_STATUS_ALLOC_FAILED']:
            self.assertIsNotNone(m.oom_evidence(message),message)
        for message in ['RuntimeError: eig failed','room size=10','Converged: False','phase unconverged']:
            self.assertIsNone(m.oom_evidence(message),message)

    def test_real_subprocess_captures_stderr_and_oom_without_allocating_memory(self):
        self.out.mkdir()
        log=self.out/'child.log'
        command=[sys.executable,'-c',"import sys; print('completed point'); print('torch.OutOfMemoryError: CUDA out of memory.',file=sys.stderr); sys.exit(1)"]
        with contextlib.redirect_stdout(io.StringIO()):code=m.run_child(command,log)
        self.assertEqual(code,1)
        self.assertIn('completed point',log.read_text(encoding='utf-8'))
        self.assertIsNotNone(m.oom_evidence(m.log_tail(log)))

    def test_additive_and_multiplicative_growth(self):
        self.assertEqual(m.next_resolution(96,SimpleNamespace(step=16,factor=None)),112)
        self.assertEqual(m.next_resolution(960,SimpleNamespace(step=None,factor=2.)),1920)
        self.assertEqual(m.next_resolution(3,SimpleNamespace(step=None,factor=1.1)),4)

    def test_checkpoint_lock(self):
        self.out.mkdir()
        with m.campaign_lock(self.out):
            with self.assertRaises(RuntimeError):
                with m.campaign_lock(self.out):pass
        with m.campaign_lock(self.out):pass

    def test_original_physics_signature_is_unchanged(self):
        baseline=json.loads((STUDY/'validation/fff1d/legacy_source_baseline.json').read_text())
        self.assertEqual(self.source,baseline['source_signature'])
        self.assertEqual({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in STUDY.glob('*.py')},baseline['root_py_hashes'])


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(AutoTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    out=STUDY/'validation/fff1d/automation_tests.json'
    out.write_text(json.dumps(dict(passed=result.wasSuccessful(),tests_run=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),
        oom_tests_are_simulated=True,real_cuda_oom_triggered=False),indent=2)+'\n',encoding='utf-8')
    raise SystemExit(0 if result.wasSuccessful() else 1)
