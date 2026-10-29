"""Regression tests for continuing convergence scheduling; no GPU is needed."""
from pathlib import Path
import contextlib
import csv
import importlib.util
import io
import json
import sys
import unittest
from unittest.mock import patch
import uuid

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('continue_convergence',HERE.parent/'continue_convergence.py')
cc=importlib.util.module_from_spec(spec);spec.loader.exec_module(cc)
SCRATCH=Path.cwd()/'work/continue_convergence_tests'/uuid.uuid4().hex
SCRATCH.mkdir(parents=True)


def result(m,q):
    r=.1+.00001*m+.000000001*q
    return dict(status='completed',mean_R=r,rcwa_seconds=.01,worker_seconds=.02,
                rows=[dict(wavelength_nm=550.,reflectance=r,transmittance=.2,absorptance_total=.8-r)])


class ContinueTests(unittest.TestCase):
    def test_order_range_is_inclusive(self):
        self.assertEqual(cc.numbers('4:40:2',True),list(range(4,41,2)))
        self.assertEqual(len(cc.numbers('4:40:2',True)),19)
        for text in ('4:40:0','4:39:2','0,2','nan'):
            with self.assertRaises(ValueError):cc.numbers(text,True)

    def test_nonphysical_skips_remaining_grid_then_next_order(self):
        calls=[]
        def execute(m,q):
            calls.append((m,q))
            return dict(status='nonphysical') if m==4 else result(m,q)
        rows=cc.drive_conditions([4,6],[512,1024],execute)
        self.assertEqual(calls,[(4,512),(6,512),(6,1024)])
        self.assertEqual(rows[1]['reason'],'nonphysical_at_same_order')

    def test_oom_stops_remaining_material_conditions(self):
        calls=[]
        def execute(m,q):
            calls.append((m,q))
            return dict(status='cuda_oom') if (m,q)==(6,1024) else result(m,q)
        rows=cc.drive_conditions([4,6,8],[512,1024],execute)
        self.assertEqual(calls,[(4,512),(4,1024),(6,512),(6,1024)])
        self.assertEqual([r['status'] for r in rows[-2:]],['skipped','skipped'])

    def test_optional_dominance_policy_keeps_smaller_grid(self):
        calls=[]
        def execute(m,q):
            calls.append((m,q))
            return dict(status='cuda_oom') if (m,q)==(6,1024) else result(m,q)
        rows=cc.drive_conditions([4,6,8],[512,1024],execute,oom_policy='dominated')
        self.assertIn((8,512),calls)
        self.assertNotIn((8,1024),calls)
        self.assertEqual(rows[-1]['reason'],'dominated_by_oom_condition')

    def test_cpu_oom_also_skips_heavier_conditions(self):
        rows=cc.drive_conditions([4,6],[512,1024],lambda m,q:dict(status='cpu_oom'))
        self.assertEqual([r['status'] for r in rows],['cpu_oom','skipped','skipped','skipped'])

    def test_resume_uses_cache_and_retry_failures_does_not_repeat_successes(self):
        rows=cc.drive_conditions([4,6],[512,1024],lambda m,q:dict(status='nonphysical') if m==4 else result(m,q))
        cache={(r['order'],r['grid']):r for r in rows}
        calls=[]
        def execute(m,q):calls.append((m,q));return result(m,q)
        resumed=cc.drive_conditions([4,6],[512,1024],execute,cache)
        self.assertEqual(calls,[])
        self.assertTrue(resumed[-1]['cache_hit'])
        retried=cc.drive_conditions([4,6],[512,1024],execute,cache,retry_failures=True)
        self.assertEqual(calls,[(4,512),(4,1024)])
        self.assertTrue(all(r['status']=='completed' for r in retried))

    def test_worker_loss_is_not_called_oom_and_skips_remaining_material(self):
        rows=cc.drive_conditions([4,6],[512,1024],lambda m,q:dict(status='worker_lost',returncode=-9))
        self.assertEqual(rows[0]['status'],'worker_lost')
        self.assertIn('worker_loss',rows[-1]['reason'])

    def test_finite_reference_excludes_failures_and_requires_two_axes(self):
        rows=[dict(order=m,grid=q,**result(m,q)) for m in (4,6) for q in (512,1024)]
        rows.append(dict(order=8,grid=512,status='nonphysical'))
        report=cc.aggregate(rows,.00025)
        self.assertEqual(report['comparison_reference'],dict(order=6,grid=1024))
        self.assertTrue(report['numerical_convergence_passed'])
        ref=next(x for x in report['comparison'] if x['is_reference'])
        self.assertIsNone(ref['within_tolerance'])
        missing=cc.aggregate([r for r in rows if r['grid']==512],.00025)
        self.assertIsNone(missing['numerical_convergence_passed'])
        self.assertIsNone(cc.aggregate([rows[-1]],.00025)['comparison_reference'])

    def test_full_batch_continues_after_material_oom_and_releases_locks(self):
        root=SCRATCH/'batch';root.mkdir()
        specs=[]
        for name in ('Al','Ti','Cu','Au','Ag'):
            case=root/name;run=case/'results/test';run.mkdir(parents=True)
            cc.write(run/'manifest.json',{})
            specs.append(dict(material=name,case_dir=str(case),config=str(case/'config.json'),run_root=str(run)))
        cases=root/'cases.json';cc.write(cases,specs)
        calls=[]
        def fake_launch(folder,request):
            if request['mode']=='prepare':
                context=dict(tool=cc.TOOL,material=request['material'],device='cpu',lattice='hexagonal',
                             config=dict(verification_tolerance=.00025),training_signature='fixed',
                             logits=[0.,0.,0.],waves=[550.])
                return dict(status='ready',context=context)
            context=cc.read(request['context']);m,q=request['order'],request['grid']
            calls.append((context['material'],m,q))
            if context['material']=='Al' and m==4 and q==512:return dict(status='cuda_oom')
            return result(m,q)
        argv=['--root',str(root),'--cases-file',str(cases),'--device','cpu','--orders','4,6','--grids','512,1024','--wavelengths','550']
        with patch.object(cc,'launch',side_effect=fake_launch),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cc.main(argv),0)
        self.assertEqual([x for x in calls if x[0]=='Al'],[('Al',4,512)])
        for name in ('Ti','Cu','Au','Ag'):self.assertEqual(len([x for x in calls if x[0]==name]),4)
        self.assertTrue(all(not (Path(x['run_root'])/'controller.lock').exists() for x in specs))
        summary=cc.read(root/'convergence_continue_summary.json')
        self.assertEqual(summary['materials']['Al']['counts']['skipped'],3)
        self.assertEqual(summary['materials']['Au']['status'],'completed')
        calls.clear()
        with patch.object(cc,'launch',side_effect=fake_launch),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cc.main(argv),0)
        self.assertEqual(calls,[])

    def test_child_classifies_oom_and_nonphysical_and_preserves_error_text(self):
        request=SCRATCH/'classification_request.json'
        cc.write(request,dict(mode='evaluate',device='cuda:0'))
        for name,status in [('OutOfMemoryError','cuda_oom'),('NonphysicalResult','nonphysical'),('ValueError','error')]:
            error=type(name,(RuntimeError,),{})('injected failure')
            output=SCRATCH/(status+'.json')
            with patch.object(cc,'worker_child',side_effect=error),contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(cc.child_main(request,output),0)
            self.assertEqual(cc.read(output)['status'],status)
            self.assertEqual(cc.read(output)['error'],'injected failure')

    def test_source_free_parent_does_not_import_torch(self):
        self.assertNotIn('torch',sys.modules)


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(ContinueTests)
    result_run=unittest.TextTestRunner(verbosity=2).run(suite)
    report=dict(status='passed' if result_run.wasSuccessful() else 'failed',tests_run=result_run.testsRun,
        failures=len(result_run.failures),errors=len(result_run.errors),cuda_execution_performed=False,
        oom_tests='simulated structured failures, no hardware exhaustion')
    cc.write(HERE.parent/'validation_reports/continue_convergence_test_report.json',report)
    print(json.dumps(report,indent=2))
    raise SystemExit(0 if result_run.wasSuccessful() else 1)
