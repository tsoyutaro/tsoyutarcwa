"""CPU regression tests for guarded Adam; no production experiment is modified."""
from pathlib import Path
import copy
import importlib.util
import json
import math
import sys
import unittest
import uuid

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from optimizer import engine, geometry, common, physics

torch.set_num_threads(1)
BASE = json.loads((HERE/'config.json').read_text(encoding='utf-8-sig'))
SCRATCH = HERE/'validation_runs'/uuid.uuid4().hex
SCRATCH.mkdir(parents=True)


def config(**updates):
    c = copy.deepcopy(BASE)
    c.update(layers=2,adam_steps=6,patience=1000,scheduler_patience=2,pruner_interval_steps=1,
             physical_backtracking_max_retries=12,physical_backtracking_min_lr=1e-6,
             optimization_wavelengths_nm=[550.])
    c.update(updates)
    return c


def params(c):
    radii = geometry.radii_values(geometry.baseline_logits(c), c)
    return dict(lr=.03,init_top=radii[0],delta_ratio=(radii[-1]-radii[0])/(c['radius_max_nm']-radii[0]))


def initial(c, p):
    bottom=p['init_top']+p['delta_ratio']*(c['radius_max_nm']-p['init_top'])
    return torch.tensor(geometry.initial_logits(p['init_top'],bottom,c),dtype=torch.float64)


def request(name, c):
    return dict(config=c,lattice='hexagonal',params=params(c),trial_folder=str(SCRATCH/name),
                signature='guarded-test-v4',trial_number=0,deadline=None)


def checkpoint(req):
    return torch.load(Path(req['trial_folder'])/'adam_checkpoint.pt',map_location='cpu',weights_only=True)


class Evaluator:
    def __init__(self, origin, target_delta=.01, bound=math.inf, interrupt_call=None, unexpected=False):
        self.origin=origin.clone()
        self.target=origin.clone()
        self.target[0] += target_delta
        self.bound=bound
        self.interrupt_call=interrupt_call
        self.unexpected=unexpected
        self.calls=0
    def __call__(self, x, *args, gradient):
        self.calls+=1
        if self.calls == self.interrupt_call:
            if self.unexpected:
                raise RuntimeError('injected coding error')
            raise KeyboardInterrupt('injected interruption')
        if abs(float(x[0].detach()-self.origin[0])) > self.bound:
            raise physics.NonphysicalResult('injected nonpassive candidate')
        y=x.detach()-self.target
        value=float((y*y).sum())
        return dict(objective=value,gradient=2*y if gradient else None,
                    rows=[dict(wavelength_nm=550.,reflectance=value,power_into_substrate=.1)],seconds=.001)


class GuardedAdamTests(unittest.TestCase):
    def test_no_rejection_matches_legacy_adam_trajectory(self):
        c=config(adam_steps=8)
        a,b=request('legacy',c),request('guarded',c)
        source=HERE/'validation/reference_engine_v3.py'
        spec=importlib.util.spec_from_file_location('optimizer._reference_validation_engine',source)
        legacy=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy)
        p0=initial(c,a['params'])
        old=legacy.train(a,'cpu',evaluator=Evaluator(p0,target_delta=.25))
        new=engine.train(b,'cpu',evaluator=Evaluator(p0,target_delta=.25))
        sa,sb=checkpoint(a),checkpoint(b)
        torch.testing.assert_close(sa['logits'],sb['logits'],rtol=0,atol=1e-14)
        self.assertAlmostEqual(old['best_loss'],new['best_loss'],places=14)
        self.assertEqual(sa['optimizer']['param_groups'],sb['optimizer']['param_groups'])
        self.assertEqual(sa['scheduler'],sb['scheduler'])
        self.assertEqual(new['rejected_proposals'],0)
        self.assertEqual(sb['stats']['evaluation_calls'],c['adam_steps']+1)

    def test_rejection_restores_adam_moments_and_step(self):
        c=config(adam_steps=1)
        req=request('rollback',c)
        p0=initial(c,req['params'])
        result=engine.train(req,'cpu',evaluator=Evaluator(p0,bound=.02))
        saved=checkpoint(req)
        expected=torch.nn.Parameter(p0.clone())
        opt=torch.optim.Adam([expected],lr=.015)
        target=p0.clone();target[0]+=.01
        expected.grad=2*(expected.detach()-target)
        opt.step()
        torch.testing.assert_close(saved['logits'],expected.detach(),rtol=0,atol=1e-14)
        actual_state=saved['optimizer']['state'][0]
        expected_state=opt.state_dict()['state'][0]
        for key in ('step','exp_avg','exp_avg_sq'):
            torch.testing.assert_close(actual_state[key],expected_state[key],rtol=0,atol=0)
        self.assertEqual(result['rejected_proposals'],1)
        self.assertEqual(saved['scheduler']['last_epoch'],1)
        self.assertTrue(saved['current_state_validated'])
        self.assertEqual(saved['next_step'],1)

    def test_later_rejection_restores_existing_adam_moments(self):
        c=config(adam_steps=2)
        req=request('later_rejection',c)
        p0=initial(c,req['params'])
        result=engine.train(req,'cpu',evaluator=Evaluator(p0,target_delta=.1,bound=.04))
        saved=checkpoint(req)
        expected=torch.nn.Parameter(p0.clone())
        target=p0.clone();target[0]+=.1
        opt=torch.optim.Adam([expected],lr=.03)
        expected.grad=2*(expected.detach()-target)
        opt.step()
        opt.param_groups[0]['lr']=.0075
        expected.grad=2*(expected.detach()-target)
        opt.step()
        torch.testing.assert_close(saved['logits'],expected.detach(),rtol=0,atol=1e-14)
        for key in ('step','exp_avg','exp_avg_sq'):
            torch.testing.assert_close(saved['optimizer']['state'][0][key],opt.state_dict()['state'][0][key],rtol=0,atol=0)
        self.assertEqual(result['rejected_proposals'],2)
        self.assertEqual(saved['scheduler']['last_epoch'],2)

    def test_interrupted_retry_resumes_same_state_and_lr(self):
        c=config(adam_steps=6)
        full,part=request('full',c),request('resume',c)
        p0=initial(c,full['params'])
        engine.train(full,'cpu',evaluator=Evaluator(p0,bound=.02))
        evaluator=Evaluator(p0,bound=.02,interrupt_call=3)
        with self.assertRaises(KeyboardInterrupt):
            engine.train(part,'cpu',evaluator=evaluator)
        paused=checkpoint(part)
        torch.testing.assert_close(paused['logits'],p0,rtol=0,atol=0)
        self.assertEqual(paused['optimizer']['param_groups'][0]['lr'],.015)
        self.assertEqual(paused['next_step'],0)
        engine.train(part,'cpu',evaluator=evaluator)
        a,b=checkpoint(full),checkpoint(part)
        torch.testing.assert_close(a['logits'],b['logits'],rtol=0,atol=0)
        self.assertEqual(a['history'],b['history'])
        self.assertEqual(a['rejections'],b['rejections'])
        self.assertEqual(a['scheduler'],b['scheduler'])
        for key in ('step','exp_avg','exp_avg_sq'):
            torch.testing.assert_close(a['optimizer']['state'][0][key],b['optimizer']['state'][0][key],rtol=0,atol=0)

    def test_retry_exhaustion_preserves_valid_best_and_is_not_completed(self):
        c=config(adam_steps=1,physical_backtracking_max_retries=2)
        req=request('exhaustion',c)
        p0=initial(c,req['params'])
        result=engine.train(req,'cpu',evaluator=Evaluator(p0,bound=1e-12))
        state=checkpoint(req)
        self.assertEqual(result['status'],'backtracking_exhausted')
        self.assertEqual(result['completed_steps'],0)
        self.assertEqual(result['rejected_proposals'],3)
        torch.testing.assert_close(state['logits'],p0,rtol=0,atol=0)
        self.assertEqual(state['optimizer']['state'],{})
        self.assertEqual(state['scheduler']['last_epoch'],0)
        self.assertTrue(state['current_state_validated'])
        self.assertEqual(result['best_logits'],p0.tolist())
        best=json.loads((Path(req['trial_folder'])/'best.json').read_text(encoding='utf-8'))
        self.assertEqual(best['status'],'backtracking_exhausted')

    def test_invalid_initial_state_does_not_shrink_updates(self):
        c=config(adam_steps=1)
        req=request('initial_invalid',c)
        p0=initial(c,req['params'])
        with self.assertRaises(physics.NonphysicalResult):
            engine.train(req,'cpu',evaluator=Evaluator(p0,bound=-1.))
        state=checkpoint(req)
        self.assertFalse(state['current_state_validated'])
        self.assertEqual(state['stats']['evaluation_calls'],1)
        self.assertEqual(state['stats']['rejected_proposals'],0)

    def test_unexpected_errors_are_not_hidden(self):
        c=config(adam_steps=1)
        req=request('unexpected',c)
        p0=initial(c,req['params'])
        with self.assertRaisesRegex(RuntimeError,'injected coding error'):
            engine.train(req,'cpu',evaluator=Evaluator(p0,interrupt_call=2,unexpected=True))
        state=checkpoint(req)
        torch.testing.assert_close(state['logits'],p0,rtol=0,atol=0)
        self.assertTrue(state['current_state_validated'])
        self.assertEqual(state['stats']['rejected_proposals'],0)

    def test_nonfinite_initial_gradient_is_rejected(self):
        c=config(adam_steps=1)
        req=request('nan_gradient',c)
        p0=initial(c,req['params'])
        valid=Evaluator(p0)
        def broken(*args,**kwargs):
            result=valid(*args,**kwargs)
            result['gradient'][0]=float('nan')
            return result
        with self.assertRaises(physics.NonphysicalResult):
            engine.train(req,'cpu',evaluator=broken)
        self.assertFalse(checkpoint(req)['current_state_validated'])

    def test_real_ag_rcwa_cpu_two_accepted_updates(self):
        c=common.load_config(HERE/'config.json')
        c.update(layers=5,optimization_order=2,optimization_grid=48,
                 optimization_wavelengths_nm=[400.,550.,700.],adam_steps=2,patience=1000)
        req=request('real_ag',c)
        result=engine.train(req,'cpu')
        state=checkpoint(req)
        self.assertEqual(result['status'],'completed')
        self.assertEqual(result['completed_steps'],2)
        self.assertTrue(state['current_state_validated'])
        self.assertTrue(bool(torch.isfinite(state['cached_valid_evaluation']['gradient']).all()))
        for row in state['cached_valid_evaluation']['rows']:
            for key in ('reflectance','power_into_substrate','absorptance_total','motheye_absorptance'):
                self.assertGreaterEqual(row[key],-c['passivity_tolerance'])
                self.assertLessEqual(row[key],1+c['passivity_tolerance'])
        radii=result['radii_nm']
        self.assertTrue(all(a < b for a,b in zip(radii,radii[1:])))


if __name__ == '__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(GuardedAdamTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report=dict(status='passed' if result.wasSuccessful() else 'failed',tests_run=result.testsRun,
                failures=len(result.failures),errors=len(result.errors),cuda_execution_performed=False,
                torch=str(torch.__version__),scratch=str(SCRATCH))
    (HERE/'validation_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
