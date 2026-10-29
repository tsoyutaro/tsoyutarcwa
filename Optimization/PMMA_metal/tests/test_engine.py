from pathlib import Path
import copy,json,sys,unittest,uuid
import torch

ROOT=Path(__file__).resolve().parents[1]
CASE=ROOT/'Al/pmma_al_motheye_P200_H500_Nz15'
sys.path.insert(0,str(CASE))
from optimizer import common,geometry,engine,physics

torch.set_num_threads(1)
BASE=common.load_config(CASE/'config_hex_trial1_M8_Q512.json')
SCRATCH=ROOT/'validation_runs'/uuid.uuid4().hex
SCRATCH.mkdir(parents=True)


def request(name,steps=6,**extra):
    c=copy.deepcopy(BASE);c.update(adam_steps=steps,patience=1000,**extra)
    return dict(config=c,lattice='hexagonal',params=dict(lr=.03),trial_folder=str(SCRATCH/name),
                signature='dual-radius-unit-v1',trial_number=0,deadline=None)


def state(req):return torch.load(Path(req['trial_folder'])/'adam_checkpoint.pt',map_location='cpu',weights_only=True)


class Evaluator:
    def __init__(self,origin,delta=.01,bound=float('inf'),interrupt=None,unexpected=False):
        self.origin=origin.clone();self.target=origin.clone();self.target[0]+=delta
        self.target[len(origin)//2]-=.005
        self.bound=bound;self.interrupt=interrupt;self.unexpected=unexpected;self.calls=0
    def __call__(self,x,*args,gradient):
        self.calls+=1
        if self.calls==self.interrupt:
            if self.unexpected:raise RuntimeError('injected runtime error')
            raise KeyboardInterrupt('injected interruption')
        if abs(float(x[0].detach()-self.origin[0]))>self.bound:raise physics.NonphysicalResult('injected nonphysical candidate')
        y=x.detach()-self.target;loss=.5+float((y*y).sum())
        return dict(objective=loss,gradient=2*y if gradient else None,seconds=.001,
            rows=[dict(wavelength_nm=550.,reflectance=.4*loss,transmittance=.6*loss,absorptance_total=1-loss)])


class EngineTests(unittest.TestCase):
    def test_both_profiles_optimized_and_absorption_is_negative_loss_gradient(self):
        req=request('both',steps=2);x=torch.tensor(geometry.baseline_logits(req['config']),dtype=torch.float64)
        evaluator=Evaluator(x,delta=.1,bound=1.)
        out=engine.train(req,'cpu',evaluator=evaluator);saved=state(req)
        expected=torch.nn.Parameter(x.clone());opt=torch.optim.Adam([expected],lr=.03)
        for i in range(2):expected.grad=2*(expected.detach()-evaluator.target);opt.step()
        torch.testing.assert_close(expected.detach(),saved['logits'],rtol=0,atol=1e-14)
        self.assertEqual(saved['logits'].numel(),32)
        self.assertTrue(bool((saved['logits'][:16]!=x[:16]).any()))
        self.assertTrue(bool((saved['logits'][16:]!=x[16:]).any()))
        self.assertAlmostEqual(out['best_absorptance'],1-out['best_loss'],places=14)
        self.assertEqual(len(out['pmma_radii_nm']),15)
        self.assertEqual(len(out['metal_outer_radii_nm']),15)
        self.assertTrue(all(a<b for a,b in zip(out['pmma_radii_nm'],out['metal_outer_radii_nm'])))
        for row in saved['history']:self.assertAlmostEqual(row['mean_absorptance'],1-row['objective'],places=14)

    def test_rejection_restores_existing_adam_state(self):
        req=request('rollback',steps=2);x=torch.tensor(geometry.baseline_logits(req['config']),dtype=torch.float64)
        evaluator=Evaluator(x,delta=.1,bound=.04)
        out=engine.train(req,'cpu',evaluator=evaluator);saved=state(req)
        expected=torch.nn.Parameter(x.clone());opt=torch.optim.Adam([expected],lr=.03)
        for lr in (.03,.0075):
            opt.param_groups[0]['lr']=lr;expected.grad=2*(expected.detach()-evaluator.target);opt.step()
        torch.testing.assert_close(expected.detach(),saved['logits'],rtol=0,atol=1e-14)
        for key in ('step','exp_avg','exp_avg_sq'):
            torch.testing.assert_close(saved['optimizer']['state'][0][key],opt.state_dict()['state'][0][key],rtol=0,atol=0)
        self.assertEqual(out['rejected_proposals'],2)
        self.assertEqual(saved['scheduler']['last_epoch'],2)

    def test_interruption_resume_restores_lr_gradient_and_two_profiles(self):
        full=request('full');part=request('resume');x=torch.tensor(geometry.baseline_logits(full['config']),dtype=torch.float64)
        engine.train(full,'cpu',evaluator=Evaluator(x,bound=.02))
        evaluator=Evaluator(x,bound=.02,interrupt=3)
        with self.assertRaises(KeyboardInterrupt):engine.train(part,'cpu',evaluator=evaluator)
        paused=state(part);self.assertEqual(paused['optimizer']['param_groups'][0]['lr'],.015)
        torch.testing.assert_close(paused['logits'],x,rtol=0,atol=0)
        engine.train(part,'cpu',evaluator=evaluator)
        a,b=state(full),state(part)
        torch.testing.assert_close(a['logits'],b['logits'],rtol=0,atol=0)
        self.assertEqual(a['history'],b['history']);self.assertEqual(a['rejections'],b['rejections'])
        self.assertTrue(b['current_state_validated'])

    def test_exhaustion_keeps_valid_profile_and_reports_failure(self):
        req=request('exhaustion',steps=1,physical_backtracking_max_retries=2)
        x=torch.tensor(geometry.baseline_logits(req['config']),dtype=torch.float64)
        result=engine.train(req,'cpu',evaluator=Evaluator(x,bound=1e-12));saved=state(req)
        self.assertEqual(result['status'],'backtracking_exhausted')
        self.assertEqual(result['completed_steps'],0)
        self.assertEqual(result['rejected_proposals'],3)
        torch.testing.assert_close(saved['logits'],x,rtol=0,atol=0)
        self.assertEqual(saved['optimizer']['state'],{})
        self.assertEqual(result['radii_nm'],geometry.radii_values(x,req['config']))

    def test_invalid_initial_gradient_stops_without_retry(self):
        req=request('nan',steps=1);x=torch.tensor(geometry.baseline_logits(req['config']),dtype=torch.float64)
        valid=Evaluator(x)
        def nan(*args,**kwargs):
            result=valid(*args,**kwargs);result['gradient'][0]=float('nan');return result
        with self.assertRaises(physics.NonphysicalResult):engine.train(req,'cpu',evaluator=nan)
        saved=state(req);self.assertFalse(saved['current_state_validated'])
        self.assertEqual(saved['stats']['rejected_proposals'],0)

    def test_unexpected_exception_is_not_hidden(self):
        req=request('exception',steps=1);x=torch.tensor(geometry.baseline_logits(req['config']),dtype=torch.float64)
        with self.assertRaisesRegex(RuntimeError,'injected runtime error'):
            engine.train(req,'cpu',evaluator=Evaluator(x,interrupt=2,unexpected=True))
        saved=state(req);torch.testing.assert_close(saved['logits'],x,rtol=0,atol=0)
        self.assertEqual(saved['stats']['rejected_proposals'],0)


if __name__=='__main__':
    run=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(EngineTests))
    report=dict(status='passed' if run.wasSuccessful() else 'failed',tests_run=run.testsRun,
                failures=len(run.failures),errors=len(run.errors),cuda_execution_performed=False)
    folder=ROOT/'validation_reports';folder.mkdir(exist_ok=True)
    (folder/'engine_tests.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2));raise SystemExit(0 if run.wasSuccessful() else 1)
