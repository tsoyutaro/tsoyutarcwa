"""Adaptive order control: physical rejection, immutable attempts, reset and resume."""
from pathlib import Path
import importlib.util,json,sys,tempfile,unittest
from unittest.mock import patch
import torch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('adaptive_order',ROOT/'optimize_adaptive_order.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
from test_engine import request,state,Evaluator,geometry,engine,physics

class AdaptiveOrderTests(unittest.TestCase):
    def test_escalates_only_nonphysical_in_steps_of_two(self):
        called=[]
        def run(m):called.append(m);return dict(status='nonphysical' if m<12 else 'completed')
        result=a.drive_orders(8,32,run)
        self.assertEqual(called,[8,10,12]);self.assertEqual(result['selected_order'],12)
    def test_never_runs_above32(self):
        called=[]
        result=a.drive_orders(30,32,lambda m:(called.append(m) or dict(status='nonphysical')))
        self.assertEqual(called,[30,32]);self.assertEqual(result['status'],'max_order_reached')
    def test_oom_does_not_increase_order(self):
        called=[]
        result=a.drive_orders(16,32,lambda m:(called.append(m) or dict(status='cuda_oom')))
        self.assertEqual(called,[16]);self.assertEqual(result['status'],'cuda_oom')
    def test_generic_error_does_not_increase_order(self):
        called=[]
        result=a.drive_orders(8,32,lambda m:(called.append(m) or dict(status='error')))
        self.assertEqual(called,[8]);self.assertEqual(result['status'],'error')
    def test_resume_skips_failed_physical_orders(self):
        called=[];history=[dict(order=8,status='nonphysical'),dict(order=10,status='nonphysical')]
        result=a.drive_orders(8,32,lambda m:(called.append(m) or dict(status='completed')),history)
        self.assertEqual(called,[12]);self.assertEqual([r['order'] for r in result['attempts']],[8,10,12])
    def test_completed_attempt_is_reused(self):
        result=a.drive_orders(8,32,lambda m:self.fail('must not rerun'),[dict(order=8,status='completed')])
        self.assertEqual(result['selected_order'],8)
    def test_paused_attempt_resumes_at_same_order(self):
        called=[];history=[dict(order=8,status='nonphysical'),dict(order=10,status='paused')]
        result=a.drive_orders(8,32,lambda m:(called.append(m) or dict(status='completed')),history)
        self.assertEqual(called,[10]);self.assertEqual(len(result['attempts']),2)
    def test_oom_attempt_resumes_at_same_order(self):
        called=[]
        a.drive_orders(8,32,lambda m:(called.append(m) or dict(status='completed')),[dict(order=8,status='cpu_oom')])
        self.assertEqual(called,[8])
    def test_invalid_initial_and_candidate_are_classified(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);a.write(p/'workflow.json',dict(status='error'))
            outcome=p/'hexagonal/trials/trial_0000/outcome.json'
            for status in ['error','nonphysical']:
                a.write(outcome,dict(status=status,error_type='NonphysicalResult',error='invalid RTA'))
                self.assertEqual(a.classify(p,1)['status'],'nonphysical')
    def test_error_message_alone_is_not_a_physical_failure(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);a.write(p/'workflow.json',dict(status='error'))
            a.write(p/'hexagonal/trials/trial_0000/outcome.json',dict(status='error',error_type='RuntimeError',error='nonphysical word in unrelated message'))
            self.assertEqual(a.classify(p,1)['status'],'error')
    def test_prepare_does_not_replace_real_history(self):
        with tempfile.TemporaryDirectory() as t:
            output=Path(t);a.write(output/'campaign_state.json',dict(materials={'Cr':dict(status='completed',attempts=[dict(order=8,status='completed')])}))
            before=(output/'campaign_state.json').read_bytes()
            with patch('sys.stderr'), self.assertRaises(SystemExit) as err:
                a.main(['--device','cpu','--materials','Cr','--start-orders','Cr:8','--prepare-only','--output-root',str(output)])
            self.assertEqual(err.exception.code,2)
            self.assertEqual((output/'campaign_state.json').read_bytes(),before)
            self.assertFalse((output/'controller.lock').exists())

    def test_default_start_orders_and_normal_off_configs(self):
        expected={'Al':8,'Ti':8,'Cr':8,'Cu':12,'Au':16}
        args=a.parser().parse_args([])
        self.assertEqual(a.starts_from(args.start_orders),expected);self.assertEqual(args.max_order,32)
        self.assertEqual(a.canonical('CU'),'Cu')
        for m,order in expected.items():
            path=ROOT/m/f'pmma_{m.lower()}_motheye_P200_H500_Nz15'/f'config_adaptive_trial1_M{order}_Q512.json'
            c=a.read(path);self.assertEqual(c['optimization_order'],order)
            self.assertEqual(c['normal_mode'],'off');self.assertEqual(c['trials'],1)
            self.assertEqual(c['nonphysical_action'],'restart-higher-order')

class AdaptiveEngineTests(unittest.TestCase):
    def test_first_invalid_candidate_stops_without_lr_backtracking(self):
        req=request('adaptive_candidate',steps=3,nonphysical_action='restart-higher-order')
        x=torch.tensor(geometry.baseline_logits(req['config']),dtype=torch.float64)
        ev=Evaluator(x,delta=.1,bound=1e-12)
        out=engine.train(req,'cpu',evaluator=ev);saved=state(req)
        self.assertEqual(out['status'],'nonphysical');self.assertEqual(out['error_type'],'NonphysicalResult')
        self.assertEqual(ev.calls,2);self.assertEqual(out['completed_steps'],0)
        self.assertEqual(out['rejected_proposals'],1)
        torch.testing.assert_close(saved['logits'],x,rtol=0,atol=0)
        self.assertEqual(saved['optimizer']['state'],{});self.assertEqual(saved['optimizer']['param_groups'][0]['lr'],.03)
    def test_each_new_order_restarts_from_nominal_shape_and_fresh_adam(self):
        initial=[];checkpoints=[]
        def execute(m):
            req=request('adaptive_order'+str(m),steps=1,optimization_order=m,nonphysical_action='restart-higher-order')
            x=torch.tensor(geometry.baseline_logits(req['config']),dtype=torch.float64)
            ev=Evaluator(x,delta=.1,bound=1e-12 if m==8 else 1.)
            def evaluate(logits,*args,**kwargs):
                if ev.calls==0:initial.append(logits.detach().clone())
                return ev(logits,*args,**kwargs)
            out=engine.train(req,'cpu',evaluator=evaluate);checkpoints.append(state(req));return out
        result=a.drive_orders(8,10,execute)
        self.assertEqual(result['selected_order'],10)
        torch.testing.assert_close(initial[0],initial[1],rtol=0,atol=0)
        self.assertEqual(checkpoints[0]['optimizer']['state'],{})
        self.assertTrue(all(float(v['step'])==1 for v in checkpoints[1]['optimizer']['state'].values()))
    def test_invalid_initial_stops_without_retry(self):
        req=request('adaptive_initial',steps=1,nonphysical_action='restart-higher-order')
        def invalid(*args,**kwargs):raise physics.NonphysicalResult('injected initial invalidity')
        with self.assertRaises(physics.NonphysicalResult):engine.train(req,'cpu',evaluator=invalid)
        saved=state(req);self.assertFalse(saved['current_state_validated']);self.assertEqual(saved['stats']['evaluation_calls'],1)
    def test_invalid_action_is_not_silently_backtracked(self):
        req=request('adaptive_invalid_action',nonphysical_action='typo')
        with self.assertRaisesRegex(ValueError,'nonphysical_action'):engine.train(req,'cpu')

if __name__=='__main__':unittest.main(verbosity=2)
