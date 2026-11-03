"""Order routing and real-journal controller cleanup without GPU optimization."""
from pathlib import Path
import importlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import uuid

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
for key in ('MOTHEYE_DEPENDENCY_PATH','RCWA_OPT_DEPENDENCY_PATH'):
    if os.environ.get(key):sys.path.append(os.environ[key])
import off_order_controller as adapter
import run_metal_off_optuna30 as batch

def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')

class RoutingTests(unittest.TestCase):
    def run_batch(self,behavior,extra=()):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);case=root/'Au';case.mkdir()
            dump(case/'config.json',{'verification_orders':[8,16],'verification_grid':1024})
            calls=[]
            def child(command,**kwargs):
                args=command[command.index('--')+1:]
                cfg=Path(args[args.index('--config')+1]);output=Path(args[args.index('--output-root')+1])
                c=json.loads(cfg.read_text());order=c['optimization_order']
                attempt=command[command.index('--attempt-id')+1]
                calls.append((order,cfg.name,output.name,c))
                action=behavior(order)
                if action in ('nonphysical','stale'):
                    dump(output/'order_retry.json',dict(policy=adapter.POLICY,status='retry_higher_order',
                        attempt_id=attempt if action=='nonphysical' else uuid.uuid4().hex,
                        order=order,grid=512,normal_mode='off'))
                    code=adapter.RETRY_EXIT_CODE
                else:code=action
                return subprocess.CompletedProcess(command,code)
            with patch.object(batch,'ROOT',root),patch.object(batch,'CASES',{'Au':('Au',14)}),patch.object(batch.subprocess,'run',side_effect=child):
                code=batch.main(['--materials','Au','--auto-order','--grid','512',*extra])
            return code,calls

    def test_nonphysical_restarts_separate_higher_order_study(self):
        code,calls=self.run_batch(lambda m:'nonphysical' if m==14 else 0)
        self.assertEqual(code,0)
        self.assertEqual([x[0] for x in calls],[14,16])
        self.assertNotEqual(calls[0][2],calls[1][2])
        self.assertTrue(all(x[3]['trials']==30 and 'initial_logits' not in x[3] for x in calls))

    def test_explicit_start_order(self):
        code,calls=self.run_batch(lambda m:'nonphysical' if m==16 else 0,['--start-order','16'])
        self.assertEqual(code,0);self.assertEqual([x[0] for x in calls],[16,18])
        self.assertGreaterEqual(max(calls[-1][3]['verification_orders']),18)

    def test_oom_stops_without_higher_order(self):
        code,calls=self.run_batch(lambda m:3)
        self.assertEqual(code,1);self.assertEqual([x[0] for x in calls],[14])

    def test_order_ceiling(self):
        code,calls=self.run_batch(lambda m:'nonphysical',['--max-order','16'])
        self.assertEqual(code,1);self.assertEqual([x[0] for x in calls],[14,16])

    def test_unconfirmed_or_stale_failure_does_not_raise_order(self):
        for action in (1,'stale'):
            code,calls=self.run_batch(lambda m:action)
            self.assertEqual(code,1);self.assertEqual([x[0] for x in calls],[14])

    def test_interrupt_stops_batch(self):
        code,calls=self.run_batch(lambda m:130)
        self.assertEqual(code,130);self.assertEqual(len(calls),1)

    def test_geometry_survives_automatic_order_increase(self):
        code,calls=self.run_batch(lambda m:'nonphysical' if m==14 else 0,
                                  ['--period-nm','100','--height-nm','300'])
        self.assertEqual(code,0)
        self.assertEqual([x[0] for x in calls],[14,16])
        for _,_,_,c in calls:
            self.assertEqual((c['period_nm'],c['height_nm']),(100.,300.))
            self.assertEqual(c['radius_max_nm'],49.999)

class GeometryTests(unittest.TestCase):
    def test_all_45_conditions_are_isolated_and_fit_the_period(self):
        geometries=[(200.,h) for h in (100.,200.,300.,400.)]
        geometries += [(p,500.) for p in (100.,150.,250.,300.,350.)]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);outputs=set();configs=set()
            for material,(relative,_) in batch.CASES.items():
                source=json.loads((HERE/relative/'config.json').read_text(encoding='utf-8-sig'))
                dump(root/relative/'config.json',source)
            with patch.object(batch,'ROOT',root):
                for material,(relative,_) in batch.CASES.items():
                    legacy=batch.prepare(material)
                    original_bytes=legacy[1].read_bytes()
                    for period,height in geometries:
                        case,cfg,output=batch.prepare(material,period_nm=period,height_nm=height)
                        self.assertEqual(case,root/relative)
                        self.assertEqual(cfg.parent.parent.name,'geometry_cases')
                        c=json.loads(cfg.read_text())
                        self.assertEqual((c['period_nm'],c['height_nm'],c['layers']),(period,height,15))
                        self.assertLess(c['baseline_base_nm'],c['radius_max_nm'])
                        self.assertLess(c['radius_max_nm'],period/2)
                        self.assertLess(c['init_top_max_nm'],c['radius_max_nm'])
                        self.assertAlmostEqual(c['baseline_tip_nm']/period,.025)
                        self.assertAlmostEqual(c['baseline_base_nm']/period,.475)
                        self.assertNotIn('initial_logits',c)
                        if material=='Cr':
                            self.assertEqual((cfg.parent/c['material_csv']).resolve(),
                                             (case/'data/Johnson.csv').resolve())
                        self.assertEqual(batch.prepare(material,period_nm=period,height_nm=height),
                                         (case,cfg,output))
                        configs.add(cfg);outputs.add(output)
                    self.assertEqual(legacy[1].read_bytes(),original_bytes)
            self.assertEqual(len(configs),45);self.assertEqual(len(outputs),45)

    def test_invalid_geometry_is_rejected_before_writing(self):
        for period,height in [(0,500),(200,-1),(float('nan'),500),(200,float('inf')),(2,500)]:
            with self.assertRaises(ValueError):
                batch.prepare('Au',period_nm=period,height_nm=height)

class ControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case=HERE/'Au/au_motheye_P200_H500_Nz15'
        sys.path.insert(0,str(cls.case))
        cls.driver=importlib.import_module('optimizer.driver')

    def test_real_journal_marks_fail_before_stopping_and_cleans_lock(self):
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);cfg=base/'config.json';output=base/'run'
            c=json.loads((self.case/'config.json').read_text(encoding='utf-8-sig'));c.update(trials=3,adam_steps=1)
            dump(cfg,c)
            train_calls=[]
            def child(folder,request):
                if request['mode']=='preflight':
                    return dict(status='ready',runtime_environment={'device':'cpu'})
                train_calls.append(request['trial_number'])
                return dict(status='error',error_type='NonphysicalResult',error='Invalid shared power flux.')
            argv=['--config',str(cfg),'--output-root',str(output),'--device','cpu','--mode','optimize','--lattices','hexagonal','--timeout-hours','0']
            first=uuid.uuid4().hex
            with patch.object(self.driver,'launch',side_effect=child):
                code=adapter.run(self.driver,argv,first)
            self.assertEqual(code,adapter.RETRY_EXIT_CODE)
            self.assertEqual(train_calls,[0])
            self.assertFalse((output/'controller.lock').exists())
            self.assertEqual(json.loads((output/'workflow.json').read_text())['status'],'nonphysical_order_retry')
            marker=json.loads((output/'order_retry.json').read_text())
            self.assertEqual(marker['failed_trial'],0);self.assertEqual(marker['attempt_id'],first)
            storage,pruner=self.driver.storage_and_pruner(output/'hexagonal',c)
            study=optuna.load_study(study_name='motheye_nv_hexagonal',storage=storage,pruner=pruner)
            self.assertEqual([(t.number,t.state.name) for t in study.trials],[(0,'FAIL')])
            # The same rejected order does not run trial 1 on a restart.
            train_calls.clear();second=uuid.uuid4().hex
            with patch.object(self.driver,'launch',side_effect=child):
                self.assertEqual(adapter.run(self.driver,argv,second),adapter.RETRY_EXIT_CODE)
            self.assertEqual(train_calls,[])
            self.assertTrue((output/'order_retry_history'/f'{first}.json').exists())
            self.assertTrue((output/'order_retry_history'/f'{second}.json').exists())

    def test_oom_and_other_errors_keep_original_handling(self):
        for status,error_type,message,expected in [('cuda_oom','OutOfMemoryError','CUDA out of memory.',3),
                ('error','ValueError','configuration failure',1)]:
            with tempfile.TemporaryDirectory() as directory:
                base=Path(directory);cfg=base/'config.json';output=base/'run'
                c=json.loads((self.case/'config.json').read_text(encoding='utf-8-sig'));c.update(trials=1,adam_steps=1)
                dump(cfg,c)
                def child(folder,request):
                    if request['mode']=='preflight':return dict(status='ready',runtime_environment={'device':'cpu'})
                    return dict(status=status,error_type=error_type,error=message)
                argv=['--config',str(cfg),'--output-root',str(output),'--device','cpu','--mode','optimize','--lattices','hexagonal','--timeout-hours','0']
                with patch.object(self.driver,'launch',side_effect=child):
                    self.assertEqual(adapter.run(self.driver,argv,uuid.uuid4().hex),expected)
                self.assertFalse((output/'order_retry.json').exists())
                self.assertFalse((output/'controller.lock').exists())

    def test_later_trial_failure_and_higher_order_restart_at_zero(self):
        import optuna
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory)
            c=json.loads((self.case/'config.json').read_text(encoding='utf-8-sig'));c.update(trials=3,adam_steps=1)
            observed=[]
            def child(folder,request):
                if request['mode']=='preflight':return dict(status='ready',runtime_environment={'device':'cpu'})
                m,t=request['config']['optimization_order'],request['trial_number'];observed.append((m,t))
                if m==14 and t==0:
                    logits=self.driver.baseline_logits(request['config'])
                    return dict(status='completed',best_loss=.2,best_logits=logits,best_step=0,
                                completed_steps=1,radii_nm=self.driver.radii_values(logits,request['config']))
                return dict(status='error',error_type='NonphysicalResult',error='Invalid shared power flux.')
            for order in (14,16):
                cfg=base/f'M{order}.json';output=base/f'run_M{order}';dump(cfg,{**c,'optimization_order':order})
                argv=['--config',str(cfg),'--output-root',str(output),'--device','cpu','--mode','optimize','--lattices','hexagonal','--timeout-hours','0']
                with patch.object(self.driver,'launch',side_effect=child):
                    self.assertEqual(adapter.run(self.driver,argv,uuid.uuid4().hex),adapter.RETRY_EXIT_CODE)
            self.assertEqual(observed,[(14,0),(14,1),(16,0)])

    def test_only_recognized_nonphysical_results_trigger_retry(self):
        self.assertTrue(adapter.is_nonphysical(dict(status='error',error_type='NonphysicalResult',error='Invalid shared power flux.')))
        self.assertFalse(adapter.is_nonphysical(dict(status='error',error_type='ValueError',error='Invalid shared power flux.')))
        self.assertFalse(adapter.is_nonphysical(dict(status='cuda_oom',error_type='NonphysicalResult',error='Invalid shared power flux.')))

if __name__=='__main__':unittest.main()
