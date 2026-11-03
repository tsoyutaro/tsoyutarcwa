"""Batch ordering, failure handling and strict Cr identity checks without a GPU."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import continue_convergence as cc
import postprocess_metal_off_optuna30 as batch


class BatchTests(unittest.TestCase):
    def run_batch(self,oom=None,nonphysical=None,spectrum_error=None,extra=()):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);calls=[];specs={}
            cases={'Al':('Al/al_motheye_P200_H500_Nz15',8),
                   'Ti':('Ti/ti_motheye_P200_H500_Nz15',8),'Cr':('cr_motheye',8)}
            for material,(relative,_) in cases.items():
                case=root/relative;run=case/'results/hex_OFF_optuna30_v1_M8_Q512'
                cc.write(case/'config_hex_OFF_optuna30_M8_Q512.json',dict(material=material,normal_mode='off',layers=15,
                    lattices=['hexagonal'],optimization_order=8,optimization_grid=512))
                cc.write(run/'workflow.json',dict(status='completed'))
                cc.write(run/'manifest.json',{})
                cc.write(run/'hexagonal/report.json',dict(best=dict(logits=[0.]*16)))
                specs[material]=(case,run)
            def launch(command,env):
                def arg(name):return command[command.index(name)+1]
                if Path(command[1]).name=='continue_convergence.py':
                    material=arg('--materials');calls.append((material,'convergence'))
                    folder=Path(arg('--root'))
                    status='cuda_oom' if material==oom else ('nonphysical' if material==nonphysical else 'completed')
                    record=dict(order=40,grid=512,status=status)
                    report=folder/'report.json';cc.write(report,dict(records=[record]))
                    cc.write(folder/'convergence_continue_summary.json',dict(materials={material:
                        dict(status='completed' if status=='completed' else 'completed_with_skips',report=str(report))}))
                    return 0
                material=next(k for k,(case,_) in specs.items() if Path(command[1]).parent==case)
                calls.append((material,'spectrum'))
                if material==spectrum_error:return 1
                run=specs[material][1];report=run/'spectrum_report.json'
                waves=cc.numbers(arg('--wavelengths'))
                cc.write(report,dict(status='completed',wavelengths_nm=waves,spectrum_csv=str(run/'spectrum.csv'),
                    outcomes=[dict(order=int(arg('--order')),grid=int(arg('--grid')),
                                   rows=[{} for _ in waves],objective=.123)]))
                cc.write(run/'hexagonal/latest_spectrum.json',dict(report=str(report)))
                return 0
            with patch.object(batch,'ROOT',root),patch.object(batch,'CASES',cases),patch.object(batch,'launch',side_effect=launch),contextlib.redirect_stdout(io.StringIO()):
                code=batch.main(['--device','cpu',*extra])
            return code,calls,cc.read(root/'postprocess_OFF_optuna30_v1/summary.json')

    def test_all_materials_run_convergence_then_101_point_spectrum(self):
        code,calls,summary=self.run_batch()
        self.assertEqual(code,0)
        self.assertEqual(calls,[(m,mode) for m in ('Al','Ti','Cr') for mode in ('convergence','spectrum')])
        self.assertEqual(summary['orders'],list(range(4,41,2)))
        self.assertEqual(summary['convergence_wavelengths_nm'],[400.,550.,700.])
        self.assertEqual(len(summary['spectrum_wavelengths_nm']),101)
        self.assertEqual(summary['materials']['Cr']['mean_R_percent'],12.3)

    def test_oom_skips_spectrum_and_continues_to_cr(self):
        code,calls,summary=self.run_batch(oom='Ti')
        self.assertEqual(code,1)
        self.assertNotIn(('Ti','spectrum'),calls)
        self.assertIn(('Cr','spectrum'),calls)
        self.assertEqual(summary['materials']['Ti']['spectrum_status'],'skipped_after_oom')

    def test_nonphysical_target_is_not_repeated_for_101_points(self):
        code,calls,summary=self.run_batch(nonphysical='Al')
        self.assertEqual(code,1)
        self.assertNotIn(('Al','spectrum'),calls)
        self.assertIn(('Cr','spectrum'),calls)
        self.assertEqual(summary['materials']['Al']['spectrum_status'],'skipped_target_convergence_failed')

    def test_spectrum_failure_continues_to_next_material(self):
        code,calls,summary=self.run_batch(spectrum_error='Al')
        self.assertEqual(code,1)
        self.assertIn(('Cr','spectrum'),calls)
        self.assertEqual(summary['materials']['Al']['status'],'error')

    def test_prepare_only_does_not_launch_jobs(self):
        code,calls,summary=self.run_batch(extra=['--prepare-only'])
        self.assertEqual(code,0);self.assertEqual(calls,[])
        self.assertEqual(summary['status'],'prepared')


class CrChecks(unittest.TestCase):
    def test_cr_preserves_strict_signature_and_runtime_policy(self):
        identity=dict(version=1,config={'material':'Cr'},source_sha256={'a':'b'},requested_device='cuda:0')
        common=SimpleNamespace(signature_for=cc.digest,sources=lambda:{'a':'b'})
        previous=dict(signature=cc.digest(identity))
        self.assertEqual(cc.execution_sources(common,None),{'a':'b'})
        self.assertIs(cc.check_identity(common,None,previous,identity),previous)
        with self.assertRaises(ValueError):
            cc.check_identity(common,None,previous,{**identity,'requested_device':'cpu'})
        cc.check_environment(None,{'device':'cuda:0'},{'device':'cuda:0'})
        with self.assertRaises(ValueError):
            cc.check_environment(None,{'device':'cuda:0'},{'device':'cuda:1'})


if __name__=='__main__':unittest.main()
