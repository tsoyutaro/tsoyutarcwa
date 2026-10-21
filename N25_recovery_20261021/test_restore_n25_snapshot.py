"""Standard-library tests for the Git-delivered recovery files."""
from pathlib import Path
import contextlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0,str(HERE))
import restore_n25_snapshot as recovery


def project_fixture(directory):
    project = Path(directory)/'project'
    project.mkdir()
    index,_ = recovery.validate_snapshot(HERE)
    for name in index['expected_project_sources_sha256_lf']:
        target = project/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(PROJECT/name,target)
    shutil.copy2(PROJECT/'run_n25_nv_optuna_adam.py',project/'run_n25_nv_optuna_adam.py')
    return project,index


class GitRecoveryTests(unittest.TestCase):
    def test_actual_cli_restores_bytes_and_preserves_original_results(self):
        with tempfile.TemporaryDirectory() as directory:
            project,index = project_fixture(directory)
            old = project/'results/incident.txt'
            old.parent.mkdir()
            old.write_text('preserve this incident')
            result = subprocess.run([sys.executable,str(HERE/'restore_n25_snapshot.py'),
                '--project-root','.'],cwd=str(project),text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)
            target = project/recovery.DEFAULT_OUTPUT
            for name,expected in index['files'].items():
                self.assertEqual(recovery.digest((target/name).read_bytes()),expected)
            self.assertEqual(old.read_text(),'preserve this incident')
            self.assertFalse((target/'controller.lock').exists())
            self.assertFalse((target/'hexagonal').exists())
            self.assertEqual(index['snapshot']['next_adam_step'],156)
            self.assertEqual(index['snapshot']['completed_square_trials'],list(range(13)))
            recovery.validate_project(project,index)
            self.assertIn('--timeout-hours 0',result.stdout)

    def test_existing_destination_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            project,_ = project_fixture(directory)
            target = project/'existing'
            target.mkdir()
            marker = target/'keep.txt'
            marker.write_text('keep')
            with self.assertRaisesRegex(ValueError,'already exists'):
                recovery.restore(project,target,HERE)
            self.assertEqual(marker.read_text(),'keep')

    def test_changed_payload_rejected_before_result_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            project,_ = project_fixture(directory)
            bundle = Path(directory)/'bundle'
            shutil.copytree(HERE,bundle,ignore=shutil.ignore_patterns('__pycache__'))
            (bundle/'snapshot/square/sampler.pkl').write_bytes(b'changed')
            target = project/'recovered'
            with self.assertRaisesRegex(ValueError,'checksum differs'):
                recovery.restore(project,target,bundle)
            self.assertFalse(target.exists())

    def test_changed_physics_rejected_before_result_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            project,_ = project_fixture(directory)
            (project/'rcwa_ext/nvm.py').write_text('# modified')
            target = project/'recovered'
            with self.assertRaisesRegex(ValueError,'source differs'):
                recovery.restore(project,target,HERE)
            self.assertFalse(target.exists())

    def test_bad_paths_and_outside_project_rejected(self):
        for name in ('../outside','/absolute','C:/outside','..\\outside'):
            with self.subTest(name=name),self.assertRaises(ValueError):
                recovery.safe_relative(name)
        with tempfile.TemporaryDirectory() as directory:
            project,_ = project_fixture(directory)
            with self.assertRaises(ValueError):
                recovery.restore(project,Path(directory)/'outside',HERE)

    @unittest.skipUnless(shutil.which('git'),'Git executable is required for checkout test')
    def test_git_checkout_with_autocrlf_preserves_all_snapshot_bytes(self):
        def git(*args):
            result = subprocess.run(['git',*args],text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)
            return result
        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory)/'original'
            original.mkdir()
            shutil.copytree(HERE,original/HERE.name,ignore=shutil.ignore_patterns('__pycache__'))
            git('init','-q',str(original))
            git('-C',str(original),'config','core.autocrlf','true')
            git('-C',str(original),'add','--',HERE.name)
            git('-C',str(original),'-c','user.name=Recovery test','-c','user.email=test@example.invalid',
                '-c','commit.gpgsign=false','commit','-qm','Temporary recovery checkout test')
            clone = Path(directory)/'checkout'
            clone.mkdir()
            # Exercise Git's checkout filters directly. A local clone can invoke
            # an MSYS shell that is unavailable inside some Windows sandboxes.
            git('-C',str(original),'--work-tree',str(clone),'-c','core.autocrlf=true',
                'checkout-index','--all','--force')
            index,_ = recovery.validate_snapshot(clone/HERE.name)
            self.assertEqual(len(index['files']),124)
            result = subprocess.run([sys.executable,str(clone/HERE.name/'restore_n25_snapshot.py'),
                '--inspect-only'],text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(json.loads(result.stdout)['next_adam_step'],156)
            self.assertFalse(list((clone/HERE.name).rglob('*.zip')))


if __name__ == '__main__':
    unittest.main(verbosity=2)
