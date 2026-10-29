from pathlib import Path
import importlib.util
import json
import math
import unittest

import torch

torch.set_num_threads(1)
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('pmma_metal_geometry',HERE.parent/'Al/pmma_al_motheye_P200_H500_Nz15/optimizer/geometry.py')
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
CONFIG=dict(layers=15,radius_min_nm=.001,radius_max_nm=99.999,min_shell_thickness_nm=.001,
            softmax_gap_floor=1e-6,period_nm=200.,height_nm=500.)


def assert_geometry(test,r,c=CONFIG):
    test.assertEqual(tuple(r.shape),(c['layers'],2))
    test.assertTrue(bool(torch.isfinite(r).all()))
    test.assertTrue(bool((r[1:]-r[:-1]>0).all()))
    test.assertTrue(bool((r[:,1]-r[:,0]>=c['min_shell_thickness_nm']-2e-12).all()))
    test.assertGreater(float(r[:,0].min().detach()),c['radius_min_nm'])
    test.assertLess(float(r[:,1].max().detach()),c['radius_max_nm'])


class GeometryTests(unittest.TestCase):
    def test_exact_default_linear_profiles(self):
        x=torch.tensor(g.baseline_logits(CONFIG),dtype=torch.float64)
        self.assertEqual(x.numel(),32)
        r=g.radii_tensor(x,CONFIG)
        expected=torch.stack((torch.linspace(5.,65.,15,dtype=torch.float64),
                              torch.linspace(35.,95.,15,dtype=torch.float64)),dim=1)
        torch.testing.assert_close(r,expected,rtol=0,atol=3e-14)
        assert_geometry(self,r)

    def test_random_and_extreme_logits_are_always_feasible(self):
        generator=torch.Generator().manual_seed(54320)
        probes=[torch.randn(32,generator=generator,dtype=torch.float64)*scale for scale in (.01,1.,20.,1000.) for _ in range(64)]
        probes += [torch.full((32,),1000.,dtype=torch.float64),torch.full((32,),-1000.,dtype=torch.float64),
                   torch.linspace(-1000,1000,32,dtype=torch.float64),
                   torch.tensor([1000.,-1000.]*16,dtype=torch.float64)]
        for x in probes:assert_geometry(self,g.radii_tensor(x,CONFIG))

    def test_python_and_tensor_agree(self):
        generator=torch.Generator().manual_seed(9765)
        for scale in (1.,30.,1000.):
            for _ in range(16):
                x=torch.randn(32,generator=generator,dtype=torch.float64)*scale
                a=g.radii_tensor(x,CONFIG)
                b=torch.tensor(g.radii_values(x.tolist(),CONFIG),dtype=torch.float64)
                torch.testing.assert_close(a,b,rtol=0,atol=8e-14)

    def test_arbitrary_nonconstant_thickness_profile_roundtrip(self):
        core=[5.+.4*j+.05*j*j for j in range(15)]
        outer=[30.+.9*j+.08*j*j for j in range(15)]
        x=g.initial_logits_from_radii(core,outer,CONFIG)
        actual=g.radii_values(x,CONFIG)
        for a,b in zip(actual,zip(core,outer)):
            self.assertAlmostEqual(a[0],b[0],places=12)
            self.assertAlmostEqual(a[1],b[1],places=12)
        thickness=[b-a for a,b in actual]
        self.assertGreater(max(thickness)-min(thickness),1.)

    def test_decreasing_core_outer_ratio_is_representable(self):
        x=g.initial_logits(5.,20.,10.,90.,CONFIG)
        r=g.radii_tensor(torch.tensor(x,dtype=torch.float64),CONFIG)
        assert_geometry(self,r)
        ratios=r[:,0]/r[:,1]
        self.assertTrue(bool((ratios[1:]<ratios[:-1]).all()))
        self.assertAlmostEqual(float(ratios[0]),.5,places=12)
        self.assertAlmostEqual(float(ratios[-1]),2/9,places=12)

    def test_crossing_latents_and_minimum_shell_ties(self):
        a=g.initial_logits(5.,95.,8.,98.,CONFIG)[:16]
        b=g.initial_logits(35.,65.,38.,68.,CONFIG)[:16]
        x=torch.tensor(a+b,dtype=torch.float64,requires_grad=True)
        r=g.radii_tensor(x,CONFIG)
        assert_geometry(self,r)
        self.assertAlmostEqual(float(r[0,0].detach()),5.,places=12)
        self.assertAlmostEqual(float(r[-1,0].detach()),65.,places=12)
        gradient=torch.autograd.grad((r*r).sum(),x)[0]
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        # Equal latent profiles encode a shell exactly at its numerical minimum.
        y=torch.tensor(a+a,dtype=torch.float64,requires_grad=True)
        equal=g.radii_tensor(y,CONFIG)
        assert_geometry(self,equal)
        torch.testing.assert_close(equal[:,1]-equal[:,0],torch.full((15,),.001,dtype=torch.float64),rtol=0,atol=2e-12)

    def test_full_jacobian_matches_finite_differences_away_from_ties(self):
        x=torch.tensor(g.baseline_logits(CONFIG),dtype=torch.float64,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda z:g.radii_tensor(z,CONFIG), (x,),eps=1e-6,atol=1e-6,rtol=1e-5))

    def test_finite_difference_directional_objective(self):
        x=torch.tensor(g.baseline_logits(CONFIG),dtype=torch.float64,requires_grad=True)
        d=torch.sin(torch.arange(1,33,dtype=torch.float64));d/=torch.linalg.vector_norm(d)
        def loss(z):
            r=g.radii_tensor(z,CONFIG)/100.
            return (r[:,0].square()*torch.linspace(.4,1.2,15,dtype=torch.float64)+.4*torch.sin(r[:,1])).mean()
        gradient=torch.autograd.grad(loss(x),x)[0]
        h=1e-5
        finite=float((loss(x.detach()+h*d)-loss(x.detach()-h*d))/(2*h))
        autodiff=float((gradient*d).sum())
        self.assertLess(abs(finite-autodiff),1e-9)

    def test_profile_output_and_other_layer_counts(self):
        for n in (2,5,10,15,20):
            c={**CONFIG,'layers':n}
            x=g.baseline_logits(c)
            self.assertEqual(len(x),2*(n+1))
            assert_geometry(self,g.radii_tensor(torch.tensor(x,dtype=torch.float64),c),c)
            rows=g.profile_rows(x,c)
            self.assertEqual(len(rows),n)
            self.assertAlmostEqual(rows[-1]['bottom_depth_nm'],500.)
            self.assertAlmostEqual(rows[0]['pmma_radius_nm'],5.)
            self.assertAlmostEqual(rows[-1]['metal_radius_nm'],95.)
            self.assertAlmostEqual(rows[0]['metal_shell_thickness_nm'],30.)

    def test_invalid_profiles_and_settings_are_rejected(self):
        for update in [dict(layers=1),dict(min_shell_thickness_nm=0),dict(radius_max_nm=100.),
                       dict(min_shell_thickness_nm=100.),dict(softmax_gap_floor=.1),dict(period_nm=float('nan'))]:
            with self.assertRaises(ValueError):g.baseline_logits({**CONFIG,**update})
        for values in [(5.,65.,4.,95.),(5.,65.,35.,64.),(5.,5.,35.,95.)]:
            with self.assertRaises(ValueError):g.initial_logits(*values,CONFIG)
        with self.assertRaises(ValueError):g.radii_tensor(torch.zeros(31),CONFIG)
        with self.assertRaises(ValueError):g.radii_values([float('nan')]*32,CONFIG)
        # A requested profile outside the configured numerical gap interior is not silently clipped.
        with self.assertRaises(ValueError):
            g.initial_logits_from_radii([5.+1e-9*j for j in range(15)],[35.+j for j in range(15)],CONFIG)


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(GeometryTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report=dict(status='passed' if result.wasSuccessful() else 'failed',tests_run=result.testsRun,
                failures=len(result.failures),errors=len(result.errors),device='cpu',torch=str(torch.__version__),
                parameter_count=32,layers=15,representation='sorted independent cumulative-softmax latent profiles',
                ratio_constraint='none',coating_thickness_constraint='minimum only; not fixed',
                numerical_domain='Each latent gap > radial_span*softmax_gap_floor for finite inverse logits.')
    (HERE.parent/'validation_reports/geometry_tests.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
