"""CPU checks of the diagnostic substitution, sampling grid, and RCWA parity."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import unittest
import validate_normal_grid_v2 as diag

parser=argparse.ArgumentParser()
parser.add_argument('--case-dir',type=Path,default=Path(__file__).resolve().parent/'Al/al_motheye_P200_H500_Nz15')
args,rest=parser.parse_known_args()

def incompatible(sim,radius,index,grid,plan):
    resolution=grid+1
    return resolution

class NormalGridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case=args.case_dir.resolve()
        cls.common,cls.physics,cls.policy=diag.saved.import_case(cls.case)
        import torch
        from optimizer import geometry
        from optimizer.materials import epsilon_at
        from rcwa_ext.acceleration import d6
        from rcwa_ext.acceleration.spectral import SharedEvaluator
        cls.torch,cls.geometry,cls.d6,cls.Evaluator=torch,geometry,d6,SharedEvaluator
        cls.c=cls.common.load_config(cls.case/'config.json')
        cls.device=cls.common.device_setup('cpu',1);cls.waves=[400.,550.,700.]
        logits=diag.saved.read(cls.case/'results/hex_trial1_v3/hexagonal/report.json')['best']['logits']
        cls.radii=geometry.radii_tensor(torch.tensor(logits,dtype=torch.float64),cls.c)/cls.c['period_nm']
        cls.model=SimpleNamespace(material=cls.c['material'],physics=cls.physics,geometry=geometry,epsilon_at=epsilon_at)
        cls.original=staticmethod(d6.native_inputs)
    def tearDown(self):self.d6.native_inputs=self.original
    def cache(self,q):
        e=self.Evaluator(self.model)
        return e,e.cache(self.c,'hexagonal',self.waves,2,q,self.device)
    def powers(self,q,n=None):
        self.d6.native_inputs=(self.original if n is None else diag.unregularized_normal_function(self.original)[0] if n=='off' else diag.fixed_normal_function(self.original,n)[0])
        e,c=self.cache(q)
        with self.torch.no_grad():
            leaves=tuple(v for r in self.radii for v in e.geometry_pair(r,c))
            return [tuple(float(v) for v in e.response(c,leaves,w,c['indices'][i],c['ports'][i],False)) for i,w in enumerate(self.waves)]
    def test_diagonal_inputs_equal_original_all_layers(self):
        _,c=self.cache(48);fixed,_=diag.fixed_normal_function(self.original,48)
        with self.torch.no_grad():
            for r in self.radii:
                a=self.original(c['template'],r,c['indices'][0],48,c['plan'])
                b=fixed(c['template'],r,c['indices'][0],48,c['plan'])
                for x,y in zip(a,b):self.assertTrue(self.torch.equal(x,y))
    def test_material_coefficients_unchanged(self):
        _,c=self.cache(48);r=self.radii[-1]
        a=diag.fixed_normal_function(self.original,48)[0](c['template'],r,c['indices'][1],48,c['plan'])
        b=diag.fixed_normal_function(self.original,64)[0](c['template'],r,c['indices'][1],48,c['plan'])
        self.assertTrue(self.torch.equal(a[0],b[0]));self.assertTrue(self.torch.equal(a[1],b[1]))
        self.assertGreater(float(self.torch.linalg.vector_norm(a[2]-b[2])),1e-5)
    def test_actual_sample_grid_is_used(self):
        from unittest.mock import patch
        fixed,_=diag.fixed_normal_function(self.original,64);calls=[];original_arange=self.torch.arange
        _,c=self.cache(48)
        def arange(*a,**kw):calls.append(a[0]);return original_arange(*a,**kw)
        with patch.object(self.torch,'arange',arange):fixed(c['template'],self.radii[0],c['indices'][0],48,c['plan'])
        self.assertEqual(calls,[48,48])
    def test_legacy_forward_parity_Q48(self):
        a,b=self.powers(48),self.powers(48,48)
        self.assertLess(max(abs(x-y) for p,q in zip(a,b) for x,y in zip(p,q)),1e-10)
    def test_legacy_forward_parity_Q64(self):
        a,b=self.powers(64),self.powers(64,64)
        self.assertLess(max(abs(x-y) for p,q in zip(a,b) for x,y in zip(p,q)),1e-10)
    def test_off_diagonal_conditions_finite(self):
        for q,n in [(48,64),(64,48)]:self.assertTrue(all(self.torch.isfinite(self.torch.tensor(v)).all() for v in self.powers(q,n)))
    def test_unexpected_source_rejected(self):
        with self.assertRaises(ValueError):diag.fixed_normal_function(incompatible,48)
    def test_comparison_axes_and_nonphysical(self):
        rows=[dict(grid=q,normal_reference_grid=n,status='completed',rows=[diag.power_row(550,.1+q*.0001+n*.00001,.2,1e-6)]) for q in [48,64] for n in [48,64]]
        r=diag.comparisons(rows,[48,64],[48,64],.00025)
        self.assertEqual(len(r['sampling_comparisons']),2);self.assertEqual(len(r['smoothing_comparisons']),2)
        self.assertAlmostEqual(r['sampling_comparisons'][0]['details']['reflectance']['max_abs_difference'],.0016)
        self.assertAlmostEqual(r['smoothing_comparisons'][0]['details']['reflectance']['max_abs_difference'],.00016)
        rows[-1]['status']='nonphysical';r=diag.comparisons(rows,[48,64],[48,64],.00025)
        self.assertIsNone(r['sampling_comparisons'][-1]['passed']);self.assertIsNone(r['smoothing_comparisons'][-1]['passed'])


    def test_off_material_coefficients_unchanged(self):
        _,c=self.cache(48);r=self.radii[-1]
        a=self.original(c['template'],r,c['indices'][1],48,c['plan'])
        b=diag.unregularized_normal_function(self.original)[0](c['template'],r,c['indices'][1],48,c['plan'])
        self.assertTrue(self.torch.equal(a[0],b[0]));self.assertTrue(self.torch.equal(a[1],b[1]))
        self.assertGreater(float(self.torch.linalg.vector_norm(a[2]-b[2])),1e-5)
    def test_off_does_not_call_smoothstep(self):
        from unittest.mock import patch
        _,c=self.cache(48)
        def forbidden(*a,**k):raise AssertionError('smoothstep called in OFF mode')
        with patch.dict(self.original.__globals__,{'_smoothstep':forbidden}):
            raw=diag.unregularized_normal_function(self.original)[0]
            raw(c['template'],self.radii[-1],c['indices'][1],48,c['plan'])
    def test_off_projection_independent_of_circle_radius(self):
        _,c=self.cache(48);raw=diag.unregularized_normal_function(self.original)[0]
        a=raw(c['template'],self.radii[0],c['indices'][1],48,c['plan'])[2]
        b=raw(c['template'],self.radii[-1],c['indices'][1],48,c['plan'])[2]
        self.assertTrue(self.torch.equal(a,b))
    def test_off_projection_is_hermitian(self):
        _,c=self.cache(48);raw=diag.unregularized_normal_function(self.original)[0]
        a=raw(c['template'],self.radii[-1],c['indices'][1],48,c['plan'])[2]
        self.assertLess(float(self.torch.linalg.vector_norm(a-a.mH)),1e-14)
    def test_off_ordinary_point_matches_exact_projector(self):
        t=self.torch
        u=t.tensor([[[3.]],[[20.]]],dtype=t.float64);v=t.tensor([[[4.]],[[30.]]],dtype=t.float64)
        d=t.topk(u*u+v*v,k=2,dim=0,largest=False).values
        fields=diag.unsmoothed_projection(t.tensor([[.6]]),t.tensor([[.8]]),u,v,d,1.,1.)
        for a,b in zip(fields,[.36,.48,.64]):self.assertAlmostEqual(float(a),b,places=6)
    def test_off_center_and_all_ties_have_explicit_conventions(self):
        import math
        t=self.torch
        def field(us,vs):
            u=t.tensor(us,dtype=t.float64).reshape(-1,1,1);v=t.tensor(vs,dtype=t.float64).reshape(-1,1,1)
            d,idx=t.topk(u*u+v*v,k=2,dim=0,largest=False)
            first=int(idx[0,0,0]);den=max(float(d[0,0,0])**.5,1e-14)
            return diag.unsmoothed_projection(u[first]/den,v[first]/den,u,v,d,1.,1.)
        for us,vs in [([0.,2.],[0.,0.]),([1.,-1.],[1.,1.]),([1.,-.5,-.5],[0.,math.sqrt(3)/2,-math.sqrt(3)/2])]:
            fields=field(us,vs)
            for a,b in zip(fields,[.5,0.,.5]):self.assertAlmostEqual(float(a),b,places=12)
    def test_off_forward_finite(self):
        for q in [48,64]:self.assertTrue(all(self.torch.isfinite(self.torch.tensor(v)).all() for v in self.powers(q,'off')))
    def test_off_unexpected_source_rejected(self):
        with self.assertRaises(ValueError):diag.unregularized_normal_function(incompatible)

if __name__=='__main__':unittest.main(argv=[__file__]+rest,verbosity=2)
