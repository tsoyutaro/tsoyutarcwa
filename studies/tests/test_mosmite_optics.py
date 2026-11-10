"""Checks of analytic flux balance and optional phase quadrature."""
import argparse,importlib.util,math,sys,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import mosmite_optics as m
from mosmite_phase_average import phase_average

def transfer_reference(indices,depths,wl):
    # Independent2x2 intensity transfer implementation of Byrnes Eqs25--30.
    L=np.eye(2)
    for i,(a,b) in enumerate(zip(indices,indices[1:])):
        r=(a-b)/(a+b);tf=2*a/(a+b);tb=2*b/(a+b)
        Rf=abs(r)**2;Rb=Rf;Tf=abs(tf)**2*b.real/a.real;Tb=abs(tb)**2*a.real/b.real
        L=L@(np.array([[1,-Rb],[Rf,Tf*Tb-Rf*Rb]])/Tf)
        if i<len(depths):
            P=np.exp(-4*np.pi*b.imag*depths[i]/wl);L=L@np.diag([1/P,P])
    return L[1,0]/L[0,0],1/L[0,0]

class OpticsTests(unittest.TestCase):
    def test_lossless_slab_closed_form(self):
        n=1.57;r=((n-1)/(n+1))**2
        out=m.finite_sample(m.interface(1,n),n,[(n,54800,True)],600)
        self.assertAlmostEqual(out['R'],2*r/(1+r),places=14)
        self.assertAlmostEqual(out['T'],(1-r)/(1+r),places=14)
        self.assertAlmostEqual(out['A'],0,places=14)
    def test_absorbing_slab_analytic_geometric_series(self):
        n=1.57+.012j;d=1234.;wl=600.;r=(1-n)/(1+n);tf=2/(1+n);tb=2*n/(1+n)
        P=np.exp(-4*np.pi*n.imag*d/wl);D=1-abs(r)**4*P**2
        R=abs(r)**2+abs(tf*tb*r)**2*P**2/D;T=abs(tf*tb)**2*P/D
        out=m.finite_sample(m.interface(1,n),n,[(n,d,True)],wl)
        self.assertAlmostEqual(out['R'],R,places=14);self.assertAlmostEqual(out['T'],T,places=14)
    def test_absorbing_coherent_slab_airy(self):
        n=1.57+.012j;d=1234.;wl=600.;r=(1-n)/(1+n);p=np.exp(2j*np.pi*n*d/wl);D=1-r*r*p*p
        R=abs(r*(1-p*p)/D)**2;T=abs(4*n/(1+n)**2*p/D)**2
        out=m.finite_sample(m.interface(1,n),n,[(n,d,True)],wl,coherence='coherent')
        self.assertAlmostEqual(out['R'],R,places=14);self.assertAlmostEqual(out['T'],T,places=14)
    def test_multilayers_against_independent_intensity_transfer(self):
        rng=np.random.default_rng(123)
        for length in [1,2,3,5]:
            for repeat in range(8):
                ns=1.2+rng.random(length)+1j*rng.random(length)*.001
                ds=1000+2000*rng.random(length);layers=list(zip(ns,ds,[True]*length))
                out=m.finite_sample(m.interface(1,ns[0]),ns[0],layers,550)
                R,T=transfer_reference([1+0j,*ns,1+0j],ds,550)
                self.assertAlmostEqual(out['R'],R,places=12);self.assertAlmostEqual(out['T'],T,places=12)
    def test_lossless_many_interfaces_conserve_power(self):
        ns=[1.57,1.5,1.46];out=m.finite_sample(m.interface(1,ns[0]),ns[0],[(n,10000,True) for n in ns],550)
        self.assertAlmostEqual(out['R']+out['T'],1,places=14)
    def test_opaque_layer_avoids_transfer_overflow(self):
        n=1.5+.1j;out=m.finite_sample(m.interface(1,n),n,[(n,1e8,True)],550)
        self.assertEqual(out['T'],0.);self.assertAlmostEqual(out['R'],abs((1-n)/(1+n))**2,places=14)
    def test_coherent_quarterwave_AR_on_incoherent_substrate(self):
        n=1.5;coat=np.sqrt(n);wl=550.
        out=m.finite_sample(m.identity(),1,[(coat,wl/(4*coat),False),(n,1e6,True)],wl)
        R=((n-1)/(n+1))**2
        self.assertAlmostEqual(out['R'],R,places=14);self.assertAlmostEqual(out['T'],1-R,places=14)
    def test_thin_coherent_gap_is_retained(self):
        n=1.5;wl=550.;base=[(n,5000,True),(n,20000,True)]
        direct=m.finite_sample(m.interface(1,n),n,base,wl)
        half=m.finite_sample(m.interface(1,n),n,[base[0],(1,wl/2,False),base[1]],wl)
        quarter=m.finite_sample(m.interface(1,n),n,[base[0],(1,wl/4,False),base[1]],wl)
        self.assertAlmostEqual(direct['T'],half['T'],places=14)
        self.assertGreater(abs(quarter['T']-direct['T']),.05)
        self.assertAlmostEqual(quarter['R']+quarter['T'],1,places=14)
    def test_zero_thickness_layers_are_absent(self):
        plain=m.finite_sample(m.interface(1,1.5),1.5,[(1.5,5000,True)],550)
        zero=m.finite_sample(m.interface(1,1.5),1.5,[(1.5,5000,True),(2+.5j,0,True)],550)
        self.assertEqual(plain,zero)
    def test_jones_feedback_preserves_polarization_coherency(self):
        n=1.5;rf=np.diag(np.sqrt([.1,.2]));tf=np.diag(np.sqrt([.7,.6]));tb=tf.copy()
        rb=np.array([[.12,.07j],[.04j,.16]],complex)
        surface=(tf/np.sqrt(n),rf,rb,tb*np.sqrt(n));out=m.finite_sample(surface,n,[(n,10000,True)],550)
        Rtail=((n-1)/(n+1))**2
        for j,label in enumerate(['x','y']):
            C=np.zeros((2,2),complex);C[j,j]=1;term=tf@C@tf.conj().T;F=term.copy()
            for k in range(40):term=Rtail*rb@term@rb.conj().T;F+=term
            R=rf@C@rf.conj().T+Rtail*tb@F@tb.conj().T;T=(1-Rtail)*F
            self.assertAlmostEqual(out['R_'+label],np.trace(R).real,places=13)
            self.assertAlmostEqual(out['T_'+label],np.trace(T).real,places=13)
        self.assertGreater(out['cross_R'],0)
    def test_no_phase_mesh_or_sample_mean_is_called(self):
        with patch.object(np,'meshgrid',side_effect=AssertionError('phase mesh')),patch.object(np,'mean',side_effect=AssertionError('sample mean')):
            out=m.finite_sample(m.interface(1,1.5),1.5,[(1.5,5000,True),(1.46,1e6,True)],550)
        self.assertTrue(0<out['T']<1)
    def test_absorbing_interface_T_is_not_forced_to_one_minus_R(self):
        a=1.5+.2j;b=1.3+.01j;power=m.power_block(m.interface(a,b),a,b)
        R=power[1][0,0].real;T=power[0][0,0].real
        self.assertGreater(abs(R+T-1),1e-4)
    def test_invalid_inputs_rejected(self):
        for layers in [[(1.5,-1,True)],[(1.5-.1j,100,True)]]:
            with self.assertRaises(ValueError):m.finite_sample(m.interface(1,1.5),1.5,layers,550)
        with self.assertRaises(ValueError):m.finite_sample(m.identity(),1,[],550,coherence='phase-average')
    def test_uncoated_adapter_and_cli_selectors(self):
        spec=importlib.util.spec_from_file_location('test_uncoated',ROOT/'mosmite_uncoated/run_spectrum.py');u=importlib.util.module_from_spec(spec);spec.loader.exec_module(u)
        args=u.parser().parse_args(['--coherent-layers','PET,adhesive']);self.assertEqual(args.coherence,'incoherent')
        layers=u.bulk_layers(args,550,lambda w:1.5,lambda w:1.4)
        self.assertEqual([r[2] for r in layers],[False,False,True])
        out=u.finite_sample(u.interface(1,1.5),1.5,[(1.5,5000,True)],550)
        ref=m.finite_sample(m.interface(1,1.5),1.5,[(1.5,5000,True)],550)
        np.testing.assert_allclose(out,[ref['R'],ref['T'],ref['A']],atol=1e-14)
        phase_args = u.parser().parse_args(['--coherence', 'phase-average', '--phase-points', '8'])
        self.assertEqual(phase_args.phase_points, 8)
        actual, diagnostic = u.evaluate_tail(phase_args, u.interface(1, 1.5), 1.5, [(1.5, 5000, True)], 550)
        np.testing.assert_allclose(actual, out, rtol=0, atol=1e-10)
        self.assertIn('phase_average_error', diagnostic)

    def test_phase_average_matches_independent_coherent_grid_with_mixed_layers(self):
        surface = m.interface(1, 1.5)
        surface[0][0, 1] = .06j
        wl, points = 550, 3
        layers = [(1.5, 5000, True), (1.38, 110, False), (1.46, 1000, True)]
        coherent = []
        for pet in range(points):
            for glass in range(points):
                shifts = [pet, 0, glass]
                shifted = [(n, d + wl * step / (2 * n * points), False)
                           for (n, d, _), step in zip(layers, shifts)]
                coherent.append(m.finite_sample(surface, 1.5, shifted, wl, coherence='coherent'))
        actual, _ = phase_average(surface, 1.5, layers, wl, points=points, tolerance=1.)
        for key in actual:
            self.assertAlmostEqual(actual[key], np.mean([row[key] for row in coherent]), places=13)

    def test_phase_average_reports_unresolved_phase_grid(self):
        surface = m.interface(1, 3)
        with self.assertRaisesRegex(RuntimeError, 'increase --phase-points'):
            phase_average(surface, 3, [(3, 550, True)], 550, points=2, tolerance=1e-6)

    def test_nonfinite_surface_is_rejected(self):
        with self.assertRaises(ValueError):m.finite_sample((float('nan'),0,0,1),1,[],550)
    def test_pointwise_comparison_keeps_measurement_oscillations(self):
        import tempfile,csv
        spec=importlib.util.spec_from_file_location('test_pointwise',ROOT/'mosmite_uncoated/compare_pointwise.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
        with tempfile.TemporaryDirectory() as folder:
            a=Path(folder)/'sim.csv';b=Path(folder)/'meas.csv'
            a.write_text('wavelength_nm,T\n500,0.9\n505,0.9\n510,0.9\n',encoding='utf-8')
            b.write_text('wavelength_nm,T_grand_mean_percent\n500,90\n505,100\n510,90\n',encoding='utf-8')
            rows,metric=c.compare(a,b,'T_grand_mean_percent')
            self.assertEqual([r['measured_T_percent'] for r in rows],[90,100,90])
            self.assertAlmostEqual(metric['RMSE_pp'],np.sqrt(100/3),places=12)
    def test_pointwise_does_not_interpolate_or_smooth(self):
        import tempfile
        spec=importlib.util.spec_from_file_location('test_pointwise2',ROOT/'mosmite_uncoated/compare_pointwise.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
        with tempfile.TemporaryDirectory() as folder:
            a=Path(folder)/'sim.csv';b=Path(folder)/'meas.csv'
            a.write_text('wavelength_nm,T\n502.5,0.9\n',encoding='utf-8')
            b.write_text('wavelength_nm,T_grand_mean_percent\n500,90\n505,100\n',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'absent'):c.compare(a,b,'T_grand_mean_percent')
    def test_reference_parity_checks_coherent_layer_selection(self):
        import tempfile,argparse,json
        spec=importlib.util.spec_from_file_location('test_gold_memory',ROOT/'mosmite_gold_deposition/memory.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
        with tempfile.TemporaryDirectory() as folder:
            old=dict(status='completed',arguments=dict(coherence='incoherent',coherent_layers=''),optics_sha256='same')
            Path(folder,'settings.json').write_text(json.dumps(old),encoding='utf-8')
            metadata=dict(arguments=dict(coherence='incoherent',coherent_layers='PET'),optics_sha256='same')
            with self.assertRaisesRegex(ValueError,'coherent_layers'):g.compare_reference(argparse.Namespace(reference_dir=folder),metadata,[])
    def test_reference_parity_checks_shared_optics_source(self):
        import tempfile,argparse,json
        spec=importlib.util.spec_from_file_location('test_gold_memory2',ROOT/'mosmite_gold_deposition/memory.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
        with tempfile.TemporaryDirectory() as folder:
            old=dict(status='completed',arguments=dict(coherence='incoherent',coherent_layers=''),optics_sha256='old')
            Path(folder,'settings.json').write_text(json.dumps(old),encoding='utf-8')
            metadata=dict(arguments=dict(coherence='incoherent',coherent_layers=''),optics_sha256='new')
            with self.assertRaisesRegex(ValueError,'analytic-optics'):g.compare_reference(argparse.Namespace(reference_dir=folder),metadata,[])
    def test_gold_adapter(self):
        spec=importlib.util.spec_from_file_location('test_gold_stack',ROOT/'mosmite_gold_deposition/stack.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
        n=1.5;layers=[(n,5000,True)]
        self.assertEqual(g.finite_sample(m.interface(1,n),n,layers,550),m.finite_sample(m.interface(1,n),n,layers,550))

if __name__=='__main__':unittest.main(verbosity=2)
