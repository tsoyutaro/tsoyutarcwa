"""Do not confuse flux admissibility, operator passivity and convergence."""
from __future__ import annotations
from types import SimpleNamespace
import unittest

from ..run import physical,make_report


class ReportingTests(unittest.TestCase):
    def report(self,rows):
        args=SimpleNamespace(methods=['slice-nvm'],polarizations=['x'],wavelengths=[700.],
                             axis='order',tolerance=5e-4,physics_tolerance=1e-6)
        return make_report(rows,args,[2,4,6],dict(passed=True))

    def rows(self,**changes):
        result=[]
        for order in (2,4,6):
            row=dict(method='slice-nvm',polarization='x',wavelength_nm=700.,axis_value=order,
                     R=.1,P_sub=.2,A_moth=.7,A_total=.9,epsilon_real=-16.,epsilon_imag=1.)
            row.update(changes)
            row['physical']=physical(row,1e-6)
            row['status']='OK' if row['physical']['passed'] else 'UNPHYSICAL'
            result.append(row)
        return result

    def test_operator_loss_failure_does_not_report_flux_failure(self):
        report=self.report(self.rows(constitutive_loss_min_eigenvalue=-.1))
        self.assertTrue(report['passive_flux_bounds_passed'])
        self.assertFalse(report['physical_checks_passed'])
        self.assertEqual(report['constitutive_loss_flagged_cases'],3)
        self.assertTrue(report['axis_numerically_converged'])
        self.assertFalse(report['axis_converged'])
        self.assertFalse(report['passed'])
        self.assertTrue(report['curves'][0]['passive_flux_bounds'])
        self.assertFalse(report['curves'][0]['physical_checks_passed'])

    def test_actual_negative_absorption_still_fails_flux_and_overall_checks(self):
        rows=self.rows(R=1.1,A_moth=-.3,A_total=-.1)
        report=self.report(rows)
        self.assertFalse(report['passive_flux_bounds_passed'])
        self.assertFalse(report['physical_checks_passed'])
        self.assertIn('A_moth',rows[0]['physical']['failed_checks'])
        self.assertFalse(report['passed'])

    def test_positive_loss_and_converged_flux_pass(self):
        report=self.report(self.rows(constitutive_loss_min_eigenvalue=.1))
        self.assertTrue(report['passive_flux_bounds_passed'])
        self.assertTrue(report['physical_checks_passed'])
        self.assertTrue(report['axis_converged'])
        self.assertTrue(report['passed'])

    def test_old_checkpoints_with_only_nested_checks_are_supported(self):
        rows=self.rows(constitutive_loss_min_eigenvalue=-.1)
        for row in rows:
            row['physical'].pop('flux_bounds_passed')
            row['physical'].pop('failed_checks')
        report=self.report(rows)
        self.assertTrue(report['passive_flux_bounds_passed'])
        self.assertFalse(report['physical_checks_passed'])


if __name__=='__main__':
    unittest.main()
