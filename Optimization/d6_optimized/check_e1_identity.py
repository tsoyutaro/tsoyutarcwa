"""Independent random-matrix check: two E1 rows equal all 12 group actions."""
from pathlib import Path
import argparse
import json
import torch
import block_assembly as assembly

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--output',type=Path,default=Path('validation_results/e1_identity.json'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(417)
    sim = assembly.make_sim('hexagonal',2,48,2.5,550.,200.,args.device)
    plan = assembly.GeometryContext(sim)
    _,group = sim._triangular_star_group_operators()
    count = 2*plan.scalar_dimension
    errors = []
    for left,right,basis_left,basis_right,partner_left,partner_right in (
        ('electric','magnetic',plan.electric,plan.magnetic,plan.electric_partner,plan.magnetic_partner),
        ('magnetic','electric',plan.magnetic,plan.electric,plan.magnetic_partner,plan.electric_partner)):
        # Deliberately non-invariant, arbitrary complex matrix.
        matrix = torch.randn(count,count,dtype=torch.complex128,device=args.device)
        averaged = sum(sim._d6_transform_intertwiner(matrix,entry,left=left,right=right) for entry in group)/len(group)
        reference = basis_left.mH@averaged@basis_right
        pair = .5*(basis_left.mH@matrix@basis_right+partner_left.mH@matrix@partner_right)
        errors.append(assembly.relative_error(pair,reference))
    assert max(errors)<2e-12,errors
    result = {'status':'passed','test':'arbitrary non-invariant complex P/Q-like matrices',
              'relative_errors':errors,'seed':417,'device':args.device}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
