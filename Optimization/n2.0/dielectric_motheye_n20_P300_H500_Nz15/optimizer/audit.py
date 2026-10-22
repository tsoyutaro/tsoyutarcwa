"""All-logit gradients, directional central differences, and order refinement."""
from __future__ import annotations

from .physics import objective


def check(request, device):
    import torch
    c = request["config"]
    logits = torch.tensor(request["logits"], dtype=torch.float64, device=device, requires_grad=True)
    orders = sorted(set([c["optimization_order"], max(c["verification_orders"])]))
    waves = c["audit_wavelengths_nm"]
    records = []
    generator = torch.Generator(device="cpu").manual_seed(c["seed"]+713)
    directions = []
    for i in range(c["audit_directions"]):
        direction = torch.randn(logits.numel(), generator=generator, dtype=torch.float64)
        direction -= direction.mean()  # Remove the softmax gauge direction.
        directions.append((direction/torch.linalg.vector_norm(direction)).to(device))
    for order in orders:
        baseline = objective(logits, c, request["lattice"], waves, order, c["optimization_grid"], device, gradient=True)
        g = baseline["gradient"]
        tests = []
        for i, d in enumerate(directions):
            differences = []
            for h in (c["audit_fd_step"], c["audit_fd_step"]/2):
                plus = objective((logits.detach()+h*d), c, request["lattice"], waves, order,
                                 c["optimization_grid"], device, gradient=False)["objective"]
                minus = objective((logits.detach()-h*d), c, request["lattice"], waves, order,
                                  c["optimization_grid"], device, gradient=False)["objective"]
                differences.append((plus-minus)/(2*h))
            ad = float(torch.dot(g, d).cpu())
            close = lambda a, b: abs(a-b) <= c["gradient_atol"]+c["gradient_rtol"]*max(abs(a), abs(b))
            tests.append({"direction": i, "autograd": ad, "fd_h": differences[0], "fd_half_h": differences[1],
                          "passed": close(ad, differences[1]) and close(*differences)})
        records.append({"order": order, "objective": baseline["objective"], "gradient_logits": g.cpu().tolist(),
                        "directional_checks": tests, "directions_passed": all(t["passed"] for t in tests)})
    a = torch.tensor(records[0]["gradient_logits"], dtype=torch.float64)
    b = torch.tensor(records[-1]["gradient_logits"], dtype=torch.float64)
    scale = max(float(torch.linalg.vector_norm(a)), float(torch.linalg.vector_norm(b)))
    difference = float(torch.linalg.vector_norm(a-b))
    cosine = float(torch.dot(a, b)/(torch.linalg.vector_norm(a)*torch.linalg.vector_norm(b))) if min(float(a.norm()), float(b.norm())) > 1e-15 else None
    passed = difference <= c["gradient_atol"]+c["gradient_rtol"]*scale and (scale <= c["gradient_atol"] or cosine is not None and cosine >= c["gradient_cosine_min"])
    return {"status": "completed", "wavelengths_nm": waves, "variable_space": "all layers+1 softmax logits",
            "direction_count": len(directions), "records": records, "order_gradient_norm_difference": difference,
            "order_gradient_relative_difference": difference/max(scale, 1e-15), "order_gradient_cosine": cosine,
            "order_gradient_passed": passed, "passed": passed and all(r["directions_passed"] for r in records),
            "full_objective_wavelength_coverage": waves == c["optimization_wavelengths_nm"],
            "limits": "Directional FD checks, not exhaustive componentwise FD proof. Recheck at intermediate geometries; sampled wavelengths do not establish whole-band gradient convergence."}
