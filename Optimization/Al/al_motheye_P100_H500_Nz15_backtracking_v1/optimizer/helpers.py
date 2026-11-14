"""Identical numerical helpers copied from the original research code."""
def zero_order_x_source(simulation):
    import torch
    zero_x = int(torch.nonzero(simulation.order_x == 0, as_tuple=False)[0, 0])
    zero_y = int(torch.nonzero(simulation.order_y == 0, as_tuple=False)[0, 0])
    harmonic = zero_x * len(simulation.order_y) + zero_y
    source = torch.zeros(2 * simulation.order_N, dtype=simulation._dtype, device=simulation._device)
    source[harmonic] = 1.0
    return source

def trapezoid_weights(wavelengths):
    if len(wavelengths) == 1:
        return [1.]
    span = wavelengths[-1] - wavelengths[0]
    weights = [0.] * len(wavelengths)
    for i, (a, b) in enumerate(zip(wavelengths, wavelengths[1:])):
        weights[i] += (b - a) / (2 * span)
        weights[i + 1] += (b - a) / (2 * span)
    return weights
