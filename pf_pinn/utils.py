import numpy as np


def compute_grad(loss, net):
    loss.backward(retain_graph=True)
    grad = []
    for p in net.parameters():
        grad += p.grad.detach().cpu().numpy().flatten().tolist()
    return np.array(grad)


def logarithmic_mean(al, ar):
    zeta = al / ar
    f = (zeta - 1) / (zeta + 1)
    u = f * f
    epsilon = 1e-2
    F = 1.0 + u/3.0 + u**2/5.0 + u**3/7.0\
        if u < epsilon\
        else np.log(zeta) / 2.0 / f
    return (al + ar) / 2.0 / F


matplotlib_configs = {
    "font.size": 14,
    "axes.titlesize": 14,
}
