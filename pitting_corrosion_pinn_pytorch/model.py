import torch
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
# from allen_cahn.sampler import GeoTimeSampler

import configparser

config = configparser.ConfigParser()
config.read("config.ini")


ALPHA_PHI = config.getfloat("PARAM", "ALPHA_PHI")
OMEGA_PHI = config.getfloat("PARAM", "OMEGA_PHI")
DD = config.getfloat("PARAM", "DD")
AA = config.getfloat("PARAM", "AA")
LP = config.getfloat("PARAM", "LP")
CSE = config.getfloat("PARAM", "CSE")
CLE = eval(config.get("PARAM", "CLE"))

TIME_COEF = config.getfloat("TRAIN", "TIME_COEF")
GEO_COEF = config.getfloat("TRAIN", "GEO_COEF")
DIM = config.getint("TRAIN", "DIM")

TIME_SPAN = eval(config.get("TRAIN", "TIME_SPAN"))
GEO_SPAN = eval(config.get("TRAIN", "GEO_SPAN"))


class PittingCorrosionNN(torch.nn.Module):
    def __init__(
        self,
        sizes: list,
        act=torch.nn.Tanh,
    ):
        super().__init__()
        self.device = torch.device("cuda"
                                   if torch.cuda.is_available()
                                   else "cpu")
        self.sizes = sizes
        self.act = act
        self.model = torch.nn.Sequential(self.make_layers()).to(self.device)

    def auto_grad(self, up, down):
        return torch.autograd.grad(inputs=down, outputs=up,
                                   grad_outputs=torch.ones_like(up),
                                   create_graph=True, retain_graph=True)[0]

    def make_layers(self):
        layers = []
        for i in range(len(self.sizes) - 1):
            layers.append((f"layer{i}",
                           torch.nn.Linear(self.sizes[i], self.sizes[i + 1])))
            if i != len(self.sizes) - 2:
                layers.append((f"act{i}", self.act()))
        return OrderedDict(layers)

    def forward(self, x):
        return self.model(x)

    def net_u(self, x):
        x = x.to(self.device)
        return self.forward(x)

    def net_pde(self, geotime):
        # geo: x, t
        # sol: phi, c
        geotime = geotime.detach().requires_grad_(True).to(self.device)
        sol = self.net_u(geotime)

        dphi_dgeotime = self.auto_grad(sol[:, 0:1], geotime)
        dc_dgeotime = self.auto_grad(sol[:, 1:2], geotime)

        dphi_dt = dphi_dgeotime[:, -1:] * TIME_COEF
        dc_dt = dc_dgeotime[:, -1:] * TIME_COEF

        dphi_dgeo = dphi_dgeotime[:, :-1] * GEO_COEF
        dc_dgeo = dc_dgeotime[:, :-1] * GEO_COEF

        nabla2phi = torch.zeros_like(dphi_dgeo[:, 0:1])
        for i in range(geotime.shape[1]-1):
            nabla2phi += self.auto_grad(dphi_dgeo[:, i:i+1],
                                        geotime)[:, i:i+1] * GEO_COEF

        nabla2c = torch.zeros_like(dphi_dgeo[:, 0:1])
        for i in range(geotime.shape[1]-1):
            nabla2c += self.auto_grad(dc_dgeo[:, i:i+1],
                                      geotime)[:, i:i+1] * GEO_COEF

        df_dphi = 12 * AA * (CSE - CLE) * sol[:, 0:1] * (sol[:, 0:1] - 1) * \
            (sol[:, 1:2] - (CSE - CLE) * (-2 * sol[:, 0:1]**3 + 3 * sol[:, 0:1]**2) - CLE) \
            + 2*OMEGA_PHI*sol[:, 0:1]*(sol[:, 0:1] - 1)*(2 * sol[:, 0:1] - 1)

        nabla2_df_dc = 2 * AA * (
            nabla2c
            + 6 * (CSE - CLE) * (
                sol[:, 0:1] * (sol[:, 0:1] - 1) * nabla2phi
                + (2*sol[:, 0:1] - 1) *
                torch.sum(dphi_dgeo ** 2, dim=1, keepdim=True)
            )
        )

        ac = dphi_dt + LP * (df_dphi - ALPHA_PHI * nabla2phi)
        ch = dc_dt - DD / 2 / AA * nabla2_df_dc

        return [ac, ch]

    def gradient(self, loss):
        loss.backward(retain_graph=True)
        return torch.cat([g.grad.view(-1) for g in self.model.parameters()])

    def adaptive_sampling(self, num, base_data, method):
        base_data = base_data.to(self.device)
        self.eval()
        if method == "rar":
            ac_residual, ch_residual = self.net_pde(base_data)
            ac_residual = ac_residual.view(-1).detach()
            ch_residual = ch_residual.view(-1).detach()
            _, ac_idx = torch.topk(ac_residual.abs(), num)
            _, ch_idx = torch.topk(ch_residual.abs(), num)

            idxs = torch.cat([ac_idx, ch_idx])
            idxs = torch.unique(torch.cat([ac_idx, ch_idx]))

            # ac_anchors = base_data[ac_idx].to(self.device)
            # ch_anchors = base_data[ch_idx].to(self.device)
            # return [ac_anchors, ch_anchors]
        elif method == "gar":
            sol = self.net_u(base_data)
            dphi_dgeotime = self.auto_grad(sol[:, 0:1], base_data)
            idxs = []
            for i in range(dphi_dgeotime.shape[1]):
                _, idx = torch.topk(dphi_dgeotime[:, i].abs(), num)
                idxs.append(idx)
                # anchors.append(base_data[idx].to(self.device))
            # return anchors
            idxs = torch.unique(torch.cat(idxs))
        else:
            raise ValueError("method must be one of 'rar' or 'gar'")
        return base_data[idxs].to(self.device)

    # def compute_jacobian(self, output, batch_size=1000):
    #     output = output.reshape(-1)
    #     grads = []
    #     for i in range(0, len(output), batch_size):
    #         # grads.append(torch.autograd.grad(output[i:i + batch_size], self.parameters(), create_graph=True))
    #         grads = torch.autograd.grad(output[i:i + batch_size], list(self.parameters()), (torch.eye(output[i:i + batch_size].shape[0]).to(
    #             self.device),), is_grads_batched=True, retain_graph=True)
    #         grad_per_layer = []
    #         for p in self.parameters():
    #             grad_per_layer += torch.autograd.grad(output[i:i + batch_size], p, retain_graph=True)

    #     return torch.cat([grad.flatten().reshape(len(output), -1) for grad in grads], 1)

    def compute_jacobian(self, output):
        output = output.reshape(-1)

        grads = torch.autograd.grad(output, list(self.parameters()), (torch.eye(output.shape[0]).to(
            self.device),), is_grads_batched=True, retain_graph=True)

        return torch.cat([grad.flatten().reshape(len(output), -1) for grad in grads], 1)

    def compute_ntk(self, jac, compute='trace'):
        if compute == 'full':
            return torch.einsum('Na,Ma->NM', jac, jac)
        elif compute == 'diag':
            return torch.einsum('Na,Na->N', jac, jac)
        elif compute == 'trace':
            return torch.einsum('Na,Na->', jac, jac)
        else:
            raise ValueError('compute must be one of "full",'
                             + '"diag", or "trace"')

    def plot_predict(self, ref_sol=None, epoch=None, ts=None,
                     mesh_points=None, ref_prefix=None):

        geo_label_suffix = f" [{1/GEO_COEF:.0e}m]"
        time_label_suffix = f" [{1/TIME_COEF:.0e}s]"

        if mesh_points is None:
            geotime = np.vstack([ref_sol["x"].values, ref_sol["t"].values]).T
            geotime = torch.from_numpy(geotime).float().to(self.device)
            sol = self.net_u(geotime).detach().cpu().numpy()

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes = axes.flatten()
            axes[0].scatter(ref_sol["x"], ref_sol["t"],
                            c=sol[:, 0], cmap="coolwarm", label="phi",
                            vmin=0, vmax=1)
            axes[0].set(xlim=GEO_SPAN, ylim=TIME_SPAN,
                        xlabel="x" + geo_label_suffix, ylabel="t" + time_label_suffix)

            diff = np.abs(sol[:, 0] - ref_sol["phi"].values)
            axes[1].scatter(ref_sol["x"], ref_sol["t"], c=diff, cmap="coolwarm",
                            vmin=0, vmax=1)
            axes[1].set(xlim=GEO_SPAN, ylim=TIME_SPAN,
                        xlabel="x" + geo_label_suffix, ylabel="t" + time_label_suffix, )

            axes[0].set_title(r"Solution $\hat\phi$"
                              + f" at epoch {epoch}")
            axes[1].set_title(r"Error $|\hat \phi - \phi_{ref}|$"
                              + f" at epoch {epoch}")
            # fig.legend()

            acc = 1 - np.mean(diff ** 2)

        else:

            fig, axes = plt.subplots(len(ts), 2, figsize=(15, 5*len(ts)))
            diffs = []
            for idx, tic in enumerate(ts):
                tic_tensor = torch.ones(mesh_points.shape[0], 1)\
                    .view(-1, 1) * tic * TIME_COEF
                mesh_tensor = torch.from_numpy(mesh_points).float()
                geotime = torch.cat([mesh_tensor, tic_tensor],
                                    dim=1).to(self.device)
                with torch.no_grad():
                    sol = self.net_u(geotime).detach().cpu().numpy()
                axes[idx, 0].scatter(mesh_points[:, 0], mesh_points[:, 1], c=sol[:, 0],
                                     cmap="coolwarm", label="phi", vmin=0, vmax=1)
                axes[idx, 0].set(xlim=GEO_SPAN[0], ylim=GEO_SPAN[1], aspect="equal",
                                 xlabel="x" + geo_label_suffix, ylabel="y" + geo_label_suffix,
                                 title="pred t = " + str(round(tic, 2)))

                truth = np.load(ref_prefix + f"{tic:.2f}" + ".npy")
                diff = np.abs(sol[:, 0] - truth[:, 0])
                axes[idx, 1].scatter(mesh_points[:, 0], mesh_points[:, 1], c=diff,
                                     cmap="coolwarm", label="error", vmin=0, vmax=1)
                axes[idx, 1].set(xlim=GEO_SPAN[0], ylim=GEO_SPAN[1], aspect="equal",
                                 xlabel="x" + geo_label_suffix, ylabel="y" + geo_label_suffix,
                                 title="error t = " + str(round(tic, 2)))
                diffs.append(diff)
            acc = 1 - np.mean(np.array(diffs) ** 2)

        return fig, axes, acc

    def plot_samplings(self, geotime, bcdata, icdata, anchors):
        geotime = geotime.detach().cpu().numpy()
        bcdata = bcdata.detach().cpu().numpy()
        icdata = icdata.detach().cpu().numpy()
        anchors = anchors.detach().cpu().numpy()

        geo_label_suffix = f" [{1/GEO_COEF:.0e}m]"
        time_label_suffix = f" [{1/TIME_COEF:.0e}s]"

        if geotime.shape[1] == 2:

            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.scatter(geotime[:, 0], geotime[:, 1],
                       c="blue", label="collcations")
            ax.scatter(anchors[:, 0], anchors[:, 1],
                       c="green", label="anchors", marker="x")
            ax.scatter(bcdata[:, 0], bcdata[:, 1],
                       c="red", label="boundary condition")
            ax.scatter(icdata[:, 0], icdata[:, 1],
                       c="orange", label="initial condition")
            ax.set(xlim=GEO_SPAN, ylim=TIME_SPAN,
                   xlabel="x" + geo_label_suffix, ylabel="t" + time_label_suffix,)
            ax.legend(bbox_to_anchor=(1.02, 1.00),
                      loc='upper left', borderaxespad=0.)
        elif geotime.shape[1] == 3:
            fig, ax = plt.subplots(1, 1, figsize=(10, 5),
                                   subplot_kw={"aspect": "equal",
                                               "xlim": GEO_SPAN[0],
                                               "ylim": GEO_SPAN[1],
                                               "zlim": TIME_SPAN,
                                               "xlabel": "x" + geo_label_suffix,
                                               "ylabel": "y" + geo_label_suffix,
                                               "zlabel": "t" + time_label_suffix,
                                               "projection": "3d"})
            ax.scatter(geotime[:, 0], geotime[:, 1], geotime[:, 2],
                       c="blue", label="collcations", alpha=0.1, s=1)
            ax.scatter(anchors[:, 0], anchors[:, 1], anchors[:, 2],
                       c="black", label="anchors", s=1, marker="x", alpha=0.5)
            ax.scatter(bcdata[:, 0], bcdata[:, 1], bcdata[:, 2],
                       c="red", label="boundary condition", s=1, marker="x", alpha=0.5)
            ax.scatter(icdata[:, 0], icdata[:, 1], icdata[:, 2],
                       c="orange", label="initial condition", s=1, marker="x", alpha=0.5)
            ax.legend(bbox_to_anchor=(1.02, 1.00),
                      loc='upper left', borderaxespad=0.)

        else:
            raise ValueError("Only 2 or 3 dimensional data is supported")
        return fig, ax

    def compute_weight(self, residuals, method, batch_size, return_ntk_info=False):
        traces = []
        jacs = []
        for res in residuals:
            if method == "random":
                if batch_size < len(res):
                    jac = self.compute_jacobian(res[
                        np.random.randint(0, len(res), batch_size)
                    ])
                    trace = self.compute_ntk(jac, compute='trace').item()
                    traces.append(trace / batch_size)
                else:
                    jac = self.compute_jacobian(res)
                    trace = self.compute_ntk(jac, compute='trace').item()
                    traces.append(trace / len(res))

            elif method == "topres":
                if batch_size < len(res):
                    jac = self.compute_jacobian(res[:batch_size])
                    trace = self.compute_ntk(jac, compute='trace').item()
                    traces.append(trace / batch_size)
                else:
                    jac = self.compute_jacobian(res)
                    trace = self.compute_ntk(jac, compute='trace').item()
                    traces.append(trace / len(res))

            elif method == "mini":
                trace = 0
                for i in range(0, len(res), batch_size):
                    jac = self.compute_jacobian(res[
                        i: min(i + batch_size, len(res))
                    ])
                    trace += self.compute_ntk(jac, compute='trace').item()
                traces.append(trace / len(res))
            elif method == "full":
                jac = self.compute_jacobian(res)
                trace = self.compute_ntk(jac, compute='trace').item()
                traces.append(trace / len(res))

            else:
                raise ValueError("method must be one of 'random', 'topres'"
                                 " 'mini', or 'full'")

            if return_ntk_info:
                jacs.append(jac)

        traces = np.array(traces)
        if return_ntk_info:
            return traces.sum() / traces, jacs
        return traces.sum() / traces
