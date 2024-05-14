from pyDOE import lhs
import matplotlib.pyplot as plt
import configparser
import pandas as pd
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
import pf_pinn as pfp
import numpy as np
import torch
import datetime
import matplotlib
import time
matplotlib.use("Agg")

config = configparser.ConfigParser()
config.read("config.ini")

LOG_NAME = config.get("TRAIN", "LOG_NAME").strip('"')
now = LOG_NAME
if LOG_NAME == "None":
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
writer = SummaryWriter(log_dir="runs/" + now)


# Define the sampler
class GeoTimeSampler:
    def __init__(
        self,
        geo_span: list,  # 1d
        time_span: list,
    ):
        self.geo_span = geo_span
        self.time_span = time_span

    def resample(self, in_num, bc_num, ic_num, strateges=["lhs", "lhs", "lhs"]):
        return self.in_sample(in_num, strateges[0]), \
            self.bc_sample(bc_num, strateges[1]), \
            self.ic_sample(ic_num, strateges[2])

    def in_sample(self, in_num, strategy: str = "lhs",):

        if strategy == "lhs":
            func = pfp.make_lhs_sampling_data
        elif strategy == "grid":
            func = pfp.make_uniform_grid_data
        elif strategy == "grid_transition":
            func = pfp.make_uniform_grid_data_transition
        else:
            raise ValueError(f"Unknown strategy {strategy}")
        geotime = func(mins=[self.geo_span[0], self.time_span[0]],
                       maxs=[self.geo_span[1], self.time_span[1]],
                       num=in_num)

        return torch.from_numpy(geotime).float().requires_grad_(True)

    def bc_sample(self, bc_num, strategy: str = "lhs",):
        if strategy == "lhs":
            ts = (lhs(1, bc_num) *
                  (self.time_span[1] - self.time_span[0]) + self.time_span[0]).reshape(-1, 1)
        elif strategy == "grid":
            ts = np.linspace(self.time_span[0], self.time_span[1], bc_num)[
                1:-1].reshape(-1, 1)
        elif strategy == "grid_transition":
            ts = np.linspace(self.time_span[0], self.time_span[1], bc_num)[
                1:-1].reshape(-1, 1)
            distance = (self.time_span[1] - self.time_span[0]) / (bc_num - 1)
            shift = np.random.uniform(-distance, distance, 1)
            ts = np.clip(ts + shift, self.time_span[0], self.time_span[1])
        else:
            raise ValueError(f"Unknown strategy {strategy}")

        xt_l = np.hstack([np.full(ts.shape[0], self.geo_span[0]).reshape(-1, 1),
                          ts.reshape(-1, 1)])
        xt_r = np.hstack([np.full(ts.shape[0], self.geo_span[1]).reshape(-1, 1),
                          ts.reshape(-1, 1)])
        xts = np.vstack([xt_l, xt_r])
        return torch.from_numpy(xts).float().requires_grad_(True)

    def ic_sample(self, ic_num, strategy: str = "lhs", local_area=[-0.1, 0.1]):
        if strategy == "lhs":
            xs = (lhs(1, ic_num) *
                  (self.geo_span[1] - self.geo_span[0]) + self.geo_span[0]).reshape(-1, 1)
            xs_local = (lhs(1, ic_num) *
                        (local_area[1] - local_area[0]) + local_area[0]).reshape(-1, 1)
        elif strategy == "grid":
            xs = np.linspace(self.geo_span[0], self.geo_span[1], ic_num)[
                1:-1].reshape(-1, 1)
            xs_local = np.linspace(local_area[0], local_area[1], ic_num)[
                1:-1].reshape(-1, 1)
        elif strategy == "grid_transition":
            xs = np.linspace(self.geo_span[0], self.geo_span[1], ic_num)[
                1:-1].reshape(-1, 1)
            xs_local = np.linspace(local_area[0], local_area[1], ic_num)[
                1:-1].reshape(-1, 1)
            distance = (self.geo_span[1] - self.geo_span[0]) / (ic_num - 1)
            shift = np.random.uniform(-distance, distance, 1)
            xs = np.clip(xs + shift, self.geo_span[0], self.geo_span[1])
        else:
            raise ValueError(f"Unknown strategy {strategy}")
        xs = np.vstack([xs, xs_local])
        xts = np.hstack(
            [xs, np.full(xs.shape[0], self.time_span[0]).reshape(-1, 1)])

        return torch.from_numpy(xts).float().requires_grad_(True)


geo_span = eval(config.get("TRAIN", "GEO_SPAN"))
time_span = eval(config.get("TRAIN", "TIME_SPAN"))
sampler = GeoTimeSampler(geo_span, time_span)
net = pfp.PFPINN(
    sizes=eval(config.get("TRAIN", "NETWORK_SIZE")),
    act=torch.nn.Tanh
)

resume = config.get("TRAIN", "RESUME").strip('"')
try:
    net.load_state_dict(torch.load(resume))
    print("Load model successfully")
except:
    print("Can not load model")
    pass


ref_sol = pd.read_csv(config.get("TRAIN", "REF_PATH"))

TIME_COEF = config.getfloat("TRAIN", "TIME_COEF")
GEO_COEF = config.getfloat("TRAIN", "GEO_COEF")
ref_sol["x"] = ref_sol["x"].apply(lambda x: x * GEO_COEF)
ref_sol["t"] = ref_sol["t"].apply(lambda t: t * TIME_COEF)


ic_weight = 1
bc_weight = 1
ac_weight = 1
ch_weight = 1

NTK_BATCH_SIZE = config.getint("TRAIN", "NTK_BATCH_SIZE")
BREAK_INTERVAL = config.getint("TRAIN", "BREAK_INTERVAL")
EPOCHS = config.getint("TRAIN", "EPOCHS")
ALPHA = config.getfloat("TRAIN", "ALPHA")
LR = config.getfloat("TRAIN", "LR")

ALPHA_PHI = config.getfloat("PARAM", "ALPHA_PHI")
OMEGA_PHI = config.getfloat("PARAM", "OMEGA_PHI")
DD = config.getfloat("PARAM", "DD")
AA = config.getfloat("PARAM", "AA")
LP = config.getfloat("PARAM", "LP")
CSE = config.getfloat("PARAM", "CSE")
CLE = eval(config.get("PARAM", "CLE"))


def bc_func(xts):
    with torch.no_grad():
        val = (0.5 - xts[:, 0:1])
    return torch.cat([val, val], dim=1)


def ic_func(xts):
    with torch.no_grad():
        phi = (1 - torch.tanh(torch.sqrt(torch.tensor(OMEGA_PHI)) /
                              torch.sqrt(2 * torch.tensor(ALPHA_PHI)) * xts[:, 0:1] / GEO_COEF)) / 2
        h_phi = -2 * phi**3 + 3 * phi**2
        c = h_phi * CSE + (1 - h_phi) * 0.0
    return torch.cat([phi, c], dim=1)


criteria = torch.nn.MSELoss()
opt = torch.optim.Adam(net.parameters(), lr=LR)

GEOTIME_SHAPE = eval(config.get("TRAIN", "GEOTIME_SHAPE"))
BCDATA_SHAPE = eval(config.get("TRAIN", "BCDATA_SHAPE"))
ICDATA_SHAPE = eval(config.get("TRAIN", "ICDATA_SHAPE"))
SAMPLING_STRATEGY = eval(config.get("TRAIN", "SAMPLING_STRATEGY"))
RAR_BASE_SHAPE = config.getint("TRAIN", "RAR_BASE_SHAPE")
RAR_SHAPE = config.getint("TRAIN", "RAR_SHAPE")

for epoch in range(EPOCHS):
    net.train()
    if epoch % BREAK_INTERVAL == 0:
        geotime, bcdata, icdata = sampler.resample(GEOTIME_SHAPE, BCDATA_SHAPE,
                                                   ICDATA_SHAPE, strateges=SAMPLING_STRATEGY)
        geotime = geotime.to(net.device)
        residual_base_data = sampler.in_sample(RAR_BASE_SHAPE, strategy="lhs")
        method = config.get("TRAIN", "ADAPTIVE_SAMPLING").strip('"')
        anchors = net.adaptive_sampling(RAR_SHAPE, residual_base_data,
                                        method=method)
        net.train()
        data = torch.cat([geotime, anchors],
                         dim=0).requires_grad_(True)

        bcdata = bcdata.to(net.device)
        icdata = icdata.to(net.device)

        fig, ax = net.plot_samplings(
            geotime, bcdata,
            icdata, anchors)
        # plt.savefig(f"./runs/{now}/sampling-{epoch}.png",
        #              bbox_inches='tight', dpi=300)
        writer.add_figure("sampling", fig, epoch)

    ac_residual, ch_residual = net.net_pde(data)
    ac_loss = criteria(ac_residual, torch.zeros_like(ac_residual))
    ch_loss = criteria(ch_residual, torch.zeros_like(ch_residual))
    bc_forward = net.net_u(bcdata)
    bc_loss = criteria(bc_forward, bc_func(bcdata).detach())
    ic_forward = net.net_u(icdata)
    ic_loss = criteria(ic_forward, ic_func(icdata).detach())

    if epoch % BREAK_INTERVAL == 0:

        ac_weight, ch_weight, bc_weight, ic_weight = \
            net.compute_weight(
                [ac_residual, ch_residual, bc_forward, ic_forward],
                method=config.get("TRAIN", "NTK_MODE").strip('"'),
                batch_size=NTK_BATCH_SIZE
            )

        print(f"epoch: {epoch}, "
              f"ic_loss: {ic_loss.item():.4e}, "
              f"bc_loss: {bc_loss.item():.4e}, "
              f"ac_loss: {ac_loss.item():.4e}, "
              f"ch_loss: {ch_loss.item():.4e}, ")

        writer.add_scalar("loss/ic", ic_loss, epoch)
        writer.add_scalar("loss/bc", bc_loss, epoch)
        writer.add_scalar("loss/ac", ac_loss, epoch)
        writer.add_scalar("loss/ch", ch_loss, epoch)

        writer.add_scalar("weight/ic", ic_weight, epoch)
        writer.add_scalar("weight/bc", bc_weight, epoch)
        writer.add_scalar("weight/ac", ac_weight, epoch)
        writer.add_scalar("weight/ch", ch_weight, epoch)

        fig, ax, acc = net.plot_predict(ref_sol=ref_sol, epoch=epoch)

        if epoch % (BREAK_INTERVAL) == 0:
            torch.save(net.state_dict(), f"./runs/{now}/model-{epoch}.pt")
        # plt.savefig(f"./runs/{now}/fig-{epoch}.png",
        #             bbox_inches='tight', dpi=300)
        # ! Saving figure is too slow

        writer.add_figure("fig/predict", fig, epoch)
        writer.add_scalar("acc", acc, epoch)

    losses = ic_weight * ic_loss \
        + bc_weight * bc_loss \
        + ac_weight * ac_loss \
        + ch_weight * ch_loss

    if epoch % (BREAK_INTERVAL) == 0:
        writer.add_scalar("loss/total", losses, epoch)

    opt.zero_grad()
    losses.backward()
    opt.step()


# torch.save(net.state_dict(), "model.pt")
torch.save(net.state_dict(), "model.pt")
print("Done")
