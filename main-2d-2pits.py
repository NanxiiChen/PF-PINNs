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
matplotlib.use("Agg")

config = configparser.ConfigParser()
config.read("config.ini")


now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
writer = SummaryWriter(log_dir="runs/" + now)


class GeoTimeSampler:
    def __init__(
        self,
        geo_span: list,  # 2d
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
        geotime = func(mins=[self.geo_span[0][0], self.geo_span[1][0], self.time_span[0]],
                       maxs=[self.geo_span[0][1], self.geo_span[1]
                             [1], self.time_span[1]],
                       num=in_num)

        return torch.from_numpy(geotime).float().requires_grad_(True)

    # TODO: bc
    def bc_sample(self, bc_num: int, strategy: str = "lhs", xspan=[-0.025, 0.025]):
        # 四条边，顺着时间变成四个面
        if strategy == "lhs":
            func = pfp.make_lhs_sampling_data
        elif strategy == "grid":
            func = pfp.make_uniform_grid_data
        elif strategy == "grid_transition":
            func = pfp.make_uniform_grid_data_transition
        else:
            raise ValueError(f"Unknown strategy {strategy}")

        xyts = pfp.make_lhs_sampling_data(mins=[-0.025, 0, self.time_span[0]],
                                          maxs=[0.025, 0.025,
                                                self.time_span[1]],
                                          num=bc_num)
        xyts = xyts[xyts[:, 0] ** 2 + xyts[:, 1] ** 2 <= 0.025 ** 2]
        xyts_left = xyts.copy()
        xyts_left[:, 0:1] -= 0.15
        xyts_right = xyts.copy()
        xyts_right[:, 0:1] += 0.15

        xts = func(mins=[self.geo_span[0][0], self.time_span[0]],
                   maxs=[self.geo_span[0][1], self.time_span[1]],
                   num=bc_num)
        top = np.hstack([xts[:, 0:1], np.full(
            xts.shape[0], self.geo_span[1][1]).reshape(-1, 1), xts[:, 1:2]])  # 顶边

        yts = func(mins=[self.geo_span[1][0], self.time_span[0]],
                   maxs=[self.geo_span[1][1], self.time_span[1]],
                   num=bc_num)
        left = np.hstack([np.full(yts.shape[0], self.geo_span[0]
                         [0]).reshape(-1, 1), yts[:, 0:1], yts[:, 1:2]])  # 左边
        right = np.hstack([np.full(yts.shape[0], self.geo_span[0]
                          [1]).reshape(-1, 1), yts[:, 0:1], yts[:, 1:2]])  # 右边

        xyts = np.vstack([xyts_left, xyts_right, top, left, right])

        return torch.from_numpy(xyts).float().requires_grad_(True)

    def ic_sample(self, ic_num, strategy: str = "lhs", local_area=[[-0.1, 0.1], [0, 0.1]]):
        if strategy == "lhs":
            xys = pfp.make_lhs_sampling_data(mins=[self.geo_span[0][0], self.geo_span[1][0]],
                                             maxs=[self.geo_span[0][1],
                                                   self.geo_span[1][1]],
                                             num=ic_num)

        elif strategy == "grid":
            xys = pfp.make_uniform_grid_data(mins=[self.geo_span[0][0], self.geo_span[1][0]],
                                             maxs=[self.geo_span[0][1],
                                                   self.geo_span[1][1]],
                                             num=ic_num)
        elif strategy == "grid_transition":
            xys = pfp.make_uniform_grid_data_transition(mins=[self.geo_span[0][0], self.geo_span[1][0]],
                                                        maxs=[
                self.geo_span[0][1], self.geo_span[1][1]],
                num=ic_num)
        else:
            raise ValueError(f"Unknown strategy {strategy}")
        xys_local = pfp.make_semi_circle_data(radius=0.1,
                                              num=ic_num*2,
                                              center=[0, 0.])
        xys_local_left = xys_local.copy()
        xys_local_left[:, 0:1] -= 0.15
        xys_local_right = xys_local.copy()
        xys_local_right[:, 0:1] += 0.15
        xys = np.vstack([xys, xys_local_left, xys_local_right])
        xyts = np.hstack([xys, np.full(xys.shape[0],
                                       self.time_span[0]).reshape(-1, 1)])
        return torch.from_numpy(xyts).float().requires_grad_(True)


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


TIME_COEF = config.getfloat("TRAIN", "TIME_COEF")
GEO_COEF = config.getfloat("TRAIN", "GEO_COEF")


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
MESH_POINTS = np.load(config.get("TRAIN", "MESH_POINTS").strip('"')) * GEO_COEF


def ic_func(xts):
    # r = torch.min(
    #     torch.sqrt((xts[:, 0:1] - 0.15)**2 + xts[:, 1:2]**2),
    #     torch.sqrt((xts[:, 0:1] + 0.15)**2 + xts[:, 1:2]**2)
    # ).detach()
    r = torch.sqrt((torch.abs(xts[:, 0:1]) - 0.15)**2
                   + xts[:, 1:2]**2).detach()
    with torch.no_grad():
        phi = 1 - (1 - torch.tanh(torch.sqrt(torch.tensor(OMEGA_PHI)) /
                                  torch.sqrt(2 * torch.tensor(ALPHA_PHI)) * (r-0.05) / GEO_COEF)) / 2
        h_phi = -2 * phi**3 + 3 * phi**2
        c = h_phi * CSE
    return torch.cat([phi, c], dim=1)


# def bc_func(xts):
#     r = torch.sqrt(torch.abs(xts[:, 0:1] - 0.15) + xts[:, 1:2]).detach()
#     with torch.no_grad():
#         phi = 1 - (1 - torch.tanh(torch.sqrt(torch.tensor(OMEGA_PHI)) /
#                                   torch.sqrt(2 * torch.tensor(ALPHA_PHI)) * (r-0.05) / GEO_COEF)) / 2
#         h_phi = -2 * phi**3 + 3 * phi**2
#         c = h_phi * CSE
#     return torch.cat([phi, c], dim=1)


def bc_func(xts):
    r = torch.sqrt((torch.abs(xts[:, 0:1]) - 0.15)**2
                   + xts[:, 1:2]**2).detach()
    with torch.no_grad():
        phi = (r > 0.05).float()
        c = phi.detach()
    return torch.cat([phi, c], dim=1)


criteria = torch.nn.MSELoss()
opt = torch.optim.Adam(net.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=25000, gamma=0.8)

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

        # shuffle
        data = data[torch.randperm(len(data))]

        bcdata = bcdata.to(net.device)
        icdata = icdata.to(net.device)

        fig, ax = net.plot_samplings(geotime, bcdata, icdata, anchors)
        # plt.savefig(f"./runs/{now}/sampling-{epoch}.png",
        #             bbox_inches='tight', dpi=300)
        writer.add_figure("sampling", fig, epoch)

    FORWARD_BATCH_SIZE = config.getint("TRAIN", "FORWARD_BATCH_SIZE")

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
                method="random",
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

        TARGET_TIMES = eval(config.get("TRAIN", "TARGET_TIMES"))

        REF_PREFIX = config.get("TRAIN", "REF_PREFIX").strip('"')

        fig, ax, acc = net.plot_predict(ts=TARGET_TIMES,
                                        mesh_points=MESH_POINTS,
                                        ref_prefix=REF_PREFIX)

        if epoch % (BREAK_INTERVAL) == 0:
            torch.save(net.state_dict(), f"./runs/{now}/model-{epoch}.pt")
            # plt.savefig(f"./runs/{now}/fig-{epoch}.png",
            #             bbox_inches='tight', dpi=300)

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
    scheduler.step()

print("Done")
