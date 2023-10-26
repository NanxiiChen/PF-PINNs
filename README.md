# PittingCorrosionPINN


## 工况

1. 检验连续初始条件的作用

Discontinuous initial condition (`FeniCS`)
```python
def ic_func(xts):
    with torch.no_grad():
        phi = (xts[:, 0:1] < 0.0).float()
        c = phi * 1.0
    return torch.cat([phi, c], dim=1)
```

Continuous initial condition (`pc-PINN`)
```python
def ic_func(xts):
    with torch.no_grad():
        phi = (1 - torch.tanh(torch.sqrt(torch.tensor(OMEGA_PHI)) /
                              torch.sqrt(2 * torch.tensor(ALPHA_PHI)) * xts[:, 0:1] / GEO_COEF)) / 2
        h_phi = -2 * phi**3 + 3 * phi**2
        c = h_phi * CSE + (1 - h_phi) * 0.0
        # c = phi * 1.0
    return torch.cat([phi, c], dim=1)
```
2. 在连续边界条件上，检验是否采用多尺度采样的必要性。
3. 尝试不同大小的 `RAR_SHAPE` $|\mathcal{S}_a|$, `RAR_BASE_SHAPE` $|\mathcal{S}_{a, b}|$ 以及二者之间的倍数关系对收敛性的影响。
4. 尝试不同的 `NTK_BATCH_SIZE` 对收敛性的影响。(0, 300, 1000......)
5. 尝试不同的自适应采样准则对收敛性的影响。(residual or gradient criterion)


### 1d-activation driven
Baseline:
```ini
NETWORK_SIZE = [2] + [16]*4 + [2]
NTK_BATCH_SIZE = 256
NTK_MODE = "random"
BREAK_INTERVAL = 100
EPOCHS = 4000
GEOTIME_SHAPE = [10, 10]
BCDATA_SHAPE = 64
ICDATA_SHAPE = 64
SAMPLING_STRATEGY = ["grid_transition"] * 3
RAR_BASE_SHAPE = 5000
RAR_SHAPE = 512
ADAPTIVE_SAMPLING = "rar"
```
- 1da-case-4-1: `NTK_BATCH_SIZE=128`
- 1da-case-4-2: `NTK_BATCH_SIZE=64`
- 1da-case-4-3: `NTK_BATCH_SIZE=32`
- 1da-case-4-4: mini-batch NTK
- 1da-case-4-5: no-NTK，这里需要用前面几个工况计算稳定之后的权重作为固定权重，也能收敛。如果不借用前面的结果，全部设置成1的话，就无法收敛



## Configs

### 1d-activation driven

- 这个很容易收敛，参数取值较为随意，小网络、少采样都可以收敛

```ini
[PARAM]
ALPHA_PHI = 9.62e-5
OMEGA_PHI = 1.663e7
DD = 8.5e-10
AA = 5.35e7
LP = 1e-11
CSE = 1.
CLE = 5100/1.43e5

[TRAIN]
DIM = 1
DRIVEN = "activation"
GEO_COEF = 1e4
TIME_COEF = 1e-5
TIME_SPAN = (0, 1)
GEO_SPAN = (-0.5, 0.5)
REF_PATH = ./data/results-fenics-active.csv
ALPHA = 1.0
LR = 1e-3
RESUME = None


NETWORK_SIZE = [2] + [16]*4 + [2]
NTK_BATCH_SIZE = 32
NTK_MODE = "mini"
BREAK_INTERVAL = 100
EPOCHS = 4000
GEOTIME_SHAPE = [10, 10]
BCDATA_SHAPE = 64
ICDATA_SHAPE = 64
SAMPLING_STRATEGY = ["grid_transition"] * 3
RAR_BASE_SHAPE = 5000
RAR_SHAPE = 512
ADAPTIVE_SAMPLING = "rar"

LOG_NAME = "1da-case-4-5"
```

### 1d-dissolution driven

- 隐藏层 16x4 不太够，必须得到 16x8 以上
- `RAR_SHAPE`非常重要，而且需要注意 `RAR_BASE_SHAPE` 和 `RAR_SHAPE` 数量之间的倍数关系，因为这决定了自适应采样点是否足够 “集中”
- 学习率 5e-4 或者稍小比较合适，1e-3的话会让其他参数的鲁棒性很低
 
```ini

[PARAM]
ALPHA_PHI = 9.62e-5
OMEGA_PHI = 1.663e7
DD = 8.5e-10
AA = 5.35e7
LP = 2.0
CSE = 1.
CLE = 5100/1.43e5

[TRAIN]
DIM = 1
DRIVEN = "dissolution"
GEO_COEF = 1e4
TIME_COEF = 1e-2
TIME_SPAN = (0, 1)
GEO_SPAN = (-0.5, 0.5)
NETWORK_SIZE = [2] + [16]*8 + [2]
REF_PATH = "./data/results-fenics-diffusion.csv"
NTK_BATCH_SIZE = 400
BREAK_INTERVAL = 1000
EPOCHS = 500000
ALPHA = 1.0
LR = 5e-4

GEOTIME_SHAPE = [15, 15]
BCDATA_SHAPE = 128
ICDATA_SHAPE = 256
SAMPLING_STRATEGY = ["grid_transition"] * 3

RAR_BASE_SHAPE = 20000
RAR_SHAPE = 4000

RESUME = None
ADAPTIVE_SAMPLING = "gar"
```

### 2d-dissolution 

```ini
[PARAM]
ALPHA_PHI = 9.62e-5
OMEGA_PHI = 1.663e7
DD = 8.5e-10
AA = 5.35e7
LP = 2.0
CSE = 1.
CLE = 5100/1.43e5

[TRAIN]
DIM = 1
DRIVEN = "dissolution"
GEO_COEF = 1e4
TIME_COEF = 1e-2
TIME_SPAN = (0, 0.5)
GEO_SPAN = ((-0.5, 0.5), (0, 0.5))
NETWORK_SIZE = [3] + [16]*8 + [2]

MESH_POINTS = "./data/2d/mesh_points.npy"
REF_PREFIX = "./data/2d/sol-"
TARGET_TIMES = [0.00, 10.24, 20.48, 49.15]

; REF_PATH = "./data/results-fenics-diffusion.csv"
NTK_BATCH_SIZE = 400
BREAK_INTERVAL = 1000
EPOCHS = 800000
ALPHA = 1.0
LR = 1e-3

GEOTIME_SHAPE = [15, 15, 15]
BCDATA_SHAPE = 128
ICDATA_SHAPE = 256
SAMPLING_STRATEGY = ["grid_transition", "lhs", "lhs"]

RAR_BASE_SHAPE = 60000
RAR_SHAPE = 10000

RESUME = None
ADAPTIVE_SAMPLING = "rar"
FORWARD_BATCH_SIZE = 2000
```
