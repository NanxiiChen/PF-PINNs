# PittingCorrosionPINN

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
NETWORK_SIZE = [2] + [16]*4 + [2]
REF_PATH = ./data/results-fenics-active.csv
NTK_BATCH_SIZE = 256
BREAK_INTERVAL = 100
EPOCHS = 40000
ALPHA = 1.0
LR = 1e-3

GEOTIME_SHAPE = [10, 10]
BCDATA_SHAPE = 64
ICDATA_SHAPE = 64
SAMPLING_STRATEGY = ["grid_transition"] * 3

RAR_BASE_SHAPE = 2000
RAR_SHAPE = 256

RESUME = None
ADAPTIVE_SAMPLING = "rar"
```

### 1d-dissolution driven

- 隐藏层 16x4 不太够，必须得到 16x8 以上
- `RAR_SHAPE`非常重要，而且需要注意 `RAR_BASE_SHAPE` 和 `RAR_SHAPE` 数量之间的倍数关系，因为这决定了自适应采样点是否足够 “集中”
- 学习率 1e-3 或者稍小比较合适
 
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
NTK_BATCH_SIZE = 600
BREAK_INTERVAL = 1000
EPOCHS = 500000
ALPHA = 1.0
LR = 1e-3

GEOTIME_SHAPE = [10, 10]
BCDATA_SHAPE = 128
ICDATA_SHAPE = 128
SAMPLING_STRATEGY = ["grid_transition"] * 3

RAR_BASE_SHAPE = 30000
RAR_SHAPE = 4000

RESUME = None
ADAPTIVE_SAMPLING = "rar"
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
TIME_SPAN = (0, 0.1)
GEO_SPAN = ((-1, 1), (0, 1))
NETWORK_SIZE = [3] + [16]*4 + [2]

MESH_POINTS = "./data/2d/mesh_points.npy"
REF_PREFIX = "./data/2d/sol-"
TARGET_TIMES = [0.00, 1.02, 3.07, 10.24]

; REF_PATH = "./data/results-fenics-diffusion.csv"
NTK_BATCH_SIZE = 800
BREAK_INTERVAL = 1000
EPOCHS = 800000
ALPHA = 1.0
LR = 5e-4

GEOTIME_SHAPE = [15, 15, 15]
BCDATA_SHAPE = 256
ICDATA_SHAPE = 256
SAMPLING_STRATEGY = ["grid_transition", "lhs", "lhs"]

RAR_BASE_SHAPE = 10000
RAR_SHAPE = 512

RESUME = None
ADAPTIVE_SAMPLING = "rar"
```