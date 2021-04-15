import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt
import multiprocessing as mp
import math


class heat_model:
    def __init__(self, t_step=60, z_step=0.02, k=1, cw=2.8, T_init=18):
        self.t_step = t_step
        self.z_step = z_step
        self.k = k
        self.cw = cw * 1_000_000
        self.T_init = T_init

        self.K = self.k / self.cw
        self.earth = np.array(
            [9 for _ in np.arange(0, 3 + z_step, z_step)], dtype=float
        )
        self.earth[0] = T_init
        self.iter_num = 0

        # self.K*(self.t_step/(self.z_step**2))

    @staticmethod
    def month_amplitude(month):
        if month <= 3:
            return month * 6 / 3 + 6
        elif month <= 8:
            return month * (-2) / 5 + 66 / 5
        else:
            return month * (-4) / 3 + 62 / 3

    @staticmethod
    def month_mean_temp(month):
        return np.array(
            [
                -2.5,
                -1.0,
                3.0,
                8.5,
                13.5,
                8.5,
                13.5,
                16.5,
                18.5,
                18,
                13.5,
                8.5,
                3.5,
                -1.0,
            ]
        )[month]

    def plot_layer_heat_progression(self, ax):
        ax.plot(np.arange(0, 3 + self.z_step, self.z_step), self.earth)
        return ax

    def __str__(self):
        return str(self.earth)

    def iterate(self):
        tmp = deepcopy(self.earth)
        for i in range(1, len(self.earth) - 1):
            tmp[i] = self.earth[i] + self.K * (self.t_step / (self.z_step ** 2)) * (
                self.earth[i - 1] + self.earth[i + 1] - 2 * self.earth[i]
            )
        self.earth = tmp
        self.iter_num += 1

    def iterate_month(self, month):
        amplitude = self.month_amplitude(month)
        mean_temp = self.month_mean_temp(month)
        self.earth[0] = mean_temp + amplitude * math.sin(
            2 * math.pi * ((self.iter_num * self.t_step) / 86400)
        )
        self.iterate()

def model_for_month(m,q):
    model = heat_model()
    for _ in range(86400 // 4):
        model.iterate_month(m)
    q.put(model)
    

#ZAD4

# week = 60*60*24*7
# dts = [5, 30, 60, 120, 240, 700]

# fig, axs = plt.subplots(2,3)
# i = 0
# for dt in dts:
#     model = heat_model(t_step=dt)
#     h_steps = 60*60 // dt
#     for step in range(week//dt):
#         model.iterate_month(6)
#         if step%h_steps == 0:
#             model.plot_layer_heat_progression(axs[i//3][i%3])
#     print(i)
#     i += 1

# plt.show()

# dzs = [0.006, 0.01, 0.05, 0.1, 0.5, 1]

# fig, axs = plt.subplots(2,3)
# i = 0
# for dz in dzs:
#     model = heat_model(z_step=dz)
#     h_steps = 60
#     for step in range(week//60):
#         model.iterate_month(6)
#         if step%h_steps == 0:
#             model.plot_layer_heat_progression(axs[i//3][i%3])
#     print(i)
#     i += 1

# plt.show()

#ZAD5,6

day = 60*60*24
step = 5
day_steps = day//step
model = heat_model(t_step=step, z_step=0.02)
stable = False
i=0
prev_means = np.zeros((len(model.earth)))
while not stable:
    tmp = np.zeros((len(model.earth)))
    for _ in range(day_steps):
        model.iterate_month(6)
        tmp += model.earth
    means = tmp / day_steps
    stable = np.all(list(map(lambda x: x < 0.1 and x > -0.1, prev_means - means)))
    prev_means = means
    i += 1

print(f'steps={i*day_steps}, time={i*day_steps*step}')
fig, ax = plt.subplots()
model.plot_layer_heat_progression(ax)
plt.show()
    



# fig, ax = plt.subplots()
# q = mp.Queue()
# processes = [mp.Process(target=model_for_month, args=(m,q,)) for m in range(12)]

# for p in processes:
#     p.start()

# for p in processes:
#     p.join()

# while not q.empty():
#     q.get().plot_layer_heat_progression(ax)

# plt.show()