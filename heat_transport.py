import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt, colors as clr
import multiprocessing as mp
import math
import argparse

day = 24 * 60 * 60

month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


class heat_model:
    month_mean_temp = [
        -2.5,
        -1.0,
        3.0,
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
        self.D = math.sqrt(self.K * day / math.pi)

    def analitic(self, z, t, m):
        return self.month_mean_temp[m] + self.month_amplitude(m) * math.sin(
            2 * math.pi * t / day - z / self.D
        ) * math.e ** (-z / self.D)

    @staticmethod
    def month_amplitude(month):
        if month <= 3:
            return month * 6 / 3 + 6
        elif month <= 8:
            return month * (-2) / 5 + 66 / 5
        else:
            return month * (-4) / 3 + 62 / 3

    def plot_layer_heat_progression(self, ax, label=None, color=None):
        ax.plot(
            np.arange(0, 3 + self.z_step, self.z_step),
            self.earth,
            label=label,
            color=color,
        )
        ax.set_xlabel("Głębokość [m]")
        ax.set_ylabel("Temperatura [C]")
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
        mean_temp = self.month_mean_temp[month]
        self.earth[0] = mean_temp + amplitude * math.sin(
            2 * math.pi * ((self.iter_num * self.t_step) / 86400)
        )
        self.iterate()


def zad4():
    week = 60 * 60 * 24 * 7
    dts = [5, 30, 60, 120, 240, 700]

    fig, axs = plt.subplots(2, 3)
    i = 0
    for dt in dts:
        model = heat_model(t_step=dt)
        h_steps = 60 * 60 // dt
        for step in range(week // dt):
            model.iterate_month(6)
            if step % h_steps == 0:
                ax = model.plot_layer_heat_progression(axs[i // 3][i % 3])
                ax.set_title(f"dt = {dt}")
        print(i)
        i += 1

    plt.show()

    dzs = [0.006, 0.01, 0.05, 0.1, 0.5, 1]

    fig, axs = plt.subplots(2, 3)
    i = 0
    for dz in dzs:
        model = heat_model(z_step=dz)
        h_steps = 60
        for step in range(week // 60):
            model.iterate_month(6)
            if step % h_steps == 0:
                ax = model.plot_layer_heat_progression(axs[i // 3][i % 3])
                ax.set_title(f"dz = {dz}")
        print(i)
        i += 1

    plt.show()


def zad5_6_7():
    step = 5
    day_steps = day // step
    model = heat_model(t_step=step, z_step=0.02)
    stable = False
    i = 0
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

    print(f"steps={i*day_steps}, time={i*day_steps*step}")
    fig, ax = plt.subplots()
    model.plot_layer_heat_progression(ax)
    # plt.show()

    a = [model.analitic(x, 60 * 60 * 24, 6) for x in np.arange(0, 3.02, 0.02)]

    ax.plot(list(np.arange(0, 3.02, 0.02)), a)
    plt.show()


def zad8_parallel():
    prints_per_month = 1
    step = 60

    def model_for_month(m, q):
        model = heat_model(t_step=step)
        steps = month_days[m] * day // step
        print_steps = np.linspace(0, steps - 1, prints_per_month + 1, dtype=int)[1:]
        for i in range(steps):
            model.iterate_month(m)
            if i in print_steps:
                q.put((m, model.earth[0], deepcopy(model)))

    fig, ax = plt.subplots()
    q = mp.Queue()
    processes = [
        mp.Process(
            target=model_for_month,
            args=(
                m,
                q,
            ),
        )
        for m in range(12)
    ]

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    data = []

    while not q.empty():
        data.append(list(q.get()))

    data = sorted(data, key=lambda x: x[0] * 1000000 + x[1])
    for month, i, m in data:
        m.plot_layer_heat_progression(ax, label=f"{month}, {i}")

    plt.legend()
    plt.show()


def zad8():
    fig, ax = plt.subplots()
    t_step = 120
    plot_step = day // t_step
    m = heat_model(t_step=t_step, z_step=0.05)
    steps = np.cumsum(month_days)
    steps = steps * day // t_step
    last_month = -1
    for i in range(steps[-1]):
        month = 12 - len(steps)
        m.iterate_month(month)
        if i % plot_step == 0:
            if last_month != month:
                last_month = month
                l = month + 1
            else:
                l = None
            m.plot_layer_heat_progression(
                ax, color=clr.hsv_to_rgb([month / 12, 1, 1]),
                label=l
            )
        steps = list(filter(lambda x: x > i, steps))
    plt.legend()
    plt.show()


def zad9():
    t_step = 120
    year_steps = 365 * day // t_step
    plot_step = day // t_step
    model = heat_model(t_step=t_step, z_step=0.05)
    stable = False
    i = 0
    prev_means = np.zeros((len(model.earth)))
    while not stable:
        fig, ax = plt.subplots()
        tmp = np.zeros((len(model.earth)))
        steps = np.cumsum(month_days)
        steps = steps * day // t_step
        last_month = -1
        for j in range(steps[-1]):
            month = 12 - len(steps)
            model.iterate_month(month)
            if j % plot_step == 0:
                if last_month != month:
                    last_month = month
                    l = month + 1
                else:
                    l = None
                model.plot_layer_heat_progression(
                    ax, color=clr.hsv_to_rgb([month / 12, 1, 1]),
                    label=l
                )
            tmp += model.earth
            steps = list(filter(lambda x: x > j, steps))
        means = tmp / steps[-1]
        print(np.max(np.abs(prev_means - means)))
        stable = np.all(list(map(lambda x: x < 0.1 and x > -0.1, prev_means - means)))
        prev_means = means
        i += 1
        plt.legend()
        plt.show()
        plt.clf()

    print(f"steps={i*year_steps}, time={i*year_steps*t_step}")


zads = {
    "4": zad4,
    "5": zad5_6_7,
    "6": zad5_6_7,
    "7": zad5_6_7,
    "8": zad8,
    "8p": zad8_parallel,
    "9": zad9,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("zad")

    args = parser.parse_args()
    zads[args.zad]()
