import numpy as np
from scipy import integrate


INFINITISMEL = 1e-10
INTEGRATE_SCALE = 7
BETA = 5
LR = 0.0001


def p(mean=0, std=1):
    def _p(x):
        return (
            np.exp(-(x - mean)**2 / (2 * std**2)) /
            np.sqrt(2 * np.pi * std**2)
        )
    return _p


def J(alpha=1):
    def _J(x, lbd):
        mu = np.exp(lbd)
        return np.sum(
            np.exp(x - mu / (alpha * mu)) /
            ((alpha * mu)**2 * (1 + np.exp((x - mu) / alpha * mu))**2)
        )
    return _J


def f(alpha=1):
    p_std = p(0, 1)

    def _f(lbd):
        ys = []
        mu = np.exp(lbd)
        for mu_i in mu:
            def _func(x):
                return p_std(x) / (1 + np.exp(-(x-mu_i)/alpha*mu_i))
            ys.append(integrate.quad(_func,
                                     -INTEGRATE_SCALE, INTEGRATE_SCALE)[0])
        return np.array(ys)
    return _f


def CL_1(beta=BETA, C_1=0.82*6):
    f_1 = f(1)

    def _CL_1(lbd):
        return beta * np.linalg.norm(C_1 - np.sum(f_1(lbd)))
    return _CL_1


def CL_2(beta=BETA, C_2=60):
    def _CL_2(lbd):
        mu = np.exp(lbd)
        return beta * np.linalg.norm(C_2 - np.sum(1 / mu))
    return _CL_2


def L(lbd):
    p_std = p(0, 1)
    J_1 = J(1)

    def _func(x):
        return p_std(x) * np.log(J_1(x, lbd) + INFINITISMEL)
    return -integrate.quad(_func, -INTEGRATE_SCALE, INTEGRATE_SCALE)[0]


DELTA = 0.00001


def grad(L, lbd):
    N = lbd.shape[0]
    dlbd = np.zeros(N)
    for i in range(N):
        lbd_u, lbd_l = lbd.copy(), lbd.copy()
        lbd_u[i] += DELTA
        lbd_l[i] -= DELTA
        # print(L(lbd_u))
        # print(L(lbd_l))
        dlbd[i] = (L(lbd_u) - L(lbd_l)) / (2 * DELTA)
    return dlbd


class Main():
    def optimize(self):
        lbd = 3 + 0.1 * np.random.rand(12)
        lbd = np.zeros(12) + 0.1 * np.random.rand(12) - 0.05
        # lbd = 4 * np.ones((12))
        lbd = [-1.5311583911721351, -1.7488273351202217, -1.9155950728614572, -1.7726186131272468, -1.5175487042809355, -1.4163656365528485, -1.7737755163210878, -1.5157499774173089, -1.4030243226800037, -1.6634171843600214, -1.6849462072274184, -1.5526528237937502]
        lbd = np.array(lbd)
        CL_1_ = CL_1()
        CL_2_ = CL_2()

        import json

        for i in range(1000):
            dlbdL = grad(L, lbd)
            dlbdCL_1 = grad(CL_1_, lbd)
            dlbdCL_2 = grad(CL_2_, lbd)
            print(i)
            print('L', L(lbd))
            print('CL_1', CL_1_(lbd))
            print('CL_2', CL_2_(lbd))
            print('dlblL', dlbdL)
            print('dlblCL_1', dlbdCL_1)
            print('dlbdCL_2', dlbdCL_2)
            print('lbd', json.dumps(lbd.tolist()))
            lbd -= LR * (dlbdL + dlbdCL_1 + dlbdCL_2)

    def plot(self):
        lbd_0 = np.array([-1.53, -1.53])  # noqa

        CL_1_ = CL_1()
        CL_2_ = CL_2()

        def Loss(lbd):
            return L(lbd) + CL_1_(lbd) + CL_2_(lbd)

        x = np.linspace(-2, -1, 100)
        y = np.linspace(-2, -1, 100)
        xv, yv = np.meshgrid(x, y)
        rest = [np.ones(xv.shape) for _ in range(10)]
        lbd = np.stack([xv, yv] + rest, axis=2)
        zv = np.zeros(xv.shape)
        for i in range(100):
            for j in range(100):
                zv[i, j] = Loss(lbd[i, j, :])
                print(i, j)

        import matplotlib.pyplot as plt

        plt.contourf(x, y, zv)
        plt.colorbar()
        plt.show()


if __name__ == '__main__':
    import fire
    fire.Fire(Main)
