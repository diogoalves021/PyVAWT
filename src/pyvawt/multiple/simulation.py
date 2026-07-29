import math
import os
import h5py
import numpy as np
from numba import njit
from scipy.optimize import root
from typing import Callable, Tuple

# ==============================================================================
# Cache em Memória RAM para Matrizes pré-calculadas
# ==============================================================================
_MATRIX_CACHE = {}

# ==============================================================================
# Quadraturas e Integração de Painéis Numba
# ==============================================================================

GL_NODES = np.array([
    -0.9739065285171717, -0.8650633666889845, -0.6794095682990244,
    -0.4333953941292472, -0.1488743389816312,  0.1488743389816312,
     0.4333953941292472,  0.6794095682990244,  0.8650633666889845,
     0.9739065285171717
], dtype=np.float64)

GL_WEIGHTS = np.array([
    0.0666713443086881, 0.1494513491505806, 0.2190863625159820,
    0.2692667193099963, 0.2955242247147529, 0.2955242247147529,
    0.2692667193099963, 0.2190863625159820, 0.1494513491505806,
    0.0666713443086881
], dtype=np.float64)


@njit(fastmath=True)
def fast_trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Integração trapezoidal acelerada via Numba."""
    n = len(x)
    total = 0.0
    for i in range(n - 1):
        total += 0.5 * (y[i] + y[i + 1]) * (x[i + 1] - x[i])
    return total


@njit(fastmath=True)
def trapz(x: np.ndarray, y: np.ndarray) -> float:
    n = len(x)
    integral = 0.0
    for i in range(n - 1):
        integral += (x[i + 1] - x[i]) * 0.5 * (y[i] + y[i + 1])
    return integral


@njit(fastmath=True)
def pInt(theta: np.ndarray, f: np.ndarray) -> float:
    integral = trapz(theta, f)
    dtheta = 2.0 * theta[0]
    integral += dtheta * 0.5 * (f[0] + f[-1])
    return integral


@njit(fastmath=True)
def Dxintegrand(x: float, y: float, phi: float) -> float:
    v1 = x + math.sin(phi)
    v2 = y - math.cos(phi)
    denom = 2.0 * math.pi * (v1 * v1 + v2 * v2)
    if denom == 0.0:
        return 0.0
    return (v1 * math.sin(phi) - v2 * math.cos(phi)) / denom


@njit(fastmath=True)
def Ayintegrand(x: float, y: float, phi: float) -> float:
    v1 = x + math.sin(phi)
    v2 = y - math.cos(phi)
    if abs(v1) < 1e-12 and abs(v2) < 1e-12:
        return 0.0
    denom = 2.0 * math.pi * (v1 * v1 + v2 * v2)
    return (v1 * math.cos(phi) + v2 * math.sin(phi)) / denom


@njit(fastmath=True)
def panelIntegration_numba(xvec: np.ndarray, yvec: np.ndarray, thetavec: np.ndarray, is_ay: bool) -> np.ndarray:
    nx = len(xvec)
    ntheta = len(thetavec)
    dtheta = thetavec[1] - thetavec[0]
    half_dtheta = dtheta / 2.0
    A = np.zeros((nx, ntheta), dtype=np.float64)

    for i in range(nx):
        xi = xvec[i]
        yi = yvec[i]
        for j in range(ntheta):
            m = thetavec[j]
            res = 0.0
            for k in range(10):
                phi = m + half_dtheta * GL_NODES[k]
                if is_ay:
                    val = Ayintegrand(xi, yi, phi)
                else:
                    val = Dxintegrand(xi, yi, phi)
                res += GL_WEIGHTS[k] * val
            A[i, j] = half_dtheta * res
    return A


def AyIJ(xvec, yvec, thetavec):
    return panelIntegration_numba(xvec, yvec, thetavec, True)


def DxIJ(xvec, yvec, thetavec):
    return panelIntegration_numba(xvec, yvec, thetavec, False)


@njit(fastmath=True)
def WxIJ(xvec: np.ndarray, yvec: np.ndarray, thetavec: np.ndarray) -> np.ndarray:
    nx = len(xvec)
    ntheta = len(thetavec)
    dtheta = thetavec[1] - thetavec[0]
    Wx = np.zeros((nx, ntheta), dtype=np.float64)
    theta_bounds = thetavec + dtheta / 2.0

    for i in range(nx):
        x = xvec[i]
        y = yvec[i]
        if -1.0 <= y <= 1.0 and x >= 0.0 and (x * x + y * y) >= 1.0:
            thetak = math.acos(y)
            k = np.searchsorted(theta_bounds, thetak, side='right')
            if 0 <= k < ntheta:
                Wx[i, k] = -1.0
                Wx[i, ntheta - k - 1] = 1.0
    return Wx


@njit(fastmath=True)
def DxII(thetavec: np.ndarray) -> np.ndarray:
    ntheta = len(thetavec)
    dtheta = thetavec[1] - thetavec[0]
    Rx = (dtheta / (4.0 * np.pi)) * np.ones((ntheta, ntheta), dtype=np.float64)
    half_n = ntheta // 2
    inv_n = 1.0 / ntheta
    for i in range(ntheta):
        if i < half_n:
            Rx[i, i] = (-1.0 + inv_n) / 2.0
        else:
            Rx[i, i] = (1.0 + inv_n) / 2.0
    return Rx


@njit(fastmath=True)
def WxII(thetavec: np.ndarray) -> np.ndarray:
    ntheta = len(thetavec)
    Wx = np.zeros((ntheta, ntheta), dtype=np.float64)
    half_n = ntheta // 2
    for i in range(half_n, ntheta):
        Wx[i, ntheta - 1 - i] = -1.0
    return Wx


def precomputeMatrices(ntheta: int, modulepath: str) -> str:
    dtheta = 2.0 * np.pi / ntheta
    theta = np.arange(dtheta / 2.0, 2.0 * np.pi, dtheta)

    Dxself = DxII(theta)
    Wxself = WxII(theta)
    Ayself = AyIJ(-np.sin(theta), np.cos(theta), theta)

    filepath = f'{modulepath}/theta-{ntheta}.h5'
    with h5py.File(filepath, 'w') as file:
        file.create_dataset('theta', data=theta)
        file.create_dataset('Dx', data=Dxself)
        file.create_dataset('Wx', data=Wxself)
        file.create_dataset('Ay', data=Ayself)

    return filepath


def matrixAssemble(centerX, centerY, radii, ntheta):
    # Otimização 1: Cache em Memória RAM para evitar leitura repetida do arquivo HDF5
    cache_key = (tuple(centerX), tuple(centerY), tuple(radii), ntheta)
    if cache_key in _MATRIX_CACHE:
        return _MATRIX_CACHE[cache_key]

    file = f'theta-{ntheta}.h5'
    modulepath = os.getcwd()
    if not os.path.isfile(file):
        filepath = precomputeMatrices(ntheta, modulepath)
    else:
        filepath = os.path.join(modulepath, file)

    with h5py.File(filepath, 'r') as f:
        theta = f['theta'][:]
        Dxself = f['Dx'][:]
        Wxself = f['Wx'][:]
        Ayself = f['Ay'][:]

    nturbines = len(radii)
    Dx = np.zeros((nturbines * ntheta, nturbines * ntheta))
    Wx = np.zeros((nturbines * ntheta, nturbines * ntheta))
    Ay = np.zeros((nturbines * ntheta, nturbines * ntheta))

    for I in range(nturbines):
        for J in range(nturbines):
            x = (centerX[I] - radii[I] * np.sin(theta) - centerX[J]) / radii[J]
            y = (centerY[I] + radii[I] * np.cos(theta) - centerY[J]) / radii[J]

            if I == J:
                Dxsub = Dxself
                Wxsub = Wxself
                Aysub = Ayself
            elif J < I and radii[I] == radii[J]:
                Dxsub = Dx[J * ntheta:(J + 1) * ntheta, I * ntheta:(I + 1) * ntheta]
                Aysub = Ay[J * ntheta:(J + 1) * ntheta, I * ntheta:(I + 1) * ntheta]
                Wxsub = WxIJ(x, y, theta)
            else:
                Dxsub = DxIJ(x, y, theta)
                Wxsub = WxIJ(x, y, theta)
                Aysub = AyIJ(x, y, theta)

            Dx[I * ntheta:(I + 1) * ntheta, J * ntheta:(J + 1) * ntheta] = Dxsub
            Wx[I * ntheta:(I + 1) * ntheta, J * ntheta:(J + 1) * ntheta] = Wxsub
            Ay[I * ntheta:(I + 1) * ntheta, J * ntheta:(J + 1) * ntheta] = Aysub

    Ax = Dx + Wx
    result = (Ax, Ay, theta)
    _MATRIX_CACHE[cache_key] = result
    return result

# ==============================================================================
# Classes e Otimização do Solver de Forças
# ==============================================================================

class Turbine:
    def __init__(self, r: float, chord: float, twist: float, delta: float, B: int,
                 af: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]], Omega: float,
                 centerX: float, centerY: float):
        self.r = r
        self.chord = chord
        self.twist = twist
        self.delta = delta
        self.B = B
        self.af = af
        self.Omega = Omega
        self.centerX = centerX
        self.centerY = centerY


class Environment:
    def __init__(self, Vinf: float, rho: float, mu: float):
        self.Vinf = Vinf
        self.rho = rho
        self.mu = mu


@njit(fastmath=True)
def _radialforce_kinematics(uvec: np.ndarray, vvec: np.ndarray, thetavec: np.ndarray,
                            Vinf: float, Omega: float, r: float, twist: float):
    rotation = 1.0 if Omega >= 0.0 else -1.0
    abs_Omega = abs(Omega)
    n = len(thetavec)

    W = np.empty(n, dtype=np.float64)
    phi = np.empty(n, dtype=np.float64)
    alpha = np.empty(n, dtype=np.float64)

    for i in range(n):
        sin_t = math.sin(thetavec[i])
        cos_t = math.cos(thetavec[i])
        u_i = uvec[i]
        v_i = vvec[i]

        vn = Vinf * (1.0 + u_i) * sin_t - Vinf * v_i * cos_t
        vt = rotation * (Vinf * (1.0 + u_i) * cos_t + Vinf * v_i * sin_t) + abs_Omega * r
        w = math.sqrt(vn * vn + vt * vt)
        p = math.atan2(vn, vt)

        W[i] = w
        phi[i] = p
        alpha[i] = p - twist

    return rotation, W, phi, alpha


@njit(fastmath=True)
def _radialforce_postprocess(cl: np.ndarray, cd: np.ndarray, phi: np.ndarray, W: np.ndarray,
                              thetavec: np.ndarray, Vinf: float, rho: float, r: float,
                              chord: float, B: int, rotation: float, delta: float, Omega: float):
    n = len(thetavec)
    q = np.empty(n, dtype=np.float64)
    Rp = np.empty(n, dtype=np.float64)
    Tp = np.empty(n, dtype=np.float64)
    Zp = np.empty(n, dtype=np.float64)
    integrand = np.empty(n, dtype=np.float64)

    sigma = B * chord / r
    cos_delta = math.cos(delta)
    tan_delta = math.tan(delta)
    abs_Omega = abs(Omega)
    inv_vinf = 1.0 / Vinf

    for i in range(n):
        p = phi[i]
        w = W[i]
        cos_p = math.cos(p)
        sin_p = math.sin(p)

        cn_i = cl[i] * cos_p + cd[i] * sin_p
        ct_i = cl[i] * sin_p - cd[i] * cos_p

        w_ratio = w * inv_vinf
        w_ratio_sq = w_ratio * w_ratio

        q[i] = (sigma / (4.0 * math.pi)) * cn_i * w_ratio_sq

        qd = 0.5 * rho * w * w
        Rp[i] = -cn_i * qd * chord
        Tp[i] = ct_i * qd * chord / cos_delta
        Zp[i] = -cn_i * qd * chord * tan_delta

        sin_t = math.sin(thetavec[i])
        cos_t = math.cos(thetavec[i])
        integrand[i] = w_ratio_sq * (cn_i * sin_t - rotation * ct_i * cos_t / cos_delta)

    CT = (sigma / (4.0 * math.pi)) * fast_trapz(integrand, thetavec)

    if CT > 2.0:
        a = 0.5 * (1.0 + math.sqrt(1.0 + CT))
        ka = 1.0 / (a - 1.0)
    elif CT > 0.96:
        a = (1.0 / 7.0) * (1.0 + 3.0 * math.sqrt(3.5 * CT - 3.0))
        ka = 18.0 * a / (7.0 * a * a - 2.0 * a + 4.0)
    else:
        a = 0.5 * (1.0 - math.sqrt(1.0 - CT))
        ka = 1.0 / (1.0 - a)

    H = 1.0
    Sref = 2.0 * r * H
    Q = r * Tp
    P = abs_Omega * B / (2.0 * math.pi) * fast_trapz(Q, thetavec)
    CP = P / (0.5 * rho * Vinf**3 * Sref)

    return q, ka, CT, CP, Rp, Tp, Zp


def radialforce(uvec, vvec, thetavec, turbine: Turbine, env: Environment):
    rotation, W, phi, alpha = _radialforce_kinematics(
        uvec, vvec, thetavec, env.Vinf, turbine.Omega, turbine.r, turbine.twist
    )
    cl, cd = turbine.af(alpha)
    return _radialforce_postprocess(
        cl, cd, phi, W, thetavec, env.Vinf, env.rho,
        turbine.r, turbine.chord, turbine.B, rotation, turbine.delta, turbine.Omega
    )


@njit(fastmath=True)
def _compute_residual_fast_single(A: np.ndarray, q: np.ndarray, ka: float, w: np.ndarray) -> np.ndarray:
    """Otimização 2: Residual ultra-rápido para 1 turbina em Numba sem alocação."""
    res = A @ q
    for i in range(len(w)):
        res[i] = res[i] * ka - w[i]
    return res


@njit(fastmath=True)
def _compute_residual_output(A: np.ndarray, q: np.ndarray, k: np.ndarray, ntheta: int, w: np.ndarray) -> np.ndarray:
    nturbines = len(k)
    kmult = np.empty(2 * nturbines * ntheta, dtype=np.float64)
    for i in range(nturbines):
        ki = k[i]
        start_u = i * ntheta
        end_u = (i + 1) * ntheta
        start_v = nturbines * ntheta + i * ntheta
        end_v = nturbines * ntheta + (i + 1) * ntheta

        kmult[start_u:end_u] = ki
        kmult[start_v:end_v] = ki

    return (A @ q) * kmult - w


def residual(w, A, theta, k, turbines, env):
    ntheta = len(theta)
    nturbines = int(len(w) / (2 * ntheta))
    q = np.empty(ntheta * nturbines, dtype=np.float64)

    for i in range(nturbines):
        u = w[i * ntheta : (i + 1) * ntheta]
        v = w[nturbines * ntheta + i * ntheta : nturbines * ntheta + (i + 1) * ntheta]
        q_i, ka, *_ = radialforce(u, v, theta, turbines[i], env)
        q[i * ntheta : (i + 1) * ntheta] = q_i
        if nturbines == 1:
            k = np.array([ka])

    return _compute_residual_output(A, q, k, ntheta, w)


def actuatorcylinder(turbines, env, ntheta, w0=None):
    centerX = np.array([turbine.centerX for turbine in turbines])
    centerY = np.array([turbine.centerY for turbine in turbines])
    radii = np.array([turbine.r for turbine in turbines])

    Ax, Ay, theta = matrixAssemble(centerX, centerY, radii, ntheta)

    ntheta = len(theta)
    nturbines = len(turbines)
    tol = 1e-6

    CT = np.zeros(nturbines)
    CP = np.zeros(nturbines)
    Rp = np.zeros((ntheta, nturbines))
    Tp = np.zeros((ntheta, nturbines))
    Zp = np.zeros((ntheta, nturbines))
    k = np.zeros(nturbines)

    # Otimização 3: Pré-empilhamento das matrizes FORA das funções do solver
    A_full = np.vstack([Ax, Ay])

    if w0 is not None and nturbines > 1:
        w0_coupled = w0
        for i in range(nturbines):
            u_start = w0[i * ntheta : (i + 1) * ntheta]
            v_start = w0[ntheta * nturbines + i * ntheta : ntheta * nturbines + (i + 1) * ntheta]
            _, k[i], *_ = radialforce(u_start, v_start, theta, turbines[i], env)
    else:
        w0_coupled = np.zeros(nturbines * ntheta * 2) if w0 is None else w0

        for i in range(nturbines):
            if w0 is not None and nturbines == 1:
                w0_single = w0
            else:
                w0_single = np.zeros(ntheta * 2)

            idx = np.arange(i * ntheta, (i + 1) * ntheta)
            
            # Pré-calcula a matriz do bloco individual apenas uma vez
            A_single = np.vstack([Ax[idx][:, idx], Ay[idx][:, idx]])

            def resid_single(x):
                u = x[:ntheta]
                v = x[ntheta:]
                q_val, ka_val, *_ = radialforce(u, v, theta, turbines[i], env)
                return _compute_residual_fast_single(A_single, q_val, ka_val, x)

            result = root(resid_single, w0_single, method='lm', tol=tol)
            w_single = result.x
            if not result.success:
                print(f'Solver não convergiu para a turbina {i + 1}. Mensagem: {result.message}')

            u = w_single[:ntheta]
            v = w_single[ntheta:]
            _, k[i], CT[i], CP[i], Rp[:, i], Tp[:, i], Zp[:, i] = radialforce(u, v, theta, turbines[i], env)

            if nturbines > 1:
                w0_coupled[i * ntheta : (i + 1) * ntheta] = u
                w0_coupled[ntheta * nturbines + i * ntheta : ntheta * nturbines + (i + 1) * ntheta] = v

        if nturbines == 1:
            return CT, CP, Rp, Tp, Zp, theta, w_single

    def resid_multiple(x):
        return residual(x, A_full, theta, k, turbines, env)

    result = root(resid_multiple, w0_coupled, method='lm', tol=tol)
    w_coupled = result.x
    if not result.success:
        print(f'Solver não convergiu para o sistema acoplado. Mensagem: {result.message}')

    for i in range(nturbines):
        idx = list(range(i * ntheta, (i + 1) * ntheta))
        u = w_coupled[idx]
        v = w_coupled[ntheta * nturbines + np.array(idx)]
        _, _, CT[i], CP[i], Rp[:, i], Tp[:, i], Zp[:, i] = radialforce(u, v, theta, turbines[i], env)

    return CT, CP, Rp, Tp, Zp, theta, w_coupled
