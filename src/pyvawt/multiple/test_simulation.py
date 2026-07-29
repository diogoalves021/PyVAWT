import numpy as np
import pytest

try:
    from .simulation import Turbine, Environment, actuatorcylinder
except ImportError:
    from simulation import Turbine, Environment, actuatorcylinder


@pytest.fixture
def env():
    return Environment(Vinf=1.0, rho=1.225, mu=1.7894e-05)


def mock_airfoil_naca0018(alpha: np.ndarray):
    cl = 2.0 * np.pi * alpha
    cd = 0.01 + 0.1 * alpha**2
    return cl, cd


# ==============================================================================
# Dados de Referência Esperados [TSR, CP, CT, Rp, Tp, Zp]
# ==============================================================================

BASELINE_TURBINE_1 = [
    (1.000000, 0.332005, 0.386430, 0.014456, -0.006043, 0.000000),
    (1.315789, 0.379531, 0.460457, 0.060484, -0.007502, 0.000000),
    (1.631579, 0.420755, 0.533016, 0.118757, -0.008549, 0.000000),
    (1.947368, 0.452228, 0.599751, 0.189062, -0.009196, 0.000000),
    (2.263158, 0.473529, 0.659266, 0.270857, -0.009485, 0.000000),
    (2.578947, 0.485159, 0.711145, 0.363377, -0.009482, 0.000000),
    (2.894737, 0.487981, 0.755452, 0.465629, -0.009280, 0.000000),
    (3.210526, 0.482998, 0.792547, 0.576408, -0.008993, 0.000000),
    (3.526316, 0.471226, 0.823000, 0.694335, -0.008747, 0.000000),
    (3.842105, 0.453610, 0.847522, 0.817921, -0.008679, 0.000000),
    (4.157895, 0.430965, 0.866910, 0.945659, -0.008920, 0.000000),
    (4.473684, 0.403932, 0.881977, 1.076115, -0.009586, 0.000000),
    (4.789474, 0.372967, 0.893505, 1.208006, -0.010776, 0.000000),
    (5.105263, 0.338343, 0.902203, 1.340256, -0.012560, 0.000000),
    (5.421053, 0.300175, 0.908682, 1.472009, -0.014986, 0.000000),
    (5.736842, 0.258721, 0.913834, 1.602128, -0.018107, 0.000000),
    (6.052632, 0.213924, 0.918192, 1.730073, -0.021930, 0.000000),
    (6.368421, 0.165410, 0.921831, 1.855928, -0.026425, 0.000000),
    (6.684211, 0.112834, 0.924823, 1.979758, -0.031562, 0.000000),
    (7.000000, 0.055875, 0.927236, 2.101618, -0.037313, 0.000000),
]

BASELINE_TURBINE_2 = BASELINE_TURBINE_1

# Estruturação da matriz do parametrizador contendo [TSR, dados_t1, dados_t2]
SWEEP_PAIR_DATA = [
    (t1_row[0], t1_row[1:], t2_row[1:])
    for t1_row, t2_row in zip(BASELINE_TURBINE_1, BASELINE_TURBINE_2)
]


# ==============================================================================
# Suíte de Testes Parametrizada
# ==============================================================================

@pytest.mark.parametrize("tsr, exp_t1, exp_t2", SWEEP_PAIR_DATA)
def test_two_turbines_sweep_fidelity(env, tsr, exp_t1, exp_t2):
    vinf = env.Vinf
    r = 3.0
    omega = (tsr * vinf) / r

    # Instanciação das duas turbinas no domínio
    t1 = Turbine(r=r, chord=0.25, twist=0.0, delta=0.0, B=3,
                 af=mock_airfoil_naca0018, Omega=omega, centerX=0.0, centerY=6.0)
    t2 = Turbine(r=r, chord=0.25, twist=0.0, delta=0.0, B=3,
                 af=mock_airfoil_naca0018, Omega=omega, centerX=14.0, centerY=0.0)

    # Execução do solver para o par de turbinas
    CT, CP, Rp, Tp, Zp, theta, _ = actuatorcylinder([t1, t2], env, ntheta=36)

    # Desempacotamento dos valores esperados
    exp_cp1, exp_ct1, exp_rp1, exp_tp1, exp_zp1 = exp_t1
    exp_cp2, exp_ct2, exp_rp2, exp_tp2, exp_zp2 = exp_t2

    # --- 1. Validações da Primeira Turbina (Índice 0) ---
    np.testing.assert_allclose(CP[0], exp_cp1, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(CT[0], exp_ct1, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(Rp[0, 0], exp_rp1, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(Tp[0, 0], exp_tp1, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(Zp[0, 0], exp_zp1, rtol=1e-3, atol=1e-4)

    # --- 2. Validações da Segunda Turbina (Índice 1) ---
    np.testing.assert_allclose(CP[1], exp_cp2, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(CT[1], exp_ct2, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(Rp[0, 1], exp_rp2, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(Tp[0, 1], exp_tp2, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(Zp[0, 1], exp_zp2, rtol=1e-3, atol=1e-4)

    # --- 3. Checagem de dimensões das matrizes resultantes ---
    assert Rp.shape == (36, 2)
    assert Tp.shape == (36, 2)
    assert Zp.shape == (36, 2)
