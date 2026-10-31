"""(f) 큰 입자가 빠른 방전에서 손해를 보는 까닭은 '반응 겉넓이'일까, '리튬이 퍼지는 거리'일까?

시뮬레이션 e에서 큰 입자(반지름 200 nm)는 작은 입자(50 nm)보다 고율(빠른 방전) 성능을 훨씬 많이 잃었어요.
PyBaMM에서 입자를 크게 하면 두 가지가 한꺼번에 바뀌어요.
  (1) 전하 이동(반응): 같은 활물질 부피에 반응할 겉넓이가 1/R로 줄어요(a = 3 × 활물질 부피 비율 ÷ R).
  (2) 물질 이동(고체 확산): 리튬이 입자 안쪽까지 퍼지는 시간이 R²/D로 길어져요.
어느 쪽이 10C 손해를 만드는지 가르려고, 200 nm 입자에 '코팅'과 '도핑'을 흉내 낸 손잡이를 따로 돌려 봐요.
  - '코팅' 손잡이 = 양극 교환 전류 밀도(반응 속도 상수)에 k를 곱해요.
  - '도핑' 손잡이 = 양극 입자 확산계수에 D를 곱해요.
  둘 다 모델 안의 가정이에요. 실제 탄소 코팅·도핑이 이 값을 이만큼 바꾼다는 뜻이 아니에요.
  D×16이면 R²/D가 50 nm와 같아지고, k×4면 전극 부피당 교환 전류(a × 교환 전류 밀도)가 50 nm와 같아져요.

- 설계 A(한 가지 크기): d와 같은 DFN 하프셀·상수·식. 공극률 0.38·0.30에서 50 nm 기준과 200 nm × (D, k) 8가지.
- 설계 B(섞은 전극): PyBaMM의 두 입자 종류 기능("particle phases": ("1", "2"))으로 50 nm(Primary)와
  200 nm(Secondary)를 부피 20%·50%로 섞고, 큰 입자 쪽에만 D×16 또는 k×4를 줘요.
- 1C와 유지율은 d와 같아요(설계마다 자기 0.1C 용량으로 1C를 정하고, 유지율 = xC 용량 ÷ 0.1C 용량 × 100).
- 보정하지 않은 모델 값이에요. 실제 제품 성능이 아니라 손잡이끼리 비교한 경향만 봐요.

사용법 (저장소 루트에서):
  계산 + 그림:  C:\\p\\Scripts\\python.exe simulations\\pybamm\\f_lfp_ct_vs_mass.py [--workers 4]
  그림만 다시:  .venv\\Scripts\\python.exe simulations\\pybamm\\f_lfp_ct_vs_mass.py --plot-only
  (PyBaMM 가상환경에 matplotlib이 없으면 계산 뒤 그림 단계만 건너뛰어요. 그때 두 번째 명령을 써요.)
"""
import csv
import datetime
import json
import os
import platform
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
sys.path.insert(0, HERE)
import d_lfp_density_rate as dstudy  # noqa: E402  (d의 상수·식을 그대로 써요. 가져올 때 계산은 돌지 않아요.)

JSON_PATH = os.path.join(RES, "f_ct_vs_mass.json")
CSV_PATH = os.path.join(RES, "f_ct_vs_mass.csv")
FIG_RET = os.path.join(RES, "f_retention_by_variant.png")
FIG_SHARE = os.path.join(RES, "f_gap_recovered_share.png")
FIG_BLEND = os.path.join(RES, "f_blend_large_phase.png")
D_JSON = os.path.join(RES, "d_density_rate.json")
E_JSON = os.path.join(RES, "e_size_mix.json")

# ---- 실험 설계 ----
POROSITIES = [0.38, 0.30]        # 0.38: 전해질이 모자라지 않는 쪽, 0.30: d의 입자 크기 곁가지 실험과 비교
C_RATES = dstudy.C_RATES         # 0.1C, 1C, 2C, 5C, 10C
SHARE_RATES = [1.0, 2.0, 5.0, 10.0]
R_SMALL_NM, R_LARGE_NM = 50, 200
# (이름, 반지름 nm, 확산계수 배수 D, 교환 전류 밀도 배수 k)
VARIANTS_A = [
    ("R50", 50, 1, 1),
    ("R200", 200, 1, 1),
    ("D2", 200, 2, 1), ("D4", 200, 4, 1), ("D16", 200, 16, 1),
    ("k2", 200, 1, 2), ("k4", 200, 1, 4),
    ("D16k4", 200, 16, 4), ("D4k2", 200, 4, 2),
]
BLEND_FRACTIONS = [0.2, 0.5]     # 큰 입자(200 nm)의 부피 비율
VARIANTS_B = [("base", 1, 1), ("D16", 16, 1), ("k4", 1, 4)]   # 큰 입자 쪽에만 주는 배수
OPT_SINGLE = dstudy.OPTIONS
OPT_COMPOSITE = {"working electrode": "positive", "particle phases": ("1", "2")}
KEY_D = "Positive particle diffusivity [m2.s-1]"
KEY_J0 = "Positive electrode exchange-current density [A.m-2]"
# 두 입자 종류 모델이 "Primary: "/"Secondary: "를 붙여 따로 찾는 이름(PyBaMM 26.8에서 모델을 만들며 확인한 8개)
PHASE_KEYS = [
    "Maximum concentration in positive electrode [mol.m-3]",
    "Positive particle radius [m]",
    KEY_D,
    "Positive electrode active material volume fraction",
    KEY_J0,
    "Positive electrode OCP [V]",
    "Initial concentration in positive electrode [mol.m-3]",
    "Positive electrode OCP entropic change [V.K-1]",
]
# 격자 점검: 가장 힘든 10C를 칸 수를 늘려 다시 풀어요.
MESH_CHECK_A = ["R200", "D16", "k4", "D16k4"]
MESH_CHECK_B = [(0.5, "base"), (0.5, "D16"), (0.5, "k4")]
# 설계 B 안정성 점검(공극률 0.30·10C, 양극 두께 방향 칸 수만 늘림): (설계, 이름, 큰 입자 비율)
STABILITY_DESIGNS = [("A", "R50", None), ("B_check", "same50", 0.5), ("B", "base", 0.2), ("B", "D16", 0.2),
                     ("B", "base", 0.5), ("B", "D16", 0.5)]
STABILITY_XP = [40, 80]
TEST_T_K = 298.15


def rp_for(radius_nm):
    """d의 규칙: 입자 안은 반지름 50 nm마다 20칸."""
    return max(20, int(round(20 * radius_nm / 50)))


def var_pts_single(radius_nm):
    return dstudy.var_pts_for(radius_nm)


def var_pts_composite(r1_nm, r2_nm):
    return {"x_s": 20, "x_p": 20, "r_p_prim": rp_for(r1_nm), "r_p_sec": rp_for(r2_nm)}


def scale_param(value, factor):
    """값이 함수면 lambda *a: factor * f(*a)처럼 감싸고, 숫자면 곱해요. 배수 1이면 그대로 둬요."""
    if factor == 1:
        return value
    if callable(value):
        def scaled(*args):
            return factor * value(*args)
        scaled.__name__ = f"x{factor:g}_{getattr(value, '__name__', 'function')}"
        return scaled
    return factor * value


# ---------------------------------------------------------- PyBaMM 준비 ----
def context():
    """d와 같은 하프셀 기본 값 묶음. e의 context()가 d의 run_study 1–4단계와 같은 순서로 만들어요."""
    import e_lfp_size_mix as estudy
    return estudy.context()


def evaluate_param(value, args):
    """파라미터를 시험 점에서 계산해요(함수면 pybamm.Scalar로 불러서 값으로)."""
    import pybamm
    if callable(value):
        out = value(*[pybamm.Scalar(a) for a in args])
        return float(out.evaluate()) if hasattr(out, "evaluate") else float(out)
    return float(value)


def processed_param(p, key, inputs):
    """PyBaMM이 모델 안에서 쓰는 길(ParameterValues.process_symbol)로 같은 시험 점의 값을 계산해요."""
    import pybamm
    sym = pybamm.FunctionParameter(key, {k: pybamm.Scalar(v) for k, v in inputs.items()})
    return float(p.process_symbol(sym).evaluate())


def test_points(ctx):
    base = ctx["base"]
    c_max = ctx["c_max"]
    c_e0 = float(base["Initial concentration in electrolyte [mol.m-3]"])
    return {
        KEY_J0: {"args": (c_e0, 0.5 * c_max, c_max, TEST_T_K),
                 "inputs": {"Electrolyte concentration [mol.m-3]": c_e0,
                            "Positive particle surface concentration [mol.m-3]": 0.5 * c_max,
                            "Maximum positive particle surface concentration [mol.m-3]": c_max,
                            "Temperature [K]": TEST_T_K},
                 "point": {"c_e_mol_m3": c_e0, "c_s_surf_mol_m3": 0.5 * c_max, "c_s_max_mol_m3": c_max, "T_K": TEST_T_K}},
        KEY_D: {"args": (0.5, TEST_T_K),
                "inputs": {"Positive particle stoichiometry": 0.5, "Temperature [K]": TEST_T_K},
                "point": {"stoichiometry": 0.5, "T_K": TEST_T_K}},
    }


def proof_entry(p, key, base_value, factor, tp, phase_prefix=""):
    """배수가 정말 들어갔는지: 바꾼 파라미터를 시험 점에서 직접 계산하고, PyBaMM 처리 경로로도 계산해요."""
    full = phase_prefix + key
    new_value = p[full]
    v_base = evaluate_param(base_value, tp["args"])
    v_new = evaluate_param(new_value, tp["args"])
    out = {
        "parameter": full,
        "base_is_function": callable(base_value),
        "base_function": getattr(base_value, "__name__", None) if callable(base_value) else None,
        "updated_is_function": callable(new_value),
        "updated_function": getattr(new_value, "__name__", None) if callable(new_value) else None,
        "test_point": tp["point"],
        "base_value": v_base,
        "updated_value": v_new,
        "ratio": v_new / v_base,
        "expected_factor": factor,
    }
    try:
        # 두 입자 종류 이름("Primary: ...")은 FunctionParameter 이름으로 그대로 찾을 수 있어요.
        # 상수(확산계수)도 함수와 똑같이 처리 경로를 거쳐요. 저장된 숫자를 다시 읽으면 따로 확인한 게 아니에요.
        pv = processed_param(p, full, tp["inputs"])
        out["pybamm_processed_value"] = pv
        out["pybamm_processed_ratio"] = pv / v_base
    except Exception as err:  # noqa: BLE001 - 처리 경로 점검이 안 되면 이유를 남겨요
        out["pybamm_processed_error"] = f"{type(err).__name__}: {str(err)[:200]}"
    ratios = [out["ratio"]] + ([out["pybamm_processed_ratio"]] if "pybamm_processed_ratio" in out else [])
    out["ok"] = all(abs(r - factor) <= 1e-12 * max(1.0, factor) for r in ratios)
    return out


def make_param(task):
    """설계 하나의 ParameterValues, 모델 선택지, 배수 증명을 만들어요."""
    ctx = context()
    base = ctx["base"]
    eps = task["porosity"]
    act = dstudy.active_fraction(eps)
    tps = test_points(ctx)
    p = base.copy()
    p["Positive electrode porosity"] = eps
    p["Positive electrode active material volume fraction"] = act
    proof = []
    if task["kind"] == "single":
        p["Positive particle radius [m]"] = task["radius_nm"] * 1e-9
        p[KEY_D] = scale_param(base[KEY_D], task["D"])
        p[KEY_J0] = scale_param(base[KEY_J0], task["k"])
        proof.append(proof_entry(p, KEY_D, base[KEY_D], task["D"], tps[KEY_D]))
        proof.append(proof_entry(p, KEY_J0, base[KEY_J0], task["k"], tps[KEY_J0]))
        return p, OPT_SINGLE, proof
    # 두 입자 종류: Primary = 작은 입자, Secondary = 큰 입자. 활물질 부피 비율은 같은 전체 값을 부피 비율로 나눠요.
    phi = task["large_fraction"]
    r1, r2 = task["radii_nm"]
    for ph in ("Primary: ", "Secondary: "):
        p.update({ph + k: base[k] for k in PHASE_KEYS}, check_already_exists=False)
    p["Primary: Positive particle radius [m]"] = r1 * 1e-9
    p["Secondary: Positive particle radius [m]"] = r2 * 1e-9
    p["Primary: Positive electrode active material volume fraction"] = (1 - phi) * act
    p["Secondary: Positive electrode active material volume fraction"] = phi * act
    p["Secondary: " + KEY_D] = scale_param(base[KEY_D], task["D"])
    p["Secondary: " + KEY_J0] = scale_param(base[KEY_J0], task["k"])
    for ph, fac_d, fac_k in (("Primary: ", 1, 1), ("Secondary: ", task["D"], task["k"])):
        proof.append(proof_entry(p, KEY_D, base[KEY_D], fac_d, tps[KEY_D], ph))
        proof.append(proof_entry(p, KEY_J0, base[KEY_J0], fac_k, tps[KEY_J0], ph))
    return p, OPT_COMPOSITE, proof


# 계산기(solver) 대안 순서는 d와 같아요: 기본 IDAKLU → 허용 오차를 줄인 IDAKLU → CasadiSolver(safe).
SOLVER_NAMES = ["IDAKLUSolver", "IDAKLUSolver(rtol=1e-8,atol=1e-10)", "CasadiSolver(safe)"]


def make_solver(name):
    import pybamm
    return {
        SOLVER_NAMES[0]: lambda: pybamm.IDAKLUSolver(),
        SOLVER_NAMES[1]: lambda: pybamm.IDAKLUSolver(rtol=1e-8, atol=1e-10),
        SOLVER_NAMES[2]: lambda: pybamm.CasadiSolver(mode="safe"),
    }[name]()


def _var(sol, name):
    try:
        return sol[name].entries
    except Exception:  # noqa: BLE001 - 없는 변수는 None으로 남겨요
        return None


def _at(arr, i):
    """시간 칸 i의 값. 공간 축이 있으면 그 축들을 평균해요(변수마다 모양이 달라서요)."""
    import numpy as np
    a = np.asarray(arr)
    return float(a[..., i].mean()) if a.ndim > 1 else float(a[i])


def _at_q(arr, q, qx):
    """방전 용량 qx 지점의 값(시간 칸 사이는 용량으로 보간). 공간 축이 있으면 먼저 평균해요."""
    import numpy as np
    a = np.asarray(arr)
    series = a.reshape(-1, a.shape[-1]).mean(axis=0) if a.ndim > 1 else a
    return float(np.interp(qx, q, series))


def solve_once(p, opt, phases, current_A, c_rate, var_pts, solver_name):
    """한 번 방전. d의 solve_once와 같은 값에, 원인을 가르는 값(끝 상태 리튬 채움, 반쯤 방전했을 때 손실 전압)을 더해요."""
    import numpy as np
    import pybamm
    ctx = context()
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    p = p.copy()
    p["Current function [A]"] = current_A
    model = pybamm.lithium_ion.DFN(opt)
    sim = pybamm.Simulation(model, parameter_values=p, solver=make_solver(solver_name),
                            var_pts={**model.default_var_pts, **var_pts})
    t0 = time.perf_counter()
    sol = sim.solve([0, 1.5 * 3600 / c_rate])
    sec = time.perf_counter() - t0
    area_m2 = ctx["area_m2"]
    q = sol["Discharge capacity [A.h]"].entries
    v = sol["Voltage [V]"].entries
    to_cm2 = 1000 / (area_m2 * 1e4)
    energy = float(trapezoid(v, q))
    cut = dstudy.SECOND_CUTOFF_V
    below = np.nonzero(v < cut)[0]
    if len(below) == 0:
        q25 = float(q[-1])
    else:
        i = int(below[0])
        q25 = 0.0 if i == 0 else float(q[i - 1] + (v[i - 1] - cut) / (v[i - 1] - v[i]) * (q[i] - q[i - 1]))
    ce_p = sol["Positive electrolyte concentration [mol.m-3]"].entries
    q_mid = q[-1] / 2   # 그 방전 용량의 절반 지점. 중간 전압과 같은 방식(보간)으로 읽어요.
    out = {
        "capacity_Ah": float(q[-1]),
        "capacity_mAh_cm2": float(q[-1]) * to_cm2,
        "energy_mWh_cm2": round(energy * to_cm2, 5),
        "average_voltage_V": round(energy / float(q[-1]), 4),
        "capacity_to_2p5V_mAh_cm2": round(q25 * to_cm2, 5),
        "min_cathode_electrolyte_conc_mol_m3": round(float(ce_p.min()), 1),
        "start_voltage_V": round(float(v[0]), 4),
        "mid_voltage_V": round(float(np.interp(q_mid, q, v)), 4),
        "end_voltage_V": round(float(v[-1]), 4),
        "end_time_s": round(float(sol.t[-1]), 2),
        "termination": str(sol.termination),
        "reached_cutoff": "Minimum voltage" in str(sol.termination),
        "seconds_incl_build": round(sec, 3),
        "solver": solver_name,
    }
    # 입자 종류별 값. 한 가지 크기는 phases = [("", "")], 두 종류는 [("_small", " primary"), ("_large", " secondary")]
    for tag, ph in phases:
        j0 = _var(sol, f"X-averaged positive electrode{ph} exchange current density [A.m-2]")
        a = _var(sol, f"X-averaged positive electrode{ph} surface area to volume ratio [m-1]")
        sto_r = _var(sol, f"R-averaged positive{ph} particle stoichiometry")
        sto_s = _var(sol, f"X-averaged positive{ph} particle surface stoichiometry")
        eta_r = _var(sol, f"X-averaged positive electrode{ph} reaction overpotential [V]")
        eta_c = _var(sol, f"Positive{ph} particle concentration overpotential [V]")
        out[f"t0_exchange_current_density_A_m2{tag}"] = _at(j0, 0) if j0 is not None else None
        out[f"surface_area_to_volume_m-1{tag}"] = float(f"{_at(a, 0):.8g}") if a is not None else None
        out[f"end_stoichiometry_mean{tag}"] = round(_at(sto_r, -1), 4) if sto_r is not None else None
        out[f"end_surface_stoichiometry_mean{tag}"] = round(_at(sto_s, -1), 4) if sto_s is not None else None
        # 반쯤 방전했을 때(용량의 절반) 손실 전압 크기(mV): 반응(전하 이동)과 입자 속 농도 차이(물질 이동)
        out[f"mid_reaction_overpotential_abs_mV{tag}"] = (round(1000 * abs(_at_q(eta_r, q, q_mid)), 2)
                                                         if eta_r is not None else None)
        out[f"mid_particle_concentration_overpotential_abs_mV{tag}"] = (
            round(1000 * abs(_at_q(eta_c, q, q_mid)), 2) if eta_c is not None else None)
    eta_e = _var(sol, "X-averaged electrolyte overpotential [V]")
    out["mid_electrolyte_overpotential_abs_mV"] = (round(1000 * abs(_at_q(eta_e, q, q_mid)), 2)
                                                   if eta_e is not None else None)
    return out


def solve_chain(p, opt, phases, current_A, c_rate, var_pts):
    attempts = []
    for name in SOLVER_NAMES:
        try:
            out = solve_once(p, opt, phases, current_A, c_rate, var_pts, name)
        except Exception as err:  # noqa: BLE001 - 실패도 결과로 남겨요
            attempts.append(f"{name}: {type(err).__name__}: {str(err)[:300]}")
            continue
        if not out["reached_cutoff"]:
            attempts.append(f"{name}: 하한 전압에 닿기 전에 멈춤 ({out['termination']})")
            continue
        out["earlier_attempts"] = attempts
        return out, attempts
    return None, attempts


def phases_for(task):
    return [("", "")] if task["kind"] == "single" else [("_small", " primary"), ("_large", " secondary")]


# ------------------------------------------------------------- 설계 하나 ----
def run_design(task):
    """0.1C로 1C를 정한 뒤(0.1% 안으로 맞을 때까지 반복) 1C·2C·5C·10C를 풀어요. d의 run_design과 같은 순서예요."""
    ctx = context()
    eps = task["porosity"]
    act = dstudy.active_fraction(eps)
    area_m2, thickness, x100, c_max = ctx["area_m2"], ctx["thickness"], ctx["x100"], ctx["c_max"]
    q_theo = act * thickness * c_max * dstudy.F / 3600 * (1 - x100) / 10
    label = task["label"]
    failures = []
    d = {k: task[k] for k in ("design", "variant", "label", "kind", "porosity", "radius_nm", "large_fraction",
                              "radii_nm", "D", "k", "var_pts", "note_ko") if k in task}
    d.update({
        "active_volume_fraction": round(act, 5),
        "conductivity_S_m": ctx["base_sigma"],
        "thickness_um": thickness * 1e6,
        "theoretical_capacity_mAh_cm2": round(q_theo, 4),
        "estimate_density_g_cm3": round(dstudy.density_estimate(eps), 4),
        "runs": [],
    })
    try:
        p, opt, proof = make_param(task)
    except Exception as err:  # noqa: BLE001
        d["status"] = "설정 실패"
        d["error"] = f"{type(err).__name__}: {str(err)[:500]}"
        failures.append({"design": label, "stage": "make_param", "error": d["error"]})
        return d, failures
    d["model_options"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in opt.items()}
    d["parameter_scaling_proof"] = proof
    phases = phases_for(task)

    def solve_safe(current, c):
        out, attempts = solve_chain(p, opt, phases, current, c, task["var_pts"])
        if out is None:
            failures.append({"design": label, "c_rate": c, "attempts": attempts})
        return out

    q_ref = q_theo
    iterations = []
    low = None
    q_used = q_ref
    for _ in range(4):
        current = 0.1 * q_ref / 1000 * 1e4 * area_m2
        low = solve_safe(current, 0.1)
        if low is None:
            break
        iterations.append(round(low["capacity_mAh_cm2"], 5))
        converged = abs(low["capacity_mAh_cm2"] - q_ref) / q_ref < 1e-3
        q_used = q_ref
        q_ref = low["capacity_mAh_cm2"]
        if converged:
            break
    d["c_rate_1C_iterations_mAh_cm2"] = iterations
    if low is None:
        d["status"] = "0.1C 계산 실패"
        return d, failures
    one_c = q_ref
    d["one_C_definition_mAh_cm2"] = round(one_c, 5)
    d["low_rate_run_used_1C_basis_mAh_cm2"] = round(q_used, 5)
    q01 = low["capacity_mAh_cm2"]
    d["capacity_0p1C_mAh_cm2"] = round(q01, 5)
    d["utilization_0p1C_pct_of_theoretical"] = round(100 * q01 / q_theo, 2)
    e01, q25_01 = low["energy_mWh_cm2"], low["capacity_to_2p5V_mAh_cm2"]
    d["runs"].append(dict(low, c_rate=0.1, current_density_mA_cm2=round(0.1 * q_used, 5),
                          capacity_mAh_cm2=round(q01, 5), retention_pct=100.0,
                          energy_retention_pct=100.0, retention_to_2p5V_pct=100.0))
    for c in C_RATES[1:]:
        out = solve_safe(c * one_c / 1000 * 1e4 * area_m2, c)
        if out is None:
            d["runs"].append({"c_rate": c, "status": "failed"})
            continue
        out.update(c_rate=c, current_density_mA_cm2=round(c * one_c, 5),
                   capacity_mAh_cm2=round(out["capacity_mAh_cm2"], 5),
                   retention_pct=round(100 * out["capacity_mAh_cm2"] / q01, 2),
                   energy_retention_pct=round(100 * out["energy_mWh_cm2"] / e01, 2),
                   retention_to_2p5V_pct=round(100 * out["capacity_to_2p5V_mAh_cm2"] / q25_01, 2))
        d["runs"].append(out)
    d["status"] = "ok" if all("status" not in r for r in d["runs"]) else "일부 실패"
    print(f"{label}: 0.1C {q01:.4f} mAh/cm2 | "
          + " ".join(f"{r['c_rate']}C {r.get('retention_pct', 'FAIL')}%" for r in d["runs"]), flush=True)
    return d, failures


def run_single(task):
    """10C 한 번만 다시 풀기(격자 점검)."""
    ctx = context()
    p, opt, _ = make_param(task)
    current = task["c_rate"] * task["one_C_mAh_cm2"] / 1000 * 1e4 * ctx["area_m2"]
    out, attempts = solve_chain(p, opt, phases_for(task), current, task["c_rate"], task["var_pts"])
    res = {k: task[k] for k in ("check", "design_label", "c_rate", "used_var_pts", "var_pts") if k in task}
    if out is None:
        res["error"] = attempts
        return res
    ref = task["reference_capacity_mAh_cm2"]
    res.update({
        "used_mesh_mAh_cm2": ref,
        "refined_mesh_mAh_cm2": round(out["capacity_mAh_cm2"], 5),
        "difference_pct": round(100 * (out["capacity_mAh_cm2"] - ref) / ref, 3),
        "retention_difference_pct_points": round(100 * (out["capacity_mAh_cm2"] - ref) / task["capacity_0p1C_mAh_cm2"], 3),
        "solver": out["solver"], "earlier_attempts": out["earlier_attempts"],
        "seconds_incl_build": out["seconds_incl_build"],
    })
    print(f"mesh {res['check']} | {res['design_label']}: {res['difference_pct']}%", flush=True)
    return res


def _call(args):
    fn_name, task = args
    return {"run_design": run_design, "run_single": run_single}[fn_name](task)


def run_tasks(fn_name, tasks, workers):
    args = [(fn_name, t) for t in tasks]
    if workers <= 1:
        return [_call(a) for a in args]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_call, args))


# ---------------------------------------------------------------- 계산 ----
def build_tasks():
    tasks = []
    for eps in POROSITIES:
        for vid, r_nm, fd, fk in VARIANTS_A:
            tasks.append({"design": "A", "kind": "single", "variant": vid, "porosity": eps, "radius_nm": r_nm,
                          "D": fd, "k": fk, "var_pts": var_pts_single(r_nm), "label": f"A eps={eps} {vid}"})
        # 점검 (4): D×16·k×4가 50 nm와 다른 까닭이 입자 안 칸 수인지 보려고, 입자 안 칸 수를 서로 바꾼 설계를 더해요.
        tasks.append({"design": "check4", "kind": "single", "variant": "D16k4_rp20", "porosity": eps, "radius_nm": 200,
                      "D": 16, "k": 4, "var_pts": {**var_pts_single(200), "r_p": 20},
                      "label": f"check4 eps={eps} D16k4 r_p=20",
                      "note_ko": "200 nm·D×16·k×4를 50 nm와 같은 입자 안 칸 수(20칸, 반지름 대비 같은 칸 간격)로 풀어요."})
        tasks.append({"design": "check4", "kind": "single", "variant": "R50_rp80", "porosity": eps, "radius_nm": 50,
                      "D": 1, "k": 1, "var_pts": {**var_pts_single(50), "r_p": 80},
                      "label": f"check4 eps={eps} R50 r_p=80",
                      "note_ko": "50 nm를 200 nm 설계와 같은 입자 안 칸 수(80칸)로 풀어요."})
        for phi in BLEND_FRACTIONS:
            for vid, fd, fk in VARIANTS_B:
                tasks.append({"design": "B", "kind": "composite", "variant": vid, "porosity": eps, "large_fraction": phi,
                              "radii_nm": [R_SMALL_NM, R_LARGE_NM], "D": fd, "k": fk,
                              "var_pts": var_pts_composite(R_SMALL_NM, R_LARGE_NM),
                              "label": f"B eps={eps} large={phi:.0%} {vid}"})
        # 설계 B 구현 점검: 두 종류를 같은 크기로 두면 한 가지 크기(설계 A)와 같아야 해요.
        for r_nm in (R_SMALL_NM, R_LARGE_NM):
            tasks.append({"design": "B_check", "kind": "composite", "variant": f"same{r_nm}", "porosity": eps,
                          "large_fraction": 0.5, "radii_nm": [r_nm, r_nm], "D": 1, "k": 1,
                          "var_pts": var_pts_composite(r_nm, r_nm),
                          "label": f"B_check eps={eps} both {r_nm}nm 50/50",
                          "note_ko": f"두 종류 모두 {r_nm} nm, 부피 50:50. 설계 A의 {r_nm} nm와 같아야 해요."})
    return tasks


def run_study(workers):
    os.environ.setdefault("PYBAMM_DISABLE_TELEMETRY", "true")
    import casadi
    import pybamm

    t_start = time.perf_counter()
    ctx = context()
    tasks = build_tasks()
    out = run_tasks("run_design", tasks, workers)
    designs = [d for d, _ in out]
    failures = [f for _, fs in out for f in fs]
    t_designs = time.perf_counter() - t_start

    def find(design, eps, variant, phi=None):
        return next((x for x in designs if x["design"] == design and x["porosity"] == eps and x["variant"] == variant
                     and (phi is None or x.get("large_fraction") == phi)), None)

    def run_at(d, c):
        return next((r for r in (d or {}).get("runs", []) if r["c_rate"] == c and "capacity_mAh_cm2" in r), None)

    def ret_full(d, c):
        """반올림하지 않은 유지율(%): 몫 계산에 써요."""
        r = run_at(d, c)
        return 100 * r["capacity_mAh_cm2"] / d["capacity_0p1C_mAh_cm2"] if r else None

    # ---- 격자 점검 (2) ----
    mesh_tasks = []
    for eps in POROSITIES:
        for vid in MESH_CHECK_A:
            d = find("A", eps, vid)
            r = run_at(d, 10.0)
            if r is None:
                continue
            vp = d["var_pts"]
            for name, fine in (("r_p x2 (입자 안 칸만 2배)", {**vp, "r_p": 2 * vp["r_p"]}),
                               ("x_s·x_p·r_p x2 (모든 방향 2배)", {k: 2 * v for k, v in vp.items()})):
                mesh_tasks.append({**{k: d[k] for k in ("kind", "porosity", "radius_nm", "D", "k")},
                                   "check": name, "design_label": d["label"], "c_rate": 10.0,
                                   "used_var_pts": vp, "var_pts": fine, "one_C_mAh_cm2": d["one_C_definition_mAh_cm2"],
                                   "reference_capacity_mAh_cm2": r["capacity_mAh_cm2"],
                                   "capacity_0p1C_mAh_cm2": d["capacity_0p1C_mAh_cm2"]})
        for phi, vid in MESH_CHECK_B:
            d = find("B", eps, vid, phi)
            r = run_at(d, 10.0)
            if r is None:
                continue
            vp = d["var_pts"]
            for name, fine in (("r_p x2 (입자 안 칸만 2배)", {**vp, "r_p_prim": 2 * vp["r_p_prim"], "r_p_sec": 2 * vp["r_p_sec"]}),
                               ("x_s·x_p·r_p x2 (모든 방향 2배)", {k: 2 * v for k, v in vp.items()})):
                mesh_tasks.append({**{k: d[k] for k in ("kind", "porosity", "large_fraction", "radii_nm", "D", "k")},
                                   "check": name, "design_label": d["label"], "c_rate": 10.0,
                                   "used_var_pts": vp, "var_pts": fine, "one_C_mAh_cm2": d["one_C_definition_mAh_cm2"],
                                   "reference_capacity_mAh_cm2": r["capacity_mAh_cm2"],
                                   "capacity_0p1C_mAh_cm2": d["capacity_0p1C_mAh_cm2"]})
    # 설계 B 안정성 점검: 두 종류를 같은 50 nm로 둔 B_check가 공극률 0.30·10C(전해질이 바닥나는 조건)에서만
    # 설계 A와 달라서(처음 실행에서 확인), 양극 두께 방향 칸 수(x_p)를 40·80칸으로 늘려 그 차이가 줄어드는지 봐요.
    stab_tasks = []
    for design, variant, phi in STABILITY_DESIGNS:
        d = find(design, 0.30, variant, phi)
        r = run_at(d, 10.0)
        if r is None:
            continue
        for xp in STABILITY_XP:
            stab_tasks.append({**{k: d[k] for k in ("kind", "porosity", "radius_nm", "large_fraction", "radii_nm", "D", "k")
                                  if k in d},
                               "check": f"x_p {xp}", "design_label": d["label"], "c_rate": 10.0,
                               "used_var_pts": d["var_pts"], "var_pts": {**d["var_pts"], "x_p": xp},
                               "one_C_mAh_cm2": d["one_C_definition_mAh_cm2"],
                               "reference_capacity_mAh_cm2": r["capacity_mAh_cm2"],
                               "capacity_0p1C_mAh_cm2": d["capacity_0p1C_mAh_cm2"]})
    extra = run_tasks("run_single", mesh_tasks + stab_tasks, workers)
    mesh_check, stab_out = extra[:len(mesh_tasks)], extra[len(mesh_tasks):]

    # ---- 설계 A 요약: 유지율과 '되찾은 몫' ----
    def share(v, lo, hi):
        if v is None or lo is None or hi is None or abs(hi - lo) < 1e-9:
            return None
        return round(100 * (v - lo) / (hi - lo), 2)

    table_a = []
    for eps in POROSITIES:
        r50, r200 = find("A", eps, "R50"), find("A", eps, "R200")
        for vid, r_nm, fd, fk in VARIANTS_A:
            d = find("A", eps, vid)
            row = {"porosity": eps, "variant": vid, "radius_nm": r_nm, "D_factor": fd, "k_factor": fk,
                   "capacity_0p1C_mAh_cm2": d.get("capacity_0p1C_mAh_cm2") if d else None}
            for c in SHARE_RATES:
                r = run_at(d, c)
                row[f"retention_{c:g}C_pct"] = r["retention_pct"] if r else None
            r10 = run_at(d, 10.0)
            row["energy_retention_10C_pct"] = r10["energy_retention_pct"] if r10 else None
            row["capacity_10C_mAh_cm2"] = r10["capacity_mAh_cm2"] if r10 else None
            row["gap_recovered_share_pct"] = {f"{c:g}C": share(ret_full(d, c), ret_full(r200, c), ret_full(r50, c))
                                              for c in SHARE_RATES}
            if r10:
                for key in ("end_stoichiometry_mean", "end_surface_stoichiometry_mean",
                            "mid_reaction_overpotential_abs_mV", "mid_particle_concentration_overpotential_abs_mV",
                            "mid_electrolyte_overpotential_abs_mV", "min_cathode_electrolyte_conc_mol_m3"):
                    row[f"10C_{key}"] = r10.get(key)
            table_a.append(row)

    # 확산(D×16)과 반응(k×4)을 따로·함께 줬을 때 되찾은 몫과 상호작용(함께 − 따로의 합)
    decomposition = []
    for eps in POROSITIES:
        rows = {t["variant"]: t for t in table_a if t["porosity"] == eps}
        for c in SHARE_RATES:
            key = f"{c:g}C"
            sd, sk, sb = (rows[v]["gap_recovered_share_pct"][key] for v in ("D16", "k4", "D16k4"))
            decomposition.append({
                "porosity": eps, "c_rate": c,
                "retention_R50_pct": rows["R50"][f"retention_{c:g}C_pct"],
                "retention_R200_pct": rows["R200"][f"retention_{c:g}C_pct"],
                "gap_pct_points": (round(rows["R50"][f"retention_{c:g}C_pct"] - rows["R200"][f"retention_{c:g}C_pct"], 2)
                                   if rows["R50"][f"retention_{c:g}C_pct"] is not None
                                   and rows["R200"][f"retention_{c:g}C_pct"] is not None else None),
                "share_D16_pct": sd, "share_k4_pct": sk, "share_D16k4_pct": sb,
                "interaction_pct": round(sb - sd - sk, 2) if None not in (sd, sk, sb) else None,
            })

    # ---- 배수 증명 모음 (모델 안 값 포함) ----
    scaling_proof = {"rule_ko": ("함수로 된 파라미터(교환 전류 밀도 f(c_e, c_s_surf, c_s_max, T))는 lambda *a: factor * f(*a)처럼 감싸고, "
                                 "숫자로 된 파라미터(이 값 묶음의 확산계수 5.9e-18 m²/s, 상수)는 그대로 곱했어요. "
                                 "바꾼 파라미터를 시험 점에서 직접 계산(updated_value)하고, 함수·상수 모두 PyBaMM이 모델 안에서 쓰는 처리 경로"
                                 "(ParameterValues.process_symbol)로도 계산(pybamm_processed_value)해서 배수와 같은지 봤어요."),
                     "test_points": {k: v["point"] for k, v in test_points(ctx).items()},
                     "per_design": [], "in_model_t0": [], "all_ok": True}
    for d in designs:
        pr = d.get("parameter_scaling_proof", [])
        scaling_proof["per_design"].append({"design": d["label"], "entries": [
            {k: e.get(k) for k in ("parameter", "expected_factor", "ratio", "pybamm_processed_ratio", "ok",
                                   "updated_function", "pybamm_processed_error")} for e in pr]})
        if not all(e.get("ok") for e in pr):
            scaling_proof["all_ok"] = False
    # 모델 안: 같은 처음 상태(t=0)에서 교환 전류 밀도는 k배, 겉넓이 비율(a)은 반지름에 반비례해야 해요.
    for eps in POROSITIES:
        ref200, ref50 = find("A", eps, "R200"), find("A", eps, "R50")
        j_ref = run_at(ref200, 0.1)["t0_exchange_current_density_A_m2"] if run_at(ref200, 0.1) else None
        a50 = run_at(ref50, 0.1)["surface_area_to_volume_m-1"] if run_at(ref50, 0.1) else None
        for vid, r_nm, fd, fk in VARIANTS_A:
            r = run_at(find("A", eps, vid), 0.1)
            if r is None or j_ref is None:
                continue
            scaling_proof["in_model_t0"].append({
                "design": f"A eps={eps} {vid}", "expected_k": fk,
                "t0_exchange_current_density_A_m2": r["t0_exchange_current_density_A_m2"],
                "ratio_to_R200": round(r["t0_exchange_current_density_A_m2"] / j_ref, 12),
                "surface_area_to_volume_m-1": r["surface_area_to_volume_m-1"],
                "surface_area_ratio_to_R50": round(r["surface_area_to_volume_m-1"] / a50, 8),
                "expected_surface_area_ratio_to_R50": R_SMALL_NM / r_nm,
                "exchange_current_per_volume_ratio_to_R50": round(
                    r["surface_area_to_volume_m-1"] * r["t0_exchange_current_density_A_m2"]
                    / (a50 * run_at(ref50, 0.1)["t0_exchange_current_density_A_m2"]), 8),
            })
        for phi in BLEND_FRACTIONS:
            bb = run_at(find("B", eps, "base", phi), 0.1)
            for vid, fd, fk in VARIANTS_B:
                r = run_at(find("B", eps, vid, phi), 0.1)
                if r is None or bb is None:
                    continue
                scaling_proof["in_model_t0"].append({
                    "design": f"B eps={eps} large={phi:.0%} {vid}", "expected_k_small": 1, "expected_k_large": fk,
                    "ratio_small_phase_to_base": round(r["t0_exchange_current_density_A_m2_small"]
                                                       / bb["t0_exchange_current_density_A_m2_small"], 12),
                    "ratio_large_phase_to_base": round(r["t0_exchange_current_density_A_m2_large"]
                                                       / bb["t0_exchange_current_density_A_m2_large"], 12),
                    "surface_area_to_volume_m-1_small": r["surface_area_to_volume_m-1_small"],
                    "surface_area_to_volume_m-1_large": r["surface_area_to_volume_m-1_large"],
                })
    scaling_proof["in_model_note_ko"] = (
        "모든 설계가 같은 처음 상태(만충, 전해질 1200 mol/m³)에서 시작해서, t=0의 교환 전류 밀도 비는 k와 같아야 해요. "
        "확산계수는 PyBaMM 결과 변수('Positive particle effective diffusivity')로 읽으면 약 1.1e-16 m²/s(2^-53) 단위의 계단으로만 나와요"
        "(기준 5.9e-18은 0, D×16의 9.44e-17은 1.11e-16. IDAKLU·CasadiSolver 모두 같음). 그래서 결과 변수 대신 위 처리 경로 계산으로 확인했어요. "
        "모델 안 값이 정상인 것은 D 배수에 따라 용량이 계단 없이 바뀌는 것으로도 보여요.")

    # ---- 점검 (1): d와 맞춰 보기 ----
    with open(D_JSON, encoding="utf-8") as f:
        dres = json.load(f)

    def d_design(study, eps, r_nm):
        return next((x for x in dres["designs"] if x["study"] == study and x["porosity"] == eps
                     and x["conductivity_factor"] == 1.0 and x["particle_radius_nm"] == r_nm), None)

    def compare(fd, dd, name):
        if fd is None or dd is None or "capacity_0p1C_mAh_cm2" not in fd:
            return {"comparison": name, "status": "비교할 결과가 없어요"}
        fr = {r["c_rate"]: r for r in fd["runs"]}
        dr = {r["c_rate"]: r for r in dd["runs"]}
        rows = []
        for c in C_RATES[1:]:
            a, b = fr.get(c, {}), dr.get(c, {})
            if "capacity_mAh_cm2" not in a or "capacity_mAh_cm2" not in b:
                rows.append({"c_rate": c, "status": "한쪽 결과 없음"})
                continue
            rows.append({"c_rate": c, "study_d_retention_pct": b["retention_pct"], "study_f_retention_pct": a["retention_pct"],
                         "retention_difference_pct_points": round(a["retention_pct"] - b["retention_pct"], 2),
                         "capacity_difference_pct": round(100 * (a["capacity_mAh_cm2"] - b["capacity_mAh_cm2"])
                                                          / b["capacity_mAh_cm2"], 4),
                         "study_d_solver": b.get("solver"), "study_f_solver": a.get("solver")})
        qd, qf = dd["capacity_0p1C_mAh_cm2"], fd["capacity_0p1C_mAh_cm2"]
        return {"comparison": name,
                "capacity_0p1C_mAh_cm2": {"study_d": qd, "study_f": qf, "difference_pct": round(100 * (qf - qd) / qd, 4)},
                "one_C_mAh_cm2": {"study_d": dd["one_C_definition_mAh_cm2"], "study_f": fd["one_C_definition_mAh_cm2"]},
                "rates": rows,
                "max_abs_retention_difference_pct_points": max(
                    [abs(r["retention_difference_pct_points"]) for r in rows if "retention_difference_pct_points" in r] or [None])}

    check1 = [
        compare(find("A", 0.30, "R50"), d_design("main", 0.30, 50), "A eps=0.30 R50 vs d main eps=0.30 R=50nm"),
        compare(find("A", 0.30, "R200"), d_design("particle_radius", 0.30, 200), "A eps=0.30 R200 vs d particle_radius eps=0.30 R=200nm"),
        compare(find("A", 0.38, "R50"), d_design("main", 0.38, 50), "A eps=0.38 R50 vs d main eps=0.38 R=50nm (추가)"),
    ]

    # ---- 점검 (3): 모든 방전이 하한 전압에 닿았나 ----
    all_runs = [(d["label"], r) for d in designs for r in d["runs"]]
    check3 = {
        "designs": len(designs),
        "runs_total": len(all_runs),
        "runs_reached_cutoff": sum(1 for _, r in all_runs if r.get("reached_cutoff")),
        "runs_failed": [{"design": lab, "c_rate": r["c_rate"]} for lab, r in all_runs if "status" in r],
        "designs_not_ok": [{"design": d["label"], "status": d.get("status"), "error": d.get("error")}
                           for d in designs if d.get("status") != "ok"],
        "mesh_check_runs": len(mesh_check),
        "mesh_check_failed": [m for m in mesh_check if "error" in m],
        "stability_check_runs": len(stab_out),
        "stability_check_failed": [m for m in stab_out if "error" in m],
        "solver_usage": {n: sum(1 for _, r in all_runs if r.get("solver") == n) for n in SOLVER_NAMES},
        "runs_with_earlier_solver_failures": [
            {"design": lab, "c_rate": r["c_rate"], "solver": r["solver"], "earlier_attempts": r["earlier_attempts"]}
            for lab, r in all_runs if r.get("earlier_attempts")],
        "note_ko": ("d와 같이, 계산기가 오류를 내거나 하한 전압(2.0 V)에 닿기 전에 멈추면 다음 계산기로 다시 풀었어요. "
                    "runs_reached_cutoff가 runs_total과 같으면 결과에 쓴 방전은 모두 하한 전압에서 끝난 거예요. "
                    "앞선 계산기가 실패한 방전은 runs_with_earlier_solver_failures에 이유와 함께 남겼어요."),
    }

    # ---- 점검 (4): D×16·k×4 vs 50 nm ----
    def cmp_rates(a, b):
        rows = []
        for c in C_RATES:
            ra, rb = run_at(a, c), run_at(b, c)
            if ra is None or rb is None:
                rows.append({"c_rate": c, "status": "한쪽 결과 없음"})
                continue
            rows.append({"c_rate": c,
                         "capacity_difference_pct": round(100 * (ra["capacity_mAh_cm2"] - rb["capacity_mAh_cm2"])
                                                          / rb["capacity_mAh_cm2"], 4),
                         "retention_difference_pct_points": round(ret_full(a, c) - ret_full(b, c), 4)})
        return rows

    check4 = []
    for eps in POROSITIES:
        r50, both = find("A", eps, "R50"), find("A", eps, "D16k4")
        both20, r50_80 = find("check4", eps, "D16k4_rp20"), find("check4", eps, "R50_rp80")
        mech = {}
        for name, d in (("R50 (r_p 20)", r50), ("D16k4 (r_p 80)", both), ("D16k4 (r_p 20)", both20),
                        ("R50 (r_p 80)", r50_80), ("R200", find("A", eps, "R200"))):
            r10 = run_at(d, 10.0)
            if r10:
                mech[name] = {k: r10.get(k) for k in ("end_stoichiometry_mean", "end_surface_stoichiometry_mean",
                                                      "mid_reaction_overpotential_abs_mV",
                                                      "mid_particle_concentration_overpotential_abs_mV",
                                                      "mid_electrolyte_overpotential_abs_mV", "end_time_s")}
        entry = {
            "porosity": eps,
            "D16k4_rp80_vs_R50_rp20": cmp_rates(both, r50),
            "D16k4_rp20_vs_R50_rp20_same_relative_mesh": cmp_rates(both20, r50),
            "D16k4_rp80_vs_R50_rp80_same_relative_mesh": cmp_rates(both, r50_80),
            "R50_rp80_vs_R50_rp20_mesh_effect_alone": cmp_rates(r50_80, r50),
            "capacity_0p1C_mAh_cm2": {n: (d or {}).get("capacity_0p1C_mAh_cm2")
                                      for n, d in (("R50", r50), ("D16k4", both), ("D16k4_rp20", both20), ("R50_rp80", r50_80))},
            "10C_mechanism_values": mech,
        }

        def max_abs(rows):
            vals = [abs(r["retention_difference_pct_points"]) for r in rows if "retention_difference_pct_points" in r]
            return max(vals) if vals else None
        entry["max_abs_retention_difference_pct_points"] = {
            "D16k4_rp80_vs_R50_rp20": max_abs(entry["D16k4_rp80_vs_R50_rp20"]),
            "same_relative_mesh_rp20": max_abs(entry["D16k4_rp20_vs_R50_rp20_same_relative_mesh"]),
            "same_relative_mesh_rp80": max_abs(entry["D16k4_rp80_vs_R50_rp80_same_relative_mesh"]),
            "mesh_effect_alone_R50": max_abs(entry["R50_rp80_vs_R50_rp20_mesh_effect_alone"]),
        }
        check4.append(entry)
    check4_note = (
        "식으로 보면 D×16·k×4인 200 nm는 50 nm와 같은 문제예요. 입자 안 좌표를 반지름으로 나눠 보면 확산 시간 R²/D가 같고, "
        "겉넓이 비율 a가 1/4이라 같은 전류에서 겉넓이당 반응 속도가 4배인데, 교환 전류 밀도도 4배라 버틀러–볼머 식의 비(반응 속도 ÷ 교환 전류 밀도)와 "
        "과전압이 같아요. 입자 겉면의 농도 기울기(반응 속도 ÷ D × R)도 같아요. 그래서 식에 남는 차이는 없고, "
        "계산 결과의 차이는 계산 칸(입자 안 칸 수)과 계산기 허용 오차에서만 올 수 있어요. "
        "이를 확인하려고 입자 안 칸 수를 서로 맞춘 설계(check4)를 더했어요. 같은 상대 칸 간격끼리 비교한 차이(same_relative_mesh)가 "
        "원래 비교(D16k4_rp80_vs_R50_rp20)보다 훨씬 작으면, 남은 차이는 버틀러–볼머 비선형성이나 농도 의존성이 아니라 계산 칸에서 온 거예요.")

    # ---- 설계 B 요약 ----
    table_b = []
    for eps in POROSITIES:
        r50 = find("A", eps, "R50")
        for phi in BLEND_FRACTIONS:
            bb = find("B", eps, "base", phi)
            for vid, fd, fk in VARIANTS_B:
                d = find("B", eps, vid, phi)
                row = {"porosity": eps, "large_fraction": phi, "variant": vid, "large_D_factor": fd, "large_k_factor": fk,
                       "status": (d or {}).get("status"),
                       "capacity_0p1C_mAh_cm2": (d or {}).get("capacity_0p1C_mAh_cm2")}
                for c in SHARE_RATES:
                    r = run_at(d, c)
                    row[f"retention_{c:g}C_pct"] = r["retention_pct"] if r else None
                row["gap_recovered_share_vs_blend_base_pct"] = {
                    f"{c:g}C": share(ret_full(d, c), ret_full(bb, c), ret_full(r50, c)) for c in SHARE_RATES}
                r10 = run_at(d, 10.0)
                if r10:
                    for key in ("end_stoichiometry_mean_small", "end_stoichiometry_mean_large",
                                "end_surface_stoichiometry_mean_small", "end_surface_stoichiometry_mean_large",
                                "mid_reaction_overpotential_abs_mV_small", "mid_reaction_overpotential_abs_mV_large",
                                "min_cathode_electrolyte_conc_mol_m3"):
                        row[f"10C_{key}"] = r10.get(key)
                table_b.append(row)
    b_checks = []
    for eps in POROSITIES:
        for r_nm, vid in ((R_SMALL_NM, "R50"), (R_LARGE_NM, "R200")):
            bc, a = find("B_check", eps, f"same{r_nm}"), find("A", eps, vid)
            b_checks.append({"comparison": f"B_check eps={eps} both {r_nm} nm vs A {vid}", "rates": cmp_rates(bc, a),
                             "capacity_0p1C_mAh_cm2": {"B_check": (bc or {}).get("capacity_0p1C_mAh_cm2"),
                                                       "A": (a or {}).get("capacity_0p1C_mAh_cm2")}})
    only_large_scaled = []
    for d in designs:
        if d["design"] != "B":
            continue
        pr = d.get("parameter_scaling_proof", [])
        only_large_scaled.append({
            "design": d["label"],
            "primary_ratios": {e["parameter"]: e["ratio"] for e in pr if e["parameter"].startswith("Primary: ")},
            "secondary_ratios": {e["parameter"]: e["ratio"] for e in pr if e["parameter"].startswith("Secondary: ")},
            "ok": all(e.get("ok") for e in pr),
        })
    # e(로그정규 분포 폭 0.25)의 큰 입자 20%와 견주기(같아야 하는 값은 아니에요)
    vs_e = []
    try:
        with open(E_JSON, encoding="utf-8") as f:
            eres = json.load(f)
        for eps in POROSITIES:
            et = next((t for t in eres["summary_table"] if t["porosity"] == eps and t["large_fraction"] == 0.2), None)
            fb = next((t for t in table_b if t["porosity"] == eps and t["large_fraction"] == 0.2 and t["variant"] == "base"), None)
            if et and fb:
                vs_e.append({"porosity": eps, "large_fraction": 0.2,
                             **{f"retention_{c:g}C_pct": {"study_e_distribution": et.get(f"retention_{c:g}C_pct"),
                                                          "study_f_two_phases": fb.get(f"retention_{c:g}C_pct")}
                                for c in SHARE_RATES}})
    except (OSError, KeyError) as err:
        vs_e = [{"status": f"e 결과를 읽지 못했어요: {err}"}]
    b_ok = all(d.get("status") == "ok" for d in designs if d["design"] in ("B", "B_check"))

    # 설계 B 안정성 점검 정리: 공극률 0.30·10C를 양극 두께 방향 칸 수 20·40·80칸으로
    stability = []
    for design, variant, phi in STABILITY_DESIGNS:
        d = find(design, 0.30, variant, phi)
        r = run_at(d, 10.0)
        if r is None:
            continue
        row = {"design": d["label"], "x_p_20_mAh_cm2": r["capacity_mAh_cm2"], "x_p_20_retention_pct": r["retention_pct"],
               "min_cathode_electrolyte_conc_10C_mol_m3": r["min_cathode_electrolyte_conc_mol_m3"]}
        for s in stab_out:
            if s.get("design_label") != d["label"]:
                continue
            xp = s["check"].split()[-1]
            if "refined_mesh_mAh_cm2" in s:
                row[f"x_p_{xp}_mAh_cm2"] = s["refined_mesh_mAh_cm2"]
                row[f"x_p_{xp}_retention_pct"] = round(100 * s["refined_mesh_mAh_cm2"] / d["capacity_0p1C_mAh_cm2"], 2)
                row[f"x_p_{xp}_solver"] = s["solver"]
            else:
                row[f"x_p_{xp}_error"] = s.get("error")
        stability.append(row)
    a_row = next((s for s in stability if s["design"] == "A eps=0.3 R50"), None)
    c_row = next((s for s in stability if s["design"] == "B_check eps=0.3 both 50nm 50/50"), None)
    gap_by_xp = {}
    if a_row and c_row:
        for xp in [20] + STABILITY_XP:
            key = f"x_p_{xp}_mAh_cm2"
            if key in a_row and key in c_row:
                gap_by_xp[f"x_p {xp}"] = round(100 * (c_row[key] - a_row[key]) / a_row[key], 3)
    start_v_offsets = []
    for eps in POROSITIES:
        for r_nm, vid in ((R_SMALL_NM, "R50"), (R_LARGE_NM, "R200")):
            bc, a = find("B_check", eps, f"same{r_nm}"), find("A", eps, vid)
            for c in C_RATES:
                rb, ra = run_at(bc, c), run_at(a, c)
                if rb and ra:
                    start_v_offsets.append(round(1000 * (rb["start_voltage_V"] - ra["start_voltage_V"]), 2))
    stability_summary = {
        "rows_porosity_0p30_10C": stability,
        "same_size_composite_minus_single_capacity_pct_by_x_p": gap_by_xp,
        "start_voltage_offset_composite_minus_single_mV_range": ([min(start_v_offsets), max(start_v_offsets)]
                                                                 if start_v_offsets else None),
        "note_ko": ("두 종류를 모두 50 nm로 둔 전극(B_check)은 공극률 0.38에서는 설계 A와 용량이 소수 다섯째 자리까지 같았지만, "
                    "공극률 0.30·10C(양극 빈틈의 전해질이 바닥나는 조건)에서는 달랐어요. 그래서 양극 두께 방향 칸 수만 40·80칸으로 늘려 "
                    "같은 크기 두 종류와 한 종류의 용량 차이가 줄어드는지 봤어요(same_size_composite_minus_single_capacity_pct_by_x_p). "
                    "줄어들면 두 모델의 차이는 식이 아니라 계산 칸 때문이고, 설계 B의 공극률 0.30·10C 숫자는 그만큼 대략값으로 읽어요. "
                    "또 처음 전압(start_voltage)이 두 종류 모델에서 조금 높게 나와서 그 차이 범위(mV)도 적었어요. 원인은 찾지 못했어요."),
    }
    design_b = {
        "status": "ok" if b_ok else "일부 실패 또는 건너뜀",
        "model_options": {"working electrode": "positive", "particle phases": ["1", "2"]},
        "how_ko": ("PyBaMM 26.8의 DFN은 하프셀(working electrode: positive)과 두 입자 종류(particle phases: (\"1\", \"2\"))를 함께 만들 수 있었어요. "
                   "'음극'(리튬 금속)은 한 종류, 양극은 두 종류예요. 모델이 'Primary: '/'Secondary: '를 붙여 따로 찾는 이름 8개"
                   "(phase_parameter_keys)를 기본 값에서 그대로 복사한 뒤, 반지름(Primary 50 nm, Secondary 200 nm)과 "
                   "활물질 부피 비율(같은 전체 값을 (1−큰 입자 비율) : 큰 입자 비율로 나눔)만 바꾸고, 배수는 Secondary 쪽에만 줬어요. "
                   "공극률·전도도·두께와 1C 정하는 법은 설계 A와 같아요."),
        "phase_parameter_keys": PHASE_KEYS,
        "var_pts": var_pts_composite(R_SMALL_NM, R_LARGE_NM),
        "summary_table": table_b,
        "only_large_phase_scaled_check": only_large_scaled,
        "implementation_check_same_size_vs_design_A": b_checks,
        "stability_check_porosity_0p30_10C": stability_summary,
        "compare_with_study_e_20pct": vs_e,
        "compare_with_study_e_note_ko": ("e는 봉우리마다 크기가 퍼진 로그정규 분포(상대 표준편차 0.25)이고, f의 설계 B는 크기가 딱 두 가지예요. "
                                         "그래서 같지 않아도 돼요. 참고로만 적었어요."),
        "share_definition_ko": ("되찾은 몫 = (손잡이를 준 섞은 전극 유지율 − 손잡이 없는 섞은 전극 유지율) ÷ "
                                "(50 nm만 쓴 전극 유지율 − 손잡이 없는 섞은 전극 유지율) × 100"),
    }

    # ---- 계산한 숫자로 만든 요약 문장 ----
    findings = []
    dominant = []
    for eps in POROSITIES:
        rows = {t["variant"]: t for t in table_a if t["porosity"] == eps}

        def rr(v, c, rows=rows):
            return rows[v][f"retention_{c:g}C_pct"]

        def sh(v, c, rows=rows):
            return rows[v]["gap_recovered_share_pct"][f"{c:g}C"]
        for c in (5.0, 10.0):
            findings.append(
                f"공극률 {eps:.2f}, {c:g}C: 50 nm {rr('R50', c)}%, 손잡이 없는 200 nm {rr('R200', c)}%. "
                f"확산계수만 ×16 → {rr('D16', c)}%(되찾은 몫 {sh('D16', c)}%), 교환 전류 밀도만 ×4 → {rr('k4', c)}%"
                f"(몫 {sh('k4', c)}%), 둘 다 → {rr('D16k4', c)}%(몫 {sh('D16k4', c)}%).")
        if None not in (sh("D16", 10.0), sh("k4", 10.0)):
            dominant.append(sh("D16", 10.0) >= 90 and abs(sh("k4", 10.0)) <= 10)
        r200, r50, k4 = rows["R200"], rows["R50"], rows["k4"]
        findings.append(
            f"공극률 {eps:.2f}, 10C 방전 끝 리튬 채움 정도(0–1, 1이 가득): 200 nm는 겉면 {r200.get('10C_end_surface_stoichiometry_mean')}, "
            f"입자 평균 {r200.get('10C_end_stoichiometry_mean')} / 50 nm는 겉면 {r50.get('10C_end_surface_stoichiometry_mean')}, "
            f"평균 {r50.get('10C_end_stoichiometry_mean')} / k×4는 겉면 {k4.get('10C_end_surface_stoichiometry_mean')}, "
            f"평균 {k4.get('10C_end_stoichiometry_mean')}. 반쯤 방전했을 때(그 방전 용량의 절반, 보간) 반응 과전압: "
            f"200 nm {r200.get('10C_mid_reaction_overpotential_abs_mV')} mV, "
            f"k×4 {k4.get('10C_mid_reaction_overpotential_abs_mV')} mV, 50 nm {r50.get('10C_mid_reaction_overpotential_abs_mV')} mV.")
    if dominant and all(dominant):
        conclusion = ("이 모델(단순 확산, Prada2013 값)에서는 큰 입자의 5C·10C 손해가 거의 모두 고체 확산 거리(물질 이동) 때문이에요. "
                      "두 공극률 모두 10C에서 확산계수 ×16이 차이의 90% 이상을 되찾았고, 교환 전류 밀도 ×4는 10% 안쪽만 바꿨어요(기준: 이 두 문턱값). "
                      "방전은 큰 입자 겉면이 리튬으로 가득 차서(겉면 채움 정도가 1에 가까움) 전압이 급히 떨어질 때 끝나고, "
                      "그때 200 nm 입자 평균 채움 정도는 낮아요(findings의 숫자). 교환 전류 밀도 ×4는 끝나는 겉면 채움 정도를 거의 바꾸지 않아서 "
                      "남는 용량도 그대로예요. 겉면이 늦게 차려면 리튬이 안쪽으로 빨리 퍼져야 하고, 그건 확산계수가 정해요.")
    else:
        conclusion = ("두 문턱값(확산 ×16이 10C 차이의 90% 이상, 반응 ×4가 10% 이하)을 두 공극률 모두에서 만족하지는 않았어요. "
                      "decomposition_D16_vs_k4의 숫자를 그대로 읽어요.")
    for e in check4:
        m = e["max_abs_retention_difference_pct_points"]
        findings.append(
            f"점검 4, 공극률 {e['porosity']:.2f}, 유지율 차이의 최댓값(모든 방전 속도): D×16·k×4(입자 안 80칸) − 50 nm(20칸) "
            f"{m['D16k4_rp80_vs_R50_rp20']}%p / 입자 안 칸 수를 맞춘 20칸끼리 {m['same_relative_mesh_rp20']}%p, "
            f"80칸끼리 {m['same_relative_mesh_rp80']}%p / 50 nm만 20칸→80칸 {m['mesh_effect_alone_R50']}%p.")
    for eps in POROSITIES:
        r50v = next(t["retention_10C_pct"] for t in table_a if t["porosity"] == eps and t["variant"] == "R50")
        for phi in BLEND_FRACTIONS:
            tb = {t["variant"]: t for t in table_b if t["porosity"] == eps and t["large_fraction"] == phi}
            if not all(tb.get(v, {}).get("retention_10C_pct") is not None for v, _, _ in VARIANTS_B):
                continue
            findings.append(
                f"설계 B, 공극률 {eps:.2f}, 큰 입자 {phi:.0%}, 10C: 손잡이 없음 {tb['base']['retention_10C_pct']}%, "
                f"큰 입자만 D×16 {tb['D16']['retention_10C_pct']}%(몫 {tb['D16']['gap_recovered_share_vs_blend_base_pct']['10C']}%), "
                f"큰 입자만 k×4 {tb['k4']['retention_10C_pct']}%(몫 {tb['k4']['gap_recovered_share_vs_blend_base_pct']['10C']}%). "
                f"50 nm만 쓴 전극은 {r50v}%.")
    if gap_by_xp:
        findings.append("설계 B 안정성(공극률 0.30·10C), 10C 용량 차이(같은 50 nm 두 종류 − 한 종류), 양극 두께 방향 칸 수별: "
                        + ", ".join(f"{k}칸 {v}%" for k, v in gap_by_xp.items()) + ".")

    result = {
        "title": "큰 입자의 고율 손해: 반응 겉넓이(전하 이동)와 고체 확산 거리(물질 이동) 가르기 (PyBaMM DFN 하프셀)",
        "created": datetime.date.today().isoformat(),
        "script": "simulations/pybamm/f_lfp_ct_vs_mass.py",
        "python": sys.version.split()[0], "pybamm": pybamm.__version__, "casadi": casadi.__version__,
        "platform": platform.platform(), "cpu_count": os.cpu_count(), "workers": workers,
        "question_ko": ("시뮬레이션 e에서 큰 입자(200 nm)는 작은 입자(50 nm)보다 고율 성능을 훨씬 많이 잃었어요. "
                        "PyBaMM에서 입자가 커지면 (1) 부피당 반응 겉넓이가 1/R로 줄고(전하 이동), (2) 고체 속 확산 시간이 R²/D로 길어져요(물질 이동). "
                        "10C 손해는 어느 쪽 때문인가요? 코팅·도핑을 흉내 낸 손잡이로 하나씩 되돌려 봐요."),
        "stand_in_assumptions_ko": [
            "'코팅' = 양극 교환 전류 밀도(반응 속도 상수)에 k를 곱한 것으로 가정해요. 실제 탄소 코팅이 반응 속도를 이만큼 올린다는 뜻이 아니에요.",
            "'도핑' = 양극 입자 확산계수에 D를 곱한 것으로 가정해요. 실제 도핑이 확산계수를 이만큼 올린다는 뜻이 아니에요.",
            "실제 코팅은 전자 전도도·겉면 막 저항도 바꾸고, 도핑은 용량·전압도 바꿀 수 있어요. 여기서는 한 값만 바꿔 원인을 가르는 계산 실험이에요.",
        ],
        "knob_targets_ko": {
            "D16": f"확산계수 ×16이면 R²/D가 50 nm와 같아요((200/50)² = 16).",
            "k4": "교환 전류 밀도 ×4이면 전극 부피당 교환 전류(a × 교환 전류 밀도, a = 3 × 활물질 부피 비율 ÷ R)가 50 nm와 같아요(200/50 = 4).",
            "D16k4": "둘 다 주면 식이 50 nm와 같아져요(check4_note_ko).",
        },
        "derived_scales": {
            f"{vid}": {"radius_nm": r_nm, "D_factor": fd, "k_factor": fk,
                       "diffusion_time_R2_over_D_s": round((r_nm * 1e-9) ** 2 / (fd * float(ctx["base"][KEY_D])), 2),
                       "surface_area_relative_to_R50": R_SMALL_NM / r_nm,
                       "exchange_current_per_volume_relative_to_R50": fk * R_SMALL_NM / r_nm}
            for vid, r_nm, fd, fk in VARIANTS_A},
        "derived_scales_note_ko": ("확산 시간 R²/D는 확산계수 5.9e-18 m²/s(Prada2013)로 계산했어요. 10C 방전은 360초, 5C는 720초예요. "
                                   "부피당 교환 전류의 비는 같은 농도(같은 교환 전류 밀도 함수 값)에서 a × k를 50 nm와 견준 값이에요."),
        "cell": "half-cell: LFP working electrode vs lithium metal (coin-cell style), study d와 같음",
        "model": "DFN",
        "model_options_design_A": OPT_SINGLE,
        "parameter_set": "Prada2013 (A123 LFP/graphite 26650), study d와 같음",
        "borrowed_from_Xu2019_added": ctx["added"],
        "borrowed_from_Xu2019_replaced": ctx["replaced"],
        "temperature_K": dstudy.TEMPERATURE_K,
        "voltage_window_V": ctx["v_window"],
        "initial_state": {"x100_positive_stoichiometry": ctx["x100"], "note_ko": "d와 같아요(양극 OCP 3.6 V, 리튬 금속 기준)."},
        "conductivity_S_m": ctx["base_sigma"],
        "base_positive_particle_diffusivity_m2_s": float(ctx["base"][KEY_D]),
        "base_exchange_current_density_function": getattr(ctx["base"][KEY_J0], "__name__", str(ctx["base"][KEY_J0])),
        "design_rule_ko": ("d와 같아요. 두께 80 µm 고정, 활물질 부피 비율 = 0.374 × (1−공극률) ÷ (1−0.426), 기준 전자 전도도. "
                           "같은 공극률이면 입자 크기·손잡이와 상관없이 활물질 양(이론 용량)이 같아요."),
        "c_rate_definition_ko": ("d와 같아요. 설계마다 0.1C 방전 용량으로 1C 전류를 정하고, 0.1C 용량이 0.1% 안으로 같아질 때까지 반복했어요. "
                                 "유지율 = xC 용량 ÷ 0.1C 용량 × 100."),
        "var_pts_ko": ("d의 규칙과 같아요. 두께 방향 x_s 20, x_p 20, 입자 안은 반지름 50 nm마다 20칸(50 nm 20칸, 200 nm 80칸). "
                       "설계 B는 작은 입자 r_p_prim 20칸, 큰 입자 r_p_sec 80칸이에요."),
        "solver": SOLVER_NAMES,
        "solver_note_ko": "d와 같은 순서예요. 기본 IDAKLU가 실패하거나 하한 전압에 닿기 전에 멈추면 다음 계산기로 다시 풀어요.",
        "rates_C": C_RATES, "porosities": POROSITIES,
        "variants_design_A": [{"variant": v, "radius_nm": r, "D_factor": fd, "k_factor": fk} for v, r, fd, fk in VARIANTS_A],
        "share_definition_ko": ("되찾은 몫 = (손잡이를 준 200 nm 유지율 − 손잡이 없는 200 nm 유지율) ÷ (50 nm 유지율 − 손잡이 없는 200 nm 유지율) × 100. "
                                "100%면 50 nm만큼 되찾은 것, 0%면 하나도 못 되찾은 거예요. 같은 공극률·같은 방전 속도끼리 계산하고, "
                                "반올림하지 않은 유지율로 계산했어요."),
        "conclusion_ko": conclusion,
        "findings_ko": findings,
        "summary_table_design_A": table_a,
        "decomposition_D16_vs_k4": decomposition,
        "decomposition_note_ko": ("share_D16은 확산만, share_k4는 반응만, share_D16k4는 둘 다 되돌렸을 때 되찾은 몫이에요. "
                                  "interaction = 둘 다 − (확산만 + 반응만). 0이면 두 효과가 따로 더해지고, 0이 아니면 서로 얽혀 있다는 뜻이에요."),
        "parameter_scaling_proof": scaling_proof,
        "design_B": design_b,
        "checks": {
            "check1_baselines_vs_study_d": check1,
            "check2_mesh_refinement": mesh_check,
            "check2_note_ko": ("10C를 칸 수를 늘려 다시 풀고 본 계산 용량과의 차이(%)를 적었어요. 설계 A는 200 nm 기준·D×16·k×4·D×16·k×4, "
                               "설계 B는 큰 입자 50%의 기준·D×16·k×4를 두 공극률에서 점검했어요. "
                               "'r_p x2'는 입자 안 칸만, '모든 방향 2배'는 두께 방향(x_s, x_p)과 입자 안을 함께 2배로 늘린 거예요."),
            "check3_reached_cutoff": check3,
            "check4_D16k4_vs_R50": check4,
            "check4_note_ko": check4_note,
        },
        "designs": designs,
        "failures": failures,
        "figures": {
            os.path.basename(FIG_RET): "설계 A: 손잡이별 5C·10C 용량 유지율(공극률 0.38·0.30 막대).",
            os.path.basename(FIG_SHARE): "설계 A: 50 nm와 200 nm 사이 유지율 차이 가운데 손잡이가 되찾은 몫(방전 속도별, 공극률별 칸).",
            os.path.basename(FIG_BLEND): "설계 B: 큰 입자 20%·50% 섞은 전극에서 큰 입자에만 손잡이를 줬을 때 5C·10C 유지율.",
        },
        "caveats_ko": [
            "보정하지 않은 모델 값이에요. 실제 LFP 코인셀·제품의 숫자와 바로 비교하면 안 돼요.",
            "코팅·도핑은 한 파라미터에 배수를 곱한 대역이에요(stand_in_assumptions_ko). 실제 공정의 효과 크기가 아니에요.",
            "입자 안 리튬 이동을 단순 확산(상수 확산계수)으로 계산해요. LFP의 두 상(FePO4/LiFePO4) 경계 이동은 다루지 않아요. "
            "'확산이 원인'이라는 결론은 이 단순 확산 가정 안에서만 읽어요.",
            "교환 전류 밀도 함수(Kashkooli2017)와 버틀러–볼머 식은 Prada2013 값 묶음 그대로예요. 겉면 막 저항·이중층은 없어요.",
            "전해질 농도 최솟값이 0보다 아주 조금 작게 나온 것은 계산 오차예요. 0으로 읽어요.",
        ] + [
            f"격자 점검: {m['design_label']} 10C 용량이 {m['check']}에서 {m['difference_pct']}% 달라져요. 이 값은 대략값으로 읽어요."
            for m in mesh_check if abs(m.get("difference_pct", 0)) > 1
        ],
        "timing": {
            "design_runs_wall_seconds": round(t_designs, 1),
            "total_wall_seconds": round(time.perf_counter() - t_start, 1),
            "sum_of_solve_seconds": round(sum(r.get("seconds_incl_build", 0) for _, r in all_runs)
                                          + sum(m.get("seconds_incl_build", 0) for m in mesh_check + stab_out), 1),
            "solve_count_in_results": len(all_runs) + len(mesh_check) + len(stab_out),
            "note_ko": "sum_of_solve_seconds는 결과에 쓰인 계산 한 번 한 번의 시간 합이에요(실패한 시도와 0.1C 반복은 빠짐).",
        },
    }
    os.makedirs(RES, exist_ok=True)
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        cols = ["current_density_mA_cm2", "capacity_mAh_cm2", "retention_pct", "energy_mWh_cm2", "energy_retention_pct",
                "average_voltage_V", "retention_to_2p5V_pct", "min_cathode_electrolyte_conc_mol_m3",
                "end_stoichiometry_mean", "end_surface_stoichiometry_mean",
                "end_stoichiometry_mean_small", "end_stoichiometry_mean_large",
                "mid_reaction_overpotential_abs_mV", "mid_particle_concentration_overpotential_abs_mV",
                "mid_reaction_overpotential_abs_mV_small", "mid_reaction_overpotential_abs_mV_large",
                "mid_electrolyte_overpotential_abs_mV", "end_time_s", "reached_cutoff", "solver", "seconds_incl_build"]
        w.writerow(["design", "variant", "kind", "porosity", "radius_nm", "large_volume_fraction", "radii_nm",
                    "D_factor", "k_factor", "var_pts", "capacity_0p1C_mAh_cm2", "c_rate"] + cols + ["status", "termination"])
        for d in designs:
            for r in d["runs"] or [{"c_rate": "", "status": d.get("status")}]:
                w.writerow([d["design"], d["variant"], d["kind"], d["porosity"], d.get("radius_nm", ""),
                            d.get("large_fraction", ""), ";".join(str(x) for x in d.get("radii_nm", [])),
                            d["D"], d["k"], json.dumps(d["var_pts"]), d.get("capacity_0p1C_mAh_cm2", ""), r["c_rate"]]
                           + [("" if r.get(c) is None else r.get(c)) for c in cols]
                           + [r.get("status", "ok"), r.get("termination", "")])
    print("saved", JSON_PATH)
    print("saved", CSV_PATH)
    print("failures:", len(failures), "| wall seconds:", result["timing"]["total_wall_seconds"])


# ---------------------------------------------------------------- 그림 ----
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
POROSITY_COLOR = {0.38: "#86b6ef", 0.30: "#1c5cab"}      # e와 같은 공극률 색(촘촘할수록 어두워요)
# 손잡이 종류별 색: 확산(D) = 파랑, 반응(k) = 주황, 둘 다 = 보라. 같은 종류 안에서는 배수가 클수록 어두워요.
SHARE_STYLE = {
    "D2": ("#86b6ef", "o", "-"), "D4": ("#2a78d6", "o", "-"), "D16": ("#104281", "o", "-"),
    "k2": ("#f2a07b", "s", "-"), "k4": ("#c2491b", "s", "-"),
    "D4k2": ("#9085e9", "D", "--"), "D16k4": ("#3d2f9e", "D", "--"),
}
VARIANT_ORDER = ["R50", "R200", "D2", "D4", "D16", "k2", "k4", "D4k2", "D16k4"]
BLEND_COLOR = {"base": "#898781", "D16": "#2a78d6", "k4": "#eb6834"}


def variant_label(vid, ko=True):
    names = {"R50": ("50 nm (작은 입자)", "50 nm (small)"), "R200": ("200 nm, 손잡이 없음", "200 nm, no knob"),
             "D2": ("200 nm · D×2", None), "D4": ("200 nm · D×4", None), "D16": ("200 nm · D×16", None),
             "k2": ("200 nm · k×2", None), "k4": ("200 nm · k×4", None),
             "D4k2": ("200 nm · D×4 · k×2", None), "D16k4": ("200 nm · D×16 · k×4", None)}
    ko_name, en_name = names[vid]
    return ko_name if ko or en_name is None else en_name


def make_figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    with open(JSON_PATH, encoding="utf-8") as f:
        res = json.load(f)
    korean = any(fm.name == "Malgun Gothic" for fm in font_manager.fontManager.ttflist)
    plt.rcParams.update({
        "font.family": "Malgun Gothic" if korean else "DejaVu Sans",
        "axes.unicode_minus": False,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "font.size": 10,
    })
    T = (lambda ko, en: ko) if korean else (lambda ko, en: en)
    table = res["summary_table_design_A"]
    note = T("PyBaMM DFN 하프셀(LFP 양극 vs 리튬 금속), Prada2013 값 기반, 양극 두께 80 µm, 기준 전자 전도도, 25 °C. 보정하지 않은 모델 값이에요.\n"
             "D = 양극 입자 확산계수 배수('도핑' 가정), k = 교환 전류 밀도 배수('코팅' 가정). 실제 코팅·도핑의 효과 크기가 아니에요.",
             "PyBaMM DFN half-cell (LFP vs Li metal), Prada2013-based, 80 µm cathode, base conductivity, 25 °C. Uncalibrated.\n"
             "D = particle diffusivity factor ('doping' stand-in), k = exchange-current density factor ('coating' stand-in).")

    def val(eps, vid, key):
        return next((t[key] for t in table if t["porosity"] == eps and t["variant"] == vid), None)

    # 그림 1: 손잡이별 5C·10C 유지율 (가로 막대, 공극률 두 개)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.6), sharey=True)
    n = len(VARIANT_ORDER)
    h = 0.36
    for ax, (key, title) in zip(axes, [("retention_5C_pct", T("5C 용량 유지율 (12분 방전)", "5C capacity retention")),
                                       ("retention_10C_pct", T("10C 용량 유지율 (6분 방전)", "10C capacity retention"))]):
        for j, eps in enumerate(POROSITIES):
            ys = [i + (j - 0.5) * (h + 0.04) for i in range(n)]
            vals = [val(eps, v, key) for v in VARIANT_ORDER]
            ax.barh(ys, [v if v is not None else 0 for v in vals], height=h, color=POROSITY_COLOR[eps],
                    edgecolor=SURFACE, linewidth=1.0,
                    label=T(f"공극률 {eps:.2f}", f"porosity {eps:.2f}"))
            for y, v in zip(ys, vals):
                ax.text((v or 0) + 1.0, y, T("실패", "fail") if v is None else f"{v:.1f}", va="center",
                        fontsize=8, color=INK2)
        for sep in (1.5, 4.5, 6.5):   # 기준 | 확산 | 반응 | 둘 다
            ax.axhline(sep, color=AXIS, linewidth=0.8)
        ax.set_xlim(0, 108)
        ax.set_title(title, fontsize=11, loc="left", color=INK)
        ax.set_xlabel(T("0.1C 대비 유지율 (%)", "Retention vs 0.1C (%)"), fontsize=9)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(n))
    axes[0].set_yticklabels([variant_label(v, korean) for v in VARIANT_ORDER], fontsize=9)
    axes[0].invert_yaxis()
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper left", bbox_to_anchor=(0.2, 0.915), ncol=2, frameon=False, fontsize=9)
    fig.suptitle(T("200 nm 입자에 확산(D)·반응(k) 손잡이를 따로 줬을 때 빠른 방전에서 남는 용량",
                   "200 nm particles with diffusion (D) or reaction (k) knobs: rate capability"),
                 x=0.02, y=0.985, ha="left", fontsize=13, color=INK)
    fig.text(0.02, 0.01, note, fontsize=8, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.07, 1, 0.88))
    fig.savefig(FIG_RET, dpi=150)
    plt.close(fig)

    # 그림 2: 되찾은 몫 (방전 속도별, 공극률별 칸)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharey=True)
    xs = list(range(len(SHARE_RATES)))
    lines_order = ["D2", "D4", "D16", "k2", "k4", "D4k2", "D16k4"]
    all_vals = [val(eps, v, "gap_recovered_share_pct")[f"{c:g}C"] for eps in POROSITIES for v in lines_order
                for c in SHARE_RATES if val(eps, v, "gap_recovered_share_pct")]
    all_vals = [v for v in all_vals if v is not None]
    y_lo = min(-5, min(all_vals) - 5) if all_vals else -5
    y_hi = max(108, max(all_vals) + 5) if all_vals else 108
    for ax, eps in zip(axes, POROSITIES):
        ax.axhline(0, color=AXIS, linewidth=1)
        ax.axhline(100, color=AXIS, linewidth=1, linestyle=":")
        ends = []
        for vid in lines_order:
            color, marker, ls = SHARE_STYLE[vid]
            shares = val(eps, vid, "gap_recovered_share_pct") or {}
            ys = [shares.get(f"{c:g}C") for c in SHARE_RATES]
            pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
            if not pts:
                continue
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, marker=marker, linestyle=ls, linewidth=2,
                    markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.2, label=variant_label(vid, korean))
            ends.append([pts[-1][1], pts[-1][1], vid, color])
        # 오른쪽 끝 이름표: 겹치지 않게 최소 간격을 두고 위아래로 비켜요.
        ends.sort(key=lambda e: e[0])
        gap = (y_hi - y_lo) * 0.045
        for i in range(1, len(ends)):
            ends[i][1] = max(ends[i][1], ends[i - 1][1] + gap)
        for y_true, y_lab, vid, color in ends:
            ax.annotate(f"{variant_label(vid, korean).replace('200 nm · ', '')}  {y_true:.0f}%", (xs[-1], y_true),
                        xytext=(xs[-1] + 0.12, y_lab), textcoords="data", va="center", fontsize=8.5, color=INK2,
                        arrowprops=dict(arrowstyle="-", color=AXIS, linewidth=0.6) if abs(y_lab - y_true) > gap * 0.3 else None)
        gap_txt = []
        for c in SHARE_RATES:
            a, b = val(eps, "R50", f"retention_{c:g}C_pct"), val(eps, "R200", f"retention_{c:g}C_pct")
            gap_txt.append(f"{c:g}C\n" + T(f"(차이 {a - b:.1f}%p)", f"(gap {a - b:.1f} pp)") if a is not None and b is not None else f"{c:g}C")
        ax.set_xticks(xs)
        ax.set_xticklabels(gap_txt, fontsize=9)
        ax.set_xlim(-0.3, len(xs) - 1 + 1.25)
        ax.set_ylim(y_lo, y_hi)
        ax.set_title(T(f"공극률 {eps:.2f}", f"porosity {eps:.2f}"), fontsize=11, loc="left", color=INK)
        ax.set_xlabel(T("방전 속도 (괄호: 50 nm와 200 nm의 유지율 차이)", "C-rate (bracket: 50 nm vs 200 nm retention gap)"),
                      fontsize=9)
    axes[0].set_ylabel(T("50 nm와의 차이 가운데 되찾은 몫 (%)", "Share of the 50-vs-200 nm gap recovered (%)"))
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper left", bbox_to_anchor=(0.06, 0.915), ncol=7, frameon=False, fontsize=8.5)
    fig.text(0.06, 0.845, T("D×16과 D×16·k×4, D×4와 D×4·k×2, k×2와 k×4는 값이 거의 같아서 선이 겹쳐요(오른쪽 이름표에 둘 다 적었어요).",
                            "Some lines overlap (D×16 vs D×16·k×4, D×4 vs D×4·k×2, k×2 vs k×4); both are labelled on the right."),
             fontsize=9, color=INK2, ha="left")
    fig.suptitle(T("200 nm와 50 nm의 유지율 차이 가운데 손잡이가 되찾은 몫",
                   "Share of the 50 nm vs 200 nm retention gap recovered by each knob"),
                 x=0.06, y=0.985, ha="left", fontsize=13, color=INK)
    # 맑은 고딕에는 빼기 기호(U+2212) 글자가 없어서 그림 글자에는 '-'를 써요.
    fig.text(0.06, 0.01, note + T("\n되찾은 몫 = (손잡이 준 200 nm - 손잡이 없는 200 nm) ÷ (50 nm - 손잡이 없는 200 nm) × 100. 점선 100% = 50 nm만큼.",
                                  "\nshare = (variant - 200 nm) / (50 nm - 200 nm) × 100; dotted 100% = as good as 50 nm."),
             fontsize=8, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.1, 1, 0.83))
    fig.savefig(FIG_SHARE, dpi=150)
    plt.close(fig)
    print("saved", FIG_RET)
    print("saved", FIG_SHARE)

    # 그림 3: 설계 B (있을 때만)
    tb = res.get("design_B", {}).get("summary_table", [])
    if tb and any(t.get("retention_10C_pct") is not None for t in tb):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey="row")
        w = 0.26
        for row, c in enumerate((5.0, 10.0)):
            for col, eps in enumerate(POROSITIES):
                ax = axes[row][col]
                for j, (vid, _, _) in enumerate(VARIANTS_B):
                    xs_b = [i + (j - 1) * (w + 0.02) for i in range(len(BLEND_FRACTIONS))]
                    vals = [next((t[f"retention_{c:g}C_pct"] for t in tb if t["porosity"] == eps and t["large_fraction"] == phi
                                  and t["variant"] == vid), None) for phi in BLEND_FRACTIONS]
                    lab = {"base": T("손잡이 없음", "no knob"), "D16": T("큰 입자만 D×16", "large phase D×16"),
                           "k4": T("큰 입자만 k×4", "large phase k×4")}[vid]
                    ax.bar(xs_b, [v or 0 for v in vals], width=w, color=BLEND_COLOR[vid], edgecolor=SURFACE, linewidth=1.0,
                           label=lab)
                    for x, v in zip(xs_b, vals):
                        if v is None:
                            ax.text(x, 1.2, T("실패", "fail"), ha="center", fontsize=8, color=INK2)
                        else:   # 값은 막대 안쪽 위에 적어요(기준 점선과 겹치지 않게)
                            ax.text(x, v - 1.5, f"{v:.1f}", ha="center", va="top", fontsize=8.5, color=SURFACE)
                for vid, ls in (("R50", "--"), ("R200", ":")):
                    rv = val(eps, vid, f"retention_{c:g}C_pct")
                    if rv is not None:
                        ax.axhline(rv, color=INK2, linewidth=1, linestyle=ls)
                        ax.text(len(BLEND_FRACTIONS) + 0.02, rv,
                                T(f"50 nm만\n{rv:.1f}", f"50 nm only\n{rv:.1f}") if vid == "R50"
                                else T(f"200 nm만\n{rv:.1f}", f"200 nm only\n{rv:.1f}"),
                                ha="right", va="center", fontsize=8, color=INK2,
                                bbox=dict(boxstyle="square,pad=0.15", facecolor=SURFACE, edgecolor="none"))
                ax.set_xlim(-0.5, len(BLEND_FRACTIONS) + 0.05)
                ax.set_xticks(range(len(BLEND_FRACTIONS)))
                ax.set_xticklabels([T(f"큰 입자 {100 * phi:.0f}%", f"large {100 * phi:.0f}%") for phi in BLEND_FRACTIONS])
                ax.set_ylim(0, 108)
                ax.grid(axis="x", visible=False)
                ax.set_title(T(f"{c:g}C 용량 유지율 · 공극률 {eps:.2f}", f"{c:g}C retention · porosity {eps:.2f}"),
                             fontsize=11, loc="left", color=INK)
            axes[row][0].set_ylabel(T("0.1C 대비 유지율 (%)", "Retention vs 0.1C (%)"))
        handles, labels_ = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels_, loc="upper left", bbox_to_anchor=(0.06, 0.93), ncol=3, frameon=False, fontsize=9)
        fig.suptitle(T("50 nm와 200 nm를 섞은 전극: 큰 입자에만 확산(D)·반응(k) 손잡이를 줬을 때",
                       "Blended 50/200 nm electrode: knobs on the large phase only"),
                     x=0.06, y=0.985, ha="left", fontsize=13, color=INK)
        fig.text(0.06, 0.01, note + T("\n두 입자 종류(PyBaMM particle phases) 모델. 큰 입자 비율은 부피 비율이에요. 점선: 한 가지 크기만 쓴 설계 A.",
                                      "\nTwo-phase (particle phases) model; fractions by volume. Dashed/dotted: single-size design A."),
                 fontsize=8, color=MUTED, ha="left", va="bottom")
        fig.tight_layout(rect=(0, 0.08, 1, 0.9))
        fig.savefig(FIG_BLEND, dpi=150)
        plt.close(fig)
        print("saved", FIG_BLEND)
    print("labels:", "Korean (Malgun Gothic)" if korean else "English (Malgun Gothic not found)")


if __name__ == "__main__":
    if "--plot-only" not in sys.argv:
        n_workers = 4
        if "--workers" in sys.argv:
            n_workers = int(sys.argv[sys.argv.index("--workers") + 1])
        run_study(n_workers)
    try:
        make_figures()
    except ImportError as err:
        print(f"그림 단계를 건너뛰었어요 ({err}). 프로젝트 .venv로 --plot-only를 실행하면 그림이 만들어져요.")
