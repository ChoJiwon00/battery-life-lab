"""(e) 고밀도 LFP에서 흔한 방법인 '큰 입자와 작은 입자 섞기(입도 배합)'가 빠른 방전(고율)에 어떻게 작용하는지
PyBaMM으로 계산해요.

배경: 특허 심층 리포트(고밀도 LFP와 표면 코팅)에서 입도 배합이 가장 흔한 고밀도화 방법이었어요.
섞는 가장 큰 이득은 작은 입자가 큰 입자 사이 빈틈을 채워 공극률이 낮아지는 것(패킹)이에요.
그런데 PyBaMM은 입자가 쌓이는 방식을 계산하지 않아요. 그래서 두 가지를 나눠서 봐요.
  (1) 같은 공극률에서 작은 입자(반지름 50 nm)를 큰 입자(200 nm)로 바꾸면 고율 성능을 얼마나 잃나?
  (2) 공극률 x 배합 비율 표: "배합으로 공극률을 X까지 낮출 수 있다면(가정이에요, 계산하지 않아요) 고율 성능은?"

- 모델: d와 같은 DFN 하프셀(LFP 작업 전극 vs 리튬 금속), Prada2013 값, 25 °C, 2.0–3.6 V, 기준 전자 전도도.
  d의 상수와 식(활물질 비율, 밀도 추정, 방전 속도)은 d 스크립트에서 그대로 가져와요.
- 입자 크기 분포: PyBaMM의 "particle size": "distribution" 선택지예요. 하프셀("working electrode": "positive")과 함께 쓸 수 있어요.
  반지름 50 nm·200 nm를 중심으로 한 좁은 로그정규 분포 두 개를 '부피(=질량) 비율'로 섞어요.
- PyBaMM은 '면적 기준' 분포를 입력으로 받아요. 공 모양 입자는 면적 기준 분포 = 부피 기준 분포 ÷ 반지름이라
  그렇게 바꿔 넣고, PyBaMM이 거꾸로 계산한 부피 비율이 처음 비율과 같은지 JSON에 점검해 둬요.
- 1C와 유지율은 d와 같아요(설계마다 자기 0.1C 용량으로 1C를 정하고, 유지율 = xC 용량 ÷ 0.1C 용량).
- 보정하지 않은 모델 값이에요. 실제 제품 성능이 아니라 설계끼리 비교한 경향만 봐요.

사용법 (저장소 루트에서):
  계산 + 그림:  C:\\p\\Scripts\\python.exe simulations\\pybamm\\e_lfp_size_mix.py [--workers 4]
  그림만 다시:  .venv\\Scripts\\python.exe simulations\\pybamm\\e_lfp_size_mix.py --plot-only
  (계산은 설계마다 따로 돌아서 여러 프로세스로 나눠 돌려요. --workers 1이면 한 프로세스로 차례대로 돌아요.
   --cache 폴더를 주면 끝난 설계를 그 폴더에 남겨서, 중간에 멈춰도 다시 실행하면 이어서 돌아요.)
"""
import csv
import datetime
import json
import math
import os
import platform
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
sys.path.insert(0, HERE)
import d_lfp_density_rate as dstudy  # noqa: E402  (d의 상수·식을 그대로 써요. 가져올 때 계산은 돌지 않아요.)

JSON_PATH = os.path.join(RES, "e_size_mix.json")
CSV_PATH = os.path.join(RES, "e_size_mix.csv")
FIG_BLEND = os.path.join(RES, "e_retention_vs_large_fraction.png")
FIG_MAP = os.path.join(RES, "e_10C_porosity_blend_map.png")
D_JSON = os.path.join(RES, "d_density_rate.json")

# ---- 실험 설계 ----
POROSITIES = [0.38, 0.34, 0.30, 0.26]
LARGE_FRACTIONS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]     # 큰 입자의 부피(=질량) 비율
C_RATES = dstudy.C_RATES                              # 0.1C, 1C, 2C, 5C, 10C
R_SMALL_NM, R_LARGE_NM = 50, 200                      # 두 봉우리의 부피 기준 중앙 반지름(D50에 해당)
REL_SD = 0.25                                         # 봉우리마다 부피 기준 분포의 상대 표준편차
TRUNC_SIGMAS = 3.0                                    # 분포를 중앙값의 exp(±3σ_ln)에서 잘라요


def sigma_ln(rel_sd):
    """로그정규 분포의 상대 표준편차 -> log 공간 표준편차."""
    return math.sqrt(math.log(1 + rel_sd ** 2))


R_LO_NM = R_SMALL_NM * math.exp(-TRUNC_SIGMAS * sigma_ln(REL_SD))
R_HI_NM = R_LARGE_NM * math.exp(TRUNC_SIGMAS * sigma_ln(REL_SD))
SPLIT_NM = math.sqrt(R_SMALL_NM * R_LARGE_NM)         # 100 nm: 두 봉우리 사이(로그 눈금 가운데)
N_SIZE_BINS = 30                                      # 크기 칸 수(PyBaMM 기본값과 같아요). 로그 간격으로 나눠요.
R_P_POINTS = 80                                       # 입자 안 칸 수: d의 규칙(50 nm마다 20칸)을 200 nm 봉우리에 맞춤
BASE_GRID = {"x_s": 20, "x_p": 20, "r_p": R_P_POINTS, "R_p": N_SIZE_BINS}
OPTIONS = {"working electrode": "positive", "particle size": "distribution"}

# 격자 점검: 가장 힘든 설계들의 10C를 방향마다 칸 수를 2배로 다시 풀어요.
# (0.26, 1.0)은 큰 입자만 있고 전해질도 바닥나는, 10C 유지율이 가장 낮은 설계라 검증 때 더했어요.
GRID_CHECK_DESIGNS = [(0.26, 0.2), (0.26, 0.6), (0.30, 0.4), (0.38, 1.0), (0.26, 1.0)]
GRID_REFINEMENTS = {
    "r_p x2 (입자 안 칸 160)": {"r_p": 2 * R_P_POINTS},
    "R_p x2 (크기 칸 60)": {"R_p": 2 * N_SIZE_BINS},
    "x_s·x_p x2 (두께 방향 40)": {"x_s": 40, "x_p": 40},
    # 위 두께 방향 2배는 전해질이 바닥나는 설계에서 세 계산기 모두 실패해서, 양극 쪽만 늘린 점검을 더했어요.
    "x_p x2 (양극 두께 방향만 40)": {"x_p": 40},
    "x_p x1.5 (양극 두께 방향만 30)": {"x_p": 30},
}
# 구현 점검: 아주 좁은 분포(상대 표준편차 0.02) 한 봉우리로 d의 한 가지 크기 결과를 다시 만들어 봐요.
NARROW_REL_SD = 0.02
NARROW_BINS = 10
IMPLEMENTATION_CHECKS = [(0.30, 50, 20), (0.30, 200, 80)]   # (공극률, 반지름 nm, 입자 안 칸 수)
R_TYP_CHECK = (0.30, 0.4, 100)                              # (공극률, 큰 입자 비율, 기준 반지름 nm)
SOLVER_CHECK_DESIGNS = [(0.38, 0.6), (0.26, 1.0), (0.34, 0.4)]  # 10C를 다른 허용 오차 계산기로 다시 풀어 볼 설계


# ------------------------------------------------------- 분포 도우미 (numpy) ----
def size_grid(r_lo_nm, r_hi_nm, n_bins):
    """크기 칸: 로그 간격 경계(edges), 칸 가운데(nodes), 칸 너비(widths). 단위 m."""
    import numpy as np
    r_lo, r_hi = r_lo_nm * 1e-9, r_hi_nm * 1e-9
    edges = np.exp(np.linspace(math.log(r_lo), math.log(r_hi), n_bins + 1))
    edges[0], edges[-1] = r_lo, r_hi  # PyBaMM은 양 끝 경계가 최소·최대 반지름과 정확히 같은지 확인해요
    return edges, 0.5 * (edges[1:] + edges[:-1]), np.diff(edges)


def lognormal_np(R, median, sig):
    import numpy as np
    return np.exp(-(np.log(R / median)) ** 2 / (2 * sig ** 2)) / (R * sig * math.sqrt(2 * math.pi))


def mode_volume_weights(spec, n_bins):
    """봉우리마다 크기 칸별 부피 비중(합 1). PyBaMM과 같은 칸 가운데·너비로 계산해요."""
    _, nodes, widths = size_grid(spec["r_lo_nm"], spec["r_hi_nm"], n_bins)
    sig = sigma_ln(spec["rel_sd"])
    out = []
    for median_nm, weight in spec["modes"]:
        v = lognormal_np(nodes, median_nm * 1e-9, sig) * widths
        out.append((median_nm, weight, v / v.sum()))
    return nodes, widths, out


def mode_area_fractions(spec, n_bins):
    """봉우리별 반응 표면적 비율. 면적 기준 분포 ∝ 부피 기준 분포 ÷ R라서, 봉우리마다 Σ(칸별 부피 비중 ÷ R)로 계산해요.
    (100 nm로 잘라 센 표면적 비율과 달리, 작은 봉우리의 큰 꼬리와 큰 봉우리의 작은 꼬리를 제 봉우리로 셉니다.)"""
    nodes, _, modes = mode_volume_weights(spec, n_bins)
    parts = {m: w * float((wv / nodes).sum()) for m, w, wv in modes}
    total = sum(parts.values())
    return {m: v / total for m, v in parts.items()}


def make_area_weighted_pdf(spec, n_bins):
    """부피 기준 혼합 분포 f_v -> PyBaMM 입력인 면적 기준 분포 f_a = f_v / R (칸 위에서 합이 1이 되게 다시 맞춤).

    봉우리마다 같은 칸 위에서 먼저 합을 1로 맞춰서, 칸으로 나눈 뒤에도 부피 비율이 정확히 지정값이 되게 해요.
    """
    import pybamm
    _, nodes, widths = size_grid(spec["r_lo_nm"], spec["r_hi_nm"], n_bins)
    sig = sigma_ln(spec["rel_sd"])
    terms = []
    for median_nm, weight in spec["modes"]:
        if weight <= 0:
            continue
        med = median_nm * 1e-9
        z = float((lognormal_np(nodes, med, sig) * widths).sum())
        terms.append((weight / z, med))
    fv_nodes = sum(c * lognormal_np(nodes, med, sig) for c, med in terms)
    k = float((fv_nodes / nodes * widths).sum())   # 면적 기준 분포의 칸 위 합

    def f_a(R):
        fv = sum(c * pybamm.exp(-(pybamm.log(R / med)) ** 2 / (2 * sig ** 2)) / (R * sig * math.sqrt(2 * math.pi))
                 for c, med in terms)
        return fv / R / k

    return f_a, {"input_area_pdf_sum_on_grid": float((fv_nodes / nodes / k * widths).sum()),
                 "input_volume_pdf_sum_on_grid": float((fv_nodes * widths).sum())}


def blend_spec(phi):
    return {"modes": [[R_SMALL_NM, round(1 - phi, 10)], [R_LARGE_NM, phi]], "rel_sd": REL_SD,
            "r_lo_nm": R_LO_NM, "r_hi_nm": R_HI_NM}


def narrow_spec(radius_nm):
    s = sigma_ln(NARROW_REL_SD)
    return {"modes": [[radius_nm, 1.0]], "rel_sd": NARROW_REL_SD,
            "r_lo_nm": radius_nm * math.exp(-TRUNC_SIGMAS * s), "r_hi_nm": radius_nm * math.exp(TRUNC_SIGMAS * s)}


def analytic_means_nm(spec):
    """잘라내지 않은 연속 분포로 계산한 부피 기준 평균 반지름과 면적 기준 평균(=자우터 평균, R32)."""
    g = math.exp(sigma_ln(spec["rel_sd"]) ** 2 / 2)
    rv = sum(w * m * g for m, w in spec["modes"])
    inv = sum(w * g / m for m, w in spec["modes"])
    return rv, 1 / inv


def analytic_fraction_above(spec, split_nm):
    """연속 분포(양 끝을 잘라 다시 1로 맞춤)에서 split_nm보다 큰 입자의 부피 비율."""
    sig = sigma_ln(spec["rel_sd"])

    def cdf(r, m):
        return 0.5 * (1 + math.erf(math.log(r / m) / (sig * math.sqrt(2))))
    total = 0.0
    for m, w in spec["modes"]:
        lo, hi = cdf(spec["r_lo_nm"], m), cdf(spec["r_hi_nm"], m)
        mid = min(max(cdf(split_nm, m), lo), hi)
        total += w * (hi - mid) / (hi - lo)
    return total


# ---------------------------------------------------------- PyBaMM 준비 ----
_CTX = None


def context():
    """d와 같은 하프셀 기본 값 묶음을 프로세스마다 한 번만 만들어요(d의 run_study 1–4단계와 같은 순서)."""
    global _CTX
    if _CTX is not None:
        return _CTX
    os.environ.setdefault("PYBAMM_DISABLE_TELEMETRY", "true")
    import pybamm

    prada = pybamm.ParameterValues("Prada2013")
    xu = pybamm.ParameterValues("Xu2019")

    def describe(v):
        return f"function {getattr(v, '__name__', repr(v))}" if callable(v) else v

    # 1) d와 똑같이: 한 가지 크기 하프셀 모델이 요구하는데 Prada2013에 없는 이름만 Xu2019에서 빌려요.
    base = prada.copy()
    added = {}
    for _ in range(20):
        try:
            pybamm.Simulation(pybamm.lithium_ion.DFN(dstudy.OPTIONS), parameter_values=base.copy()).build()
            break
        except KeyError as err:
            msg = str(err)
            key = next((k for k in xu.keys() if k in msg and k not in base.keys()), None)
            if key is None:
                raise
            base.update({key: xu[key]}, check_already_exists=False)
            added[key] = describe(xu[key])
    # 2) 하프셀의 '음극' 이름은 리튬 금속 박이에요(d와 같음).
    replaced = {}
    for key in ["Negative electrode conductivity [S.m-1]", "Negative electrode thickness [m]"]:
        replaced[key] = {"prada2013_graphite_value": prada[key], "xu2019_lithium_metal_value": xu[key]}
        base[key] = xu[key]
    # 3) 25 °C 등온
    base["Ambient temperature [K]"] = dstudy.TEMPERATURE_K
    base["Initial temperature [K]"] = dstudy.TEMPERATURE_K
    # 4) 만충: 양극 OCP가 3.6 V가 되는 화학량론을 이분법으로 (d와 같음)
    ocp = base["Positive electrode OCP [V]"]
    c_max = base["Maximum concentration in positive electrode [mol.m-3]"]
    v_max = base["Upper voltage cut-off [V]"]
    lo, hi = 1e-9, 0.5
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if float(ocp(mid)) > v_max:
            lo = mid
        else:
            hi = mid
    x100 = 0.5 * (lo + hi)
    base["Initial concentration in positive electrode [mol.m-3]"] = x100 * c_max
    thickness = base["Positive electrode thickness [m]"]
    assert abs(thickness - 80e-6) < 1e-12 and abs(base["Positive electrode porosity"] - dstudy.BASE_POROSITY) < 1e-12
    _CTX = {
        "base": base, "added": added, "replaced": replaced, "x100": x100, "c_max": c_max,
        "thickness": thickness, "area_m2": base["Electrode height [m]"] * base["Electrode width [m]"],
        "base_sigma": base["Positive electrode conductivity [S.m-1]"],
        "r_typ_default": base["Positive particle radius [m]"],
        "v_window": [base["Lower voltage cut-off [V]"], v_max],
    }
    return _CTX


def make_param(eps, spec, n_bins, r_typ_nm=None):
    ctx = context()
    p = ctx["base"].copy()
    p["Positive electrode porosity"] = eps
    p["Positive electrode active material volume fraction"] = dstudy.active_fraction(eps)
    f_a, input_check = make_area_weighted_pdf(spec, n_bins)
    p.update({
        "Positive minimum particle radius [m]": spec["r_lo_nm"] * 1e-9,
        "Positive maximum particle radius [m]": spec["r_hi_nm"] * 1e-9,
        "Positive area-weighted particle-size distribution [m-1]": f_a,
    }, check_already_exists=False)
    if r_typ_nm is not None:
        p["Positive particle radius [m]"] = r_typ_nm * 1e-9
    return p, input_check


# 계산기(solver) 대안 순서. d는 기본 IDAKLU → rtol 1e-8·atol 1e-10 IDAKLU → CasadiSolver(safe)였어요.
# 이 모델은 상태 변수가 약 4만 8천 개라, 그 두 대안이 한 설계의 10C에서 끝나지 않아 뺐어요(first_run_record_ko 참고).
# 대안에는 '멈춤 감지'를 둬요: 연속 NO_PROGRESS_STEPS 걸음 동안 계산 시간이 NO_PROGRESS_S초도 나아가지 않으면 그만둬요.
NO_PROGRESS_STEPS, NO_PROGRESS_S = 500, 0.5
# 그래도 끝없이 도는 경우가 있어서, 계산 한 번을 따로 프로세스로 돌리고 이 시간(초)이 지나면 멈추고 다음 대안으로 넘어가요.
ATTEMPT_TIMEOUT_S = 600
SOLVER_NAMES = ["IDAKLUSolver",
                "IDAKLUSolver(rtol=1e-6,atol=1e-8,guarded)",
                "IDAKLUSolver(rtol=1e-5,atol=1e-7,guarded)"]


def make_solver(name):
    import pybamm
    guard = {"num_steps_no_progress": NO_PROGRESS_STEPS, "t_no_progress": NO_PROGRESS_S}
    return {
        SOLVER_NAMES[0]: lambda: pybamm.IDAKLUSolver(),
        SOLVER_NAMES[1]: lambda: pybamm.IDAKLUSolver(rtol=1e-6, atol=1e-8, options=guard),
        SOLVER_NAMES[2]: lambda: pybamm.IDAKLUSolver(rtol=1e-5, atol=1e-7, options=guard),
    }[name]()


def solve_once(p, current_A, c_rate, spec, grid, solver_name, dist_check=False):
    """한 번 방전. d의 solve_once와 같은 값을 내고, 크기 분포용 값(봉우리별 리튬 채움 정도 등)을 더해요."""
    import numpy as np
    import pybamm
    ctx = context()
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    p = p.copy()
    p["Current function [A]"] = current_A
    model = pybamm.lithium_ion.DFN(OPTIONS)
    edges, nodes, widths = size_grid(spec["r_lo_nm"], spec["r_hi_nm"], grid["R_p"])
    submesh = dict(model.default_submesh_types)
    submesh["positive particle size"] = pybamm.MeshGenerator(pybamm.UserSupplied1DSubMesh,
                                                             submesh_params={"edges": edges})
    sim = pybamm.Simulation(model, parameter_values=p, solver=make_solver(solver_name),
                            var_pts={**model.default_var_pts, **grid}, submesh_types=submesh)
    t0 = time.perf_counter()
    sol = sim.solve([0, 1.5 * 3600 / c_rate])
    sec = time.perf_counter() - t0
    bm = sim.built_model
    n_states = int(bm.concatenated_rhs.shape[0] + bm.concatenated_algebraic.shape[0])
    area_m2, c_max = ctx["area_m2"], ctx["c_max"]
    q = sol["Discharge capacity [A.h]"].entries
    v = sol["Voltage [V]"].entries
    to_cm2 = 1000 / (area_m2 * 1e4)
    energy = float(trapezoid(v, q))
    cut = dstudy.SECOND_CUTOFF_V
    below = np.nonzero(v < cut)[0]
    if len(below) == 0:
        q25, e25 = float(q[-1]), energy
    else:
        i = int(below[0])
        if i == 0:
            q25, e25 = 0.0, 0.0
        else:
            frac = (v[i - 1] - cut) / (v[i - 1] - v[i])
            q25 = float(q[i - 1] + frac * (q[i] - q[i - 1]))
            e25 = float(trapezoid(v[:i], q[:i])) + 0.5 * (v[i - 1] + cut) * (q25 - q[i - 1])
    ce_p = sol["Positive electrolyte concentration [mol.m-3]"].entries
    sto_p = sol["R-averaged positive particle concentration [mol.m-3]"].entries / c_max
    x_p = sol["x_p [m]"].entries[:, -1]
    i_min = int(np.argmin(sto_p[:, -1]))
    x_rel = (x_p - x_p.min()) / (x_p.max() - x_p.min())
    # 크기 칸별 끝 상태 리튬 채움 정도(두께 방향 평균; 두께 방향 칸은 고른 간격이고 활물질 비율도 고르게 같아요)
    sto_dist = sol["R-averaged positive particle stoichiometry distribution"].entries  # (크기 칸, 두께 칸, 시간)
    end_by_size = sto_dist[:, :, -1].mean(axis=1)
    _, _, modes = mode_volume_weights(spec, grid["R_p"])
    mode_end = {}
    for median_nm, weight, wv in modes:
        mode_end[f"end_stoichiometry_mode_{median_nm}nm"] = round(float((wv * end_by_size).sum()), 4) if weight > 0 else None
    fv_pyb = sol["Positive volume-weighted particle-size distribution [m-1]"].entries[:, 0, 0]
    out = {
        "min_cathode_electrolyte_conc_mol_m3": round(float(ce_p.min()), 1),
        "end_electrolyte_conc_cc_side_mol_m3": round(float(ce_p[-1, -1]), 1),
        "end_stoichiometry_separator_side": round(float(sto_p[0, -1]), 4),
        "end_stoichiometry_current_collector_side": round(float(sto_p[-1, -1]), 4),
        "end_stoichiometry_min": round(float(sto_p[i_min, -1]), 4),
        "end_stoichiometry_min_position": round(float(x_rel[i_min]), 3),
        "end_stoichiometry_mean": round(float(sto_p[:, -1].mean()), 4),
        **mode_end,
        "end_stoichiometry_mean_from_size_bins": round(float((fv_pyb * widths * end_by_size).sum()), 4),
        "capacity_Ah": float(q[-1]),
        "capacity_mAh_cm2": float(q[-1]) * to_cm2,
        "energy_mWh_cm2": round(energy * to_cm2, 5),
        "average_voltage_V": round(energy / float(q[-1]), 4),
        "capacity_to_2p5V_mAh_cm2": round(q25 * to_cm2, 5),
        "energy_to_2p5V_mWh_cm2": round(e25 * to_cm2, 5),
        "start_voltage_V": round(float(v[0]), 4),
        "mid_voltage_V": round(float(np.interp(q[-1] / 2, q, v)), 4),
        "end_voltage_V": round(float(v[-1]), 4),
        "end_time_s": round(float(sol.t[-1]), 2),
        "termination": str(sol.termination),
        "reached_cutoff": "Minimum voltage" in str(sol.termination),
        "seconds_incl_build": round(sec, 3),
        "solver": solver_name,
        "n_states": n_states,
    }
    if dist_check:
        out["_dist_check"] = distribution_check(sol, spec, grid["R_p"], nodes, widths)
    return out


def distribution_check(sol, spec, n_bins, nodes, widths):
    """PyBaMM이 입력(면적 기준)에서 거꾸로 계산한 부피 기준 분포가 지정한 부피 비율을 되살리는지 확인해요."""
    import numpy as np
    first = lambda name: sol[name].entries[:, 0, 0]  # noqa: E731  (크기 칸 축, 두께 첫 칸, 처음 시각)
    R = first("Positive particle sizes [m]")
    fv = first("Positive volume-weighted particle-size distribution [m-1]")
    fa = first("Positive area-weighted particle-size distribution [m-1]")
    fn = first("Positive number-based particle-size distribution [m-1]")
    _, _, modes = mode_volume_weights(spec, n_bins)
    basis = np.array([wv / widths for _, _, wv in modes]).T          # 봉우리별 부피 기준 밀도(칸 위 합 1)
    coef, *_ = np.linalg.lstsq(basis, fv, rcond=None)
    fit = basis @ coef
    big = R > SPLIT_NM * 1e-9
    rv_an, r32_an = analytic_means_nm(spec)
    rv_pyb = float(sol["Positive volume-weighted mean particle radius [m]"].entries[0]) * 1e9
    ra_pyb = float(sol["Positive area-weighted mean particle radius [m]"].entries[0]) * 1e9
    a_pyb = float(sol["X-averaged positive electrode surface area to volume ratio [m-1]"].entries[0])
    eps_s = float(sol["X-averaged positive electrode active material volume fraction"].entries[0])
    res = {
        "size_nodes_match_script_grid": bool(np.allclose(R, nodes, rtol=1e-12, atol=0)),
        "volume_pdf_integral": round(float((fv * widths).sum()), 12),
        "area_pdf_integral": round(float((fa * widths).sum()), 12),
        "recovered_mode_volume_fractions_least_squares": {
            f"{m}nm": round(float(c / coef.sum()), 8) for (m, _, _), c in zip(modes, coef)},
        "target_mode_volume_fractions": {f"{m}nm": w for m, w, _ in modes},
        "least_squares_max_relative_residual": float(np.max(np.abs(fv - fit)) / np.max(np.abs(fv))),
        f"volume_fraction_above_{SPLIT_NM:.0f}nm_pybamm_grid": round(float((fv * widths)[big].sum()), 6),
        f"volume_fraction_above_{SPLIT_NM:.0f}nm_continuous_target": round(analytic_fraction_above(spec, SPLIT_NM), 6),
        f"surface_area_fraction_above_{SPLIT_NM:.0f}nm": round(float((fa * widths)[big].sum()), 6),
        f"number_fraction_above_{SPLIT_NM:.0f}nm": float(f"{float((fn * widths)[big].sum()):.4g}"),
        "volume_weighted_mean_radius_nm_pybamm": round(rv_pyb, 3),
        "volume_weighted_mean_radius_nm_continuous": round(rv_an, 3),
        "area_weighted_mean_radius_nm_pybamm_R32": round(ra_pyb, 3),
        "area_weighted_mean_radius_nm_continuous_R32": round(r32_an, 3),
        "surface_area_to_volume_ratio_m-1": float(f"{a_pyb:.6g}"),
        "surface_area_to_volume_ratio_check_3eps_over_R32": float(f"{3 * eps_s / (ra_pyb * 1e-9):.6g}"),
    }
    return res


def _attempt_entry(conn, args):
    """따로 띄운 프로세스 안에서 한 번 풀어요(값 묶음은 여기서 다시 만들어요. 분포 함수는 프로세스 사이로 넘길 수 없어서예요)."""
    try:
        os.environ.setdefault("PYBAMM_DISABLE_TELEMETRY", "true")
        eps, spec, grid, r_typ_nm, current_A, c_rate, solver_name, dist_check = args
        p, _ = make_param(eps, spec, grid["R_p"], r_typ_nm)
        conn.send(("ok", solve_once(p, current_A, c_rate, spec, grid, solver_name, dist_check=dist_check)))
    except Exception as err:  # noqa: BLE001
        conn.send(("err", f"{type(err).__name__}: {str(err)[:300]}"))
    finally:
        conn.close()


def solve_attempt(args, timeout_s):
    """계산기 한 번을 따로 프로세스에서 돌리고, timeout_s초 안에 안 끝나면 그 프로세스를 끝내요.
    IDAKLU가 아주 작은 걸음으로 끝없이 도는 경우가 있어서(first_run_record_ko) 계산기 바깥에서 시간을 재요."""
    import multiprocessing as mp
    mpc = mp.get_context("spawn")
    recv, send = mpc.Pipe(duplex=False)
    proc = mpc.Process(target=_attempt_entry, args=(send, args))
    proc.start()
    send.close()
    try:
        if recv.poll(timeout_s):
            kind, val = recv.recv()
        else:
            kind, val = "timeout", f"{timeout_s}초 안에 끝나지 않아 멈춤"
    except EOFError:
        kind, val = "err", f"계산 프로세스가 결과 없이 끝남 (exit code {proc.exitcode})"
    if proc.is_alive():
        proc.terminate()
    proc.join()
    recv.close()
    return kind, val


def solve_chain(eps, spec, grid, r_typ_nm, current_A, c_rate, dist_check=False):
    attempts = []
    for name in SOLVER_NAMES:
        kind, val = solve_attempt((eps, spec, grid, r_typ_nm, current_A, c_rate, name, dist_check), ATTEMPT_TIMEOUT_S)
        if kind != "ok":
            attempts.append(f"{name}: {val}")
            continue
        if not val["reached_cutoff"]:
            attempts.append(f"{name}: 하한 전압에 닿기 전에 멈춤 ({val['termination']})")
            continue
        val["earlier_attempts"] = attempts
        return val, attempts
    return None, attempts


# ------------------------------------------------------------- 설계 하나 ----
def run_design(task):
    """0.1C로 1C를 정한 뒤(0.1% 안으로 맞을 때까지 반복) 1C·2C·5C·10C를 풀어요. d의 run_design과 같은 순서예요."""
    ctx = context()
    eps, spec, grid = task["porosity"], task["spec"], task["grid"]
    _, input_check = make_area_weighted_pdf(spec, grid["R_p"])
    act = dstudy.active_fraction(eps)
    area_m2, thickness, x100, c_max = ctx["area_m2"], ctx["thickness"], ctx["x100"], ctx["c_max"]
    q_theo = act * thickness * c_max * dstudy.F / 3600 * (1 - x100) / 10
    label = task["label"]
    failures = []
    d = {
        "study": task["study"],
        "label": label,
        "porosity": eps,
        "large_fraction": task.get("large_fraction"),
        "distribution": spec,
        "active_volume_fraction": round(act, 5),
        "inactive_volume_fraction_binder_carbon": round((1 - eps) - act, 5),
        "conductivity_S_m": ctx["base_sigma"],
        "thickness_um": thickness * 1e6,
        "theoretical_capacity_mAh_cm2": round(q_theo, 4),
        "var_pts": grid,
        "estimate_density_g_cm3": round(dstudy.density_estimate(eps), 4),
        "estimate_lfp_loading_mg_cm2": round(act * thickness * 100 * dstudy.RHO_LFP * 1000, 3),
        "input_distribution_check": input_check,
        "runs": [],
    }
    if "compare_d_radius_nm" in task:
        d["compare_d_radius_nm"] = task["compare_d_radius_nm"]

    def solve_safe(current, c, dist_check=False):
        out, attempts = solve_chain(eps, spec, grid, task.get("r_typ_nm"), current, c, dist_check=dist_check)
        if out is None:
            failures.append({"design": label, "c_rate": c, "attempts": attempts})
        return out

    q_ref = q_theo
    iterations = []
    low = None
    q_used = q_ref
    for k in range(4):
        current = 0.1 * q_ref / 1000 * 1e4 * area_m2
        low = solve_safe(current, 0.1, dist_check=(k == 0))
        if low is None:
            break
        if "_dist_check" in low:
            d["distribution_check"] = low.pop("_dist_check")
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
    low.pop("_dist_check", None)
    one_c = q_ref
    d["n_states"] = low["n_states"]
    d["one_C_definition_mAh_cm2"] = round(one_c, 5)
    d["one_C_current_density_mA_cm2"] = round(one_c, 5)
    d["low_rate_run_used_1C_basis_mAh_cm2"] = round(q_used, 5)
    q01 = low["capacity_mAh_cm2"]
    d["capacity_0p1C_mAh_cm2"] = round(q01, 5)
    d["volumetric_capacity_0p1C_mAh_cm3_coating"] = round(q01 / (thickness * 100), 2)
    d["utilization_0p1C_pct_of_theoretical"] = round(100 * q01 / q_theo, 2)
    e01, q25_01 = low["energy_mWh_cm2"], low["capacity_to_2p5V_mAh_cm2"]
    d["energy_0p1C_mWh_cm2"] = e01
    d["capacity_to_2p5V_0p1C_mAh_cm2"] = q25_01
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
          + " ".join(f"{r['c_rate']}C {r.get('retention_pct', 'FAIL')}%" for r in d["runs"])
          + f" | states {d['n_states']} | solve {sum(r.get('seconds_incl_build', 0) for r in d['runs']):.0f}s",
          flush=True)
    return d, failures


def run_single(task):
    """10C 한 번만 다시 풀기(격자 점검, 기준 반지름 점검)."""
    ctx = context()
    current = task["c_rate"] * task["one_C_mAh_cm2"] / 1000 * 1e4 * ctx["area_m2"]
    if task.get("force_solver"):
        name = task["force_solver"]
        kind, val = solve_attempt((task["porosity"], task["spec"], task["grid"], task.get("r_typ_nm"), current,
                                   task["c_rate"], name, False), ATTEMPT_TIMEOUT_S)
        if kind != "ok":
            out, attempts = None, [f"{name}: {val}"]
        elif not val["reached_cutoff"]:
            out, attempts = None, [f"{name}: 하한 전압에 닿기 전에 멈춤 ({val['termination']})"]
        else:
            out, attempts = val, []
    else:
        out, attempts = solve_chain(task["porosity"], task["spec"], task["grid"], task.get("r_typ_nm"),
                                    current, task["c_rate"])
    res = {k: task[k] for k in ("kind", "check", "design", "c_rate", "grid", "reference_solver", "force_solver") if k in task}
    if task.get("r_typ_nm") is not None:
        res["r_typ_nm"] = task["r_typ_nm"]
    if out is None:
        res["error"] = attempts
        return res
    ref = task["reference_capacity_mAh_cm2"]
    res.update({
        "used_mAh_cm2": ref,
        "check_mAh_cm2": round(out["capacity_mAh_cm2"], 5),
        "difference_pct": round(100 * (out["capacity_mAh_cm2"] - ref) / ref, 3),
        "retention_difference_pct_points": round(100 * (out["capacity_mAh_cm2"] - ref) / task["capacity_0p1C_mAh_cm2"], 3),
        "solver": out["solver"], "n_states": out["n_states"], "seconds_incl_build": out["seconds_incl_build"],
    })
    print(f"check {res['check']} | {res['design']}: {res['difference_pct']}% ({out['n_states']} states, "
          f"{out['seconds_incl_build']:.0f}s)", flush=True)
    return res


def _cached_call(args):
    """--cache 폴더가 있으면 끝난 계산을 파일로 남기고, 다시 돌릴 때 같은 계산은 파일에서 읽어요(긴 계산이 중간에 멈춰도 이어서)."""
    fn_name, task, cache_dir = args
    fn = {"run_design": run_design, "run_single": run_single}[fn_name]
    if not cache_dir:
        return fn(task)
    import hashlib
    key = hashlib.sha1(json.dumps({"fn": fn_name, "task": task, "solvers": SOLVER_NAMES},
                                  sort_keys=True).encode()).hexdigest()[:16]
    path = os.path.join(cache_dir, f"{fn_name}_{key}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            out = json.load(f)
        (out[0] if fn_name == "run_design" else out)["loaded_from_cache"] = True
        return out
    out = fn(task)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    os.replace(path + ".tmp", path)
    return out


def run_tasks(fn_name, tasks, workers, cache_dir=None):
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    args = [(fn_name, t, cache_dir) for t in tasks]
    if workers <= 1:
        return [_cached_call(a) for a in args]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_cached_call, args))


# ---------------------------------------------------------------- 계산 ----
def run_study(workers, cache_dir=None):
    os.environ.setdefault("PYBAMM_DISABLE_TELEMETRY", "true")
    import casadi
    import pybamm

    t_start = time.perf_counter()
    ctx = context()
    tasks = []
    for eps in POROSITIES:
        for phi in LARGE_FRACTIONS:
            tasks.append({"study": "main", "porosity": eps, "large_fraction": phi, "spec": blend_spec(phi),
                          "grid": dict(BASE_GRID), "label": f"main eps={eps} large={phi:.0%}"})
    for eps, r_nm, rp in IMPLEMENTATION_CHECKS:
        tasks.append({"study": "implementation_check", "porosity": eps, "spec": narrow_spec(r_nm),
                      "grid": {"x_s": 20, "x_p": 20, "r_p": rp, "R_p": NARROW_BINS}, "compare_d_radius_nm": r_nm,
                      "label": f"narrow {r_nm}nm eps={eps}"})
    out = run_tasks("run_design", tasks, workers, cache_dir)
    designs = [d for d, _ in out]
    failures = [f for _, fs in out for f in fs]
    main = {(d["porosity"], d["large_fraction"]): d for d in designs if d["study"] == "main"}

    def run10(d):
        return next((r for r in d["runs"] if r["c_rate"] == 10.0 and "capacity_mAh_cm2" in r), None)

    # 격자 점검 + 기준 반지름(R_typ) 점검
    checks, skipped = [], []
    for eps, phi in GRID_CHECK_DESIGNS:
        d = main.get((eps, phi))
        r = run10(d) if d and "one_C_definition_mAh_cm2" in d else None
        if r is None:
            skipped.append(f"eps={eps} large={phi}: 본 계산 10C가 없어 점검을 건너뛰었어요.")
            continue
        for name, change in GRID_REFINEMENTS.items():
            checks.append({"kind": "grid", "check": name, "design": d["label"], "porosity": eps, "spec": d["distribution"],
                           "grid": {**BASE_GRID, **change}, "c_rate": 10.0,
                           "one_C_mAh_cm2": d["one_C_definition_mAh_cm2"],
                           "reference_capacity_mAh_cm2": r["capacity_mAh_cm2"],
                           "capacity_0p1C_mAh_cm2": d["capacity_0p1C_mAh_cm2"]})
    eps, phi, rtyp = R_TYP_CHECK
    d = main.get((eps, phi))
    if d and run10(d):
        checks.append({"kind": "r_typ", "check": f"기준 반지름 R_typ {rtyp} nm", "design": d["label"], "porosity": eps,
                       "spec": d["distribution"], "grid": dict(BASE_GRID), "c_rate": 10.0, "r_typ_nm": rtyp,
                       "one_C_mAh_cm2": d["one_C_definition_mAh_cm2"],
                       "reference_capacity_mAh_cm2": run10(d)["capacity_mAh_cm2"],
                       "capacity_0p1C_mAh_cm2": d["capacity_0p1C_mAh_cm2"]})
    # 계산기 점검: 본 계산 10C를 다른 허용 오차의 계산기로 다시 풀어, 계산기에 따라 숫자가 얼마나 달라지는지 봐요.
    for eps, phi in SOLVER_CHECK_DESIGNS:
        d = main.get((eps, phi))
        r = run10(d) if d else None
        if r is None:
            skipped.append(f"계산기 점검 eps={eps} large={phi}: 본 계산 10C가 없어 건너뛰었어요.")
            continue
        other = SOLVER_NAMES[1] if r["solver"] == SOLVER_NAMES[0] else SOLVER_NAMES[0]
        checks.append({"kind": "solver", "check": f"계산기 {other}", "design": d["label"], "porosity": eps,
                       "spec": d["distribution"], "grid": dict(BASE_GRID), "c_rate": 10.0, "force_solver": other,
                       "reference_solver": r["solver"], "one_C_mAh_cm2": d["one_C_definition_mAh_cm2"],
                       "reference_capacity_mAh_cm2": r["capacity_mAh_cm2"],
                       "capacity_0p1C_mAh_cm2": d["capacity_0p1C_mAh_cm2"]})
    check_out = run_tasks("run_single", checks, workers, cache_dir)
    grid_check = [c for c in check_out if c.get("kind") == "grid"]
    r_typ_check = [c for c in check_out if c.get("kind") == "r_typ"]
    solver_check = [c for c in check_out if c.get("kind") == "solver"]

    # d와 맞춰 보기
    with open(D_JSON, encoding="utf-8") as f:
        dres = json.load(f)

    def d_design(study, eps, r_nm):
        return next((x for x in dres["designs"] if x["study"] == study and x["porosity"] == eps
                     and x["conductivity_factor"] == 1.0 and x["particle_radius_nm"] == r_nm), None)

    def compare(e_d, d_d):
        if e_d is None or d_d is None or "capacity_0p1C_mAh_cm2" not in e_d:
            return {"status": "비교할 결과가 없어요"}
        er = {r["c_rate"]: r for r in e_d["runs"]}
        dr = {r["c_rate"]: r for r in d_d["runs"]}
        rows = []
        for c in C_RATES[1:]:
            a, b = er.get(c, {}), dr.get(c, {})
            if "retention_pct" not in a or "retention_pct" not in b:
                rows.append({"c_rate": c, "status": "한쪽 결과 없음"})
                continue
            rows.append({"c_rate": c, "study_d_retention_pct": b["retention_pct"], "study_e_retention_pct": a["retention_pct"],
                         "difference_pct_points": round(a["retention_pct"] - b["retention_pct"], 2),
                         "study_d_energy_retention_pct": b["energy_retention_pct"],
                         "study_e_energy_retention_pct": a["energy_retention_pct"],
                         "energy_difference_pct_points": round(a["energy_retention_pct"] - b["energy_retention_pct"], 2)})
        qd, qe = d_d["capacity_0p1C_mAh_cm2"], e_d["capacity_0p1C_mAh_cm2"]
        return {"study_d_design": f"{d_d['study']} eps={d_d['porosity']} R={d_d['particle_radius_nm']}nm",
                "study_e_design": e_d["label"],
                "capacity_0p1C_mAh_cm2": {"study_d": qd, "study_e": qe, "difference_pct": round(100 * (qe - qd) / qd, 3)},
                "rates": rows}

    consistency = {
        "blend_0pct_vs_d_50nm": [compare(main.get((eps, 0.0)), d_design("main", eps, 50)) for eps in POROSITIES],
        "blend_100pct_vs_d_200nm": [compare(main.get((eps, 1.0)), d_design("particle_radius", eps, 200))
                                    for eps in POROSITIES if d_design("particle_radius", eps, 200)],
        "narrow_distribution_vs_d": [
            compare(next((x for x in designs if x["study"] == "implementation_check" and x["porosity"] == eps
                          and x["compare_d_radius_nm"] == r_nm), None),
                    d_design("main" if r_nm == 50 else "particle_radius", eps, r_nm))
            for eps, r_nm, _ in IMPLEMENTATION_CHECKS],
    }

    # 요약 표와 행렬
    def rr(d, c, key):
        return next((r.get(key) for r in d["runs"] if r["c_rate"] == c), None) if d else None

    table = []
    for eps in POROSITIES:
        for phi in LARGE_FRACTIONS:
            d = main.get((eps, phi))
            dc = (d or {}).get("distribution_check", {})
            table.append({
                "porosity": eps, "large_fraction": phi,
                "estimate_density_g_cm3": d["estimate_density_g_cm3"] if d else None,
                "capacity_0p1C_mAh_cm2": d.get("capacity_0p1C_mAh_cm2") if d else None,
                "retention_1C_pct": rr(d, 1.0, "retention_pct"), "retention_2C_pct": rr(d, 2.0, "retention_pct"),
                "retention_5C_pct": rr(d, 5.0, "retention_pct"), "retention_10C_pct": rr(d, 10.0, "retention_pct"),
                "capacity_5C_mAh_cm2": rr(d, 5.0, "capacity_mAh_cm2"), "capacity_10C_mAh_cm2": rr(d, 10.0, "capacity_mAh_cm2"),
                "energy_retention_5C_pct": rr(d, 5.0, "energy_retention_pct"),
                "energy_retention_10C_pct": rr(d, 10.0, "energy_retention_pct"),
                "average_voltage_10C_V": rr(d, 10.0, "average_voltage_V"),
                "min_cathode_electrolyte_conc_10C_mol_m3": rr(d, 10.0, "min_cathode_electrolyte_conc_mol_m3"),
                "end_stoichiometry_mode_50nm_10C": rr(d, 10.0, "end_stoichiometry_mode_50nm"),
                "end_stoichiometry_mode_200nm_10C": rr(d, 10.0, "end_stoichiometry_mode_200nm"),
                # 큰 봉우리(200 nm)가 가진 반응 표면적 비율. 예전에는 100 nm보다 큰 입자의 비율을 이 이름으로 적었어요.
                "surface_area_fraction_large": (round(mode_area_fractions(d["distribution"], d["var_pts"]["R_p"])[R_LARGE_NM], 6)
                                                if d else None),
                "surface_area_fraction_above_100nm": dc.get(f"surface_area_fraction_above_{SPLIT_NM:.0f}nm"),
                "area_weighted_mean_radius_nm_R32": dc.get("area_weighted_mean_radius_nm_pybamm_R32"),
            })

    def matrix(key):
        return [[next(t[key] for t in table if t["porosity"] == eps and t["large_fraction"] == phi)
                 for phi in LARGE_FRACTIONS] for eps in POROSITIES]

    matrices = {"rows_porosity": POROSITIES, "columns_large_fraction": LARGE_FRACTIONS}
    for key in ["capacity_0p1C_mAh_cm2", "retention_5C_pct", "retention_10C_pct", "capacity_5C_mAh_cm2",
                "capacity_10C_mAh_cm2", "energy_retention_10C_pct", "average_voltage_10C_V"]:
        matrices[key] = matrix(key)
    same_porosity_cost = []
    for eps in POROSITIES:
        ref = main.get((eps, 0.0))
        for phi in LARGE_FRACTIONS:
            d = main.get((eps, phi))
            row = {"porosity": eps, "large_fraction": phi}
            for c in (5.0, 10.0):
                a, b = rr(d, c, "retention_pct"), rr(ref, c, "retention_pct")
                row[f"retention_{c:g}C_pct"] = a
                row[f"change_vs_all_small_{c:g}C_pct_points"] = round(a - b, 2) if a is not None and b is not None else None
            same_porosity_cost.append(row)
    conversion_summary = []
    for phi in LARGE_FRACTIONS:
        d = main.get((0.30, phi))
        if d and "distribution_check" in d:
            conversion_summary.append({"large_fraction": phi, "from_design": d["label"], **d["distribution_check"]})

    sig = sigma_ln(REL_SD)
    result = {
        "title": "고밀도 LFP 양극: 큰 입자·작은 입자 섞기(입도 배합)와 율속 특성 (PyBaMM DFN 하프셀, 입자 크기 분포)",
        "created": datetime.date.today().isoformat(),
        "script": "simulations/pybamm/e_lfp_size_mix.py",
        "python": sys.version.split()[0], "pybamm": pybamm.__version__, "casadi": casadi.__version__,
        "platform": platform.platform(), "cpu_count": os.cpu_count(), "workers": workers,
        "question_ko": ("고밀도 LFP에서 흔한 방법인 큰 입자와 작은 입자 섞기(입도 배합)는 고율 성능에 어떻게 작용하나요? "
                        "(1) 같은 공극률에서 작은 입자를 큰 입자로 바꾸면 고율에서 얼마나 잃는지, "
                        "(2) 배합으로 공극률을 낮출 수 있다고 '가정'하면 어느 공극률에서 어떤 고율 성능이 되는지 봐요."),
        "background_ko": ("특허 심층 리포트(patent-radar reports/deep/high-density-lfp-2026-10.md)에서 입도 배합이 가장 흔한 고밀도화 방법이었어요. "
                          "배합의 주된 이득은 작은 입자가 큰 입자 사이 빈틈을 채워 공극률을 낮추는 것(패킹)인데, PyBaMM은 패킹을 계산하지 않아요. "
                          "그래서 공극률은 입력값으로 따로 바꾸고, 배합이 공극률을 얼마나 낮추는지는 이 계산에 들어 있지 않아요."),
        "packing_not_modelled_ko": ("공극률 x 배합 표에서 같은 배합 열을 따라 아래(더 낮은 공극률)로 내려가는 것은 "
                                    "'배합으로 그만큼 잘 채워졌다면'이라는 가정이에요. 어느 배합이 어느 공극률을 만드는지는 실험이나 패킹 모델로 따로 정해야 해요."),
        "cell": "half-cell: LFP working electrode vs lithium metal (coin-cell style), study d와 같음",
        "model": "DFN",
        "model_options": OPTIONS,
        "model_support_note_ko": ("PyBaMM 26.8에서 DFN은 하프셀(working electrode: positive)과 입자 크기 분포(particle size: distribution)를 함께 지원해요. "
                                  "그래서 가장 가까운 다른 모델을 찾을 필요가 없었어요."),
        "solver": SOLVER_NAMES,
        "solver_note_ko": ("기본 IDAKLU(d와 같음)로 풀고, 실패하거나 하한 전압에 닿기 전에 멈추면 허용 오차를 줄인 IDAKLU 두 가지"
                           "(rtol 1e-6·atol 1e-8, 그다음 rtol 1e-5·atol 1e-7)로 차례로 다시 풀어요. "
                           f"두 대안에는 '멈춤 감지'(연속 {NO_PROGRESS_STEPS}걸음 동안 계산 시간이 {NO_PROGRESS_S}초도 나아가지 않으면 그만둠)를 뒀어요. "
                           f"또 계산 한 번을 따로 프로세스로 돌려서, {ATTEMPT_TIMEOUT_S}초 안에 끝나지 않으면 멈추고 다음 대안으로 넘어가요. "
                           "d의 대안이던 rtol 1e-8·atol 1e-10 IDAKLU와 CasadiSolver(safe)는 뺐어요(first_run_record_ko). "
                           "설계마다 어느 계산기로 풀었는지와 앞선 실패는 runs[].solver, runs[].earlier_attempts, "
                           "runs_with_earlier_solver_failures에 남겼고, 계산기에 따른 차이는 solver_check에서 확인했어요."),
        "first_run_record_ko": (
            "이 결과를 얻기 전에 두 번 실행을 멈췄어요. 1차(d와 같은 대안 순서: 기본 IDAKLU → rtol 1e-8·atol 1e-10 IDAKLU → CasadiSolver, 프로세스 5개)에서는 "
            "26개 설계 중 25개가 끝났지만 공극률 0.30·큰 입자 0% 설계가 15분 넘게 끝나지 않았어요(그 프로세스 메모리 약 5.8 GB). "
            "1차 로그에는 주로 방전 끝 무렵의 IDA 수렴 실패 메시지가 35번 남았고, 끝난 25개 설계는 두 번째 계산기로 풀렸어요. "
            "2차(두 번째 대안에 걸음 수 상한 20000, CasadiSolver 대신 rtol 1e-6 IDAKLU)도 같은 설계에서 멈췄어요(메모리 약 12 GB). "
            "따로 풀어 보니 이 설계의 10C(1C 기준 2.21385 mA/cm²)에서 기본 IDAKLU는 수렴에 실패했고, rtol 1e-8·atol 1e-10 IDAKLU는 "
            "걸음 수 상한을 둬도 10분 가까이 끝나지 않았어요(1C 기준을 소수 다섯째 자리로 반올림한 값으로 넣으면 64초에 끝나기도 해서, "
            "이 계산기는 이 조건에서 아주 작은 차이에도 민감했어요). "
            "같은 조건에서 rtol 1e-6·atol 1e-8 IDAKLU는 836걸음, 47.7초에 하한 전압까지 풀렸어요. "
            "3차(지금의 대안 순서, 멈춤 감지만 있고 시간 제한은 없음)에서는 26개 중 25개가 끝났지만 공극률 0.26·큰 입자 20% 설계가 "
            "다시 끝나지 않았어요(15분 넘게, 메모리 약 8.5 GB). 멈춤 감지는 대안 계산기에만 있고, 따로 풀어 본 2차의 경우에도 "
            "멈춤 감지(연속 2000걸음에 0.01초)가 작동하지 않았어요. 그래서 계산 한 번마다 따로 프로세스로 돌리고 "
            f"{ATTEMPT_TIMEOUT_S}초 시간 제한을 두었어요. 4차(지금 코드)에서는 3차에서 끝난 25개 설계를 같은 대안 순서로 끝난 결과 그대로 "
            "캐시에서 다시 쓰고(cache_reuse_check_ko), 남은 설계 1개와 점검 계산을 새로 풀었어요. 이때 그 설계의 1C 기본 IDAKLU가 "
            f"{ATTEMPT_TIMEOUT_S}초 시간 제한에 걸려 다음 대안으로 풀렸어요(runs_with_earlier_solver_failures). "
            "그 뒤 양극 두께 방향 격자 점검(grid_check_note_ko)을 더해 다시 실행했어요. "
            "마지막으로 검증 때 공극률 0.26·큰 입자 100% 설계를 격자 점검에 더해 그 점검만 새로 풀었어요. "
            "이 파일을 만든 마지막 실행은 모두 캐시에서 읽어서 total_wall_seconds가 짧아요(timing_note_ko)."),
        "parameter_set": "Prada2013 (A123 LFP/graphite 26650), study d와 같음",
        "borrowed_from_Xu2019_added": ctx["added"],
        "borrowed_from_Xu2019_replaced": ctx["replaced"],
        "temperature_K": dstudy.TEMPERATURE_K,
        "voltage_window_V": ctx["v_window"],
        "initial_state": {"x100_positive_stoichiometry": ctx["x100"],
                          "note_ko": "d와 같아요. 모든 크기의 입자가 같은 리튬 농도(만충, OCP 3.6 V)에서 시작해요."},
        "conductivity_S_m": ctx["base_sigma"],
        "conductivity_note_ko": "기준 전자 전도도(Prada2013, d의 ×1)만 썼어요.",
        "design_rule_ko": ("d와 같아요. 두께 80 µm 고정, 활물질 부피 비율 = 0.374 × (1−공극률) ÷ (1−0.426). "
                           "같은 공극률이면 배합 비율과 상관없이 활물질 양(이론 용량)이 같아요."),
        "c_rate_definition_ko": ("d와 같아요. 설계마다 0.1C 방전 용량으로 1C 전류를 정하고, 0.1C 용량이 0.1% 안으로 같아질 때까지 반복했어요. "
                                 "유지율 = xC 용량 ÷ 0.1C 용량 × 100."),
        "rates_C": C_RATES, "porosities": POROSITIES, "large_volume_fractions": LARGE_FRACTIONS,
        "particle_size_distribution": {
            "definition_ko": ("두 봉우리 로그정규 분포를 부피(=질량, 같은 LFP라 밀도가 같아요) 비율로 섞었어요. "
                              f"봉우리마다 부피 기준 중앙 반지름(D50에 해당)이 {R_SMALL_NM} nm·{R_LARGE_NM} nm이고, "
                              f"상대 표준편차는 {REL_SD}(log 공간 표준편차 {sig:.4f})예요. "
                              f"분포는 작은 봉우리 중앙값×exp(−{TRUNC_SIGMAS:g}σ)부터 큰 봉우리 중앙값×exp(+{TRUNC_SIGMAS:g}σ)까지로 잘랐어요 "
                              f"({R_LO_NM:.2f}에서 {R_HI_NM:.2f} nm). 잘린 꼬리는 봉우리마다 약 0.13%라 다시 1로 맞췄어요."),
            "small_mode_median_radius_nm": R_SMALL_NM, "large_mode_median_radius_nm": R_LARGE_NM,
            "relative_standard_deviation": REL_SD, "sigma_ln": sig,
            "radius_range_nm": [R_LO_NM, R_HI_NM], "split_radius_for_checks_nm": SPLIT_NM,
            "size_bins": N_SIZE_BINS, "size_bin_spacing": "log-uniform (UserSupplied1DSubMesh)",
            "conversion_ko": ("PyBaMM 입력은 면적 기준 분포 f_a예요. 공 모양 입자는 f_a ∝ f_v ÷ R(f_v는 부피 기준 분포)이라, "
                              "봉우리마다 크기 칸 위에서 합을 1로 맞춘 f_v를 비율대로 더하고 R로 나눈 뒤 칸 위 합이 1이 되게 다시 맞췄어요. "
                              "PyBaMM은 안에서 f_v = R·f_a ÷ ∫R·f_a로 거꾸로 계산해요. 그 결과를 두 봉우리 모양으로 최소제곱 분해해 "
                              "되살린 부피 비율을 distribution_conversion_check에 적었어요."),
            "surface_area_note_ko": ("PyBaMM은 비표면적을 3 × 활물질 부피 비율 ÷ (면적 기준 평균 반지름 R32)으로 계산해요. "
                                     "그래서 큰 입자를 섞으면 반응 면적이 줄어요. 입자 크기마다 같은 전해질·전위에서 따로 반응하고, 따로 확산해요."),
        },
        "var_pts": BASE_GRID,
        "var_pts_note_ko": (f"두께 방향은 d와 같은 x_s 20, x_p 20이에요. 입자 안 칸 수(r_p)는 크기 칸 모두에 같은 수가 쓰여서 "
                            f"(PyBaMM은 입자마다 자기 반지름을 r_p칸으로 나눠요), d의 규칙(50 nm마다 20칸)을 큰 봉우리 중앙 200 nm에 맞춘 {R_P_POINTS}칸으로 했어요. "
                            f"그보다 큰 꼬리 입자는 칸이 조금 성기고, 이 영향은 grid_check의 r_p x2로 확인했어요. "
                            f"크기 칸은 {N_SIZE_BINS}칸(PyBaMM 기본값과 같은 수)을 로그 간격으로 나눴어요."),
        "design_changes_ko": [
            ("크기 칸을 PyBaMM 기본(고른 간격) 대신 로그 간격으로 나눴어요. 50 nm와 200 nm 봉우리가 같은 상대 해상도"
             f"(칸 하나마다 반지름이 약 {100 * ((R_HI_NM / R_LO_NM) ** (1 / N_SIZE_BINS) - 1):.1f}%씩 커져요)를 갖게 하려고요. "
             "고른 간격이면 작은 봉우리가 칸 몇 개로만 표현돼요."),
            f"분포를 중앙값의 exp(±{TRUNC_SIGMAS:g}σ)에서 잘랐어요. 더 넓게 자르면 가장 큰 꼬리 입자 때문에 입자 안 칸 수를 크게 늘려야 해서예요.",
            "봉우리를 '평균'이 아니라 '부피 기준 중앙값(D50에 해당)'으로 정했어요. 분말 사양서가 보통 D50으로 크기를 적기 때문이에요.",
            "d와 맞춰 보는 점검에 더해, 아주 좁은 분포(상대 표준편차 0.02)로 d의 한 가지 크기 결과를 다시 만드는 구현 점검을 더했어요. "
            "0%·100% 배합과 d의 차이 중 얼마가 분포 폭 때문인지 가르려고요.",
            "기준 반지름(Positive particle radius, 입자 안 좌표의 눈금으로만 쓰여요)을 50 nm에서 100 nm로 바꿔도 결과가 같은지 점검했어요.",
            "그림 2에 10C 유지율 지도 옆에 10C 면적당 용량(mAh/cm²) 지도를 더했어요. 공극률이 낮으면 0.1C 용량 자체가 커서, "
            "'배합으로 공극률을 낮춘다면'을 읽으려면 유지율과 함께 실제로 남는 용량을 봐야 해서예요.",
            f"계산을 설계마다 따로 여러 프로세스(workers={workers})로 나눠 돌렸어요. 그래서 runs[].seconds_incl_build는 한 프로세스로 돌릴 때보다 조금 길 수 있어요.",
            "계산기 대안 순서를 d와 다르게 바꿨어요(solver_note_ko, first_run_record_ko).",
        ],
        "summary_table": table,
        "matrices": matrices,
        "same_porosity_cost": same_porosity_cost,
        "distribution_conversion_check": conversion_summary,
        "consistency_vs_study_d": consistency,
        "consistency_note_ko": ("0% 배합(작은 봉우리만)과 d의 50 nm 한 가지 크기, 100% 배합과 d의 200 nm(d는 공극률 0.426·0.30에서만 계산)를 맞춰 봤어요. "
                                "분포 폭(상대 표준편차 0.25) 때문에 같지는 않아요. 아주 좁은 분포로 다시 푼 narrow_distribution_vs_d가 d와 거의 같으면, "
                                "차이는 PyBaMM 구현이 아니라 분포 폭에서 온 거예요. "
                                "100% 배합이 d의 200 nm보다 1C·2C에서는 낮고 5C·10C에서는 높은 것은 용량이 반지름에 따라 줄어드는 모양 때문이에요. "
                                "느린 방전에서는 200 nm보다 큰 꼬리 입자가 잃는 용량이 작은 쪽 입자가 더 내는 용량보다 크고, "
                                "빠른 방전에서는 큰 입자가 이미 거의 못 쓰여서 작은 쪽 입자가 더 내는 용량이 더 커요. "
                                "0% 배합(50 nm 봉우리)은 10C에서도 아직 느린 방전 쪽 모양이라, 모든 방전 속도에서 d의 50 nm보다 조금 낮아요."),
        "grid_check": grid_check,
        "grid_check_note_ko": (f"가장 힘든 설계 {len(GRID_CHECK_DESIGNS)}개의 10C를 방향마다 칸 수를 늘려 다시 풀고, 본 계산 용량과의 차이(%)를 적었어요. "
                               "실패한 점검은 error에 계산기별 실패 이유를 그대로 남겼어요. 분리막·양극 두께 방향을 함께 2배로 늘린 점검은 "
                               "전해질이 바닥나는 설계 일부에서 세 계산기 모두 실패해서, 양극 쪽만 2배·1.5배로 늘린 점검을 더했어요. "
                               "공극률 0.26·큰 입자 100% 설계(10C 유지율이 가장 낮음)는 검증 때 더한 설계예요."),
        "grid_check_skipped": skipped,
        "r_typ_check": r_typ_check,
        "solver_check": solver_check,
        "cache_reuse_check_ko": {
            "designs_loaded_from_cache": [d["label"] for d in designs if d.get("loaded_from_cache")],
            "checks_loaded_from_cache": [c["design"] + " / " + c["check"] for c in check_out if c.get("loaded_from_cache")],
            "max_successful_solve_seconds_in_cached_designs": max(
                [r.get("seconds_incl_build", 0) for d in designs if d.get("loaded_from_cache") for r in d["runs"]] or [0]),
            "attempt_timeout_s": ATTEMPT_TIMEOUT_S,
            "note_ko": ("캐시에서 다시 쓴 설계 가운데 일부는 시간 제한을 넣기 전에 같은 계산기 순서로 끝난 결과예요(first_run_record_ko). "
                        "그 안의 성공한 계산이 모두 시간 제한보다 짧게 끝났다면(위 최댓값), 시간 제한을 두고 다시 풀어도 같은 계산기가 "
                        "같은 결과를 내요. 실패한 시도는 시간 제한이 있어도 실패라서 다음 대안으로 넘어가는 순서가 같아요."),
        },
        "solver_usage": {name: sum(1 for d in designs for r in d["runs"] if r.get("solver") == name) for name in SOLVER_NAMES},
        "runs_with_earlier_solver_failures": [
            {"design": d["label"], "c_rate": r["c_rate"], "solver": r["solver"], "earlier_attempts": r["earlier_attempts"]}
            for d in designs for r in d["runs"] if r.get("earlier_attempts")],
        "designs": designs,
        "failures": failures,
        "figures": {
            os.path.basename(FIG_BLEND): ("큰 입자 부피 비율에 따른 5C·10C 용량 유지율(공극률별 선). "
                                          "점이 겹치지 않게 공극률별로 가로 위치를 ±1.8%p 안에서 조금씩 비켜 그렸어요."),
            os.path.basename(FIG_MAP): ("공극률 x 큰 입자 비율의 10C 용량 유지율 지도와 10C 면적당 용량 지도. "
                                        "배합이 공극률을 얼마나 낮추는지는 계산하지 않았어요(가정으로 읽어요)."),
        },
        "caveats_ko": [
            "보정하지 않은 모델 값이에요. 실제 LFP 코인셀·제품의 숫자와 바로 비교하면 안 돼요.",
            "패킹(작은 입자가 빈틈을 채우는 효과)은 계산하지 않았어요. 공극률은 입력값이고, 브루그만 지수 1.5도 모든 배합에 같게 썼어요. "
            "실제로는 배합에 따라 기공 모양과 구불구불한 정도도 달라져요.",
            "입자 안 리튬 이동을 단순 확산으로 계산해요. LFP의 두 상 반응은 다루지 않아요. 큰 입자의 손해는 이 가정에 특히 크게 기대요(d와 같음).",
            "크기마다 같은 확산계수·반응 속도 상수를 썼어요. 실제 큰 입자와 작은 입자는 탄소 코팅·결함이 달라 값이 다를 수 있어요.",
            "같은 두께 방향 위치에서는 큰 입자와 작은 입자가 고르게 섞여 있다고 가정해요(층이 갈라지는 분리는 없음).",
            "전해질 농도 최솟값이 0보다 아주 조금 작게 나온 것은 계산 오차예요. 0으로 읽어요.",
        ] + [
            f"격자 점검: {m['design']} 10C 용량이 {m['check']}에서 {m['difference_pct']}% 달라져요. 이 값은 대략값으로 읽어요."
            for m in grid_check if abs(m.get("difference_pct", 0)) > 1
        ],
        "total_wall_seconds": round(time.perf_counter() - t_start, 1),
        "timing_note_ko": ("total_wall_seconds는 마지막 실행 한 번의 시간이에요. 캐시에서 다시 쓴 설계가 있으면 그만큼 짧아요. "
                           "sum_of_solve_seconds는 캐시 것까지 포함한, 결과에 쓰인 계산 한 번 한 번의 시간 합이에요(실패한 시도와 0.1C 반복은 빠짐)."),
        "sum_of_solve_seconds": round(sum(r.get("seconds_incl_build", 0) for d in designs for r in d["runs"])
                                      + sum(c.get("seconds_incl_build", 0) for c in check_out), 1),
        "solve_count": sum(len(d.get("c_rate_1C_iterations_mAh_cm2", [])) - 1 + len(d["runs"]) for d in designs) + len(check_out),
    }
    os.makedirs(RES, exist_ok=True)
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        cols = ["current_density_mA_cm2", "capacity_mAh_cm2", "retention_pct", "energy_mWh_cm2",
                "energy_retention_pct", "average_voltage_V", "capacity_to_2p5V_mAh_cm2", "retention_to_2p5V_pct",
                "min_cathode_electrolyte_conc_mol_m3", "end_stoichiometry_separator_side",
                "end_stoichiometry_current_collector_side", "end_stoichiometry_mean",
                "end_stoichiometry_mode_50nm", "end_stoichiometry_mode_200nm", "seconds_incl_build", "solver", "n_states"]
        w.writerow(["study", "porosity", "large_volume_fraction", "distribution_modes_nm_weight", "rel_sd",
                    "active_volume_fraction", "estimate_density_g_cm3", "conductivity_S_m", "c_rate"]
                   + cols + ["status", "termination"])
        for d in designs:
            modes = ";".join(f"{m}:{wt:g}" for m, wt in d["distribution"]["modes"])
            for r in d["runs"]:
                w.writerow([d["study"], d["porosity"], d["large_fraction"] if d["large_fraction"] is not None else "",
                            modes, d["distribution"]["rel_sd"], d["active_volume_fraction"], d["estimate_density_g_cm3"],
                            round(d["conductivity_S_m"], 5), r["c_rate"]]
                           + [("" if r.get(c) is None else r.get(c)) for c in cols]
                           + [r.get("status", "ok"), r.get("termination", "")])
    print("saved", JSON_PATH)
    print("saved", CSV_PATH)
    print("failures:", len(failures), "| wall seconds:", result["total_wall_seconds"],
          "| solves:", result["solve_count"])


# ---------------------------------------------------------------- 그림 ----
# 공극률은 크기 순서가 있는 값이라 파란색 한 가지를 밝음→어두움으로 써요(어두울수록 촘촘해요). 모양도 함께 바꿔요.
POROSITY_STYLE = {0.38: ("#86b6ef", "o"), 0.34: ("#3987e5", "s"), 0.30: ("#1c5cab", "^"), 0.26: ("#0d366b", "D")}
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def make_figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.colors import LinearSegmentedColormap

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
    table = res["summary_table"]
    dens = {t["porosity"]: t["estimate_density_g_cm3"] for t in table}
    note = T("PyBaMM DFN 하프셀(LFP 양극 vs 리튬 금속), Prada2013 값 기반, 양극 두께 80 µm, 기준 전자 전도도, 25 °C. 보정하지 않은 모델 값이에요.\n"
             f"입자: 부피 기준 중앙 반지름 {R_SMALL_NM} nm·{R_LARGE_NM} nm 로그정규 분포(상대 표준편차 {REL_SD})를 부피 비율로 섞음. "
             "괄호 안 밀도는 LFP 3.6·비활물질 1.9 g/cm³를 가정한 추정값이에요.",
             "PyBaMM DFN half-cell (LFP vs Li metal), Prada2013-based, 80 µm cathode, base conductivity, 25 °C. Uncalibrated.\n"
             f"Particles: lognormal modes at volume-median radius {R_SMALL_NM}/{R_LARGE_NM} nm (rel. SD {REL_SD}), mixed by volume.")

    def val(eps, phi, key):
        return next((t[key] for t in table if t["porosity"] == eps and t["large_fraction"] == phi), None)

    # 그림 1: 큰 입자 비율에 따른 5C·10C 유지율
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharey=True)
    xs = [100 * f for f in LARGE_FRACTIONS]
    # 공극률별 선이 거의 겹치는 곳(특히 5C)에서도 점이 보이게, 가로로 아주 조금씩(±1.8%p 안) 비켜 그려요.
    dodge = {eps: (k - (len(POROSITIES) - 1) / 2) * 1.2 for k, eps in enumerate(POROSITIES)}
    for ax, (key, title) in zip(axes, [("retention_5C_pct", T("5C 용량 유지율 (12분 방전)", "5C capacity retention")),
                                       ("retention_10C_pct", T("10C 용량 유지율 (6분 방전)", "10C capacity retention"))]):
        for eps in POROSITIES:
            color, marker = POROSITY_STYLE[eps]
            pts = [(x + dodge[eps], val(eps, phi, key)) for x, phi in zip(xs, LARGE_FRACTIONS)
                   if val(eps, phi, key) is not None]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, marker=marker, markersize=7, linewidth=2,
                    markeredgecolor=SURFACE, markeredgewidth=1.2,
                    label=T(f"공극률 {eps:.2f} ({dens[eps]:.2f} g/cm³)", f"porosity {eps:.2f} ({dens[eps]:.2f} g/cm³)"))
        ax.set_title(title, fontsize=11, loc="left", color=INK)
        if key == "retention_5C_pct":
            ax.text(0.02, 0.04, T("5C에서는 네 공극률의 선이 거의 겹쳐요. 점이 보이게 가로로 조금씩 비켜 그렸어요.",
                                  "At 5C the four porosity lines nearly overlap; points are nudged sideways."),
                    transform=ax.transAxes, fontsize=8.5, color=INK2, ha="left", va="bottom")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{x:.0f}%" for x in xs])
        ax.set_xlabel(T(f"큰 입자({R_LARGE_NM} nm) 부피 비율  (나머지는 {R_SMALL_NM} nm)",
                        f"Volume fraction of large ({R_LARGE_NM} nm) particles"), fontsize=9)
    axes[0].set_ylabel(T("0.1C 대비 유지율 (%)", "Retention vs 0.1C (%)"))
    axes[0].set_ylim(0, 100)
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper left", bbox_to_anchor=(0.06, 0.915), ncol=4, frameon=False, fontsize=9)
    fig.suptitle(T("같은 공극률에서 작은 입자를 큰 입자로 바꿀 때 빠른 방전에서 남는 용량",
                   "Rate capability vs large-particle fraction at fixed porosity"),
                 x=0.06, y=0.985, ha="left", fontsize=13, color=INK)
    fig.text(0.06, 0.01, note, fontsize=8, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.08, 1, 0.88))
    fig.savefig(FIG_BLEND, dpi=150)
    plt.close(fig)

    # 그림 2: 공극률 x 배합 지도 (10C 유지율, 10C 면적당 용량)
    cmap = LinearSegmentedColormap.from_list("blues", BLUE_RAMP)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    panels = [("retention_10C_pct", T("10C 용량 유지율 (%)", "10C capacity retention (%)"), "{:.1f}"),
              ("capacity_10C_mAh_cm2", T("10C 면적당 용량 (mAh/cm²)", "10C areal capacity (mAh/cm²)"), "{:.2f}")]
    for ax, (key, title, fmt) in zip(axes, panels):
        grid = [[val(eps, phi, key) for phi in LARGE_FRACTIONS] for eps in POROSITIES]
        flat = [v for row in grid for v in row if v is not None]
        vmin, vmax = min(flat), max(flat)
        data = [[float("nan") if v is None else v for v in row] for row in grid]
        ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        for i, row in enumerate(grid):
            for j, v in enumerate(row):
                if v is None:
                    ax.text(j, i, T("실패", "fail"), ha="center", va="center", fontsize=9, color=INK2)
                    continue
                dark = (v - vmin) / (vmax - vmin + 1e-12) > 0.55
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=9.5,
                        color=SURFACE if dark else INK)
        ax.set_xticks(range(len(LARGE_FRACTIONS)))
        ax.set_xticklabels([f"{100 * f:.0f}%" for f in LARGE_FRACTIONS])
        ax.set_yticks(range(len(POROSITIES)))
        ax.set_yticklabels([f"{e:.2f}\n({dens[e]:.2f})" for e in POROSITIES], fontsize=9)
        ax.set_xlabel(T(f"큰 입자({R_LARGE_NM} nm) 부피 비율", f"Large ({R_LARGE_NM} nm) volume fraction"), fontsize=9)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, fontsize=11, loc="left", color=INK)
        ax.tick_params(length=0)
    axes[0].set_ylabel(T("양극 공극률 (괄호: 추정 밀도 g/cm³)", "Porosity (bracket: est. density g/cm³)"),
                       fontsize=9, labelpad=8)
    fig.suptitle(T("공극률 x 큰 입자 비율: 10C에서 남는 용량", "Porosity x blend: what remains at 10C"),
                 x=0.06, y=0.985, ha="left", fontsize=13, color=INK)
    fig.text(0.06, 0.915, T("아래 줄로 갈수록 촘촘해요. 배합이 공극률을 얼마나 낮추는지는 계산하지 않았어요. 같은 열에서 아래로 내려가는 것은 "
                            "'배합으로 그만큼 촘촘해졌다면'이라는 가정이에요.",
                            "Packing is not modelled: moving down a column assumes the blend reached that porosity."),
             fontsize=9, color=INK2, ha="left")
    fig.text(0.06, 0.01, note, fontsize=8, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.08, 1, 0.9))
    fig.savefig(FIG_MAP, dpi=150)
    plt.close(fig)
    print("saved", FIG_BLEND)
    print("saved", FIG_MAP)
    print("labels:", "Korean (Malgun Gothic)" if korean else "English (Malgun Gothic not found)")


if __name__ == "__main__":
    if "--plot-only" not in sys.argv:
        n_workers = 4
        if "--workers" in sys.argv:
            n_workers = int(sys.argv[sys.argv.index("--workers") + 1])
        cache = sys.argv[sys.argv.index("--cache") + 1] if "--cache" in sys.argv else None
        run_study(n_workers, cache)
    try:
        make_figures()
    except ImportError as err:
        print(f"그림 단계를 건너뛰었어요 ({err}). 프로젝트 .venv로 --plot-only를 실행하면 그림이 만들어져요.")
