"""(d) 고밀도 LFP 양극의 맞바꿈: 전극을 더 촘촘하게 누르면(공극률을 낮추면) 같은 두께에 활물질이 더 들어가요.
대신 빠른 방전(율속 특성)에서 손해를 볼 수 있어요. 이 손해가 양극의 전자 전도도
(탄소 코팅·도전재가 만드는 전자 길)에 따라 얼마나 달라지는지 PyBaMM으로 계산해요.

- 모델: DFN 하프셀(LFP 작업 전극 vs 리튬 금속). 코인셀 시험과 같은 구성이에요.
- 값: Prada2013(A123 LFP/흑연 26650 셀)의 양극·전해질·분리막 값을 쓰고,
  리튬 금속 쪽에 필요한 값만 Xu2019에서 빌려요. 빌린 이름은 결과 JSON에 모두 적어요.
- 양극 두께 80 µm는 고정하고 공극률·전자 전도도·입자 크기만 바꿔요.
- 1C는 설계마다 자기 0.1C 방전 용량으로 정해요. 그래서 "유지율 = xC 용량 / 0.1C 용량"으로
  설계끼리 공정하게 비교할 수 있어요(LFP 논문의 10C/0.1C 비교와 같은 방식).
- 보정(실측에 맞추기)을 하지 않은 모델 값이에요. 실제 제품 성능으로 읽으면 안 돼요.

사용법 (저장소 루트에서):
  계산 + 그림:  C:\\p\\Scripts\\python.exe simulations\\pybamm\\d_lfp_density_rate.py
  그림만 다시:  .venv\\Scripts\\python.exe simulations\\pybamm\\d_lfp_density_rate.py --plot-only
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
JSON_PATH = os.path.join(RES, "d_density_rate.json")
CSV_PATH = os.path.join(RES, "d_density_rate.csv")
FIG_RATE = os.path.join(RES, "d_retention_vs_porosity.png")
FIG_TRADE = os.path.join(RES, "d_capacity_vs_10C_retention.png")

# ---- 실험 설계 ----
BASE_POROSITY = 0.426
BASE_ACTIVE = 0.374
POROSITIES = [0.426, 0.38, 0.34, 0.30, 0.26]
COND_FACTORS = [0.1, 1.0, 10.0]          # 기준 전도도(0.338 S/m)의 0.1배·1배·10배
RADII_NM = [50, 100, 200]                # 입자 반지름 곁가지 실험 (50 nm가 Prada2013 기준)
RADIUS_POROSITIES = [0.426, 0.30]
C_RATES = [0.1, 1.0, 2.0, 5.0, 10.0]
TEMPERATURE_K = 298.15                   # 25 °C, 등온
OPTIONS = {"working electrode": "positive"}
# 계산 칸(격자): 분리막·양극 두께 방향은 PyBaMM 기본값(20칸)을 써요. 입자 안(r_p)은 반지름 50 nm마다 20칸으로
# 늘려요(50 nm 20칸, 100 nm 40칸, 200 nm 80칸). 큰 입자를 20칸으로 풀면 10C 유지율이 약 1%p 높게 나왔어요.
BASE_VAR_PTS = {"x_s": 20, "x_p": 20, "r_p": 20}


def var_pts_for(radius_nm):
    return {**BASE_VAR_PTS, "r_p": max(20, int(round(20 * radius_nm / 50)))}


def refined_var_pts(var_pts):
    """격자 점검용: 모든 방향의 칸 수를 2배로."""
    return {k: 2 * v for k, v in var_pts.items()}
# 기본 하한 전압은 Prada2013의 2.0 V예요. LFP 코인셀은 2.5 V에서 멈추는 경우도 많아서,
# 같은 방전 곡선에서 2.5 V에 처음 닿는 지점까지의 용량·에너지도 함께 읽어 둬요(추가 계산 없음).
SECOND_CUTOFF_V = 2.5
# 전극 밀도 "추정"에만 쓰는 가정 (모델 계산에는 쓰지 않아요)
RHO_LFP = 3.6       # g/cm3
RHO_INACTIVE = 1.9  # g/cm3, 바인더+탄소 평균으로 가정
F = 96485.33212


def active_fraction(eps):
    """바인더+탄소(비활물질)와 활물질의 비율을 기준 설계와 같게 유지해요."""
    return BASE_ACTIVE * (1 - eps) / (1 - BASE_POROSITY)


def density_estimate(eps):
    act = active_fraction(eps)
    inact = (1 - eps) - act
    return act * RHO_LFP + inact * RHO_INACTIVE


# ---------------------------------------------------------------- 계산 ----
def run_study():
    os.environ.setdefault("PYBAMM_DISABLE_TELEMETRY", "true")
    import numpy as np
    import pybamm

    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    t_start = time.perf_counter()
    prada = pybamm.ParameterValues("Prada2013")
    xu = pybamm.ParameterValues("Xu2019")

    def describe(v):
        return f"function {getattr(v, '__name__', repr(v))}" if callable(v) else v

    # 1) 하프셀 모델이 요구하는데 Prada2013에 없는 이름만 찾아서 Xu2019에서 빌려요.
    base = prada.copy()
    added = {}
    for _ in range(20):
        try:
            pybamm.Simulation(pybamm.lithium_ion.DFN(OPTIONS), parameter_values=base.copy()).build()
            break
        except KeyError as err:
            msg = str(err)
            key = next((k for k in xu.keys() if k in msg and k not in base.keys()), None)
            if key is None:
                raise
            base.update({key: xu[key]}, check_already_exists=False)
            added[key] = describe(xu[key])
    # 2) 하프셀에서는 "Negative electrode ..." 이름이 흑연이 아니라 리튬 금속 박을 뜻해요.
    #    전압 손실(i*L/σ)에만 쓰이고 영향은 아주 작지만, 흑연 값을 리튬 금속 값으로 바꿔 둬요.
    replaced = {}
    for key in ["Negative electrode conductivity [S.m-1]", "Negative electrode thickness [m]"]:
        replaced[key] = {"prada2013_graphite_value": prada[key], "xu2019_lithium_metal_value": xu[key]}
        base[key] = xu[key]
    # 3) 25 °C 등온
    base["Ambient temperature [K]"] = TEMPERATURE_K
    base["Initial temperature [K]"] = TEMPERATURE_K

    v_max = base["Upper voltage cut-off [V]"]
    v_min = base["Lower voltage cut-off [V]"]
    ocp = base["Positive electrode OCP [V]"]
    c_max = base["Maximum concentration in positive electrode [mol.m-3]"]
    thickness = base["Positive electrode thickness [m]"]
    base_sigma = base["Positive electrode conductivity [S.m-1]"]
    base_radius = base["Positive particle radius [m]"]
    area_m2 = base["Electrode height [m]"] * base["Electrode width [m]"]
    assert abs(thickness - 80e-6) < 1e-12 and abs(base["Positive electrode porosity"] - BASE_POROSITY) < 1e-12

    # 4) 만충 상태: 양극 OCP가 상한 전압(3.6 V)이 되는 화학량론(x100)을 이분법으로 찾아요.
    #    PyBaMM의 하프셀 SOC 계산은 OCP가 하한 2.0 V에 닿는 x0도 찾아야 하는데,
    #    이 OCP 함수는 완전 리튬화(x=1)에서도 약 2.49 V라 x0가 없어서 직접 계산해요.
    lo, hi = 1e-9, 0.5
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if float(ocp(mid)) > v_max:
            lo = mid
        else:
            hi = mid
    x100 = 0.5 * (lo + hi)
    base["Initial concentration in positive electrode [mol.m-3]"] = x100 * c_max

    def make_param(eps, sigma, radius):
        p = base.copy()
        p["Positive electrode porosity"] = eps
        p["Positive electrode active material volume fraction"] = active_fraction(eps)
        p["Positive electrode conductivity [S.m-1]"] = sigma
        p["Positive particle radius [m]"] = radius
        return p

    # 전해질 농도가 0에 가까워지는 조건에서는 계산기(solver)가 아주 작은 차이로 실패하기도 해요.
    # 그래서 기본 IDAKLU → 허용 오차를 줄인 IDAKLU → CasadiSolver 순서로 다시 시도해요.
    solver_factories = {
        "IDAKLUSolver": lambda: pybamm.IDAKLUSolver(),
        "IDAKLUSolver(rtol=1e-8,atol=1e-10)": lambda: pybamm.IDAKLUSolver(rtol=1e-8, atol=1e-10),
        "CasadiSolver(safe)": lambda: pybamm.CasadiSolver(mode="safe"),
    }

    def solve_once(p, current_A, c_rate, var_pts=None, solver_name="IDAKLUSolver"):
        p = p.copy()
        p["Current function [A]"] = current_A
        model = pybamm.lithium_ion.DFN(OPTIONS)
        solver = solver_factories[solver_name]()
        kw = {"var_pts": {**model.default_var_pts, **var_pts}} if var_pts else {}
        sim = pybamm.Simulation(model, parameter_values=p, solver=solver, **kw)
        t0 = time.perf_counter()
        sol = sim.solve([0, 1.5 * 3600 / c_rate])
        sec = time.perf_counter() - t0
        q = sol["Discharge capacity [A.h]"].entries
        v = sol["Voltage [V]"].entries
        to_cm2 = 1000 / (area_m2 * 1e4)  # Ah -> mAh/cm2 (Wh -> mWh/cm2)
        energy = float(trapezoid(v, q))
        # 같은 방전 곡선에서 2.5 V에 처음 닿는 지점까지의 용량·에너지 (2.5 V 하한으로 시험한 것과 같아요)
        below = np.nonzero(v < SECOND_CUTOFF_V)[0]
        if len(below) == 0:
            q25, e25 = float(q[-1]), energy
        else:
            i = int(below[0])
            if i == 0:
                q25, e25 = 0.0, 0.0
            else:
                frac = (v[i - 1] - SECOND_CUTOFF_V) / (v[i - 1] - v[i])
                q25 = float(q[i - 1] + frac * (q[i] - q[i - 1]))
                e25 = float(trapezoid(v[:i], q[:i])) + 0.5 * (v[i - 1] + SECOND_CUTOFF_V) * (q25 - q[i - 1])
        # 왜 멈췄는지 보기: 양극 안 전해질 농도와, 분리막 쪽·집전체 쪽 입자에 리튬이 얼마나 찼는지
        ce_p = sol["Positive electrolyte concentration [mol.m-3]"].entries
        sto_p = sol["R-averaged positive particle concentration [mol.m-3]"].entries / c_max
        # 끝 상태의 리튬 찬 정도가 가장 낮은 칸: 집전체 맨 끝 칸이 아니라 조금 안쪽일 때가 있어요.
        x_p = sol["x_p [m]"].entries[:, -1] if sto_p.shape[0] > 1 else np.array([0.0])
        i_min = int(np.argmin(sto_p[:, -1]))
        x_rel = (x_p - x_p.min()) / (x_p.max() - x_p.min()) if x_p.max() > x_p.min() else x_p * 0
        return {
            "min_cathode_electrolyte_conc_mol_m3": round(float(ce_p.min()), 1),
            "end_electrolyte_conc_cc_side_mol_m3": round(float(ce_p[-1, -1]), 1),
            "end_stoichiometry_separator_side": round(float(sto_p[0, -1]), 4),
            "end_stoichiometry_current_collector_side": round(float(sto_p[-1, -1]), 4),
            "end_stoichiometry_min": round(float(sto_p[i_min, -1]), 4),
            "end_stoichiometry_min_position": round(float(x_rel[i_min]), 3),
            "end_stoichiometry_mean": round(float(sto_p[:, -1].mean()), 4),
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
        }

    failures = []

    def solve_chain(p, current_A, c_rate, var_pts):
        attempts = []
        for solver_name in solver_factories:
            try:
                out = solve_once(p, current_A, c_rate, var_pts=var_pts, solver_name=solver_name)
                if not out["reached_cutoff"]:
                    attempts.append(f"{solver_name}: 하한 전압에 닿기 전에 멈춤 ({out['termination']})")
                    continue
                out["earlier_attempts"] = attempts
                return out, attempts
            except Exception as err:  # noqa: BLE001 - 실패도 결과로 남겨요
                attempts.append(f"{solver_name}: {type(err).__name__}: {str(err)[:300]}")
        return None, attempts

    def solve_safe(p, current_A, c_rate, label, var_pts):
        out, attempts = solve_chain(p, current_A, c_rate, var_pts)
        if out is None:
            failures.append({"design": label, "c_rate": c_rate, "attempts": attempts})
        return out

    designs = []

    def run_design(study, eps, factor, radius_nm):
        sigma = base_sigma * factor
        radius = radius_nm * 1e-9
        p = make_param(eps, sigma, radius)
        vp = var_pts_for(radius_nm)
        act = active_fraction(eps)
        label = f"{study} eps={eps} sigma={sigma:.4g} R={radius_nm}nm"
        q_theo_mAh_cm2 = act * thickness * c_max * F / 3600 * (1 - x100) / 10  # Ah/m2 -> mAh/cm2
        d = {
            "study": study,
            "porosity": eps,
            "active_volume_fraction": round(act, 5),
            "inactive_volume_fraction_binder_carbon": round((1 - eps) - act, 5),
            "conductivity_factor": factor,
            "conductivity_S_m": sigma,
            "particle_radius_nm": radius_nm,
            "thickness_um": thickness * 1e6,
            "theoretical_capacity_mAh_cm2": round(q_theo_mAh_cm2, 4),
            "var_pts": vp,
            "estimate_density_g_cm3": round(density_estimate(eps), 4),
            "estimate_lfp_loading_mg_cm2": round(act * thickness * 100 * RHO_LFP * 1000, 3),
            "runs": [],
        }
        # 0.1C: 이론 용량으로 시작해서, 측정한 0.1C 용량으로 1C를 다시 정하고 0.1% 안으로 맞을 때까지 반복
        q_ref_mAh_cm2 = q_theo_mAh_cm2
        iterations = []
        low = None
        for _ in range(4):
            current = 0.1 * q_ref_mAh_cm2 / 1000 * 1e4 * area_m2  # A
            low = solve_safe(p, current, 0.1, label, vp)
            if low is None:
                break
            iterations.append(round(low["capacity_mAh_cm2"], 5))
            converged = abs(low["capacity_mAh_cm2"] - q_ref_mAh_cm2) / q_ref_mAh_cm2 < 1e-3
            q_used = q_ref_mAh_cm2
            q_ref_mAh_cm2 = low["capacity_mAh_cm2"]
            if converged:
                break
        d["c_rate_1C_iterations_mAh_cm2"] = iterations
        if low is None:
            d["status"] = "0.1C 계산 실패"
            designs.append(d)
            return d
        one_c_mAh_cm2 = q_ref_mAh_cm2
        d["one_C_definition_mAh_cm2"] = round(one_c_mAh_cm2, 5)
        d["one_C_current_density_mA_cm2"] = round(one_c_mAh_cm2, 5)
        d["low_rate_run_used_1C_basis_mAh_cm2"] = round(q_used, 5)
        q01 = low["capacity_mAh_cm2"]
        d["capacity_0p1C_mAh_cm2"] = round(q01, 5)
        d["volumetric_capacity_0p1C_mAh_cm3_coating"] = round(q01 / (thickness * 100), 2)
        d["utilization_0p1C_pct_of_theoretical"] = round(100 * q01 / q_theo_mAh_cm2, 2)
        d["estimate_specific_capacity_0p1C_mAh_g_lfp"] = round(q01 / d["estimate_lfp_loading_mg_cm2"] * 1000, 1)
        e01 = low["energy_mWh_cm2"]
        q25_01 = low["capacity_to_2p5V_mAh_cm2"]
        d["energy_0p1C_mWh_cm2"] = e01
        d["capacity_to_2p5V_0p1C_mAh_cm2"] = q25_01
        low_run = dict(low, c_rate=0.1, current_density_mA_cm2=round(0.1 * q_used, 5),
                       capacity_mAh_cm2=round(q01, 5), retention_pct=100.0,
                       energy_retention_pct=100.0, retention_to_2p5V_pct=100.0)
        d["runs"].append(low_run)
        for c in C_RATES[1:]:
            current = c * one_c_mAh_cm2 / 1000 * 1e4 * area_m2
            out = solve_safe(p, current, c, label, vp)
            if out is None:
                d["runs"].append({"c_rate": c, "status": "failed"})
                continue
            out.update(c_rate=c, current_density_mA_cm2=round(c * one_c_mAh_cm2, 5),
                       capacity_mAh_cm2=round(out["capacity_mAh_cm2"], 5),
                       retention_pct=round(100 * out["capacity_mAh_cm2"] / q01, 2),
                       energy_retention_pct=round(100 * out["energy_mWh_cm2"] / e01, 2),
                       retention_to_2p5V_pct=round(100 * out["capacity_to_2p5V_mAh_cm2"] / q25_01, 2))
            d["runs"].append(out)
        d["status"] = "ok" if all("status" not in r for r in d["runs"]) else "일부 실패"
        designs.append(d)
        print(f"{label}: 0.1C {q01:.4f} mAh/cm2 | "
              + " ".join(f"{r['c_rate']}C {r.get('retention_pct', 'FAIL')}%" for r in d["runs"])
              + " | 10C energy {}% to2.5V {}%".format(d["runs"][-1].get("energy_retention_pct"),
                                                       d["runs"][-1].get("retention_to_2p5V_pct")), flush=True)
        return d

    for eps in POROSITIES:
        for factor in COND_FACTORS:
            run_design("main", eps, factor, 50)
    for eps in RADIUS_POROSITIES:
        for r_nm in RADII_NM:
            if r_nm == 50:
                continue  # 50 nm 기준 설계는 main에 이미 있어요
            run_design("particle_radius", eps, 1.0, r_nm)

    # 5) 격자(계산 칸) 점검: 가장 힘든 조건들을 모든 방향 칸 수 2배로 다시 풀어 용량 차이를 봐요.
    #    전해질이 바닥나는 설계(공극률 0.30·0.26)와 큰 입자(200 nm)를 모두 넣었어요.
    mesh_check = []
    hard = [("main", 0.426, 1.0, 50), ("main", 0.30, 1.0, 50), ("main", 0.30, 10.0, 50),
            ("main", 0.26, 0.1, 50), ("main", 0.26, 1.0, 50), ("main", 0.26, 10.0, 50),
            ("particle_radius", 0.426, 1.0, 200), ("particle_radius", 0.30, 1.0, 200)]
    for study, eps, factor, r_nm in hard:
        d = next((x for x in designs if x["study"] == study and x["porosity"] == eps
                  and x["conductivity_factor"] == factor and x["particle_radius_nm"] == r_nm), None)
        if d is None or "one_C_definition_mAh_cm2" not in d:
            continue
        run10 = next((r for r in d["runs"] if r["c_rate"] == 10.0 and "capacity_mAh_cm2" in r), None)
        if run10 is None:
            continue
        current = 10 * d["one_C_definition_mAh_cm2"] / 1000 * 1e4 * area_m2
        p = make_param(eps, base_sigma * factor, r_nm * 1e-9)
        fine_pts = refined_var_pts(d["var_pts"])
        name = f"{study} eps={eps} sigma_factor={factor} R={r_nm}nm"
        fine, attempts = solve_chain(p, current, 10.0, fine_pts)
        if fine is None:
            mesh_check.append({"design": name, "c_rate": 10.0, "refined_var_pts": fine_pts, "error": attempts})
            continue
        mesh_check.append({
            "design": name, "c_rate": 10.0,
            "used_var_pts": d["var_pts"],
            "used_mesh_mAh_cm2": run10["capacity_mAh_cm2"],
            "refined_var_pts": fine_pts,
            "refined_mesh_mAh_cm2": round(fine["capacity_mAh_cm2"], 5),
            "refined_solver": fine["solver"],
            "difference_pct": round(100 * (fine["capacity_mAh_cm2"] - run10["capacity_mAh_cm2"])
                                    / run10["capacity_mAh_cm2"], 3),
        })

    # 6) 보기 좋은 요약표: 공극률 x 전도도별 5C·10C 유지율
    table = []
    for d in designs:
        if d["study"] != "main":
            continue
        rr = {r["c_rate"]: r for r in d["runs"]}
        table.append({
            "porosity": d["porosity"], "conductivity_S_m": round(d["conductivity_S_m"], 5),
            "estimate_density_g_cm3": d["estimate_density_g_cm3"],
            "capacity_0p1C_mAh_cm2": d.get("capacity_0p1C_mAh_cm2"),
            "retention_1C_pct": rr.get(1.0, {}).get("retention_pct"),
            "retention_2C_pct": rr.get(2.0, {}).get("retention_pct"),
            "retention_5C_pct": rr.get(5.0, {}).get("retention_pct"),
            "retention_10C_pct": rr.get(10.0, {}).get("retention_pct"),
            "average_voltage_10C_V": rr.get(10.0, {}).get("average_voltage_V"),
            "energy_retention_10C_pct": rr.get(10.0, {}).get("energy_retention_pct"),
            "retention_10C_to_2p5V_pct": rr.get(10.0, {}).get("retention_to_2p5V_pct"),
        })

    import casadi
    result = {
        "title": "고밀도 LFP 양극: 공극률·전자 전도도·입자 크기에 따른 율속 특성 (PyBaMM DFN 하프셀)",
        "created": datetime.date.today().isoformat(),
        "script": "simulations/pybamm/d_lfp_density_rate.py",
        "python": sys.version.split()[0],
        "pybamm": pybamm.__version__,
        "casadi": casadi.__version__,
        "platform": platform.platform(),
        "question_ko": ("전극을 더 촘촘하게 누르면(공극률↓) 같은 두께에 활물질이 더 들어가지만 빠른 방전에서 손해를 볼 수 있어요. "
                        "양극의 전자 전도도(탄소 코팅·도전재)가 그 손해를 얼마나 바꾸는지 봐요."),
        "cell": "half-cell: LFP working electrode vs lithium metal (coin-cell style)",
        "model": "DFN",
        "model_options": OPTIONS,
        "solver": "IDAKLUSolver (실패하면 허용 오차를 줄인 IDAKLUSolver, 그다음 CasadiSolver safe로 다시 시도)",
        "var_pts": ("분리막·양극 두께 방향은 PyBaMM 기본값(x_s 20, x_p 20). 입자 안 r_p는 반지름 50 nm마다 20칸 "
                    "(50 nm 20, 100 nm 40, 200 nm 80). 설계마다 designs[].var_pts에 적었어요."),
        "parameter_set": "Prada2013 (A123 LFP/graphite 26650)",
        "borrowed_from_Xu2019_added": added,
        "borrowed_from_Xu2019_replaced": replaced,
        "borrowed_note_ko": ("added는 하프셀 모델이 요구하는데 Prada2013에 없어서 빌린 값이에요. "
                             "replaced는 하프셀에서 '음극' 이름이 리튬 금속 박을 뜻해서 흑연 값을 리튬 금속 값으로 바꾼 거예요. "
                             "replaced 두 값은 리튬 박의 전압 손실(전류×두께÷전도도)에만 쓰여요."),
        "unused_prada_keys_note_ko": "흑연 음극 값(OCP·확산계수·입자 크기 등)은 하프셀에서 쓰이지 않아요.",
        "temperature": {"ambient_K": TEMPERATURE_K, "initial_K": TEMPERATURE_K, "thermal": "isothermal",
                        "note_ko": "Prada2013은 298 K로 되어 있어서 25 °C(298.15 K)로 맞췄어요."},
        "voltage_window_V": [v_min, v_max],
        "initial_state": {
            "x100_positive_stoichiometry": x100,
            "initial_concentration_mol_m3": x100 * c_max,
            "definition_ko": "만충 = 양극 OCP가 상한 전압 3.6 V가 되는 상태(리튬 금속 기준)예요.",
            "why_not_pybamm_initial_soc_ko": ("PyBaMM 하프셀 SOC 계산은 OCP가 하한 2.0 V가 되는 지점도 찾아야 해요. "
                                              "그런데 이 LFP OCP 함수(Afshar2017)는 완전히 리튬이 찬 상태에서도 약 "
                                              f"{float(ocp(1.0)):.3f} V라 그 지점이 없어요. 그래서 상한 쪽만 직접 계산했어요."),
            "ocp_at_x0_V": round(float(ocp(1e-12)), 4),
            "ocp_at_x1_V": round(float(ocp(1.0)), 4),
        },
        "fixed_inputs": {
            "positive_thickness_um": thickness * 1e6,
            "separator_thickness_um": base["Separator thickness [m]"] * 1e6,
            "separator_porosity": base["Separator porosity"],
            "electrolyte_initial_concentration_mol_m3": base["Initial concentration in electrolyte [mol.m-3]"],
            "positive_bruggeman_electrolyte": base["Positive electrode Bruggeman coefficient (electrolyte)"],
            "positive_bruggeman_electrode": base["Positive electrode Bruggeman coefficient (electrode)"],
            "positive_particle_diffusivity_m2_s": base["Positive particle diffusivity [m2.s-1]"],
            "base_conductivity_S_m": base_sigma,
            "base_particle_radius_nm": base_radius * 1e9,
            "electrode_area_m2_used_for_per_cm2": area_m2,
        },
        "transport_note_ko": ("PyBaMM은 브루그만 식을 써요. 전해질 쪽 실효 이온 전도 = 값 × 공극률^1.5, "
                              "고체 쪽 실효 전자 전도 = 전도도 × (1−공극률)^1.5 예요. 그래서 촘촘하게 누르면 "
                              "이온 길은 좁아지고 전자 길은 조금 넓어져요."),
        "design_rule_ko": ("두께 80 µm 고정. 활물질 부피 비율 = 0.374 × (1−공극률) / (1−0.426). "
                           "비활물질(바인더+탄소)과 활물질의 비율은 기준 설계와 같게 유지해요. "
                           "전도도는 비활물질 양과 따로, 탄소망이 좋고 나쁨을 흉내 내는 값으로만 바꿨어요."),
        "c_rate_definition_ko": ("설계마다 0.1C 방전 용량으로 1C 전류를 정해요(1C 전류 = 0.1C 용량 ÷ 1시간). "
                                 "0.1C 전류도 그 값으로 다시 정해서, 0.1C 용량이 0.1% 안으로 같아질 때까지 반복했어요. "
                                 "유지율 = xC 용량 ÷ 0.1C 용량 × 100."),
        "capacity_unit_ko": "용량은 전극 면적 1 cm²당 mAh예요(전류 방향 단면 기준, 집전체 한 면).",
        "extra_metrics_ko": {
            "energy_retention_pct": "xC 에너지(전압×용량을 더한 값) ÷ 0.1C 에너지 × 100. 전압이 많이 떨어진 채로 버티는 경우를 가려내요.",
            "average_voltage_V": "에너지 ÷ 용량. 방전 내내 평균 전압이에요.",
            "retention_to_2p5V_pct": (f"같은 방전 곡선에서 {SECOND_CUTOFF_V} V에 처음 닿을 때까지의 용량으로 계산한 유지율이에요. "
                                      f"하한을 {SECOND_CUTOFF_V} V로 두고 시험한 것과 같아요(정전류라 그 지점까지 곡선이 같아요)."),
            "min_cathode_electrolyte_conc_mol_m3": "방전 동안 양극 기공 속 전해질 농도의 최솟값이에요. 0에 가까우면 리튬 이온이 바닥난 거예요.",
            "end_stoichiometry_*": "방전이 끝났을 때 분리막 쪽·집전체 쪽 입자에 리튬이 찬 정도(0~1)예요. 낮으면 그쪽을 다 못 쓴 거예요.",
            "end_stoichiometry_min / _min_position / _mean": ("끝났을 때 리튬이 가장 덜 찬 칸의 값과 그 위치(0 = 분리막 쪽, 1 = 집전체 쪽), "
                                                              "그리고 양극 전체 평균이에요. 가장 덜 찬 칸이 집전체 맨 끝이 아니라 조금 안쪽일 때가 있어요."),
        },
        "conductivity_reversal_note_ko": (
            "공극률 0.30·0.26의 10C에서는 전자 전도도가 낮을수록 용량 유지율이 오히려 높아요. "
            "전도도가 낮으면 반응이 처음에 집전체 쪽에 몰려서, 그쪽 전해질이 방전 초반에 바닥나요. "
            "그 뒤에는 집전체 바로 앞 얇은 층만 못 쓰고 나머지 양극은 거의 다 써요. "
            "전도도가 높으면 반응이 분리막 쪽에서 시작해 안쪽으로 나아가요. 반응이 양극 가운데쯤 왔을 때 "
            "안쪽 전해질이 바닥나서, 집전체 쪽의 더 두꺼운 층을 못 써요. "
            "runs[].end_stoichiometry_mean(양극 전체 평균)과 end_stoichiometry_min으로 확인할 수 있어요."),
        "second_cutoff_V": SECOND_CUTOFF_V,
        "density_estimate_assumption": {
            "lfp_density_g_cm3": RHO_LFP,
            "inactive_binder_carbon_density_g_cm3": RHO_INACTIVE,
            "formula": "density = active_volume_fraction*3.6 + inactive_volume_fraction*1.9 (coating only, no foil)",
            "note_ko": ("모델 계산에는 쓰지 않은 '추정' 값이에요. 공극률을 업계에서 쓰는 g/cm³ 말로 옮겨 보려고만 계산했어요. "
                        "Prada2013의 비활물질 비율(고체의 약 35 부피%)이 상용 전극보다 훨씬 커서, "
                        "같은 공극률이라도 상용 전극보다 밀도가 낮게 나와요."),
        },
        "rates_C": C_RATES,
        "porosities": POROSITIES,
        "conductivity_factors": COND_FACTORS,
        "particle_radius_side_study": {"radii_nm": RADII_NM, "porosities": RADIUS_POROSITIES,
                                       "conductivity_factor": 1.0},
        "figures": {
            "d_retention_vs_porosity.png": ("공극률에 따른 5C·10C 용량 유지율과 10C 에너지 유지율(전도도별 선). "
                                            "에너지 칸은 처음 계획에 없던 것으로, 전도도가 낮을 때 하한 2.0 V 덕에 용량은 남지만 "
                                            "평균 전압이 낮아지는 손해를 함께 보여 주려고 더했어요."),
            "d_capacity_vs_10C_retention.png": "0.1C 면적당 용량과 10C 용량 유지율(전도도별 선, 기준 전도도 점에 공극률 표시).",
        },
        "summary_table_main_R50nm": table,
        "designs": designs,
        "failures": failures,
        "mesh_check": mesh_check,
        "caveats_ko": [
            "보정하지 않은 모델 값이에요. 실제 LFP 코인셀·제품의 숫자와 바로 비교하면 안 돼요.",
            "입자 안 리튬 이동을 단순 확산으로 계산해요. LFP의 두 상(FePO4/LiFePO4) 경계 이동은 따로 다루지 않아요.",
            "전도도를 바꿀 때 비활물질 부피는 그대로 둬요. 실제로는 탄소를 늘리면 활물질 자리가 줄어요.",
            "분리막·전해질은 Prada2013 값(분리막 25 µm)이에요. 유리섬유 분리막을 쓰는 코인셀보다 이온 길이 짧아요.",
            "전해질 농도 최솟값이 0보다 아주 조금 작게(-0.x) 나온 것은 계산 오차예요. 0으로 읽어요.",
        ] + [
            f"격자 점검: {m['design']} 10C 용량이 계산 칸을 2배로 늘리면 {m['difference_pct']}% 달라져요. 이 값은 대략값으로 읽어요."
            for m in mesh_check if abs(m.get("difference_pct", 0)) > 1
        ],
        "total_seconds": round(time.perf_counter() - t_start, 1),
    }
    os.makedirs(RES, exist_ok=True)
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        cols = ["current_density_mA_cm2", "capacity_mAh_cm2", "retention_pct", "energy_mWh_cm2",
                "energy_retention_pct", "average_voltage_V", "capacity_to_2p5V_mAh_cm2", "retention_to_2p5V_pct",
                "min_cathode_electrolyte_conc_mol_m3", "end_stoichiometry_separator_side",
                "end_stoichiometry_current_collector_side", "end_stoichiometry_min",
                "end_stoichiometry_min_position", "end_stoichiometry_mean", "seconds_incl_build", "solver"]
        w.writerow(["study", "porosity", "active_volume_fraction", "inactive_volume_fraction",
                    "estimate_density_g_cm3", "conductivity_S_m", "particle_radius_nm", "c_rate"]
                   + cols + ["status", "termination"])
        for d in designs:
            for r in d["runs"]:
                w.writerow([d["study"], d["porosity"], d["active_volume_fraction"],
                            d["inactive_volume_fraction_binder_carbon"], d["estimate_density_g_cm3"],
                            round(d["conductivity_S_m"], 5), d["particle_radius_nm"], r["c_rate"]]
                           + [r.get(c, "") for c in cols] + [r.get("status", "ok"), r.get("termination", "")])
    print("saved", JSON_PATH)
    print("saved", CSV_PATH)
    print("failures:", len(failures), "| total seconds:", result["total_seconds"])


# ---------------------------------------------------------------- 그림 ----
# 색: 전도도는 크기 순서가 있는 값이라 파란색 한 가지를 밝음→어두움으로 써요(어두울수록 전도도가 높아요).
#     모양(동그라미·네모·세모)도 함께 바꿔서 색만으로 구분하지 않게 해요.
SERIES_STYLE = {0.1: ("#6da7ec", "o"), 1.0: ("#256abf", "s"), 10.0: ("#0d366b", "^")}
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


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

    main = [d for d in res["designs"] if d["study"] == "main" and d["particle_radius_nm"] == 50]

    def series(factor, key, metric="retention_pct"):
        xs, ys = [], []
        for d in sorted(main, key=lambda d: -d["porosity"]):
            if d["conductivity_factor"] != factor:
                continue
            if key == "cap":
                val = d.get("capacity_0p1C_mAh_cm2")
            else:
                val = next((r.get(metric) for r in d["runs"] if r["c_rate"] == key), None)
            if val is not None:
                xs.append(d["porosity"])
                ys.append(val)
        return xs, ys

    def label(factor):
        sigma = next(d["conductivity_S_m"] for d in main if d["conductivity_factor"] == factor)
        mult = {0.1: "×0.1", 1.0: "×1", 10.0: "×10"}[factor]
        base = T(" 기준", " base") if factor == 1.0 else ""
        return T(f"전자 전도도 {mult}{base} ({sigma:.3g} S/m)", f"Electronic conductivity {mult}{base} ({sigma:.3g} S/m)")

    dens = {d["porosity"]: d["estimate_density_g_cm3"] for d in main}

    def fmt_eps(e):
        return f"{e:.2f}" if round(e, 2) == e else f"{e:g}"  # 0.3 -> 0.30, 0.426 그대로
    note = T("PyBaMM DFN 하프셀(LFP 양극 vs 리튬 금속), Prada2013 값 기반, 양극 두께 80 µm 고정, 25 °C. "
             "보정하지 않은 모델 값이에요.\n괄호 안 밀도는 LFP 3.6·비활물질 1.9 g/cm³를 가정한 추정값이에요.",
             "PyBaMM DFN half-cell (LFP vs Li metal), Prada2013-based, 80 µm cathode, 25 °C. Uncalibrated model values.\n"
             "Densities in brackets are estimates assuming LFP 3.6 and binder+carbon 1.9 g/cm³.")

    # 그림 1: 공극률에 따른 5C·10C 용량 유지율 + 10C 에너지 유지율
    #   세 번째 칸은 전도도가 낮을 때 "용량은 남지만 전압이 낮아지는" 손해를 보이려고 더했어요.
    panels = [
        (5.0, "retention_pct", T("5C 용량 유지율 (12분 방전)", "5C capacity retention")),
        (10.0, "retention_pct", T("10C 용량 유지율 (6분 방전)", "10C capacity retention")),
        (10.0, "energy_retention_pct", T("10C 에너지 유지율 (전압까지 반영)", "10C energy retention")),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.4), sharey=True)
    for ax, (c, metric, title) in zip(axes, panels):
        for factor in COND_FACTORS:
            color, marker = SERIES_STYLE[factor]
            xs, ys = series(factor, c, metric)
            ax.plot(xs, ys, color=color, marker=marker, markersize=7, linewidth=2,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, label=label(factor))
            if xs and c == 10.0:
                ax.annotate(f"{ys[-1]:.1f}", (xs[-1], ys[-1]), xytext=(7, 0), textcoords="offset points",
                            va="center", fontsize=8.5, color=INK2)
        ax.set_title(title, fontsize=11, loc="left", color=INK)
        ax.set_xlim(0.44, 0.235)  # 오른쪽으로 갈수록 촘촘해요
        ax.set_xticks(POROSITIES)
        ax.set_xticklabels([f"{fmt_eps(e)}\n({dens[e]:.2f})" for e in POROSITIES], fontsize=8.5)
        ax.set_xlabel(T("양극 공극률 (괄호: 추정 밀도 g/cm³)  →  더 촘촘하게",
                        "Porosity (bracket: est. density g/cm³)  →  denser"), fontsize=9)
    axes[0].set_ylabel(T("0.1C 대비 유지율 (%)", "Retention vs 0.1C (%)"))
    axes[0].set_ylim(55, 100)
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper left", bbox_to_anchor=(0.045, 0.915), ncol=3, frameon=False, fontsize=9)
    fig.suptitle(T("전극을 촘촘하게 누를 때 빠른 방전에서 남는 용량과 에너지",
                   "Rate capability vs cathode porosity, by electronic conductivity"),
                 x=0.045, y=0.985, ha="left", fontsize=13, color=INK)
    fig.text(0.045, 0.01, note, fontsize=8, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.07, 1, 0.88))
    fig.savefig(FIG_RATE, dpi=150)
    plt.close(fig)

    # 그림 2: 0.1C 면적당 용량 vs 10C 유지율 (맞바꿈)
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    for factor in COND_FACTORS:
        color, marker = SERIES_STYLE[factor]
        eps_c, caps = series(factor, "cap")
        eps_r, rets = series(factor, 10.0)
        pts = [(cap, ret, e) for e, cap in zip(eps_c, caps) for e2, ret in zip(eps_r, rets) if e == e2]
        if not pts:
            continue
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, marker=marker, markersize=8,
                linewidth=2, markeredgecolor=SURFACE, markeredgewidth=1.2, label=label(factor))
        if factor == 1.0:
            # 공극률이 같으면 0.1C 용량도 같아서(전도도와 무관) 세 선의 점이 같은 x에 놓여요.
            # 그래서 공극률 이름은 점 옆이 아니라 그래프 위쪽에 x 위치마다 한 번만 적어요.
            for cap, ret, e in pts:
                ax.axvline(cap, color=GRID, linewidth=0.8, zorder=0)
                ax.text(cap, 96.2, T(f"공극률\n{fmt_eps(e)} ({dens[e]:.2f})", f"porosity\n{fmt_eps(e)} ({dens[e]:.2f})"),
                        ha="center", va="center", fontsize=8.5, color=INK2)
    ax.set_ylim(None, 98)
    ax.set_xlabel(T("0.1C 면적당 용량 (mAh/cm²)  →  같은 두께에 더 많이 담김",
                    "Areal capacity at 0.1C (mAh/cm²)  →  more stored per area"))
    ax.set_ylabel(T("10C 유지율 (0.1C 대비, %)", "10C retention vs 0.1C (%)"))
    ax.set_title(T("촘촘하게 눌러 면적당 용량을 늘리면, 어느 지점부터 10C 유지율이 꺾여요",
                   "Areal capacity vs 10C retention: the density-rate trade-off"),
                 loc="left", fontsize=12, color=INK)
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    fig.text(0.02, 0.01, note, fontsize=8, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(FIG_TRADE, dpi=150)
    plt.close(fig)
    print("saved", FIG_RATE)
    print("saved", FIG_TRADE)
    print("labels:", "Korean (Malgun Gothic)" if korean else "English (Malgun Gothic not found)")


if __name__ == "__main__":
    if "--plot-only" not in sys.argv:
        run_study()
    try:
        make_figures()
    except ImportError as err:
        print(f"그림 단계를 건너뛰었어요 ({err}). 프로젝트 .venv로 --plot-only를 실행하면 그림이 만들어져요.")
