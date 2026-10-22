"""(b) 사이클 열화 시뮬레이션: OKane2022(LG M50, NMC811/흑연+SiOx) + DFN
SEI 성장 + 리튬 석출(plating) + 입자 균열 + 활물질 손실(LAM)을 함께 켠 PyBaMM 공식 예제
(coupled-degradation) 설정을 따르고, 충전 속도만 바꿔 비교한다.

사용법: python b_degradation.py <charge_C_rate> <n_cycles> <label>
"""
import json
import os
import sys
import time

os.environ.setdefault("PYBAMM_DISABLE_TELEMETRY", "true")
import numpy as np
import pybamm

OUT = os.path.dirname(os.path.abspath(__file__))
charge_c = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
n_cycles = int(sys.argv[2]) if len(sys.argv) > 2 else 20
label = sys.argv[3] if len(sys.argv) > 3 else f"okane_{charge_c}C_{n_cycles}cyc"

options = {
    "SEI": "solvent-diffusion limited",
    "SEI porosity change": "true",
    "lithium plating": "partially reversible",
    "lithium plating porosity change": "true",
    "particle mechanics": ("swelling and cracking", "swelling only"),
    "SEI on cracks": "true",
    "loss of active material": "stress-driven",
}
model = pybamm.lithium_ion.DFN(options)
param = pybamm.ParameterValues("OKane2022")
var_pts = {"x_n": 5, "x_s": 5, "x_p": 5, "r_n": 30, "r_p": 30}

experiment = pybamm.Experiment(
    [(
        "Discharge at 1C until 2.5 V",
        "Rest for 10 minutes",
        f"Charge at {charge_c}C until 4.2 V",
        "Hold at 4.2 V until C/20",
        "Rest for 10 minutes",
    )] * n_cycles
)

t0 = time.perf_counter()
sim = pybamm.Simulation(model, parameter_values=param, experiment=experiment,
                        var_pts=var_pts, solver=pybamm.IDAKLUSolver())
sol = sim.solve()
elapsed = time.perf_counter() - t0
print(f"[{label}] solved {len(sol.cycles)} cycles in {elapsed:.1f} s", flush=True)

# 사이클별 1C 방전 용량(실제로 꺼낸 용량)
dis_cap = []
for cyc in sol.cycles:
    if cyc is None:
        continue
    q = cyc.steps[0]["Discharge capacity [A.h]"].entries
    dis_cap.append(float(q[-1] - q[0]))

sv = sol.summary_variables
names = list(getattr(sv, "all_variables", [])) or list(sv.keys())
want = [
    "Capacity [A.h]",
    "Loss of lithium inventory [%]",
    "Loss of active material in negative electrode [%]",
    "Loss of active material in positive electrode [%]",
    "Loss of capacity to negative SEI [A.h]",
    "Loss of capacity to negative SEI on cracks [A.h]",
    "Loss of capacity to negative lithium plating [A.h]",
    "Total capacity lost to side reactions [A.h]",
]
summary = {}
for w in want:
    try:
        summary[w] = [float(x) for x in np.asarray(sv[w]).ravel()]
    except Exception as e:  # noqa: BLE001
        summary[w] = f"not available: {e}"

nominal = param["Nominal cell capacity [A.h]"]
res = {
    "label": label,
    "pybamm": pybamm.__version__,
    "parameter_set": "OKane2022",
    "model": "DFN + SEI/plating/cracking/LAM",
    "charge_C": charge_c,
    "discharge_C": 1.0,
    "n_cycles_requested": n_cycles,
    "n_cycles_solved": len(dis_cap),
    "seconds": round(elapsed, 1),
    "seconds_per_cycle": round(elapsed / max(len(dis_cap), 1), 2),
    "nominal_capacity_Ah": nominal,
    "discharge_capacity_1C_Ah_first": round(dis_cap[0], 4),
    "discharge_capacity_1C_Ah_last": round(dis_cap[-1], 4),
    "discharge_capacity_fade_pct": round(100 * (1 - dis_cap[-1] / dis_cap[0]), 3),
    "discharge_capacity_1C_Ah_by_cycle": [round(x, 5) for x in dis_cap],
    "summary_last_cycle": {k: (v[-1] if isinstance(v, list) else v) for k, v in summary.items()},
    "summary_first_cycle": {k: (v[0] if isinstance(v, list) else v) for k, v in summary.items()},
    "summary_variable_names_sample": names[:60],
}
with open(os.path.join(OUT, f"b_{label}.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print(json.dumps({k: v for k, v in res.items()
                  if k not in ("discharge_capacity_1C_Ah_by_cycle", "summary_variable_names_sample")},
                 ensure_ascii=False, indent=1))
