"""(c) 시연용: LFP/흑연(Prada2013) + SEI 성장. Severson 2019와 비슷한 고속 충방전.
주의: Prada2013에는 SEI 파라미터가 없어서 OKane2022(흑연 음극)의 SEI 값을 빌려 쓴다.
보정(calibration)되지 않은 값이므로 숫자 자체는 실제 A123 셀 수명과 비교하면 안 된다.

사용법: python c_lfp_sei_demo.py <charge_C> <n_cycles>
"""
import json
import os
import sys
import time

os.environ.setdefault("PYBAMM_DISABLE_TELEMETRY", "true")
import pybamm

OUT = os.path.dirname(os.path.abspath(__file__))
charge_c = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
n_cycles = int(sys.argv[2]) if len(sys.argv) > 2 else 100

param = pybamm.ParameterValues("Prada2013")
okane = pybamm.ParameterValues("OKane2022")
borrowed = {k: v for k, v in okane.items() if k not in param.keys()}
param.update(borrowed, check_already_exists=False)
v_max = param["Upper voltage cut-off [V]"]
v_min = param["Lower voltage cut-off [V]"]

model = pybamm.lithium_ion.SPMe({"SEI": "solvent-diffusion limited"})
experiment = pybamm.Experiment(
    [(
        f"Charge at {charge_c}C until {min(v_max, 3.6)} V",
        f"Hold at {min(v_max, 3.6)} V until C/20",
        f"Discharge at 4C until {v_min} V",
    )] * n_cycles
)
t0 = time.perf_counter()
sim = pybamm.Simulation(model, parameter_values=param, experiment=experiment,
                        solver=pybamm.IDAKLUSolver())
sol = sim.solve()
elapsed = time.perf_counter() - t0
caps = []
for cyc in sol.cycles:
    q = cyc.steps[2]["Discharge capacity [A.h]"].entries
    caps.append(float(q[-1] - q[0]))
sv = sol.summary_variables
res = {
    "pybamm": pybamm.__version__,
    "parameter_set": "Prada2013 + OKane2022 SEI params (borrowed)",
    "model": "SPMe + SEI (solvent-diffusion limited)",
    "charge_C": charge_c, "discharge_C": 4.0,
    "v_max_used": min(v_max, 3.6), "v_min": v_min,
    "n_borrowed_keys": len(borrowed),
    "n_cycles_solved": len(caps),
    "seconds": round(elapsed, 1),
    "cap_first_Ah": round(caps[0], 4), "cap_last_Ah": round(caps[-1], 4),
    "fade_pct": round(100 * (1 - caps[-1] / caps[0]), 3),
    "LLI_pct_last": float(sv["Loss of lithium inventory [%]"][-1]),
    "SEI_loss_Ah_last": float(sv["Loss of capacity to negative SEI [A.h]"][-1]),
    "cap_by_cycle": [round(c, 5) for c in caps],
}
with open(os.path.join(OUT, f"c_lfp_sei_{charge_c}C_{n_cycles}cyc.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print(json.dumps({k: v for k, v in res.items() if k != "cap_by_cycle"}, indent=1))
