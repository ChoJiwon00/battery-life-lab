"""(a) LFP/graphite (Prada2013) 1C 정전류 방전: SPM / SPMe / DFN 비교, 실행 시간 측정."""
import json
import os
import platform
import sys
import time

os.environ.setdefault("PYBAMM_DISABLE_TELEMETRY", "true")
import numpy as np
import pybamm

OUT = os.path.dirname(os.path.abspath(__file__))
results = {
    "python": sys.version.split()[0],
    "pybamm": pybamm.__version__,
    "platform": platform.platform(),
    "runs": [],
}

for c_rate in [0.5, 1.0, 2.0]:
    for name, model_cls in [("SPM", pybamm.lithium_ion.SPM),
                            ("SPMe", pybamm.lithium_ion.SPMe),
                            ("DFN", pybamm.lithium_ion.DFN)]:
        model = model_cls()
        param = pybamm.ParameterValues("Prada2013")
        nominal = param["Nominal cell capacity [A.h]"]
        experiment = pybamm.Experiment(
            [f"Discharge at {c_rate}C until {param['Lower voltage cut-off [V]']} V"]
        )
        t0 = time.perf_counter()
        sim = pybamm.Simulation(model, parameter_values=param, experiment=experiment)
        sol = sim.solve()
        elapsed = time.perf_counter() - t0
        V = sol["Voltage [V]"].entries
        Q = sol["Discharge capacity [A.h]"].entries
        run = {
            "model": name,
            "c_rate": c_rate,
            "seconds_incl_build": round(elapsed, 3),
            "nominal_capacity_Ah": nominal,
            "discharged_capacity_Ah": round(float(Q[-1]), 4),
            "fraction_of_nominal": round(float(Q[-1] / nominal), 4),
            "start_voltage_V": round(float(V[0]), 4),
            "mid_voltage_V": round(float(np.interp(Q[-1] / 2, Q, V)), 4),
            "end_voltage_V": round(float(V[-1]), 4),
            "n_time_points": int(len(V)),
        }
        results["runs"].append(run)
        print(run, flush=True)
        if c_rate == 1.0:
            # 작은 출력: 방전 곡선(용량-전압) 50점만 저장
            idx = np.linspace(0, len(Q) - 1, 50).astype(int)
            with open(os.path.join(OUT, f"a_curve_{name}_1C.csv"), "w", encoding="utf-8") as f:
                f.write("discharge_capacity_Ah,voltage_V\n")
                for i in idx:
                    f.write(f"{Q[i]:.5f},{V[i]:.5f}\n")

with open(os.path.join(OUT, "a_results.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("saved", os.path.join(OUT, "a_results.json"))
