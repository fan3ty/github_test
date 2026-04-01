"""
This regression test is based on a run setup by Yuval Shmilovitz
using a script (Run_EGBE_expts.py) rather than EgbeLem. It was
originally designed to ensure that EgbeLem produces the same 
results, and also serves as a regression test for future changes.

GT Mar 2026
"""

import pickle
import numpy as np
from egbelem import EgbeLem

EARTH_GRAV_ACCEL = 9.8  # gravitational acceleration at earth's surface, m/s2

# Set up parameters
params = {
    "grid": {
        "source": "create",
        "create_grid": {
            "RasterModelGrid": {
                "shape": (100, 3),
                "xy_spacing": 500.,
                "bc": {
                    "right": "closed",
                    "top": "closed",
                    "left": "closed",
                    "bottom": "open",
                }
            }
        },
    },
    "clock": {"start": 0.0, "stop": 4.0e7, "step": 100.0},
    "output": {
        "plot_times": 4.0e7,  # float or list
        "save_times": 20000.0,  # float or list
        "report_times": 100.0,  # float or list
        "save_path": "egbe_regression_ys14",
        "clobber": True,
        "fields": None,
        "plot_to_file": True,
    },
    "initial_conditions": {
        "use_field_for_initial_topo": False,
        "initial_topo": 1.0,
        "random_topo_amp": 10.0,
        "use_field_for_initial_sed_classes": True,
        "init_grains_weight": {"_filepath": "regr_ys14_init_grain_mass.npy"},
    },
    "baselevel": {
        "uplift_rate": 0.0005,
    },
    "flow_routing": {
        "flow_metric": "D8",
        "update_flow_depressions": True,
        "depression_handler": "fill",
        "epsilon": True,
        "bankfull_runoff_rate": 5.0,
    },
    "fluvial": {
        "intermittency_factor": 0.01,
        "sediment_porosity": 0.4,
        "depth_decay_scale": 1.0,
        "plucking_coefficient": 1.0e-4,
        "epsilon": 0.2,
        "abrasion_coefficients": [0.0],
        "bedrock_abrasion_coefficient": 0.0,
        "fractions_from_plucking": {
            "_filepath": "regr_ys14_grain_size_fracs_from_plucking.npy"
        },
        "grav_accel": EARTH_GRAV_ACCEL,
        "rho_sed": 2650.0,
        "rho_water": 1000.0,
        "use_fixed_width": True,
        "fixed_width_coeff": 0.002,
        "fixed_width_expt": 0.5,
        "mannings_n": 0.05,
        "tau_star_c_median": 0.045,
        "alpha": 0.68,
        "tau_c_bedrock": 10.0,
        "d_min": 0.1,
        "grain_sizes": [0.01, 0.1],
        "plucking_by_tools_flag": False,
        "d_min": 0.1,
    },
    "hillslope": {
        "soil_creep_coefficient": 0.05,
        "rock_creep_coefficient": 0.05,
    },
}

# Instantiate EgbeLem
model = EgbeLem(params=params)


# with open(f'sim_num_14.pkl', 'rb') as f:
#    loaded_dict = pickle.load(f)

# for item in loaded_dict.keys():
#     print(item)
#     print(loaded_dict[item])

# grid = loaded_dict["grid"]
# print(grid.at_node.keys())
# print(grid.at_link.keys())

# np.save("regr_ys14_grain_size_fracs_from_plucking", loaded_dict["proportions"])
# t1 = np.load("regr_ys14_grain_size_fracs_from_plucking.npy")
# print(np.amax(t1 - loaded_dict["proportions"]))
