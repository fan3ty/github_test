#! /usr/bin/env python
# coding: utf-8
"""
EGBE-LEM: Landscape Evolution Model that implements the theory behind
the ExtendedGravelBedrockEroder.
"""

import sys

import numpy as np
from landlab import RasterModelGrid, HexModelGrid
from landlab.core import load_params
from landlab.components import (
    ExtendedGravelBedrockEroder,
    PriorityFloodFlowRouter,
    FlowAccumulator,
)
from landlab.components.soil_grading import SoilGrading

from model_base import LandlabModel, verify_input_file_and_load_params


# Default factors related to fluvial transport
EARTH_GRAV_ACCEL = 9.8  # gravitational acceleration at earth's surface, m/s2
RHO = 1000.0  # density of water, kg/m3
RHO_SED = 2650.0  # density of sediment grains, kg/m3
SEDIMENT_POROSITY = 1.0 / 3.0


class EgbeLem(LandlabModel):
    """Landscape Evolution Model using fluvial EGBE theory."""

    DEFAULT_PARAMS = {
        "grid": {
            "source": "create",
            "create_grid": {
                "RasterModelGrid": [
                    (31, 31),
                    {"xy_spacing": 1000.0},
                ],
            },
        },
        "clock": {"start": 0.0, "stop": 1000.0, "step": 10.0},
        "output": {
            "plot_times": 250.0,  # float or list
            "save_times": 1000.0,  # float or list
            "report_times": 100.0,  # float or list
            "save_path": "egbe_run",
            "clobber": True,
            "fields": None,
            "plot_to_file": True,
        },
        "initial_conditions": {
            "use_field_for_initial_topo": False,
            "initial_topo": 0.0,
            "random_topo_amp": 10.0,
            "use_field_for_initial_sed_classes": False,
            "init_grains_weight": np.array(
                [1.0 * EARTH_GRAV_ACCEL * RHO_SED * (1.0 - SEDIMENT_POROSITY)]
            ),
        },
        "baselevel": {
            "uplift_rate": 0.0001,
        },
        "flow_routing": {
            "flow_metric": "D8",
            "update_flow_depressions": True,
            "depression_handler": "fill",
            "epsilon": True,
            "bankfull_runoff_rate": 1.0,
        },
        "fluvial": {
            "intermittency_factor": 0.01,
            "sediment_porosity": SEDIMENT_POROSITY,
            "depth_decay_scale": 1.0,
            "plucking_coefficient": 1.0e-4,
            "epsilon": 0.2,
            "abrasion_coefficients": [1.0e-4],
            "bedrock_abrasion_coefficient": 1.0e-4,
            "fractions_from_plucking": [0.5],
            "grav_accel": EARTH_GRAV_ACCEL,
            "rho_sed": RHO_SED,
            "rho_water": RHO,
            "use_fixed_width": True,
            "fixed_width_coeff": 0.002,
            "fixed_width_expt": 0.5,
            "mannings_n": 0.05,
            "tau_star_c_median": 0.0495,
            "alpha": 0.68,
            "tau_c_bedrock": 10.0,
            "d_min": 0.1,
            "grain_sizes": [0.1],
            "plucking_by_tools_flag": True,
        },
    }

    def __init__(self, params: dict = {}, input_file: str = "") -> None:
        """Initialize the model."""
        if (not isinstance(params, dict)) or (not isinstance(input_file, str)):
            print("params if given must be dict & input_file if given must be str")
            raise (TypeError)
        super().__init__(params, input_file)
        params = self.params # grab any modifications to params from base class init

        # Store global dt
        self._global_dt = params["clock"]["step"]

        # Set up grid fields
        ic_params = params["initial_conditions"]
        if not ("topographic__elevation" in self.grid.at_node.keys()):
            self.grid.add_zeros("topographic__elevation", at="node")
            if ic_params["use_field_for_initial_topo"]:
                print("Warning: you selected the option use_field_for_initial_topo")
                print("but no field was provided. Using defaults instead.")
                ic_params["use_field_for_initial_topo"] = False
        num_classes = max(
            len(params["fluvial"]["grain_sizes"]),
            len(params["fluvial"]["abrasion_coefficients"]),
        )
        if not ("grains__weight" in self.grid.at_node.keys()):
            self.grid.add_field(
                "grains__weight",
                np.zeros((self.grid.number_of_nodes, num_classes)),
                at="node",
            )
            if ic_params["use_field_for_initial_sed_classes"]:
                print(
                    "Warning: you selected the option use_field_for_initial_grains_wt"
                )
                print("but no field was provided. Using defaults instead.")
                ic_params["use_field_for_initial_grains_wt"] = False
        #if not ("soil__depth" in self.grid.at_node.keys()):
        #    self.grid.add_zeros("soil__depth", at="node")
        #if not ("bedrock__elevation" in self.grid.at_node.keys()):
        #    self.grid.add_zeros("bedrock__elevation", at="node")

        self.topo = self.grid.at_node["topographic__elevation"]
        self.grains_wt = self.grid.at_node["grains__weight"].reshape(
            (self.grid.number_of_nodes, num_classes)
        )

        if not ic_params["use_field_for_initial_topo"]:
            self.topo[self.grid.core_nodes] = ic_params["initial_topo"] + ic_params[
                "random_topo_amp"
            ] * np.random.rand(self.grid.number_of_core_nodes)
        if not ic_params["use_field_for_initial_sed_classes"]:
            self.grains_wt[:] = ic_params["init_grains_weight"]
        #self.sed[:] = np.sum(self.grains_wt, axis=1) / (
        #    (1.0 - params["fluvial"]["sediment_porosity"])
        #    * params["fluvial"]["grav_accel"]
        #    * params["fluvial"]["rho_sed"]
        #)
        #self.rock[self.grid.core_nodes] = (
        #    self.topo[self.grid.core_nodes] - self.sed[self.grid.core_nodes]
        #)

        # Store parameters
        self.uplift_rate = params["baselevel"]["uplift_rate"]

        # Instantiate and initialize components: flow router
        flow_params = params["flow_routing"]
        if isinstance(self.grid, RasterModelGrid):
            self.router = PriorityFloodFlowRouter(
                self.grid,
                surface="topographic__elevation",
                flow_metric=flow_params["flow_metric"],
                update_flow_depressions=flow_params["update_flow_depressions"],
                depression_handler=flow_params["depression_handler"],
                epsilon=flow_params["epsilon"],
                accumulate_flow=True,
                runoff_rate=flow_params["bankfull_runoff_rate"],
            )
        else:
            self.router = FlowAccumulator(
                self.grid,
                runoff_rate=flow_params["bankfull_runoff_rate"],
                depression_finder="DepressionFinderAndRouter",
            )

        # Instantiate and initialize components: fluvial transport, erosion, deposition
        egbe_params = params["fluvial"]
        self.soil_grader = SoilGrading(
            self.grid,
            meansizes=egbe_params["grain_sizes"],
            grains_weight="grains__weight",
            phi=egbe_params["sediment_porosity"],
            soil_density=egbe_params["rho_sed"],
        )

        self.eroder = ExtendedGravelBedrockEroder(
            self.grid,
            intermittency_factor=egbe_params["intermittency_factor"],
            sediment_porosity=egbe_params["sediment_porosity"],
            depth_decay_scale=egbe_params["depth_decay_scale"],
            plucking_coefficient=egbe_params["plucking_coefficient"],
            epsilon=egbe_params["epsilon"],
            abrasion_coefficients=egbe_params["abrasion_coefficients"],
            bedrock_abrasion_coefficient=egbe_params["bedrock_abrasion_coefficient"],
            fractions_from_plucking=egbe_params["fractions_from_plucking"],
            rho_sed=egbe_params["rho_sed"],
            rho_water=egbe_params["rho_water"],
            use_fixed_width=egbe_params["use_fixed_width"],
            fixed_width_coeff=egbe_params["fixed_width_coeff"],
            fixed_width_expt=egbe_params["fixed_width_expt"],
            mannings_n=egbe_params["mannings_n"],
            tau_star_c_median=egbe_params["tau_star_c_median"],
            alpha=egbe_params["alpha"],
            tau_c_bedrock=egbe_params["tau_c_bedrock"],
            d_min=egbe_params["d_min"],
            plucking_by_tools_flag=egbe_params["plucking_by_tools_flag"],
        )

        self.rock = self.grid.at_node["bedrock__elevation"]
        self.sed = self.grid.at_node["soil__depth"]

    def update(self, dt=None):
        """Advance the model by one time step of duration dt."""

        if dt is None:
            dt = self._global_dt
        # print("BL Update here", self.current_time, dt, self.uplift_rate)
        # print(" topo before uplift", self.topo[self.grid.core_nodes])
        dz = self.uplift_rate * dt
        self.topo[self.grid.core_nodes] += dz
        self.rock[self.grid.core_nodes] += dz
        # print(" topo after uplift but before GBE", self.topo[self.grid.core_nodes])
        self.router.run_one_step()
        self.eroder.run_one_step(dt)
        # print(" topo after GBE", self.topo[self.grid.core_nodes])
        self.current_time += dt


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print("Running EgbeLem with parameters in", sys.argv[1])
        params = verify_input_file_and_load_params(sys.argv[1])
    else:
        print("Running EgbeLem with default parameters")
        params = {}
    elem = EgbeLem(params=params)
    elem.run()
