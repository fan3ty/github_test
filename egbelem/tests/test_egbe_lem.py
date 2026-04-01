"""
test_egbe_lem.py: external unit tests for EgbeLem and associated code.
"""

import os
import numpy as np
from numpy.testing import assert_equal, assert_almost_equal
import pytest
from landlab import RasterModelGrid
from egbelem import EgbeLem


def test_run_with_default_params_cl():
    os.system("python ../src/egbelem/egbelem.py")
    os.system("python ../src/egbelem/egbelem.py ../src/egbelem/egbedefaults.yaml")


def test_run_with_default_params():
    model = EgbeLem()
    model.run()
    model = EgbeLem(input_file="../src/egbelem/egbedefaults.yaml")
    model.run()


def test_run_pause_continue():
    model = EgbeLem()
    model.update_until(200.0, dt=100.0)
    assert_equal(model.current_time, 200.0)
    model.update_until(400.0, dt=100.0)
    assert_equal(model.current_time, 400.0)


def test_combo_specified_and_default_params():
    model = EgbeLem(params={"clock": {"stop": 400.0}})
    model.run()
    assert_equal(model.current_time, 400.0)
    model = EgbeLem(input_file="egbe_one_param.yaml")
    model.run()
    assert_equal(model.current_time, 300.0)
    assert_equal(model.uplift_rate, model.DEFAULT_PARAMS["baselevel"]["uplift_rate"])


def test_with_hex_grid():
    params = {
        "grid": {
            "source": "create",
            "create_grid": {
                "HexModelGrid": [
                    (3, 3),
                    {"spacing": 1000.0},
                ],
            },
        },
    }
    model = EgbeLem(params=params)
    model.run()


def test_pass_fields():
    frac_from_pluck = np.ones((12, 1))
    frac_from_pluck[:6, 0] = 0.1
    params = {
        "grid": {
            "source": "create",
            "create_grid": {
                "RasterModelGrid": [
                    (4, 3),
                    {"xy_spacing": 1000.0},
                ],
            },
        },
        "baselevel": {"uplift_rate": np.array([1, 2])},  # 2 core nodes
        "fluvial": {
            "plucking_coefficient": 1.0e-4
            * np.array([10, 10, 10, 10, 10, 10, 1, 1, 1, 1, 1, 1]),
            "bedrock_abrasion_coefficient": 1.0e-5
            * np.array([10, 10, 10, 10, 10, 10, 1, 1, 1, 1, 1, 1]),
            "fractions_from_plucking": frac_from_pluck,
        },
    }
    model = EgbeLem(params=params)
    model.run()


def test_run_with_predefined_grid_and_fields():

    TOPO = 4.4
    SOIL = 1.1
    grid = RasterModelGrid((3, 3), xy_spacing=100.0)
    elev = grid.add_zeros("topographic__elevation", at="node")
    elev[4] = TOPO
    gw = grid.add_field(
        "grains__weight", np.zeros((grid.number_of_nodes, 1)), at="node"
    )
    gw[4] = (
        SOIL
        * EgbeLem.DEFAULT_PARAMS["fluvial"]["rho_sed"]
        * (1.0 - EgbeLem.DEFAULT_PARAMS["fluvial"]["sediment_porosity"])
    )
    rock = grid.add_zeros("bedrock__elevation", at="node")
    rock[:] = TOPO - SOIL

    params = {
        "grid": {"source": "grid_object", "grid_object": grid},
        "clock": {"stop": 100.0},
        "initial_conditions": {
            "use_field_for_initial_topo": True,
            "use_field_for_initial_sed_classes": True,
        },
    }

    model = EgbeLem(params=params)
    unitmass = EgbeLem.DEFAULT_PARAMS["fluvial"]["rho_sed"] * (
        1.0 - EgbeLem.DEFAULT_PARAMS["fluvial"]["sediment_porosity"]
    )
    assert_equal(model.grid.at_node["soil__depth"][4], 1.1)
    assert_equal(model.grid.at_node["topographic__elevation"][4], 4.4)
    assert_almost_equal(model.grid.at_node["grains__weight"][4], 1.1 * unitmass)
    assert_almost_equal(model.grid.at_node["bedrock__elevation"][4], 3.3)
    model.run()


def test_run_with_predefined_arrays_from_files():

    params = {}
    params["fluvial"] = {}
    params["grid"] = {
        "source": "create",
        "create_grid": {
            "RasterModelGrid": [
                (3, 3),
                {"xy_spacing": 1000.0},
            ],
        },
    }

    np.save("testpc", 1.0e-4 * np.arange(9))
    params["fluvial"]["plucking_coefficient"] = {"_filepath": "testpc.npy"}
    np.save("testffp", 0.1 * np.arange(9).reshape((9, 1)))
    params["fluvial"]["fractions_from_plucking"] = {"_filepath": "testffp.npy"}
    np.save("testbac", 1.0e-4 * np.arange(9))
    params["fluvial"]["bedrock_abrasion_coefficient"] = {"_filepath": "testbac.npy"}

    model = EgbeLem(params=params)
    assert_equal(model.eroder._br_abr_coef, 1.0e-4 * np.arange(9))
    assert_equal(model.eroder._plucking_coef, 1.0e-4 * np.arange(9))
    assert_equal(
        model.eroder._fractions_from_plucking, 0.1 * np.arange(9).reshape((9, 1))
    )
    model.run()


def test_input_source_error():
    with pytest.raises(TypeError):
        EgbeLem(params=None)
        EgbeLem(input_file=None)


if __name__ == "__main__":
    # test_run_with_predefined_arrays_from_files()
    # test_input_source_error()
    # test_run_with_predefined_grid_and_fields()
    # test_pass_fields()
    # test_with_hex_grid()
    test_combo_specified_and_default_params()
    # test_run_with_default_params_cl()
    # test_run_with_default_params()
    # test_run_pause_continue()
