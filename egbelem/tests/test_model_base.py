import os
import numpy as np
from numpy.testing import assert_equal
from landlab import RasterModelGrid
from landlab.io.native_landlab import load_grid
from landlab.io.netcdf import read_netcdf
from model_base import LandlabModel


def test_grid_write():
    grid = RasterModelGrid((3, 3))
    f1 = grid.add_field("nodefield", np.arange(grid.number_of_nodes), at="node")
    f2 = grid.add_field("linkfield", np.arange(grid.number_of_links), at="link")
    params = {"grid": {"source": "grid_object", "grid_object": grid}}
    model = LandlabModel(params=params)
    model.save_state_grid_format("test", 0, 3)
    g = load_grid("test000.grid")
    assert_equal(g.at_node["nodefield"], f1)
    assert_equal(g.at_link["linkfield"], f2)
    os.remove("test000.grid")


def test_write_functions():
    grid = RasterModelGrid((3, 3))
    f1 = grid.add_field(
        "topographic__elevation", np.arange(grid.number_of_nodes), at="node"
    )
    f2 = grid.add_field("linkfield", np.arange(grid.number_of_links), at="link")
    params = {"grid": {"source": "grid_object", "grid_object": grid}}
    model = LandlabModel(params=params)

    model.save_state_grid_format("test", 0, 3)
    g = load_grid("test000.grid")
    assert_equal(g.at_node["topographic__elevation"], f1)
    assert_equal(g.at_link["linkfield"], f2)
    os.remove("test000.grid")

    model.save_state_vtk_format("test", 0, 3)
    try:
        f = open("test000.vtk", "rt")
        f.close()
    except FileNotFoundError:
        raise
    os.remove("test000.vtk")

    model.save_state_netcdf_format("test", 0, 3)
    g = read_netcdf("test000.nc")
    assert_equal(g.at_node["topographic__elevation"], f1)
    os.remove("test000.nc")


if __name__ == "__main__":
    test_write_functions()
