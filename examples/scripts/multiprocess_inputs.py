import pybamm

# load model
model = pybamm.lithium_ion.DFN()
# create geometry
geometry = model.default_geometry

# load parameter values and process model and geometry
param = model.default_parameter_values
param["Current function [A]"] = "[input]"
param.process_geometry(geometry)
param.process_model(model)

# set mesh
var = pybamm.standard_spatial_vars
var_pts = {var.x_n: 30, var.x_s: 30, var.x_p: 30, var.r_n: 10, var.r_p: 10}
mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)

# discretise model
disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model)

# solve model
t_eval = [0, 3600]
inputs = [{"Current function [A]": x} for x in range(1, 3)]
solver = pybamm.IDAKLUSolver(atol=1e-6, rtol=1e-3, options={"num_threads": 4})
solution = solver.solve(
    model,
    t_eval,
    inputs,
)

# solve the model at the given time points, passing multiple current values as inputs
