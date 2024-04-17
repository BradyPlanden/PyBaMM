import pybamm
import numpy as np
import time


# pybamm.set_logging_level("INFO")

inputs = {
    "Current function [A]": 0.222,
    "Separator porosity": 0.3,
}

# Set-up the model
model = pybamm.lithium_ion.DFN()
geometry = model.default_geometry
param = model.default_parameter_values
param.update({key: "[input]" for key in inputs.keys()})
param.process_geometry(geometry)
param.process_model(model)
var = pybamm.standard_spatial_vars
var_pts = {var.x_n: 20, var.x_s: 20, var.x_p: 20, var.r_n: 10, var.r_p: 10}
mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)
disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model)


############
# Use a short time-vector for this example, and declare which variables to track
t_eval = np.linspace(0, 3600, 10)
output_variables = [
    "Voltage [V]",
    "Current [A]",
    "Time [s]",
]

# Create the IDAKLU Solver object
idaklu_solver = pybamm.IDAKLUSolver(
    rtol=1e-8,
    atol=1e-8,
)

# Create the IDAKLU Solver object
idaklu_solver_output_vars = pybamm.IDAKLUSolver(
    rtol=1e-8,
    atol=1e-8,
    output_variables=output_variables,
)

casadi_solver_fast = pybamm.CasadiSolver(
    mode="fast with events",
    rtol=1e-8,
    atol=1e-8,
)

casadi_solver_fast_with_events = pybamm.CasadiSolver(
    mode="fast",
    rtol=1e-8,
    atol=1e-8,
)

casadi_solver = pybamm.CasadiSolver(
    rtol=1e-8,
    atol=1e-8,
)

# Instead, we Jaxify the IDAKLU solver using similar arguments...
ida_jax_solver = idaklu_solver_output_vars.jaxify(
    model,
    t_eval,
)

jax_solver = pybamm.JaxSolver(
    rtol=1e-8,
    atol=1e-8,
    method="BDF",
)

# ... and then obtain a JAX expression for the solve
f = ida_jax_solver.get_jaxpr()
print(f"JAX expression: {f}")

######################
n = 100
x = np.linspace(0.1, 0.5, n)
y = np.linspace(0.2, 0.4, n)

result = np.column_stack((x, y))

######################

start_time = time.time()
for i in result:
    sim = idaklu_solver.solve(
        model,
        t_eval,
        inputs={
            "Current function [A]": i[0],
            "Separator porosity": i[1],
        },
        # calculate_sensitivities=True,
    )
print(f"Mean IDAKLU time per solve: {(time.time() - start_time)/n}")

start_time = time.time()
for i in result:
    sim = idaklu_solver_output_vars.solve(
        model,
        t_eval,
        inputs={
            "Current function [A]": i[0],
            "Separator porosity": i[1],
        },
        # calculate_sensitivities=True,
    )
print(f"Mean IDAKLU w/ output vars time per solve: {(time.time() - start_time)/n}")

start_time = time.time()
for i in result:
    sim = casadi_solver.solve(
        model,
        t_eval,
        inputs={
            "Current function [A]": i[0],
            "Separator porosity": i[1],
        },
        # calculate_sensitivities=True,
    )
print(f"Mean Casadi time per solve: {(time.time() - start_time)/n}")

start_time = time.time()
for i in result:
    sim = casadi_solver_fast.solve(
        model,
        t_eval,
        inputs={
            "Current function [A]": i[0],
            "Separator porosity": i[1],
        },
        # calculate_sensitivities=True,
    )
print(f"Mean Casadi (Fast) time per solve: {(time.time() - start_time)/n}")


start_time = time.time()
for i in result:
    sim = casadi_solver_fast_with_events.solve(
        model,
        t_eval,
        inputs={
            "Current function [A]": i[0],
            "Separator porosity": i[1],
        },
        # calculate_sensitivities=True,
    )
print(f"Mean Casadi (Fast with Events) time per solve: {(time.time() - start_time)/n}")


start_time = time.time()
for i in result:
    # Jax IdaKLU
    data = f(
        t_eval,
        inputs={
            "Current function [A]": i[0],
            "Separator porosity": i[1],
        },
    )
print(f"Mean Jax IdaKLU time per solve: {(time.time() - start_time)/n}")


# Set-up the model for Jax
model = pybamm.lithium_ion.DFN()
model.convert_to_format = "jax"
model.events = []
geometry = model.default_geometry
param = model.default_parameter_values
param.update({key: "[input]" for key in inputs.keys()})
param.process_geometry(geometry)
param.process_model(model)
var = pybamm.standard_spatial_vars
var_pts = {var.x_n: 20, var.x_s: 20, var.x_p: 20, var.r_n: 10, var.r_p: 10}
mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)
disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model)


# Jax
start_time = time.time()
for i in result:
    jax_solver.solve(
        model,
        t_eval,
        inputs={
            "Current function [A]": i[0],
            "Separator porosity": i[1],
        },
        # calculate_sensitivities=True,
    )
print(f"Mean Jax BDF time per solve: {(time.time() - start_time)/n}")

#####################

# j = [
#         {
#             "Current function [A]": y,
#             "Separator porosity": x,
#         } for x in np.linspace(0.2, 0.4, 1000) for y in np.linspace(0.1, 0.5, 1000)
# ]


# start_time = time.time()
# # This is how we would normally perform a solve using IDAKLU
# sim = idaklu_solver.solve(
#     model,
#     t_eval,
#     inputs=j,
#     # calculate_sensitivities=True,
# )
# print(f"Time taken: {time.time() - start_time}")

# start_time = time.time()
# # This is how we would normally perform a solve using CasADi
# sim = casadi_solver.solve(
#     model,
#     t_eval,
#     inputs=j,
#     # calculate_sensitivities=True,
# )
# print(f"Time taken: {time.time() - start_time}")

# start_time = time.time()
# # Jax IdaKLU
# data = f(t_eval, j)
# print(f"Time taken: {time.time() - start_time}")
