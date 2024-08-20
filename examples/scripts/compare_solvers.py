import pybamm
import numpy as np
import time


# Create the list of input dicts
n = 1  # Number of solves
inputs = [
    {
        "Current function [A]": x,
        "Separator porosity": y,
    }
    for x, y in zip(np.linspace(0.1, 0.5, n), np.linspace(0.2, 0.4, n))
]

# Solve with gradient:
grad = False
use_jax = True
rtol = 1e-6
atol = 1e-6

if use_jax:
    import jax

    jax.config.update("jax_platform_name", "cpu")


# Function to construct the conventional model and Jax model
def build_model():
    model = pybamm.lithium_ion.DFN()
    geometry = model.default_geometry
    param = model.default_parameter_values
    param.update({key: "[input]" for key in inputs[0].keys()})
    param.process_geometry(geometry)
    param.process_model(model)
    var = pybamm.standard_spatial_vars
    var_pts = {var.x_n: 10, var.x_s: 10, var.x_p: 10, var.r_n: 10, var.r_p: 10}
    mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)

    return model


def build_jax_model():
    # Set-up the model for Jax
    jax_model = pybamm.lithium_ion.DFN()
    jax_model.convert_to_format = "jax"
    jax_model.events = []
    geometry = jax_model.default_geometry
    param = jax_model.default_parameter_values
    param.update({key: "[input]" for key in inputs[0].keys()})
    param.process_geometry(geometry)
    param.process_model(jax_model)
    var = pybamm.standard_spatial_vars
    var_pts = {var.x_n: 10, var.x_s: 10, var.x_p: 10, var.r_n: 10, var.r_p: 10}
    mesh = pybamm.Mesh(geometry, jax_model.default_submesh_types, var_pts)
    disc = pybamm.Discretisation(mesh, jax_model.default_spatial_methods)
    disc.process_model(jax_model)

    return jax_model


# Setup
t_eval = np.linspace(0, 3600, 10)
model = build_model()
jax_model = build_jax_model()

# Add IDAKLU if installed
if pybamm.have_idaklu():
    # Output variables
    output_variables = [
        "Voltage [V]",
        "Current [A]",
        "Time [s]",
    ]

# Add IDAKLU if installed
if pybamm.have_idaklu():
    # Create the IDAKLU Solver object w/o output vars
    idaklu_solver = pybamm.IDAKLUSolver(
        rtol=rtol,
        atol=atol,
    )

    # Create the IDAKLU Solver object
    idaklu_solver_output_vars = pybamm.IDAKLUSolver(
        rtol=rtol,
        atol=atol,
        output_variables=output_variables,
    )

    # If jax, add the idaklu jax solver
    if pybamm.have_jax() and use_jax:
        # Instead, we Jaxify the IDAKLU solver using similar arguments...
        ida_jax_solver = idaklu_solver_output_vars.jaxify(
            model,
            t_eval,
        )
        f = ida_jax_solver.get_jaxpr()
    else:
        ida_jax_solver = None


# Add the Casadi solvers
casadi_solver_fast_with_events = pybamm.CasadiSolver(
    mode="fast with events",
    rtol=rtol,
    atol=atol,
)

casadi_solver_fast = pybamm.CasadiSolver(
    mode="fast",
    rtol=rtol,
    atol=atol,
)

casadi_solver = pybamm.CasadiSolver(
    rtol=rtol,
    atol=atol,
)

# Add the Jax solver
if pybamm.have_jax() and use_jax:
    jax_solver = pybamm.JaxSolver(
        rtol=rtol,
        atol=atol,
        method="BDF",
    )

# ------ Serial solve benchmarks ------ #
if pybamm.have_idaklu():
    start_time = time.time()
    for i in range(n):
        sim = idaklu_solver.solve(
            model,
            t_eval,
            inputs=inputs[i],
            calculate_sensitivities=grad,
        )
        for k in output_variables:
            V = sim[k].data
    print(f"Total IDAKLU time per solve: {(time.time() - start_time)}")

    start_time = time.time()
    for i in range(n):
        sim = idaklu_solver_output_vars.solve(
            model,
            t_eval,
            inputs=inputs[i],
            calculate_sensitivities=grad,
        )
        for k in output_variables:
            V = sim[k].data
    print(f"Total IDAKLU w/ output vars time per solve: {(time.time() - start_time)}")

    if ida_jax_solver is not None:
        start_time = time.time()
        # Jax IdaKLU
        for i in range(n):
            data = f(
                t_eval,
                inputs=inputs[i],
            )
        print(f"Total Jax IDAKLU time per solve: {(time.time() - start_time)}")

start_time = time.time()
for i in range(n):
    sim = casadi_solver.solve(
        model,
        t_eval,
        inputs=inputs[i],
        calculate_sensitivities=grad,
    )
    for k in output_variables:
        V = sim[k].data
print(f"Total Casadi time per solve: {(time.time() - start_time)}")  # 0.110s

start_time = time.time()
for i in range(n):
    sim = casadi_solver_fast.solve(
        model,
        t_eval,
        inputs=inputs[i],
        calculate_sensitivities=grad,
    )
    for k in output_variables:
        V = sim[k].data
print(f"Total Casadi (Fast) time per solve: {(time.time() - start_time)}")  # 0.068s

start_time = time.time()
for i in range(n):
    sim = casadi_solver_fast_with_events.solve(
        model,
        t_eval,
        inputs=inputs[i],
        calculate_sensitivities=grad,
    )
    for k in output_variables:
        V = sim[k].data
print(
    f"Total Casadi (Fast with Events) time per solve: {(time.time() - start_time)}"
)  # 0.0675

if pybamm.have_jax() and use_jax:
    start_time = time.time()
    for i in range(n):
        sim = jax_solver.solve(
            jax_model,
            t_eval,
            inputs=inputs[i],
        )
        for k in output_variables:
            V = sim[k].data
    print(f"Total Jax BDF time per solve: {(time.time() - start_time)}")


# # ------ Multiprocess solve benchmarks ------ #
# # Rebuild model and reconstruct solver
# model = build_model()
# casadi_solver_fast_with_events = pybamm.CasadiSolver(
#     mode="fast with events",
#     rtol=1e-6,
#     atol=1e-6,
# )
#
# # Create the IDAKLU Solver object
# idaklu_solver = pybamm.IDAKLUSolver(
#     rtol=1e-6,
#     atol=1e-6,
# )
#
# # IDAKLU Multi-threaded
# if pybamm.have_idaklu():
#     start_time = time.time()
#     sim = idaklu_solver.solve(
#         model,
#         t_eval,
#         inputs=multi_inputs,
#         calculate_sensitivities=grad,
#     )
#     V = sim["Voltage [V]"].data
#     print(f"Time taken: {time.time() - start_time}")
#
#     # Jax IDAKLU Multi-threaded
#     if pybamm.have_jax() and use_jax:
#         start_time = time.time()
#         sim = ida_jax_solver.solve(
#             model,
#             t_eval,
#             inputs=multi_inputs,
#             calculate_sensitivities=grad,
#         )
#         V = sim["Voltage [V]"].data
#         print(f"Total IDA Jax time per solve: {(time.time() - start_time)}")
#         print(f"Mean IDA Jax time per solve: {(time.time() - start_time)/n}")
#
# # Casadi (Fast with Events) Multi-threaded
# start_time = time.time()
# sim = casadi_solver_fast_with_events.solve(
#     model,
#     t_eval,
#     inputs=multi_inputs,
#     calculate_sensitivities=grad,
# )
# V = sim["Voltage [V]"].data
# print(
#     f"Total Casadi (Fast with Events) time per solve: {(time.time() - start_time)}"
# )  # 192.77s
# print(
#     f"Mean Casadi (Fast with Events) time per solve: {(time.time() - start_time)/n}"
# )  # 0.0385s
#
# # Jax BDF Multi-threaded
# if pybamm.have_jax() and use_jax:
#     start_time = time.time()
#     sim = jax_solver.solve(
#         jax_model,
#         t_eval,
#         inputs=multi_inputs,
#     )
#     V = sim["Voltage [V]"].data
#     print(f"Total Jax BDF time per solve: {(time.time() - start_time)}")
#     print(f"Mean Jax BDF time per solve: {(time.time() - start_time)/n}")
