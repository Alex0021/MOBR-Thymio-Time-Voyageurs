import math
import numpy as np
from casadi import *
import do_mpc
import matplotlib.pyplot as plt

def nmpc(abs_pos, goal_position, min_x, min_y, max_x, max_y):
    
    # Define the model
    model_type = 'discrete'  # 'continuous' or 'discrete'
    model = do_mpc.model.Model(model_type)

    # Parameters
    d = 9.5

    # State and Output
    x = model.set_variable('_x', 'x')
    y = model.set_variable('_x', 'y')
    theta = model.set_variable('_x', 'theta')
    dx = model.set_variable('_x', 'dx')
    dy = model.set_variable('_x', 'dy')
    dtheta = model.set_variable('_x', 'dtheta')

    u_r = model.set_variable('_u', 'u_r')  # right wheel speed
    u_l = model.set_variable('_u', 'u_l')  # left wheel speed

    # State equations
    x_k1 = x + dx
    y_k1 = y + dy
    theta_k1 = theta - dtheta
    

    # Model constraints
    model.set_rhs('x', x_k1)
    model.set_rhs('y', y_k1)
    model.set_rhs('theta', theta_k1)
    model.set_rhs('dx', ((u_r + u_l) / 2) * cos(theta) )
    model.set_rhs('dy', ((u_r + u_l) / 2) * sin(theta) )
    model.set_rhs('dtheta', (u_r - u_l) / (2 * d))

    # Set up the model
    model.setup()

    # MPC
    mpc = do_mpc.controller.MPC(model)
    # MPC Configuration
    setup_mpc = {
        'n_horizon': 10,
        'n_robust': 0,
        'open_loop': 0,
        't_step': 0.5,
        'state_discretization': 'collocation',
        'collocation_type': 'radau',
        'collocation_deg': 3,
        'collocation_ni': 1,
        'store_full_solution': True,
        'nlpsol_opts': {'ipopt.linear_solver': 'mumps',
                        'ipopt.print_level': 0,
                        'print_time': 0,
                        'ipopt.sb': 'yes'}
    }
    mpc.set_param(**setup_mpc)

    # Goal
    rx=goal_position[0][0]
    ry=goal_position[0][1]

    # Objective function
    mterm = (x-rx)**2 + (y-ry)**2
    lterm = (x-rx)**2 + (y-ry)**2

    # Set the objective for the MPC instance
    mpc.set_objective(mterm=mterm, lterm=lterm)

    # Set rate-of-change term for the MPC instance
    mpc.set_rterm(u_r=1e-2, u_l=1e-2)  # You can adjust the value based on your specific application

    # Constraints bounds on States:
    
    mpc.bounds['upper','_x', 'x'] = 120 #100 map size
    mpc.bounds['lower','_x', 'x'] = -120
    mpc.bounds['upper','_x', 'y'] = 120
    mpc.bounds['lower','_x', 'y'] = -120
    
    
    mpc.bounds['upper','_x', 'theta'] = math.pi
    mpc.bounds['lower','_x', 'theta'] = -math.pi
    
    # Constraints bounds on inputs: (20cm/s)
    mpc.bounds['upper','_u', 'u_r'] = 20
    mpc.bounds['lower','_u', 'u_r'] = -20
    mpc.bounds['upper','_u', 'u_l'] = 20
    mpc.bounds['lower','_u', 'u_l'] = -20
    

    # Setup the MPC instance
    mpc.setup()

    # Set initial guess for the solver
    mpc.set_initial_guess()  # Provide your initial guess for the control inputs u0

    # Simulation
    simulator = do_mpc.simulator.Simulator(model)
    simulator.set_param(t_step=0.5)  # Set the simulation time step

    # Set initial conditions
    simulator.x0['x'] = abs_pos[0][0]
    simulator.x0['y'] = abs_pos[0][1]
    simulator.x0['theta'] = abs_pos[0][2]
    simulator.x0['dx'] = 0.0
    simulator.x0['dy'] = 0.0
    simulator.x0['dtheta'] = 0.0

    # Set parameter function for the simulator
    p_template = simulator.get_p_template()

    def p_fun(t_now):
        return p_template

    simulator.set_p_fun(p_fun)
    simulator.setup()

    # Simulation
    for i in range(10):
        u0 = mpc.make_step(simulator.x0)
        simulator.make_step(u0)

    controller = {var: np.array(simulator.data['_u', var]) for var in model.u.keys()}
    
    # Get futures predicted state
    states = {var: np.array(simulator.data['_x', var]) for var in model.x.keys()}
    
    time_points = np.array(simulator.data['_time'])

    # Unit transformation
    output_r = controller['u_r'][0][0]
    output_l = controller['u_l'][0][0]


    return output_r, output_l, states, time_points, controller