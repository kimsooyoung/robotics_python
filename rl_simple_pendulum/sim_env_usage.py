from pendulum_env import Simulator
from pendulum_env import PendulumPlant

# get the simulator

# Test Env
# torque_limit = 5.0
# mass = 0.57288
# length = 0.5
# damping = 0.10
# gravity = 9.81
# coulomb_fric = 0.0
# inertia = mass*length**2

# Real Values
torque_limit = 30.0
mass = 10.0
length = 0.3
damping = 0.01
gravity = 9.81
coulomb_fric = 0.0
inertia = mass*length**2

pendulum = PendulumPlant(
    mass=mass,
    length=length,
    damping=damping,
    gravity=gravity,
    coulomb_fric=coulomb_fric,
    inertia=inertia,
    torque_limit=torque_limit
)

sim = Simulator(plant=pendulum)

# simulate
x0_sim = [0.0, 0.0]
dt = 0.01
t_final = 10
integrator = "runge_kutta"
tau = 10.0

# Manual torque control
T, X, U = sim.simulate_and_animate(
    t0=0.0,
    x0=x0_sim,
    tf=t_final,
    dt=dt,
    manual_control=True,
    controller=tau,
    integrator=integrator,
    phase_plot=True,
)