import sys, os
import numpy as np
from matplotlib import pyplot as plt

def prepare_empty_data_dict(dt, tf, n=None):
    """
    Prepare empty data/trajectory dictionary.
    Used in real time experiments.

    Parameters
    ----------
        dt : float
            timestep in [s]
        tf : float
            final time in [s]

    Returns
    -------
        dict : data dictionary
            all entries are filled with zeros
    """
    if n is None:
        n = int(tf / dt)

    # create 4 empty numpy array, where measured data can be stored
    des_time = np.zeros(n)
    des_pos = np.zeros(n)
    des_vel = np.zeros(n)
    des_tau = np.zeros(n)

    meas_time = np.zeros(n)
    meas_pos = np.zeros(n)
    meas_vel = np.zeros(n)
    meas_tau = np.zeros(n)
    vel_filt = np.zeros(n)

    data_dict = {
        "des_time": des_time,
        "des_pos": des_pos,
        "des_vel": des_vel,
        "des_tau": des_tau,
        "meas_time": meas_time,
        "meas_pos": meas_pos,
        "meas_vel": meas_vel,
        "meas_tau": meas_tau,
        "vel_filt": vel_filt
    }
    return data_dict

def plot_trajectory(T, X, U, save_to=None, show=False):
    fig, ax = plt.subplots(3, 1, figsize=(18, 6), sharex="all")

    ax[0].plot(T, np.asarray(X).T[0], label=r'$\theta$')
    ax[0].set_ylabel("angle [rad]")
    ax[0].legend(loc="best")

    ax[1].plot(T, np.asarray(X).T[1], label=r'$\omega$')
    ax[1].set_ylabel("angular velocity [rad/s]")
    ax[1].legend(loc="best")
    
    ax[2].plot(T, np.asarray(U).flatten(), label=r'$\tau$')
    ax[2].set_xlabel("time [s]")
    ax[2].set_ylabel("input torque [Nm]")
    ax[2].legend(loc="best")

    if save_to is not None:
        plt.savefig(save_to)
    if show:
        plt.show()

def plot_ilqr_trace(cost_trace, redu_ratio_trace, regu_trace):
    fig, ax = plt.subplots(2, 2, figsize=(10, 6))

    ax[0, 0].plot(cost_trace)
    ax[0, 0].set_xlabel('# Iteration')
    ax[0, 0].set_ylabel('Total cost')
    ax[0, 0].set_title('Cost trace')

    delta_opt = (np.array(cost_trace) - cost_trace[-1])
    ax[0, 1].plot(delta_opt)
    ax[0, 1].set_yscale('log')
    ax[0, 1].set_xlabel('# Iteration')
    ax[0, 1].set_ylabel('Optimality gap')
    ax[0, 1].set_title('Convergence plot')

    ax[1, 0].plot(redu_ratio_trace)
    ax[1, 0].set_title('Ratio of actual reduction and expected reduction')
    ax[1, 0].set_ylabel('Reduction ratio')
    ax[1, 0].set_xlabel('# Iteration')

    ax[1, 1].plot(regu_trace)
    ax[1, 1].set_title('Regularization trace')
    ax[1, 1].set_ylabel('Regularization')
    ax[1, 1].set_xlabel('# Iteration')
    plt.tight_layout()

    plt.show()

def save_trajectory(csv_path, data_dict):
    """
    Save trajectory data to csv file.

    Parameters
    ----------
        csv_path: string
            path where a csv file containing the trajectory data
            will be stored
        data_dict : dict
            dictionary containing the trajectory data.
            expected dictionary keys:
                - "des_time"
                - "des_pos"
                - "des_vel"
                - "des_tau"
                - "meas_time"
                - "meas_pos"
                - "meas_vel"
                - "meas_tau"
                - "vel_filt"
    """
    if not os.path.exists(os.path.dirname(csv_path)):
        os.makedirs(os.path.dirname(csv_path))

    data = [
        data_dict["des_time"],
        data_dict["des_pos"],
        data_dict["des_vel"],
        data_dict["des_tau"],
        data_dict["meas_time"],
        data_dict["meas_pos"],
        data_dict["meas_vel"],
        data_dict["meas_tau"]
    ]

    data = np.asarray(data).T

    header = "des_time,des_pos,des_vel,des_tau,meas_time,meas_pos,meas_vel,meas_tau"

    np.savetxt(csv_path,
               data,
               delimiter=',',
               header=header,
               comments="")
    print(f'Saved .csv data to folder {csv_path}')
