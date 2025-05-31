"""
iLQR Calculator
===============

Large parts taken from `Russ Tedrake <https://github.com/RussTedrake/underactuated>`_.
"""

import numpy as np
import pydrake.symbolic as sym


class iLQR_Calculator():
    '''
    Class to calculate an optimal trajectory with an iterative
    linear quadratic regulator (iLQR). This implementation uses the pydrake symbolic library.
    '''
    def __init__(self, n_x=2, n_u=1):
        '''
        Class to calculate an optimal trajectory with an iterative
        linear quadratic regulator (iLQR). This implementation uses the pydrake symbolic library.

        Parameters
        ----------
        n_x : int, default=2
            The size of the state space.
        n_u : int, default=1
            The size of the control space.
        '''
        self.n_x = n_x
        self.n_u = n_u

    def set_start(self, x0):
        '''
        Set the start state for the trajectory.

        Parameters
        ----------
        x0 : array-like
            the start state. Should have the shape of (n_x,)
        '''
        self.x0 = np.asarray(x0)

    def set_discrete_dynamics(self, dynamics_func):
        '''
        Sets the dynamics function for the iLQR calculation.

        Parameters
        ----------
        danamics_func : function
            dynamics_func should be a function with inputs (x, u) and output xd
        '''
        self.discrete_dynamics = dynamics_func

    def _rollout(self, u_trj):
        x_trj = np.zeros((u_trj.shape[0]+1, self.x0.shape[0]))
        x = self.x0
        i = 0
        x_trj[i, :] = x
        for u in u_trj:
            i = i+1
            x = self.discrete_dynamics(x, u)
            x_trj[i, :] = x
        return x_trj

    def set_stage_cost(self, stage_cost_func):
        '''
        Set the stage cost (running cost) for the ilqr optimization.

        Parameters
        ----------
        stage_cost_func : function
            stage_cost_func should be a function with inputs (x, u)
            and output cost
        '''
        self.stage_cost = stage_cost_func

    def set_final_cost(self, final_cost_func):
        '''
        Set the final cost for the ilqr optimization.

        Parameters
        ----------
        final_cost_func : function
            final_cost_func should be a function with inputs x
            and output cost
        '''
        self.final_cost = final_cost_func

    def _cost_trj(self, x_trj, u_trj):
        total = 0.0
        ln = 0.0
        N = x_trj.shape[0]
        for i in range(len(x_trj)-1):
            ln = ln + self.stage_cost(x_trj[i, :], u_trj[i, :]) / N
        lf = self.final_cost(x_trj[i+1, :])
        total = ln + lf
        return total

    def init_derivatives(self):
        """
        Initialize the derivatives of the dynamics.
        """
        self.x_sym = np.array([sym.Variable("x_{}".format(i))
                               for i in range(self.n_x)])
        self.u_sym = np.array([sym.Variable("u_{}".format(i))
                               for i in range(self.n_u)])
        x = self.x_sym
        u = self.u_sym

        l_stage = self.stage_cost(x, u)
        self.l_x = sym.Jacobian([l_stage], x).ravel()
        self.l_u = sym.Jacobian([l_stage], u).ravel()
        self.l_xx = sym.Jacobian(self.l_x, x)
        self.l_ux = sym.Jacobian(self.l_u, x)
        self.l_uu = sym.Jacobian(self.l_u, u)

        l_final = self.final_cost(x)
        self.l_final_x = sym.Jacobian([l_final], x).ravel()
        self.l_final_xx = sym.Jacobian(self.l_final_x, x)

        f = self.discrete_dynamics(x, u)
        self.f_x = sym.Jacobian(f, x)
        self.f_u = sym.Jacobian(f, u)

    def _compute_stage_cost_derivatives(self, x, u):
        env = {self.x_sym[i]: x[i] for i in range(x.shape[0])}
        env.update({self.u_sym[i]: u[i] for i in range(u.shape[0])})

        l_x = sym.Evaluate(self.l_x, env).ravel()
        l_u = sym.Evaluate(self.l_u, env).ravel()
        l_xx = sym.Evaluate(self.l_xx, env)
        l_ux = sym.Evaluate(self.l_ux, env)
        l_uu = sym.Evaluate(self.l_uu, env)

        f_x = sym.Evaluate(self.f_x, env)
        f_u = sym.Evaluate(self.f_u, env)

        return l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u

    def _compute_final_cost_derivatives(self, x):
        env = {self.x_sym[i]: x[i] for i in range(x.shape[0])}
        l_final_x = sym.Evaluate(self.l_final_x, env).ravel()
        l_final_xx = sym.Evaluate(self.l_final_xx, env)
        return l_final_x, l_final_xx

    def _Q_terms(self, l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u, V_x, V_xx):
        Q_x = f_x.T.dot(V_x) + l_x
        Q_u = f_u.T.dot(V_x) + l_u
        Q_xx = l_xx + f_x.T.dot(V_xx.dot(f_x))
        Q_ux = l_ux + f_u.T.dot(V_xx.dot(f_x))
        Q_uu = l_uu + f_u.T.dot(V_xx.dot(f_u))
        return Q_x, Q_u, Q_xx, Q_ux, Q_uu

    def _gains(self, Q_uu, Q_u, Q_ux):
        Q_uu_inv = np.linalg.inv(Q_uu + Q_uu.T)
        k = -2*Q_uu_inv.dot(Q_u.T)
        K = -2*Q_uu_inv.dot(Q_ux)
        return k, K

    def _V_terms(self, Q_x, Q_u, Q_xx, Q_ux, Q_uu, K, k):
        V_x = Q_x + K.T.dot(Q_u) + 0.5*(Q_ux.T.dot(k) + K.T.dot(Q_uu.T.dot(k)))
        V_xx = Q_xx + Q_ux.T.dot(K) + K.T.dot(Q_ux) + K.T.dot(Q_uu.dot(K))
        return V_x, V_xx

    def _expected_cost_reduction(self, Q_u, Q_uu, k):
        return -Q_u.T.dot(k) - 0.5 * k.T.dot(Q_uu.dot(k))

    def _forward_pass(self, x_trj, u_trj, k_trj, K_trj):
        x_trj_new = np.zeros(x_trj.shape)
        x_trj_new[0, :] = x_trj[0, :]
        u_trj_new = np.zeros(u_trj.shape)
        for n in range(u_trj.shape[0]):
            u_trj_new[n, :] = u_trj[n] + k_trj[n] + \
                              K_trj[n].dot((x_trj_new[n] - x_trj[n]))
            x_trj_new[n+1, :] = self.discrete_dynamics(x_trj_new[n],
                                                       u_trj_new[n])
        return x_trj_new, u_trj_new

    def _backward_pass(self, x_trj, u_trj, regu):
        k_trj = np.zeros([u_trj.shape[0], u_trj.shape[1]])
        K_trj = np.zeros([u_trj.shape[0], u_trj.shape[1], x_trj.shape[1]])
        expected_cost_redu = 0
        l_final_x, l_final_xx = self._compute_final_cost_derivatives(x_trj[-1])
        V_x = l_final_x
        V_xx = l_final_xx
        for n in range(u_trj.shape[0]-1, -1, -1):
            l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u = (
                    self._compute_stage_cost_derivatives(x_trj[n], u_trj[n]))
            Q_x, Q_u, Q_xx, Q_ux, Q_uu = self._Q_terms(l_x, l_u, l_xx, l_ux,
                                                       l_uu, f_x, f_u,
                                                       V_x, V_xx)
            # We add regularization to ensure that Q_uu is invertible
            # and nicely conditioned
            Q_uu_regu = Q_uu + np.eye(Q_uu.shape[0])*regu
            k, K = self._gains(Q_uu_regu, Q_u, Q_ux)
            k_trj[n, :] = k
            K_trj[n, :, :] = K
            V_x, V_xx = self._V_terms(Q_x, Q_u, Q_xx, Q_ux, Q_uu, K, k)
            expected_cost_redu += self._expected_cost_reduction(Q_u, Q_uu, k)
        return k_trj, K_trj, expected_cost_redu

    def run_ilqr(self, N=50, init_u_trj=None, init_x_trj=None, shift=False,
                 max_iter=50, break_cost_redu=1e-6, regu_init=100):
        """
        Run the iLQR optimization and receive a optimal trajectory for the
        defined cost function.

        Parameters
        ----------
        N : int, default=50
            The number of waypoints for the trajectory
        init_u_trj : array-like, default=None
            initial guess for the control trajectory
            ignored if None
        init_x_trj : array_like, default=None
            initial guess for the state space trajectory
            ignored if None
        shift : bool, default=False
            whether to shift the initial guess trajectories by one entry
            (delete the first entry and duplicate the last entry)
        max_iter : int, default=50
            optimization iterations the alogrithm makes at every timestep
        break_cost_redu : float, default=1e-6
            cost at which the optimization breaks off early
        regu_init : float, default=100
           initialization value for the regularizer

        Returns
        -------
        x_trj : array-like
            state space trajectory
        u_trj : array-like
            control trajectory
        cost_trace : array-like
            trace of the cost development during the optimization
        regu_trace : array-like
            trace of the regularizer development during the optimization
        redu_ratio_trace : array-like
            trace of ratio of cost_reduction and expected cost reduction
             during the optimization
        redu_trace : array-like
            trace of the cost reduction development during the optimization
        """
        if init_u_trj is not None:
            u_trj = init_u_trj
            if shift:
                u_trj = np.delete(u_trj, 0, axis=0)
                u_trj = np.append(u_trj, [u_trj[-1]], axis=0)
        else:
            u_trj = np.random.randn(N-1, self.n_u)*0.0001

        if init_x_trj is not None:
            x_trj = init_x_trj
            if shift:
                x_trj = np.delete(x_trj, 0, axis=0)
                last_state = [self.discrete_dynamics(x_trj[-1],
                              np.array(u_trj[-1]))]
                x_trj = np.append(x_trj, last_state, axis=0)
        else:
            x_trj = self._rollout(u_trj)

        total_cost = self._cost_trj(x_trj, u_trj)
        regu = regu_init
        max_regu = 10000
        min_regu = 0.01

        # Setup traces
        cost_trace = [total_cost]
        redu_ratio_trace = [1]
        redu_trace = []
        regu_trace = [regu]

        # Run main loop
        for it in range(max_iter):
            # Backward and forward pass
            k_trj, K_trj, expected_cost_redu = self._backward_pass(x_trj,
                                                                   u_trj,
                                                                   regu)
            x_trj_new, u_trj_new = self._forward_pass(x_trj,
                                                      u_trj,
                                                      k_trj,
                                                      K_trj)
            # Evaluate new trajectory
            total_cost = self._cost_trj(x_trj_new, u_trj_new)
            cost_redu = cost_trace[-1] - total_cost
            redu_ratio = cost_redu / abs(expected_cost_redu)
            # Accept or reject iteration
            if cost_redu > 0:
                # Improvement! Accept new trajectories and lower regularization
                redu_ratio_trace.append(redu_ratio)
                cost_trace.append(total_cost)
                x_trj = x_trj_new
                u_trj = u_trj_new
                regu *= 0.7
            else:
                # Reject new trajectories and increase regularization
                regu *= 2.0
                cost_trace.append(cost_trace[-1])
                redu_ratio_trace.append(0)
            regu = min(max(regu, min_regu), max_regu)
            regu_trace.append(regu)
            redu_trace.append(cost_redu)

            # Early termination if expected improvement is small
            if expected_cost_redu <= break_cost_redu:
                break

        return x_trj, u_trj, cost_trace, regu_trace, \
            redu_ratio_trace, redu_trace
