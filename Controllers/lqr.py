import numpy as np


class LQR():
    def __init__(self,state_dim:int,
                      control_dim:int,
                      A:np.ndarray | float,
                      B:np.ndarray | float,
                      R:np.ndarray | float,
                      Q:np.ndarray | float,
                      P:np.ndarray | float,
                      N: int,
                 ):
        self.n = state_dim
        self.m = control_dim
        self.A = A
        self.B = B
        self.P = P
        self.Q = Q
        self.R = R
        self.time_horizon = N

        self.feedback_gains = np.zeros((N, control_dim, state_dim))
        self.cost_to_go = np.zeros((N+1, state_dim,state_dim))

    def solve_DARE(self)->None:
        self.F = self.P
        self.cost_to_go[-1,:,:] = self.F

        for k in range(self.time_horizon-1, -1, -1):
        #itteration from T-1 to 0 at 1 interval steps
            M_1 = self.R + self.B.T @ self.F @ self.B
            M_1_inv = np.linalg.inv(M_1)
            self.feedback_gains[k,:,:] = M_1_inv @ self.B.T @ self.F @ self.A
            M_2 = self.F - self.F @ self.B @ M_1_inv @ self.B.T @ self.F
            self.F = self.Q +  self.A.T @ M_2 @ self.A
            self.cost_to_go[k,:,:] = self.F

    def get_lqr_solution(self, state:np.ndarray | int,
                               min_action:np.ndarray | int,
                               max_action:np.ndarray | int,
                         )->np.ndarray:
        self.solve_DARE()
        initial_feedback_gain = self.feedback_gains[0,:,:]
        policy_output = np.clip((initial_feedback_gain @ state), min_action, max_action)
        return -policy_output
