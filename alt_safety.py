import numpy as np
import cvxpy as cp

class SafetyFilter:
    def __init__(self, joint_lower, joint_upper, theta_target=1.0, gamma=5.0, epsilon=5.0):
        self.joint_lower = joint_lower
        self.joint_upper = joint_upper
        self.theta_target = theta_target
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_joints = len(joint_lower)

    def compute_cbf_constraints(self, q, dq):
        h = (self.joint_upper - q) * (q - self.joint_lower)
        dh = (self.joint_upper - 2 * q + self.joint_lower) * dq
        return dh + self.gamma * h  # Should be >= 0

    def compute_clf_constraint(self, theta, theta_dot):
        V = (theta - self.theta_target)**2
        V_dot = 2 * (theta - self.theta_target) * theta_dot
        return -V_dot - self.epsilon * V  # Should be >= 0

    def filter(self, u_rl, q, dq, theta, theta_dot):
        u = cp.Variable(self.n_joints)
        objective = cp.Minimize(cp.sum_squares(u - u_rl))

        cbf_ineq = self.compute_cbf_constraints(q, dq)
        clf_ineq = self.compute_clf_constraint(theta, theta_dot)

        constraints = [cbf_ineq[i] + 0 * u[i] >= 0 for i in range(self.n_joints)]
        constraints += [clf_ineq + 0 * cp.sum(u) >= 0]

        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.OSQP)
            return u.value if u.value is not None else u_rl
        except:
            return u_rl
