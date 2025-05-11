import numpy as np
import cvxpy as cp
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper

class CBFController:
    def __init__(self, urdf_path, mesh_dir, joint_limits, workspace_bounds, gamma=5.0):
        self.robot = RobotWrapper.BuildFromURDF(urdf_path, package_dirs=[mesh_dir])
        self.model = self.robot.model
        self.data = self.robot.data

        self.joint_limits = joint_limits  # (theta_min, theta_max)
        self.workspace_bounds = workspace_bounds  # (p_min, p_max)
        self.gamma = gamma

        self.ee_frame = self.model.getFrameId("panda_hand")  # Adjust if needed

    def joint_limit_constraints(self, theta):
        theta_min, theta_max = self.joint_limits
        A_list, b_list = [], []

        for i in range(len(theta)):
            h_lower = theta[i] - theta_min[i]
            h_upper = theta_max[i] - theta[i]

            A_list.append(np.eye(len(theta))[i])      # Lower: +1
            b_list.append(-self.gamma * h_lower)

            A_list.append(-np.eye(len(theta))[i])     # Upper: -1
            b_list.append(-self.gamma * h_upper)

        return A_list, b_list

    def workspace_constraints(self, theta, theta_dot):
        p_min, p_max = self.workspace_bounds

        # Kinematics
        self.robot.forwardKinematics(theta)
        #self.robot.updateGeometryPlacements()
        p = self.data.oMf[self.ee_frame].translation
        J = self.robot.computeFrameJacobian(theta, self.ee_frame)[:3, :]  # 3x9

        A_list, b_list = [], []

        for i in range(3):  # x, y, z
            h_lower = p[i] - p_min[i]
            h_upper = p_max[i] - p[i]

            A_list.append(J[i, :])
            b_list.append(-self.gamma * h_lower)

            A_list.append(-J[i, :])
            b_list.append(-self.gamma * h_upper)

        return A_list, b_list

    def cbf_qp_filter(self, u_rl, A_list, b_list):
        A = np.hstack(A_list) #np.vstack(A_list)[:, :8]
        b = np.array(b_list)
        n = u_rl.shape[0]
        u = cp.Variable(n)
        objective = cp.Minimize(cp.sum_squares(u - u_rl))
        constraints = [A @ u <= b]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.OSQP)

        if u.value is None:
            #print("[Warning] CBF QP infeasible — using fallback.")
            return u_rl
        print("Feasible QP")
        return u.value

    def get_safe_action(self, theta, theta_dot, goal_position, u_rl):
        # Safety constraints
        A_joint, b_joint = self.joint_limit_constraints(theta)
        A_ws, b_ws = self.workspace_constraints(theta, theta_dot)

        # CLF constraint for goal
        A_goal, b_goal = self.goal_clf_constraint(theta, goal_position)

        A_total = A_goal
        b_total = b_goal

        print(A_total)

        return self.cbf_qp_filter(u_rl, A_total, b_total)
    
    def goal_clf_constraint(self, theta, p_goal, c=5.0):
        self.robot.forwardKinematics(theta)
        self.robot.updateGeometryPlacements()
        p = self.data.oMf[self.ee_frame].translation
        J = self.robot.computeFrameJacobian(theta, self.ee_frame)[:3, :]  # 3x9

        print("p: ", p)
        print("p_goal: ", p_goal)

        error = p - p_goal
        V = 0.5 * np.dot(error, error)
        grad_V = error.T @ J  # shape (1, 9)
        
        A = grad_V
        b = -c * V
        return A[:8], b

