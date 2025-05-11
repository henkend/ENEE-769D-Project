import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config
from robosuite.wrappers import GymWrapper
import matplotlib.pyplot as plt
from alt_agent import *
from alt_safety import SafetyFilter

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

if __name__ == '__main__':

    if not os.path.exists("tmp/td3"):
        os.makedirs("tmp/td3")

    # Create dict to hold options for the environment
    options = {}

    # Get enviornment and robot
    options["env_name"] = "Door"
    options["robots"] = ["Panda"]
    
    # Load the desired controller
    controller_name = "JOINT_VELOCITY"
    arm_controller_config = suite.load_part_controller_config(default_controller=controller_name)
    robot = options["robots"][0] if isinstance(options["robots"], list) else options["robots"]
    options["controller_configs"] = refactor_composite_controller_config(
        arm_controller_config, robot, ["right"]
    )

    # Create the environment
    env = suite.make(
        **options,
        has_renderer = False,
        has_offscreen_renderer = False,
        use_camera_obs = False,
        horizon = 300,
        reward_shaping=True,
        control_freq = 20,
    )

    env = GymWrapper(env)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # TD3 Agent
    agent = TD3Agent(
        input_dim=obs_dim,
        n_actions=act_dim,
        max_action=env.action_space.high[0],
        min_action=env.action_space.low[0],
    )

    # Replay buffer
    buffer = ReplayBuffer(max_size=500_000, input_dim=obs_dim, n_actions=act_dim)

    # Panda joint limits
    joint_limits = {
        "lower": np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]),
        "upper": np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973]),
    }

    safety_filter = SafetyFilter(joint_limits["lower"], joint_limits["upper"])

    # Hyperparameters
    n_episodes = 500
    max_steps = 300
    batch_size = 128
    best_score = 0
    returns = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_return = 0

        for t in range(max_steps):
            act = agent.choose_action(obs)

            # Extract states for safety filtering
            q = obs[14:21]
            dq = obs[35:42]

            hinge_qpos_addr = env.env.sim.model.get_joint_qpos_addr(env.env.door.joints[0])
            hinge_qvel_addr = env.env.sim.model.get_joint_qvel_addr(env.env.door.joints[0])

            theta = env.env.sim.data.qpos[hinge_qpos_addr]
            theta_dot = env.env.sim.data.qvel[hinge_qvel_addr]

            # Apply CBF/CLF filter
            filtered_act = safety_filter.filter(act[:7], q, dq, theta, theta_dot)
            filtered_act = np.hstack((filtered_act, act[7:]))

            next_obs, reward, terminated, truncated, _ = env.step(filtered_act)
            done = terminated or truncated

            buffer.store_transition(obs, filtered_act, reward, next_obs, done)
            agent.update(buffer, batch_size)

            obs = next_obs
            ep_return += reward
            if done:
                break

        if ep_return > best_score or (ep % 10 == 0):
            best_score = ep_return
            print(f"Saving model w/ score {best_score}")
            agent.save_models()

        returns.append(ep_return)
        print(f"Episode {ep + 1}/{n_episodes} — Return: {ep_return:.2f}")

    # Plotting results
    plt.figure(figsize=(10, 5))
    plt.plot(returns, label="Episode Return")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("TD3 Performance on Pendulum-v1")
    plt.grid(True)
    plt.legend()
    plt.show()