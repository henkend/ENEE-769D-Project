from alt_agent import *
import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config
from robosuite.wrappers import GymWrapper
import matplotlib.pyplot as plt

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
        has_renderer = True,
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

    # Load model
    agent.load_models()

    # Hyperparameters
    n_episodes = 25
    max_steps = 300
    batch_size = 128
    best_score = 0
    returns = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_return = 0

        for t in range(max_steps):
            act = agent.choose_action(obs)
            
            next_obs, reward, terminated, truncated, _ = env.step(act)
            env.render()

            done = terminated or truncated
            obs = next_obs
            ep_return += reward

            if done:
                break

        returns.append(ep_return)
        print(f"Episode {ep + 1}/{n_episodes} — Return: {ep_return:.2f}")

    # Plotting results
    plt.figure(figsize=(10, 5))
    plt.plot(returns, label="Episode Return")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("TD3 Performance on Panda Manipulator")
    plt.grid(True)
    plt.legend()
    plt.show()