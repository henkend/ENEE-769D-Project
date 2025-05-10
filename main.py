import time
import os
import gym
import numpy as np
import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config
from robosuite.wrappers import GymWrapper
from td3_torch import Agent
from cbf import CBFController

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

    # Initialize the Agent
    actor_learning_rate = 0.015
    critic_learning_rate = 0.015
    batch_size = 100
    layer_1_size = 256
    layer_2_size = 128

    agent = Agent(actor_learning_rate=actor_learning_rate, critic_learning_rate=critic_learning_rate, tau=0.01, input_dims=env.observation_space.shape,
                  env=env, n_actions=env.action_space.shape[0], layer1_size=layer_1_size, layer2_size=layer_2_size, batch_size=batch_size,)

    agent.load_models()

    # Load robot and controller
    theta_min = np.array([-2.9] * 7)
    theta_max = np.array([2.9] * 7)
    p_min = np.array([0.3, -0.5, 0.05])
    p_max = np.array([0.8, 0.5, 0.6])

    controller = CBFController(
        urdf_path="Panda/panda.urdf",
        mesh_dir="Panda/meshes/",
        joint_limits=(theta_min, theta_max),
        workspace_bounds=(p_min, p_max),
        gamma=5.0
    )

    # Train the model
    n_games = 200
    best_score = 0

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        iter = 0    

        while not done:
            # Sometimes the observation comes as a tuple with 
            # the actual observation as the first element
            if isinstance(observation, tuple):
                observation = observation[0]

            # Extract joint state from observation
            theta = observation[14:21]
            theta_dot = observation[35:44]
            handle_pos = observation[3:6]

            # Sample action u_rl
            u_rl = agent.choose_action(observation=observation, validation=True)

            # Filter with CBF
            u_safe = controller.get_safe_action(theta, theta_dot, handle_pos, u_rl)
            
            # Apply action
            next_observation, reward, done, done2, info = env.step(u_safe)

            # Update score
            score += reward

            # Save current context
            agent.remember(observation, u_safe, reward, next_observation, done)

            # Learn
            agent.learn()

            # Update observation
            observation = next_observation

        if score > best_score or i == (n_games - 1):
            print(f"Saving model w/ score {score}")
            best_score = score
            agent.save_models()

        print(f"{i}, {score}")