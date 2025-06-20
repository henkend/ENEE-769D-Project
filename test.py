import time
import os
import gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config
from robosuite.wrappers import GymWrapper
from td3_torch import Agent

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

    actor_learning_rate = 0.01
    critic_learning_rate = 0.01
    batch_size = 100 #video is 128
    layer_1_size = 256
    layer_2_size = 128

    agent = Agent(actor_learning_rate=actor_learning_rate, critic_learning_rate=critic_learning_rate, tau=0.01, input_dims=env.observation_space.shape,
                  env=env, n_actions=env.action_space.shape[0], layer1_size=layer_1_size, layer2_size=layer_2_size, batch_size=batch_size,)
    
    n_games = 15
    best_score = 0

    agent.load_models()

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        iter = 0    

        print(f"Starting episode {i}")

        while not done:
            # Sometimes the observation comes as a tuple with 
            # the actual observation as the first element
            if isinstance(observation, tuple):
                observation = observation[0]

            #action = agent.choose_action(observation=observation, validation=True)

            action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05]
            
            print(observation)

            next_observation, reward, done, done2, info = env.step(action)
            env.render()

            print(next_observation)

            score += reward
            observation = next_observation


        print(f"Episode: {i} score {score}")
