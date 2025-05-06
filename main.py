import time
import os
import gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config
from robosuite.wrappers import GymWrapper
from td3_torch import Agent

from networks import CriticNetwork, ActorNetwork
from buffer import ReplayBuffer

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
        has_renderer = False,
        has_offscreen_renderer = False,
        use_camera_obs = False,
        horizon = 300,
        reward_shaping=True,
        control_freq = 20,
    )

    env = GymWrapper(env)

    actor_learning_rate = 0.005
    critic_learning_rate = 0.005
    batch_size = 100
    layer_1_size = 256
    layer_2_size = 128

    agent = Agent(actor_learning_rate=actor_learning_rate, critic_learning_rate=critic_learning_rate, tau=0.005, input_dims=env.observation_space.shape,
                  env=env, n_actions=env.action_space.shape[0], layer1_size=layer_1_size, layer2_size=layer_2_size, batch_size=batch_size,)
    
    n_games = 200
    best_score = 0
    episode_identifier = f"actor_learning_rate={actor_learning_rate} critic_learning_rate={critic_learning_rate} layer_1_size={layer_1_size} layer_2_size={layer_2_size}"

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

            action = agent.choose_action(observation=observation)
            
            next_observation, reward, done, done2, info = env.step(action)

            score += reward
            agent.remember(observation, action, reward, next_observation, done)
            agent.learn()
            observation = next_observation

        if score > best_score:
            print(f"Saving model w/ score {score}")
            best_score = score
            agent.save_models()

        #if (i % 10 == 0):
        #    agent.save_models()

        print(f"Episode: {i} score {score}")