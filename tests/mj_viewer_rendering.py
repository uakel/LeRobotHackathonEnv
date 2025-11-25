from time import sleep
from lerobothackathonenv.env import LeRobot
import gymnasium as gym
import lerobothackathonenv as _

env: LeRobot = gym.make("LeRobot-v0")
env.unwrapped.render_to_window()

obs = env.reset()

while True:
    action = env.action_space.sample()
    obs = env.step(action)
    env.unwrapped.render_to_window()
