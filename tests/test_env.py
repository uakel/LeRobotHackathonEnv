from lerobothackathonenv.env import LeRobot
import gymnasium as gym
env: LeRobot = gym.make("LeRobot-v0")

def test_reset():
    obs, info = env.reset()
    keys = set(obs.keys())
    assert "qpos" in keys
    assert "qvel" in keys
    assert "actuator_force" in keys
    assert all(o.shape == (6,) for o in obs.values())

def test_rendering():
    from matplotlib.pyplot import imshow, show
    image = env.render()
    imshow(image)
    show()

