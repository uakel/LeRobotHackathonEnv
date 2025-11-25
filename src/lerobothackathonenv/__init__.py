from gymnasium.envs.registration import register

register(
    id="LeRobot-v0",
    entry_point="lerobothackathonenv.env:LeRobot",
)
