from .types import *

from numpy import clip
from numpy.linalg import norm

from gymnasium import spaces

class StdTask(ExtendedTask):
    ACTION_SPACE = spaces.Box(low=-1, high=1, shape=(6,), dtype=float64)

    RANGE_QPOS = (-3, 3)
    RANGE_QVEL = (-10, 10)
    RANGE_AF = (-3.35, 3.35)
    OBSERVATION_SPACE = spaces.Dict(
        dict(
            qpos=spaces.Box(*RANGE_QPOS,shape=(6,),dtype=float64),
            qvel=spaces.Box(*RANGE_QVEL,shape=(6,),dtype=float64),
            actuator_force=spaces.Box(*RANGE_AF, shape=(6,), dtype=float64),
        )
    )

    def __init__(self, random=None):
        super().__init__(random=random)

    def get_observation(
        self,
        physics: Physics
    ) -> Obs:
        data = physics.data
        obs = dict(
            qpos=clip(data.qpos, *self.RANGE_QPOS),
            qvel=clip(data.qvel, *self.RANGE_QVEL),
            actuator_force=clip(data.actuator_force, *self.RANGE_AF),
        )
        return obs

    def get_reward(
        self,
        physics: Physics
    ) -> float:
        qvel = physics.data.qvel
        return float(norm(qvel))

# Generate more tasks here...
