from .types import *

from pathlib import Path

from .tasks import ExtendedTask, StdTask

from gymnasium import Env
from dm_control import mujoco
from dm_control.rl import control
from mujoco import viewer

class LeRobot(Env):
    def __init__(
        self,
        mj_xml_path: Optional[Path] = None,
        dm_control_task_desc: Optional[ExtendedTask] = None,
    ):
        default_mj_xml_path = Path(__file__).parent / "models" / "xml" / "so101_tabletop_manipulation_generated.xml"
        mj_xml_path_str = str(mj_xml_path or default_mj_xml_path)
        dm_control_physics = mujoco.Physics.from_xml_path(mj_xml_path_str)
        dm_control_task: ExtendedTask = dm_control_task_desc or StdTask()
        self.dm_control_env = control.Environment(
            dm_control_physics,
            dm_control_task
        )

        self.observation_space = dm_control_task.OBSERVATION_SPACE
        self.action_space = dm_control_task.ACTION_SPACE

        self._window = None

    def step(
        self,
        action: NDArray[float64]
    ) -> StepResult:
        step, reward, discount, observation = self.dm_control_env.step(action)
        info: Dict = dict()
        terminated = trunctuated = False
        return observation, reward, terminated, trunctuated, info

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> ResetResult:
        super().reset(seed=seed)
        time_step = self.dm_control_env.reset()
        dummy_info: Dict = dict()
        return time_step.observation, dummy_info

    def render_to_window(self):
        if self._window is None:
            self._window = viewer.launch_passive(
                self.dm_control_env._physics.model.ptr,
                self.dm_control_env._physics.data.ptr
            )
        else:
            self._window.sync()

    def render(self):
        return self.dm_control_env.physics.render()



