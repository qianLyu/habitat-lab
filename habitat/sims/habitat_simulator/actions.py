#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from typing import Dict

import attr

import habitat_sim
from habitat.core.registry import registry
from habitat.core.simulator import ActionSpaceConfiguration, Config
from habitat.core.utils import Singleton


class _DefaultHabitatSimActions(Enum):
    STOP = 0
    MOVE_FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
    LOOK_UP = 4
    LOOK_DOWN = 5


@attr.s(auto_attribs=True, slots=True)
class HabitatSimActionsSingleton(metaclass=Singleton):
    r"""Implements an extendable Enum for the mapping of action names
    to their integer values.

    This means that new action names can be added, but old action names cannot
    be removed nor can their mapping be altered. This also ensures that all
    actions are always contigously mapped in :py:`[0, len(HabitatSimActions) - 1]`

    This accesible as the global singleton :ref:`HabitatSimActions`
    """

    _known_actions: Dict[str, int] = attr.ib(init=False, factory=dict)

    def __attrs_post_init__(self):
        for action in _DefaultHabitatSimActions:
            self._known_actions[action.name] = action.value

    def extend_action_space(self, name: str) -> int:
        r"""Extends the action space to accomodate a new action with
        the name :p:`name`

        :param name: The name of the new action
        :return: The number the action is registered on

        Usage:

        .. code:: py

            from habitat.sims.habitat_simulator.actions import HabitatSimActions
            HabitatSimActions.extend_action_space("MY_ACTION")
            print(HabitatSimActions.MY_ACTION)
        """
        assert (
            name not in self._known_actions
        ), "Cannot register an action name twice"
        self._known_actions[name] = len(self._known_actions)

        return self._known_actions[name]

    def has_action(self, name: str) -> bool:
        r"""Checks to see if action :p:`name` is already register

        :param name: The name to check
        :return: Whether or not :p:`name` already exists
        """

        return name in self._known_actions

    def __getattr__(self, name):
        return self._known_actions[name]

    def __getitem__(self, name):
        return self._known_actions[name]

    def __len__(self):
        return len(self._known_actions)

    def __iter__(self):
        return iter(self._known_actions)


HabitatSimActions: HabitatSimActionsSingleton = HabitatSimActionsSingleton()


@registry.register_action_space_configuration(name="v0")
class HabitatSimV0ActionSpaceConfiguration(ActionSpaceConfiguration):
    def get(self):
        return {
            HabitatSimActions.STOP: habitat_sim.ActionSpec("stop"),
            HabitatSimActions.MOVE_FORWARD: habitat_sim.ActionSpec(
                "move_forward",
                habitat_sim.ActuationSpec(
                    amount=self.config.FORWARD_STEP_SIZE
                ),
            ),
            HabitatSimActions.TURN_LEFT: habitat_sim.ActionSpec(
                "turn_left",
                habitat_sim.ActuationSpec(amount=self.config.TURN_ANGLE),
            ),
            HabitatSimActions.TURN_RIGHT: habitat_sim.ActionSpec(
                "turn_right",
                habitat_sim.ActuationSpec(amount=self.config.TURN_ANGLE),
            ),
        }


@registry.register_action_space_configuration(name="v1")
class HabitatSimV1ActionSpaceConfiguration(
    HabitatSimV0ActionSpaceConfiguration
):
    def get(self):
        config = super().get()
        new_config = {
            HabitatSimActions.LOOK_UP: habitat_sim.ActionSpec(
                "look_up",
                habitat_sim.ActuationSpec(amount=self.config.TILT_ANGLE),
            ),
            HabitatSimActions.LOOK_DOWN: habitat_sim.ActionSpec(
                "look_down",
                habitat_sim.ActuationSpec(amount=self.config.TILT_ANGLE),
            ),
        }

        config.update(new_config)

        return config


@registry.register_action_space_configuration(name="pyrobotnoisy")
class HabitatSimPyRobotActionSpaceConfiguration(ActionSpaceConfiguration):
    def get(self):
        return {
            HabitatSimActions.STOP: habitat_sim.ActionSpec("stop"),
            HabitatSimActions.MOVE_FORWARD: habitat_sim.ActionSpec(
                "pyrobot_noisy_move_forward",
                habitat_sim.PyRobotNoisyActuationSpec(
                    amount=self.config.FORWARD_STEP_SIZE,
                    robot=self.config.NOISE_MODEL.ROBOT,
                    controller=self.config.NOISE_MODEL.CONTROLLER,
                    noise_multiplier=self.config.NOISE_MODEL.NOISE_MULTIPLIER,
                ),
            ),
            HabitatSimActions.TURN_LEFT: habitat_sim.ActionSpec(
                "pyrobot_noisy_turn_left",
                habitat_sim.PyRobotNoisyActuationSpec(
                    amount=self.config.TURN_ANGLE,
                    robot=self.config.NOISE_MODEL.ROBOT,
                    controller=self.config.NOISE_MODEL.CONTROLLER,
                    noise_multiplier=self.config.NOISE_MODEL.NOISE_MULTIPLIER,
                ),
            ),
            HabitatSimActions.TURN_RIGHT: habitat_sim.ActionSpec(
                "pyrobot_noisy_turn_right",
                habitat_sim.PyRobotNoisyActuationSpec(
                    amount=self.config.TURN_ANGLE,
                    robot=self.config.NOISE_MODEL.ROBOT,
                    controller=self.config.NOISE_MODEL.CONTROLLER,
                    noise_multiplier=self.config.NOISE_MODEL.NOISE_MULTIPLIER,
                ),
            ),
            HabitatSimActions.LOOK_UP: habitat_sim.ActionSpec(
                "look_up",
                habitat_sim.ActuationSpec(amount=self.config.TILT_ANGLE),
            ),
            HabitatSimActions.LOOK_DOWN: habitat_sim.ActionSpec(
                "look_down",
                habitat_sim.ActuationSpec(amount=self.config.TILT_ANGLE),
            ),
            # The perfect actions are needed for the oracle planner
            "_forward": habitat_sim.ActionSpec(
                "move_forward",
                habitat_sim.ActuationSpec(
                    amount=self.config.FORWARD_STEP_SIZE
                ),
            ),
            "_left": habitat_sim.ActionSpec(
                "turn_left",
                habitat_sim.ActuationSpec(amount=self.config.TURN_ANGLE),
            ),
            "_right": habitat_sim.ActionSpec(
                "turn_right",
                habitat_sim.ActuationSpec(amount=self.config.TURN_ANGLE),
            ),
        }

'''
HabitatSimActions.extend_action_space("CONT_CTRL")

@attr.s(auto_attribs=True, slots=True)
class ContCtrlActuationSpec:
    pass

@habitat_sim.registry.register_move_fn(body_action=True)
class NothingAction(habitat_sim.SceneNodeControl):
    def __call__(
        self,
        scene_node: habitat_sim.SceneNode,
        actuation_spec: ContCtrlActuationSpec,
    ):
        pass

@registry.register_action_space_configuration
class ContCtrlSpace(HabitatSimV1ActionSpaceConfiguration):
    def get(self):
        return {
            HabitatSimActions.CONT_CTRL: habitat_sim.ActionSpec("nothing_action"),
        }
# import habitat
from habitat.tasks.nav.nav import SimulatorTaskAction

@registry.register_task_action
class ContCtrl(SimulatorTaskAction):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Icky, but apparently necessary...
        # Assume that there is only one robot in the episode and it is the agent
        self._robo_id_1 = 0

    def _get_uuid(self, *args, **kwargs) -> str:
        return "cont_ctrl"

    def reset(self, task, *args, **kwargs):
        task.is_stop_called = False
        if not self._sim._sim.get_existing_object_ids():
            self._sim._sim.config.sim_cfg.allow_sliding = True
            obj_templates_mgr = self._sim._sim.get_object_template_manager()
            locobot_template_id = obj_templates_mgr.load_object_configs('/nethome/nyokoyama3/habitat-sim/data/objects/locobot_merged')[0]
            self._robo_id_1 = self._sim._sim.add_object(locobot_template_id, self._sim._sim.agents[0].scene_node)
            self._sim._sim.set_object_motion_type(habitat_sim.physics.MotionType.KINEMATIC, 0)

    def step(self, task, *args, **kwargs):
        time_step = 1.

        move_amount = torch.tanh(kwargs['move']).item()
        turn_amount = torch.tanh(kwargs['turn']).item()


        # Scale actions
        move_amount = (move_amount-1.)/2.*0.25
        turn_amount *= np.pi/18.0
        self._sim._sim.config.sim_cfg.allow_sliding = True

        if abs(move_amount) < 0.1*0.25 and abs(turn_amount) < 0.1*np.pi/18.0:
            task.is_stop_called = True
        
        vel_control = habitat_sim.physics.VelocityControl()
        vel_control.controlling_lin_vel = True
        vel_control.controlling_ang_vel = True
        vel_control.lin_vel_is_local = True
        vel_control.ang_vel_is_local = True
        vel_control.linear_velocity = np.array([0.0, 0.0, move_amount])
        vel_control.angular_velocity = np.array([0.0, turn_amount, 0])

        # New code start
        previous_rigid_state = self._sim._sim.get_rigid_state(self._robo_id_1)
        # manually integrate the rigid state
        target_rigid_state = vel_control.integrate_transform(
            time_step, previous_rigid_state
        )

        # snap rigid state to navmesh and set state to object/agent
        end_pos = self._sim._sim.step_filter(
            previous_rigid_state.translation, target_rigid_state.translation
        )
        self._sim._sim.set_translation(end_pos, self._robo_id_1)
        self._sim._sim.set_rotation(target_rigid_state.rotation, self._robo_id_1)

        # Check if a collision occured
        dist_moved_before_filter = (
            target_rigid_state.translation - previous_rigid_state.translation
        ).dot()
        dist_moved_after_filter = (end_pos - previous_rigid_state.translation).dot()

        # NB: There are some cases where ||filter_end - end_pos|| > 0 when a
        # collision _didn't_ happen. One such case is going up stairs.  Instead,
        # we check to see if the the amount moved after the application of the filter
        # is _less_ than the amount moved before the application of the filter
        EPS = 1e-5
        collided = (dist_moved_after_filter + EPS) < dist_moved_before_filter
        # New code end

        ret = self._sim._sim.step_physics(time_step)
        
        return self._sim.step(HabitatSimActions.CONT_MOVE)
'''