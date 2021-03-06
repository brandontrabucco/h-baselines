"""Environment for training automated vehicles in a mixed-autonomy setting."""
import collections
import numpy as np
import random
import os
from gym.spaces import Box
from copy import deepcopy
from collections import defaultdict
from csv import DictReader
from scipy.optimize import fsolve

from flow.envs.multiagent import MultiEnv
from flow.core.params import InFlows
from flow.controllers import FollowerStopper
from flow.networks import I210SubNetwork

from hbaselines.envs.mixed_autonomy.envs.utils import get_rl_accel
from hbaselines.envs.mixed_autonomy.envs.utils import get_relative_obs
from hbaselines.envs.mixed_autonomy.envs.utils import update_rl_veh
from hbaselines.envs.mixed_autonomy.envs.utils import get_lane
from hbaselines.envs.mixed_autonomy.envs.utils import v_eq_function


BASE_ENV_PARAMS = dict(
    # scaling factor for the AV accelerations, in m/s^2
    max_accel=1,
    # whether to use the follower-stopper controller for the AVs
    use_follower_stopper=False,
    # whether to include a stopping penalty
    stopping_penalty=False,
    # whether to include a regularizing penalty for accelerations by the AVs
    acceleration_penalty=False,
    # number of observation frames to use. Additional frames are provided from
    # previous time steps.
    obs_frames=1,
)

CLOSED_ENV_PARAMS = BASE_ENV_PARAMS.copy()
CLOSED_ENV_PARAMS.update(dict(
    # range for the lengths allowed in the network. If set to None, the ring
    # length is not modified from its initial value.
    ring_length=[220, 270],
))

OPEN_ENV_PARAMS = BASE_ENV_PARAMS.copy()
OPEN_ENV_PARAMS.update(dict(
    # range for the inflows allowed in the network. If set to None, the inflows
    # are not modified from their initial value.
    inflows=[1000, 2000],
    # path to the initialized vehicle states. Cannot be set in addition to the
    # `inflows` term. This feature defines its own inflows.
    warmup_path=None,
    # the AV penetration rate, defining the portion of inflow vehicles that
    # will be automated. If "inflows" is set to None, this is irrelevant.
    rl_penetration=0.1,
    # maximum number of controllable vehicles in the network
    num_rl=5,
    # the interval (in meters) in which automated vehicles are controlled. If
    # set to None, the entire region is controllable.
    control_range=[500, 2500],
))


class AVMultiAgentEnv(MultiEnv):
    """Environment for training automated vehicles in a mixed-autonomy setting.

    Required from env_params:

    * max_accel: scaling factor for the AV accelerations, in m/s^2
    * use_follower_stopper: whether to use the follower-stopper controller for
      the AVs
    * stopping_penalty: whether to include a stopping penalty
    * acceleration_penalty: whether to include a regularizing penalty for
      accelerations by the AVs
    * obs_frames: number of observation frames to use. Additional frames are
      provided from previous time steps.

    States
        The observation consists of the speeds and bumper-to-bumper headways of
        the vehicles immediately preceding and following autonomous vehicle, as
        well as the ego speed of the autonomous vehicles.

    Actions
        The action space consists of a bounded acceleration for each autonomous
        vehicle. In order to ensure safety, these actions are bounded by
        failsafes provided by the simulator at every time step.

    Rewards
        The reward provided by the environment is equal to the negative vector
        normal of the distance between the speed of all vehicles in the network
        and a desired speed, and is offset by largest possible negative term to
        ensure non-negativity if environments terminate prematurely. This
        reward may only include two penalties:

        * acceleration_penalty: If set to True in env_params, the negative of
          the sum of squares of the accelerations by the AVs is added to the
          reward.
        * stopping_penalty: If set to True in env_params, a penalty of -5 is
          added to the reward for every RL vehicle that is not moving.

    Termination
        A rollout is terminated if the time horizon is reached or if two
        vehicles collide into one another.

    Attributes
    ----------
    leader : list of str
        the names of the vehicles leading the RL vehicles at any given step.
        Used for visualization.
    follower : list of str
        the names of the vehicles following the RL vehicles at any given step.
        Used for visualization.
    num_rl : int
        a fixed term to represent the number of RL vehicles in the network. In
        closed networks, this is the original number of RL vehicles. Otherwise,
        this value is passed via env_params.
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        """See parent class."""
        for p in BASE_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError('Env parameter "{}" not supplied'.format(p))

        super(AVMultiAgentEnv, self).__init__(
            env_params=env_params,
            sim_params=sim_params,
            network=network,
            simulator=simulator,
        )

        # this is stored to be reused during the reset procedure
        self._network_cls = network.__class__
        self._network_name = deepcopy(network.orig_name)
        self._network_net_params = deepcopy(network.net_params)
        self._network_initial_config = deepcopy(network.initial_config)
        self._network_traffic_lights = deepcopy(network.traffic_lights)
        self._network_vehicles = deepcopy(network.vehicles)

        # used for visualization: the vehicles behind and after RL vehicles
        # (ie the observed vehicles) will have a different color
        self.leader = []
        self.follower = []

        self.num_rl = deepcopy(self.initial_vehicles.num_rl_vehicles)
        self._mean_speeds = []
        self._obs_history = defaultdict(list)
        self._obs_frames = env_params.additional_params["obs_frames"]

        # dynamics controller for controlled RL vehicles. Only relevant if
        # "use_follower_stopper" is set to True.
        human_type = "human" if "human" in self.k.vehicle.type_parameters \
            else "human_0"
        self._av_controller = FollowerStopper(
            veh_id="av",
            v_des=30,
            max_accel=1,
            display_warnings=False,
            fail_safe=['obey_speed_limit', 'safe_velocity', 'feasible_accel'],
            car_following_params=self.k.vehicle.type_parameters[human_type][
                "car_following_params"],
        )

    def rl_ids(self):
        """Return the IDs of the currently observed and controlled RL vehicles.

        This is static in closed networks and dynamic in open networks.
        """
        return self.k.vehicle.get_rl_ids()

    @property
    def action_space(self):
        """See class definition."""
        if self.env_params.additional_params["use_follower_stopper"]:
            return Box(
                low=0,
                high=15,
                shape=(1,),
                dtype=np.float32)
        else:
            return Box(
                low=-1,
                high=1,
                shape=(1,),
                dtype=np.float32)

    @property
    def observation_space(self):
        """See class definition."""
        return Box(
            low=-float('inf'),
            high=float('inf'),
            shape=(5 * self._obs_frames,),
            dtype=np.float32)

    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        if self.env_params.additional_params["use_follower_stopper"]:
            for veh_id in rl_actions.keys():
                self._av_controller.veh_id = veh_id
                self._av_controller.v_des = rl_actions[veh_id][0]
                acceleration = self._av_controller.get_action(self)

                # Apply the action via the simulator.
                self.k.vehicle.apply_acceleration(veh_id, acceleration)
        else:
            rl_ids = list(rl_actions.keys())

            acceleration = get_rl_accel(
                accel=[deepcopy(rl_actions[veh_id][0]) for veh_id in rl_ids],
                vel=self.k.vehicle.get_speed(rl_ids),
                max_accel=self.env_params.additional_params["max_accel"],
                dt=self.sim_step,
            )

            # Run the action through the controller, to include failsafe
            # actions.
            for i, veh_id in enumerate(rl_ids):
                acceleration[i] = self.k.vehicle.get_acc_controller(
                    veh_id).get_action(self, acceleration=acceleration[i])

            # Apply the action via the simulator.
            self.k.vehicle.apply_acceleration(
                acc=acceleration, veh_ids=list(rl_actions.keys()))

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        # In case no vehicles were available in the current step, pass an empty
        # reward dict.
        if rl_actions is None:
            return {}

        # Compute the reward.
        return {
            rl_id: self._compute_reward_util(
                rl_actions=rl_actions[rl_id],
                veh_ids=self.k.vehicle.get_ids(),
                rl_ids=[rl_id],
                **kwargs
            )
            for rl_id in rl_actions.keys()
        }

    def _compute_reward_util(self, rl_actions, veh_ids, rl_ids, **kwargs):
        """Compute the reward over a specific list of vehicles.

        Parameters
        ----------
        rl_actions : array_like
            the actions performed by the automated vehicles
        veh_ids : list of str
            the vehicle IDs to compute the network-level rewards over
        rl_ids : list of str
            the vehicle IDs to compute the AV-level penalties over

        Returns
        -------
        float
            the computed reward
        """
        if self.env_params.evaluate or rl_actions is None:
            reward = np.mean(self.k.vehicle.get_speed(veh_ids))
        else:
            params = self.env_params.additional_params
            stopping_penalty = params["stopping_penalty"]
            acceleration_penalty = params["acceleration_penalty"]

            num_vehicles = len(veh_ids)
            vel = np.array(self.k.vehicle.get_speed(veh_ids))
            if any(vel < -100) or kwargs["fail"] or num_vehicles == 0:
                # in case of collisions or an empty network
                reward = 0
            else:
                # =========================================================== #
                # Reward high system-level average speeds.                    #
                # =========================================================== #

                reward_scale = 0.1
                reward = reward_scale * np.mean(vel) ** 2

                # =========================================================== #
                # Penalize stopped RL vehicles.                               #
                # =========================================================== #

                if stopping_penalty:
                    for veh_id in rl_ids:
                        if self.k.vehicle.get_speed(veh_id) <= 1:
                            reward -= 5

                # =========================================================== #
                # Penalize the sum of squares of the AV accelerations.        #
                # =========================================================== #

                if acceleration_penalty:
                    accel = [self.k.vehicle.get_accel(veh_id, True, True) or 0
                             for veh_id in rl_ids]
                    reward -= sum(np.square(accel))

        return reward

    def get_state(self):
        """See class definition."""
        self.leader = []
        self.follower = []

        for veh_id in self.k.vehicle.get_rl_ids():
            # Add relative observation of each vehicle.
            obs_vehicle, leader, follower = get_relative_obs(self, veh_id)
            self._obs_history[veh_id].append(obs_vehicle)

            # Maintain queue length.
            if len(self._obs_history[veh_id]) > self._obs_frames:
                self._obs_history[veh_id] = \
                    self._obs_history[veh_id][self._obs_frames:]

            # Append to the leader/follower lists.
            if veh_id in self.rl_ids():
                if leader not in ["", None]:
                    self.leader.append(leader)
                if follower not in ["", None]:
                    self.follower.append(follower)

        # Remove memory for exited vehicles.
        for key in self._obs_history.keys():
            if key not in self.k.vehicle.get_rl_ids():
                del self._obs_history[key]

        # Initialize a set on empty observations
        obs = {key: None for key in self.rl_ids()}

        for i, veh_id in enumerate(self.rl_ids()):
            # Concatenate the past n samples for a given time delta and return
            # as the final observation.
            obs_t = np.concatenate(self._obs_history[veh_id][::-1])
            obs_vehicle = np.array([0. for _ in range(5 * self._obs_frames)])
            obs_vehicle[:len(obs_t)] = obs_t

            obs[veh_id] = obs_vehicle

        return obs

    def additional_command(self):
        """See parent class.

        Define which vehicles are observed for visualization purposes.
        """
        # specify observed vehicles
        for veh_id in self.leader + self.follower:
            self.k.vehicle.set_observed(veh_id)

    def step(self, rl_actions):
        """See parent class."""
        obs, rew, done, _ = super(AVMultiAgentEnv, self).step(rl_actions)
        info = {}

        if self.time_counter > \
                self.env_params.warmup_steps * self.env_params.sims_per_step:
            self._mean_speeds.append(np.mean(
                self.k.vehicle.get_speed(self.k.vehicle.get_ids(), error=0)))

            info.update({"speed": np.mean(self._mean_speeds)})

        return obs, rew, done, info

    def reset(self, new_inflow_rate=None):
        """See parent class.

        In addition, a few variables that are specific to this class are
        emptied before they are used by the new rollout.
        """
        self._mean_speeds = []
        self.leader = []
        self.follower = []
        self._obs_history = defaultdict(list)
        return super().reset(new_inflow_rate)


class AVClosedMultiAgentEnv(AVMultiAgentEnv):
    """Closed network variant of AVMultiAgentEnv.

    This environment is suitable for training policies on a ring road.

    We attempt to train a control policy in this setting that is robust to
    changes in density by altering the number of human-driven vehicles within
    the network. The number of automated vehicles, however, are kept constant
    in order to maintain a fixed state/action space. It it worth noting that
    this leads to varying AV penetration rates across simulations.

    Moreover, we ensure that vehicles in the observation/action are sorted by
    their initial position in the network to account for any noise brought
    about by positioning of vehicles after shuffling.

    Required from env_params:

    * max_accel: scaling factor for the AV accelerations, in m/s^2
    * use_follower_stopper: whether to use the follower-stopper controller for
      the AVs
    * stopping_penalty: whether to include a stopping penalty
    * acceleration_penalty: whether to include a regularizing penalty for
      accelerations by the AVs
    * obs_frames: number of observation frames to use. Additional frames are
      provided from previous time steps.
    * ring_length: range for the lengths allowed in the network. If set to
      None, the ring length is not modified from its initial value.
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        """See parent class."""
        for p in CLOSED_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError('Env parameter "{}" not supplied'.format(p))

        super(AVClosedMultiAgentEnv, self).__init__(
            env_params=env_params,
            sim_params=sim_params,
            network=network,
            simulator=simulator,
        )

        # solve for the free flow velocity of the ring
        v_guess = 4
        self._v_eq = fsolve(
            v_eq_function, np.array(v_guess),
            args=(len(self.initial_ids), self.k.network.length()))[0]

        # for storing the distance from the free-flow-speed for a given rollout
        self._percent_v_eq = []

    def step(self, rl_actions):
        """See parent class."""
        obs, rew, done, info = super(AVClosedMultiAgentEnv, self).step(
            rl_actions)

        if self.time_counter > \
                self.env_params.warmup_steps * self.env_params.sims_per_step:
            speed = np.mean(self.k.vehicle.get_speed(self.k.vehicle.get_ids()))
            info.update({"v_eq": self._v_eq})
            info.update({"v_eq_frac": speed / self._v_eq})

        return obs, rew, done, info

    def reset(self, new_inflow_rate=None):
        """See class definition."""
        self._percent_v_eq = []

        params = self.env_params.additional_params
        if params["ring_length"] is not None:
            # Make sure restart instance is set to True when resetting.
            self.sim_params.restart_instance = True

            # Choose the network length randomly.
            length = random.randint(
                params['ring_length'][0], params['ring_length'][1])

            # Add the ring length to NetParams.
            new_net_params = deepcopy(self._network_net_params)
            new_net_params.additional_params["length"] = length

            # Update the network.
            self.network = self._network_cls(
                self._network_name,
                net_params=new_net_params,
                vehicles=self._network_vehicles,
                initial_config=self._network_initial_config,
                traffic_lights=self._network_traffic_lights,
            )

            # solve for the velocity upper bound of the ring
            v_guess = 4
            self._v_eq = fsolve(v_eq_function, np.array(v_guess),
                                args=(len(self.initial_ids), length))[0]

            print('\n-----------------------')
            print('ring length:', self.net_params.additional_params['length'])
            print('v_eq:', self._v_eq)
            print('-----------------------')

        # Perform the reset operation.
        obs = super(AVClosedMultiAgentEnv, self).reset()

        return obs


class AVOpenMultiAgentEnv(AVMultiAgentEnv):
    """Open network variant of AVMultiAgentEnv.

    In this environment, every vehicle is treated as a separate agent. This
    environment is suitable for training policies on a merge or highway
    network.

    We attempt to train a control policy in this setting that is robust to
    changes in density by altering the inflow rate of vehicles within the
    network. This is made to proportionally increase the inflow rate of both
    the human-driven and automated (or RL) vehicles in the network to maintain
    a fixed RL penetration rate.

    Moreover, in order to account for variability in the number of automated
    vehicles during training, we include a "num_rl" term and perform the
    following operations to the states and actions:

    * States: In order to maintain a fixed observation size in open networks,
      when the number of AVs in the network is less than "num_rl", the extra
      entries are filled in with zeros. Conversely, if the number of autonomous
      vehicles is greater than "num_rl", the observations from the additional
      vehicles are not included in the state space.
    * Actions: In order to account for variability in the number of autonomous
      vehicles in open networks, if n_AV < "num_rl" the additional actions
      provided by the agent are not assigned to any vehicle. Moreover, if
      n_AV > "num_rl", the additional vehicles are not provided with actions
      from the learning agent, and instead act as human-driven vehicles as
      well.

    Required from env_params:

    * max_accel: scaling factor for the AV accelerations, in m/s^2
    * use_follower_stopper: whether to use the follower-stopper controller for
      the AVs
    * stopping_penalty: whether to include a stopping penalty
    * acceleration_penalty: whether to include a regularizing penalty for
      accelerations by the AVs
    * obs_frames: number of observation frames to use. Additional frames are
      provided from previous time steps.
    * inflows: range for the inflows allowed in the network. If set to None,
      the inflows are not modified from their initial value.
    * warmup_path: path to the initialized vehicle states. Cannot be set in
      addition to the `inflows` term. This feature defines its own inflows.
    * rl_penetration: the AV penetration rate, defining the portion of inflow
      vehicles that will be automated. If "inflows" is set to None, this is
      irrelevant.
    * num_rl: maximum number of controllable vehicles in the network
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        """See parent class."""
        for p in OPEN_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError('Env parameter "{}" not supplied'.format(p))

        super(AVOpenMultiAgentEnv, self).__init__(
            env_params=env_params,
            sim_params=sim_params,
            network=network,
            simulator=simulator,
        )

        # Get the paths to all the initial state xml files
        warmup_path = env_params.additional_params["warmup_path"]
        if warmup_path is not None:
            self.warmup_paths = [
                f for f in os.listdir(warmup_path) if f.endswith(".xml")
            ]
            self.warmup_description = defaultdict(list)
            for record in DictReader(
                    open(os.path.join(warmup_path, 'description.csv'))):
                for key, val in record.items():  # or iteritems in Python 2
                    self.warmup_description[key].append(float(val))
        else:
            self.warmup_paths = None
            self.warmup_description = None

        # maximum number of controlled vehicles
        self.num_rl = env_params.additional_params["num_rl"]

        # queue of rl vehicles waiting to be controlled
        self.rl_queue = collections.deque()

        # names of the rl vehicles controlled at any step
        self.rl_veh = []

        # names of the rl vehicles past the control range
        self.removed_veh = []

        # control range, updated to be entire network if not specified
        self._control_range = \
            self.env_params.additional_params["control_range"] or \
            [0, self.k.network.length()]

        # dynamics controller for uncontrolled RL vehicles (mimics humans)
        controller = self.k.vehicle.type_parameters["human"][
            "acceleration_controller"]
        self._rl_controller = controller[0](
            veh_id="rl",
            car_following_params=self.k.vehicle.type_parameters["human"][
                "car_following_params"],
            **controller[1]
        )

        if isinstance(network, I210SubNetwork):
            # the name of the final edge, whose speed limit may be updated
            self._final_edge = "119257908#3"
            # maximum number of lanes to add vehicles across
            self._num_lanes = 5
        else:
            # the name of the final edge, whose speed limit may be updated
            self._final_edge = "highway_end"
            # maximum number of lanes to add vehicles across
            self._num_lanes = 1

    def rl_ids(self):
        """See parent class."""
        return self.rl_veh

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        # In case no vehicles were available in the current step, pass an empty
        # reward dict.
        if rl_actions is None:
            return {}

        # Collect the names of the vehicles within the control range.
        control_min = self._control_range[0]
        control_max = self._control_range[1]
        veh_ids = [
            veh_id for veh_id in self.k.vehicle.get_ids() if
            control_min <= self.k.vehicle.get_x_by_id(veh_id) <= control_max
        ]

        # Compute the reward. Penalties are only assigned for the actions of
        # the unique vehicle.
        reward = {
            rl_id: self._compute_reward_util(
                rl_actions=rl_actions[rl_id],
                veh_ids=veh_ids,
                rl_ids=[rl_id],
                **kwargs
            )
            for rl_id in rl_actions.keys()
        }

        # A separate (shared) reward is passed to every agent.
        return reward

    def additional_command(self):
        """See parent class.

        This method performs to auxiliary tasks:

        * Define which vehicles are observed for visualization purposes.
        * Maintains the "rl_veh" and "rl_queue" variables to ensure the RL
          vehicles that are represented in the state space does not change
          until one of the vehicles in the state space leaves the network.
          Then, the next vehicle in the queue is added to the state space and
          provided with actions from the policy.
        """
        super(AVOpenMultiAgentEnv, self).additional_command()

        # Update the RL lists.
        self.rl_queue, self.rl_veh, self.removed_veh = update_rl_veh(
            self,
            rl_queue=self.rl_queue,
            rl_veh=self.rl_veh,
            removed_veh=self.removed_veh,
            control_range=self._control_range,
            num_rl=self.num_rl,
            rl_ids=reversed(sorted(
                self.k.vehicle.get_rl_ids(), key=self.k.vehicle.get_x_by_id)),
        )

        # Specify actions for the uncontrolled RL vehicles based on human-
        # driven dynamics.
        for veh_id in list(
                set(self.k.vehicle.get_rl_ids()) - set(self.rl_veh)):
            self._rl_controller.veh_id = veh_id
            acceleration = self._rl_controller.get_action(self)
            self.k.vehicle.apply_acceleration(veh_id, acceleration)

    def step(self, rl_actions):
        """See parent class."""
        obs, rew, done, info = super(AVOpenMultiAgentEnv, self).step(
            rl_actions)

        if self.time_counter > \
                self.env_params.warmup_steps * self.env_params.sims_per_step:
            # Update the most recent mean speed term to match the speed of the
            # control range.
            kv = self.k.vehicle
            control_range = self._control_range
            veh_ids = [
                veh_id for veh_id in kv.get_ids()
                if control_range[0] < kv.get_x_by_id(veh_id) < control_range[1]
            ]
            self._mean_speeds[-1] = np.mean(kv.get_speed(veh_ids, error=0))

            info.update({"speed": np.mean(self._mean_speeds)})

        return obs, rew, done, info

    def reset(self, new_inflow_rate=None):
        """See class definition."""
        end_speed = None
        params = self.env_params.additional_params
        if params["inflows"] is not None or params["warmup_path"] is not None:
            # Make sure restart instance is set to True when resetting.
            self.sim_params.restart_instance = True

            if self.warmup_paths is not None:
                # Choose a random available xml file.
                xml_file = random.sample(self.warmup_paths, 1)[0]
                xml_num = int(xml_file.split(".")[0])

                # Update the choice of initial conditions.
                self.sim_params.load_state = os.path.join(
                    params["warmup_path"], xml_file)

                # Assign the inflow rate to match the xml number.
                inflow_rate = self.warmup_description["inflow"][xml_num]
                end_speed = self.warmup_description["end_speed"][xml_num]
                print("inflow: {}, end_speed: {}".format(
                    inflow_rate, end_speed))
            else:
                # New inflow rate for human and automated vehicles, randomly
                # assigned based on the inflows variable
                inflow_range = self.env_params.additional_params["inflows"]
                inflow_low = inflow_range[0]
                inflow_high = inflow_range[1]
                inflow_rate = random.randint(inflow_low, inflow_high)

            # Create a new inflow object.
            new_inflow = InFlows()

            for inflow_i in self._network_net_params.inflows.get():
                veh_type = inflow_i["vtype"]
                edge = inflow_i["edge"]
                depart_lane = inflow_i["departLane"]
                depart_speed = inflow_i["departSpeed"]

                # Get the inflow rate of the lane/edge based on whether the
                # vehicle types are human-driven or automated.
                penetration = params["rl_penetration"]
                if veh_type == "human":
                    vehs_per_hour = inflow_rate * (1 - penetration)
                else:
                    vehs_per_hour = inflow_rate * penetration

                new_inflow.add(
                    veh_type=veh_type,
                    edge=edge,
                    vehs_per_hour=vehs_per_hour,
                    depart_lane=depart_lane,
                    depart_speed=depart_speed,
                )

            # Add the new inflows to NetParams.
            new_net_params = deepcopy(self._network_net_params)
            new_net_params.inflows = new_inflow

            # Update the network.
            self.network = self._network_cls(
                self._network_name,
                net_params=new_net_params,
                vehicles=self._network_vehicles,
                initial_config=self._network_initial_config,
                traffic_lights=self._network_traffic_lights,
            )
            self.net_params = new_net_params

        # Clear all AV-related attributes.
        self._clear_attributes()

        _ = super(AVOpenMultiAgentEnv, self).reset()

        # Add automated vehicles.
        if self.warmup_paths is not None:
            self._add_automated_vehicles()

        # Update the end speed, if specified.
        if end_speed is not None:
            self.k.kernel_api.edge.setMaxSpeed(self._final_edge, end_speed)

        # Add the vehicles to their respective attributes.
        self.additional_command()

        # Recompute the initial observation.
        return self.get_state()

    def _clear_attributes(self):
        """Clear all AV-related attributes."""
        self.leader = []
        self.follower = []
        self.rl_veh = []
        self.removed_veh = []
        self.rl_queue = collections.deque()

    def _add_automated_vehicles(self):
        """Replace a portion of vehicles with automated vehicles."""
        penetration = self.env_params.additional_params["rl_penetration"]

        # Sort the initial vehicles by their positions.
        sorted_vehicles = sorted(
            self.k.vehicle.get_ids(),
            key=lambda x: self.k.vehicle.get_x_by_id(x))

        # Replace every nth vehicle with an RL vehicle.
        for lane in range(self._num_lanes):
            sorted_vehicles_lane = [
                veh for veh in sorted_vehicles if get_lane(self, veh) == lane]

            for i, veh_id in enumerate(sorted_vehicles_lane):
                if (i + 1) % int(1 / penetration) == 0:
                    # Don't add vehicles past the control range.
                    pos = self.k.vehicle.get_x_by_id(veh_id)
                    if pos < self._control_range[1]:
                        self.k.vehicle.set_vehicle_type(veh_id, "rl")


class LaneOpenMultiAgentEnv(AVOpenMultiAgentEnv):
    """Lane-level network variant of AVOpenMultiAgentEnv.

    Unlike previous environments in this file, this environment treats every
    lane as a separate agent, with the automated vehicles in any given lane
    being control by a single centralized policy. This environment is designed
    specifically for the I-210 subnetwork.

    Additional descriptions to this task can be founded in its parent classes.

    Required from env_params:

    * max_accel: scaling factor for the AV accelerations, in m/s^2
    * use_follower_stopper: whether to use the follower-stopper controller for
      the AVs
    * stopping_penalty: whether to include a stopping penalty
    * acceleration_penalty: whether to include a regularizing penalty for
      accelerations by the AVs
    * obs_frames: number of observation frames to use. Additional frames are
      provided from previous time steps.
    * inflows: range for the inflows allowed in the network. If set to None,
      the inflows are not modified from their initial value.
    * warmup_path: path to the initialized vehicle states. Cannot be set in
      addition to the `inflows` term. This feature defines its own inflows.
    * rl_penetration: the AV penetration rate, defining the portion of inflow
      vehicles that will be automated. If "inflows" is set to None, this is
      irrelevant.
    * num_rl: maximum number of controllable vehicles in the network
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        """See parent class."""
        super(LaneOpenMultiAgentEnv, self).__init__(
            env_params=env_params,
            sim_params=sim_params,
            network=network,
            simulator=simulator,
        )

        # queue of rl vehicles in each lane that are waiting to be controlled
        self.rl_queue = [collections.deque() for _ in range(self._num_lanes)]

        # names of the rl vehicles in each lane that are controlled at any step
        self.rl_veh = [[] for _ in range(self._num_lanes)]

    @property
    def action_space(self):
        """See class definition."""
        if self.env_params.additional_params["use_follower_stopper"]:
            return Box(
                low=0,
                high=15,
                shape=(self.num_rl,),
                dtype=np.float32)
        else:
            return Box(
                low=-1,
                high=1,
                shape=(self.num_rl,),
                dtype=np.float32)

    @property
    def observation_space(self):
        """See class definition."""
        return Box(
            low=-float('inf'),
            high=float('inf'),
            shape=(5 * self.num_rl,),
            dtype=np.float32)

    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        for key in rl_actions.keys():
            # Get the lane ID.
            lane = int(key.split("_")[-1])

            # Get the acceleration for the given lane.
            acceleration = get_rl_accel(
                accel=deepcopy(rl_actions[key]),
                vel=self.k.vehicle.get_speed(self.rl_ids()[lane]),
                max_accel=self.env_params.additional_params["max_accel"],
                dt=self.sim_step,
            )

            # Apply the actions to the given lane.
            self._apply_per_lane_actions(acceleration, self.rl_ids()[lane])

    def _apply_per_lane_actions(self, rl_actions, veh_ids):
        """Apply accelerations to RL vehicles on a given lane.

        Parameters
        ----------
        rl_actions : array_like
            the actions to be performed on the given lane
        veh_ids : list of str
            the names of the RL vehicles on the given lane
        """
        if self.env_params.additional_params["use_follower_stopper"]:
            accelerations = []
            for i, veh_id in enumerate(veh_ids):
                self._av_controller.veh_id = veh_id
                self._av_controller.v_des = rl_actions[i]
                accelerations.append(self._av_controller.get_action(self))
        else:
            accelerations = get_rl_accel(
                accel=deepcopy(rl_actions),
                vel=self.k.vehicle.get_speed(veh_ids),
                max_accel=self.env_params.additional_params["max_accel"],
                dt=self.sim_step,
            )

            # Run the action through the controller, to include failsafe
            # actions.
            for i, veh_id in enumerate(veh_ids):
                accelerations[i] = self.k.vehicle.get_acc_controller(
                    veh_id).get_action(self, acceleration=accelerations[i])

        # Apply the actions via the simulator.
        self.k.vehicle.apply_acceleration(
            veh_ids,
            accelerations[:len(veh_ids)])

    def get_state(self):
        """See class definition."""
        self.leader = []
        self.follower = []

        for veh_id in self.k.vehicle.get_rl_ids():
            # Add relative observation of each vehicle.
            obs_vehicle, leader, follower = get_relative_obs(self, veh_id)
            self._obs_history[veh_id].append(obs_vehicle)

            # Maintain queue length.
            if len(self._obs_history[veh_id]) > self._obs_frames:
                self._obs_history[veh_id] = \
                    self._obs_history[veh_id][self._obs_frames:]

            # Append to the leader/follower lists.
            if veh_id in self.rl_ids():
                if leader not in ["", None]:
                    self.leader.append(leader)
                if follower not in ["", None]:
                    self.follower.append(follower)

        # Remove memory for exited vehicles.
        for key in self._obs_history.keys():
            if key not in self.k.vehicle.get_rl_ids():
                del self._obs_history[key]

        # Initialize a set on empty observations
        obs = {
            "lane_{}".format(i):
                np.array([0 for _ in range(5*self._obs_frames*self.num_rl)])
            for i in range(self._num_lanes)
        }

        for lane in range(self._num_lanes):
            # Collect the names of the RL vehicles on the lane.
            rl_ids = self.rl_ids()[lane]

            for i, veh_id in enumerate(rl_ids):
                # Concatenate the past n samples for a given time delta in the
                # output observations.
                ob_t = np.concatenate(self._obs_history[veh_id][::-1])
                obs["lane_{}".format(lane)][
                    5*self._obs_frames*i:5*self._obs_frames*i+len(ob_t)] = ob_t

        return obs

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        # In case no vehicles were available in the current step, pass an empty
        # reward dict.
        if rl_actions is None:
            return {}

        reward = {}

        # Collect the names of the vehicles within the control range.
        control_min = self._control_range[0]
        control_max = self._control_range[1]
        veh_ids = [
            veh_id for veh_id in self.k.vehicle.get_ids() if
            control_min <= self.k.vehicle.get_x_by_id(veh_id) <= control_max
        ]

        for lane in range(self._num_lanes):
            # Collect the names of all vehicles on the given lane, while
            # taking into account edges with an extra lane.
            veh_ids_lane = [v for v in veh_ids if get_lane(self, v) == lane]

            # Collect the names of the RL vehicles on the lane.
            rl_ids_lane = [
                veh for veh in self.rl_ids()[lane] if veh in veh_ids_lane]

            # Collect the actions that just correspond to this lane.
            rl_actions_lane = rl_actions["lane_{}".format(lane)]

            # Compute the reward for a given lane.
            reward["lane_{}".format(lane)] = self._compute_reward_util(
                rl_actions=rl_actions_lane,
                veh_ids=veh_ids_lane,
                rl_ids=rl_ids_lane,
                **kwargs
            )

        return reward

    def additional_command(self):
        """See parent class.

        Here, the operations are done at a per-lane level.
        """
        for lane in range(self._num_lanes):
            # Collect the names of the RL vehicles on the given lane, while
            # tacking into account edges with an extra lane.
            rl_ids = [veh for veh in self.k.vehicle.get_rl_ids()
                      if get_lane(self, veh) == lane]

            # Update the RL lists.
            self.rl_queue[lane], self.rl_veh[lane], self.removed_veh = \
                update_rl_veh(
                    self,
                    rl_queue=self.rl_queue[lane],
                    rl_veh=self.rl_veh[lane],
                    removed_veh=self.removed_veh,
                    control_range=self._control_range,
                    num_rl=self.num_rl,
                    rl_ids=reversed(sorted(
                        rl_ids, key=self.k.vehicle.get_x_by_id)),
                )

            # Specify actions for the uncontrolled RL vehicles based on human-
            # driven dynamics.
            for veh_id in list(set(rl_ids) - set(self.rl_veh[lane])):
                self._rl_controller.veh_id = veh_id
                acceleration = self._rl_controller.get_action(self)
                self.k.vehicle.apply_acceleration(veh_id, acceleration)

        # Specify observed vehicles.
        for veh_id in self.leader + self.follower:
            self.k.vehicle.set_observed(veh_id)

    def _clear_attributes(self):
        """Clear all AV-related attributes."""
        self.leader = []
        self.follower = []
        self.rl_veh = [[] for _ in range(self._num_lanes)]
        self.removed_veh = []
        self.rl_queue = [collections.deque() for _ in range(self._num_lanes)]
