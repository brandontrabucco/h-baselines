"""Base goal-conditioned hierarchical policy."""
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from copy import deepcopy
import random
from functools import reduce

from hbaselines.fcnet.base import ActorCriticPolicy
from hbaselines.goal_conditioned.replay_buffer import HierReplayBuffer
from hbaselines.utils.reward_fns import negative_distance
from hbaselines.utils.misc import get_manager_ac_space, get_goal_indices
from hbaselines.utils.tf_util import get_trainable_vars
from hbaselines.utils.tf_util import reduce_std


class GoalConditionedPolicy(ActorCriticPolicy):
    r"""Goal-conditioned hierarchical reinforcement learning model.

    This policy is an implementation of the two-level hierarchy presented
    in [1], which itself is similar to the feudal networks formulation [2, 3].
    This network consists of a high-level, or Manager, pi_{\theta_H} that
    computes and outputs goals g_t ~ pi_{\theta_H}(s_t, h) every `meta_period`
    time steps, and a low-level policy pi_{\theta_L} that takes as inputs the
    current state and the assigned goals and attempts to perform an action
    a_t ~ pi_{\theta_L}(s_t,g_t) that satisfies these goals.

    The Manager is rewarded based on the original environment reward function:
    r_H = r(s,a;h).

    The Target term, h, parameterizes the reward assigned to the Manager in
    order to allow the policy to generalize to several goals within a task, a
    technique that was first proposed by [4].

    Finally, the Worker is motivated to follow the goals set by the Manager via
    an intrinsic reward based on the distance between the current observation
    and the goal observation:
    r_L (s_t, g_t, s_{t+1}) = -||s_t + g_t - s_{t+1}||_2

    Bibliography:

    [1] Nachum, Ofir, et al. "Data-efficient hierarchical reinforcement
        learning." Advances in Neural Information Processing Systems. 2018.
    [2] Dayan, Peter, and Geoffrey E. Hinton. "Feudal reinforcement learning."
        Advances in neural information processing systems. 1993.
    [3] Vezhnevets, Alexander Sasha, et al. "Feudal networks for hierarchical
        reinforcement learning." Proceedings of the 34th International
        Conference on Machine Learning-Volume 70. JMLR. org, 2017.
    [4] Schaul, Tom, et al. "Universal value function approximators."
        International Conference on Machine Learning. 2015.

    Attributes
    ----------
    manager : hbaselines.fcnet.base.ActorCriticPolicy
        the manager policy
    meta_period : int
        manger action period
    worker_reward_scale : float
        the value the intrinsic (Worker) reward should be scaled by
    relative_goals : bool
        specifies whether the goal issued by the Manager is meant to be a
        relative or absolute goal, i.e. specific state or change in state
    off_policy_corrections : bool
        whether to use off-policy corrections during the update procedure. See:
        https://arxiv.org/abs/1805.08296.
    hindsight : bool
        whether to use hindsight action and goal transitions, as well as
        subgoal testing. See: https://arxiv.org/abs/1712.00948
    subgoal_testing_rate : float
        rate at which the original (non-hindsight) sample is stored in the
        replay buffer as well. Used only if `hindsight` is set to True.
    connected_gradients : bool
        whether to connect the graph between the manager and worker
    cg_weights : float
        weights for the gradients of the loss of the worker with respect to the
        parameters of the manager. Only used if `connected_gradients` is set to
        True.
    multistep_llp : bool
        whether to use the multi-step LLP update procedure. See: TODO
    num_ensembles : int
        number of ensemble models for the Worker dynamics
    num_particles : int
        number of particles used to generate the forward estimate of the model.
        See: TODO
    use_fingerprints : bool
        specifies whether to add a time-dependent fingerprint to the
        observations
    fingerprint_range : (list of float, list of float)
        the low and high values for each fingerprint element, if they are being
        used
    fingerprint_dim : tuple of int
        the shape of the fingerprint elements, if they are being used
    centralized_value_functions : bool
        specifies whether to use centralized value functions for the Manager
        critic functions
    prev_meta_obs : array_like
        previous observation by the Manager
    meta_action : array_like
        current action by the Manager
    meta_reward : float
        current meta reward, counting as the cumulative environment reward
        during the meta period
    batch_size : int
        SGD batch size
    worker : hbaselines.fcnet.base.ActorCriticPolicy
        the worker policy
    worker_reward_fn : function
        reward function for the worker
    """

    def __init__(self,
                 sess,
                 ob_space,
                 ac_space,
                 co_space,
                 buffer_size,
                 batch_size,
                 actor_lr,
                 critic_lr,
                 verbose,
                 tau,
                 gamma,
                 layer_norm,
                 layers,
                 act_fun,
                 use_huber,
                 meta_period,
                 worker_reward_scale,
                 relative_goals,
                 off_policy_corrections,
                 hindsight,
                 subgoal_testing_rate,
                 connected_gradients,
                 cg_weights,
                 multistep_llp,
                 use_sample_not_mean,
                 max_rollout_using_model,
                 max_rollout_when_training,
                 worker_dynamics_bptt_lr,
                 worker_model_bptt_lr,
                 add_final_q_value,
                 num_ensembles,
                 num_particles,
                 use_fingerprints,
                 fingerprint_range,
                 centralized_value_functions,
                 env_name="",
                 meta_policy=None,
                 worker_policy=None,
                 additional_params=None):
        """Instantiate the goal-conditioned hierarchical policy.

        Parameters
        ----------
        sess : tf.compat.v1.Session
            the current TensorFlow session
        ob_space : gym.spaces.*
            the observation space of the environment
        ac_space : gym.spaces.*
            the action space of the environment
        co_space : gym.spaces.*
            the context space of the environment
        buffer_size : int
            the max number of transitions to store
        batch_size : int
            SGD batch size
        actor_lr : float
            actor learning rate
        critic_lr : float
            critic learning rate
        verbose : int
            the verbosity level: 0 none, 1 training information, 2 tensorflow
            debug
        tau : float
            target update rate
        gamma : float
            discount factor
        layer_norm : bool
            enable layer normalisation
        layers : list of int or None
            the size of the neural network for the policy
        act_fun : tf.nn.*
            the activation function to use in the neural network
        use_huber : bool
            specifies whether to use the huber distance function as the loss
            for the critic. If set to False, the mean-squared error metric is
            used instead
        meta_period : int
            manger action period
        worker_reward_scale : float
            the value the intrinsic (Worker) reward should be scaled by
        relative_goals : bool
            specifies whether the goal issued by the Manager is meant to be a
            relative or absolute goal, i.e. specific state or change in state
        off_policy_corrections : bool
            whether to use off-policy corrections during the update procedure.
            See: https://arxiv.org/abs/1805.08296
        hindsight : bool
            whether to include hindsight action and goal transitions in the
            replay buffer. See: https://arxiv.org/abs/1712.00948
        subgoal_testing_rate : float
            rate at which the original (non-hindsight) sample is stored in the
            replay buffer as well. Used only if `hindsight` is set to True.
        connected_gradients : bool
            whether to connect the graph between the manager and worker
        cg_weights : float
            weights for the gradients of the loss of the worker with respect to
            the parameters of the manager. Only used if `connected_gradients`
            is set to True.
        multistep_llp : bool
            whether to use the multi-step LLP update procedure. See: TODO
        use_sample_not_mean : bool
            whether to use samples for predicting future states
        max_rollout_using_model : int
            the number of states into the future to predict using a model
        max_rollout_when_training : int
            the number of states into the future to predict using a model
            when training the model
        worker_dynamics_bptt_lr : float
            dynamics learning rate
        worker_model_bptt_lr : float
            actor learning rate
        add_final_q_value: bool
            whether to add the final q value to the actor loss
        num_ensembles : int
            number of ensemble models for the Worker dynamics
        num_particles : int
            number of particles used to generate the forward estimate of the
            model. See: TODO
        use_fingerprints : bool
            specifies whether to add a time-dependent fingerprint to the
            observations
        fingerprint_range : (list of float, list of float)
            the low and high values for each fingerprint element, if they are
            being used
        centralized_value_functions : bool
            specifies whether to use centralized value functions for the
            Manager and Worker critic functions
        meta_policy : type [ hbaselines.fcnet.base.ActorCriticPolicy ]
            the policy model to use for the Manager
        worker_policy : type [ hbaselines.fcnet.base.ActorCriticPolicy ]
            the policy model to use for the Worker
        additional_params : dict
            additional algorithm-specific policy parameters. Used internally by
            the class when instantiating other (child) policies.
        """
        super(GoalConditionedPolicy, self).__init__(
            sess=sess,
            ob_space=ob_space,
            ac_space=ac_space,
            co_space=co_space,
            buffer_size=buffer_size,
            batch_size=batch_size,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            verbose=verbose,
            tau=tau,
            gamma=gamma,
            layer_norm=layer_norm,
            layers=layers,
            act_fun=act_fun,
            use_huber=use_huber
        )

        self.meta_period = meta_period
        self.worker_reward_scale = worker_reward_scale
        self.relative_goals = relative_goals
        self.off_policy_corrections = off_policy_corrections
        self.hindsight = hindsight
        self.subgoal_testing_rate = subgoal_testing_rate
        self.connected_gradients = connected_gradients
        self.cg_weights = cg_weights
        self.multistep_llp = multistep_llp
        self.use_sample_not_mean = use_sample_not_mean
        self.max_rollout_using_model = max_rollout_using_model
        self.max_rollout_when_training = max_rollout_when_training
        self.worker_dynamics_bptt_lr = worker_dynamics_bptt_lr
        self.worker_model_bptt_lr = worker_model_bptt_lr
        self.add_final_q_value = add_final_q_value
        self.num_ensembles = num_ensembles
        self.num_particles = num_particles
        self.use_fingerprints = use_fingerprints
        self.fingerprint_range = fingerprint_range
        self.fingerprint_dim = (len(self.fingerprint_range[0]),)
        self.centralized_value_functions = centralized_value_functions

        # Get the Manager's action space.
        manager_ac_space = get_manager_ac_space(
            ob_space, relative_goals, env_name,
            use_fingerprints, self.fingerprint_dim)

        # Manager observation size
        meta_ob_dim = self._get_ob_dim(ob_space, co_space)

        # Create the replay buffer.
        self.replay_buffer = HierReplayBuffer(
            buffer_size=int(buffer_size/meta_period),
            batch_size=batch_size,
            meta_period=meta_period,
            meta_obs_dim=meta_ob_dim[0],
            meta_ac_dim=manager_ac_space.shape[0],
            worker_obs_dim=ob_space.shape[0] + manager_ac_space.shape[0],
            worker_ac_dim=ac_space.shape[0],
        )

        # =================================================================== #
        # Part 1. Setup the Manager                                           #
        # =================================================================== #

        # Create the Manager policy.
        with tf.compat.v1.variable_scope("Manager"):
            self.manager = meta_policy(
                sess=sess,
                ob_space=ob_space,
                ac_space=manager_ac_space,
                co_space=co_space,
                buffer_size=buffer_size,
                batch_size=batch_size,
                actor_lr=actor_lr,
                critic_lr=critic_lr,
                verbose=verbose,
                tau=tau,
                gamma=gamma,
                layer_norm=layer_norm,
                layers=layers,
                act_fun=act_fun,
                use_huber=use_huber,
                scope="Manager",
                zero_fingerprint=False,
                fingerprint_dim=self.fingerprint_dim[0],
                **(additional_params or {}),
            )

        # a fixed goal transition function for the meta-actions in between meta
        # periods. This is used when relative_goals is set to True in order to
        # maintain a fixed absolute position of the goal.
        if relative_goals:
            def goal_transition_fn(obs0, goal, obs1):
                return obs0 + goal - obs1
        else:
            def goal_transition_fn(obs0, goal, obs1):
                return goal
        self.goal_transition_fn = goal_transition_fn

        # previous observation by the Manager
        self.prev_meta_obs = None

        # current action by the Manager
        self.meta_action = None

        # current meta reward, counting as the cumulative environment reward
        # during the meta period
        self.meta_reward = None

        # The following is redundant but necessary if the changes to the update
        # function are to be in the GoalConditionedPolicy policy and not
        # FeedForwardPolicy.
        self.batch_size = batch_size

        # Use this to store a list of observations that stretch as long as the
        # dilated horizon chosen for the Manager. These observations correspond
        # to the s(t) in the HIRO paper.
        self._observations = []

        # Use this to store the list of environmental actions that the worker
        # takes. These actions correspond to the a(t) in the HIRO paper.
        self._worker_actions = []

        # rewards provided by the policy to the worker
        self._worker_rewards = []

        # done masks at every time step for the worker
        self._dones = []

        # actions performed by the manager during a given meta period. Used by
        # the replay buffer.
        self._meta_actions = []

        # =================================================================== #
        # Part 2. Setup the Worker                                            #
        # =================================================================== #

        # Create the Worker policy.
        with tf.compat.v1.variable_scope("Worker"):
            self.worker = worker_policy(
                sess,
                ob_space=ob_space,
                ac_space=ac_space,
                co_space=manager_ac_space,
                buffer_size=buffer_size,
                batch_size=batch_size,
                actor_lr=actor_lr,
                critic_lr=critic_lr,
                verbose=verbose,
                tau=tau,
                gamma=gamma,
                layer_norm=layer_norm,
                layers=layers,
                act_fun=act_fun,
                use_huber=use_huber,
                scope="Worker",
                zero_fingerprint=self.use_fingerprints,
                fingerprint_dim=self.fingerprint_dim[0],
                **(additional_params or {}),
            )

        # Collect the state indices for the worker rewards.
        self.goal_indices = get_goal_indices(
            ob_space, env_name, use_fingerprints, self.fingerprint_dim)

        # reward function for the worker
        def worker_reward_fn(states, goals, next_states):
            return negative_distance(
                states=states,
                state_indices=self.goal_indices,
                goals=goals,
                next_states=next_states,
                relative_context=relative_goals,
                offset=0.0
            )
        self.worker_reward_fn = worker_reward_fn

        if self.connected_gradients:
            self._setup_connected_gradients()

        if self.multistep_llp:
            # Create the Worker dynamics model and training procedure.
            self._setup_multistep_llp()

        else:
            # Default values if the specific algorithm is not called.
            self.worker_obs_ph = None
            self.worker_obs1_ph = None
            self.worker_action_ph = None
            self.worker_model = None
            self.worker_model_loss = None
            self.worker_model_optimizer = None

    def initialize(self):
        """See parent class.

        This method calls the initialization methods of the manager and worker.
        """
        self.manager.initialize()
        self.worker.initialize()
        self.meta_reward = 0

    def update(self, update_actor=True, **kwargs):
        """Perform a gradient update step.

        This is done both at the level of the Manager and Worker policies.

        The kwargs argument for this method contains two additional terms:

        * update_meta (bool): specifies whether to perform a gradient update
          step for the meta-policy (i.e. Manager)
        * update_meta_actor (bool): similar to the `update_policy` term, but
          for the meta-policy. Note that, if `update_meta` is set to False,
          this term is void.

        **Note**; The target update soft updates for both the manager and the
        worker policies occur at the same frequency as their respective actor
        update frequencies.

        Parameters
        ----------
        update_actor : bool
            specifies whether to update the actor policy. The critic policy is
            still updated if this value is set to False.

        Returns
        -------
         ([float, float], [float, float])
            manager critic loss, worker critic loss
        (float, float)
            manager actor loss, worker actor loss
        """
        # Not enough samples in the replay buffer.
        if not self.replay_buffer.can_sample():
            return ([0, 0], [0, 0]), (0, 0)

        # Specifies whether to remove additional data from the replay buffer
        # sampling procedure. Since only a subset of algorithms use additional
        # data, removing it can speedup the other algorithms.
        with_additional = self.off_policy_corrections or self.multistep_llp

        # Get a batch.
        meta_obs0, meta_obs1, meta_act, meta_rew, meta_done, worker_obs0, \
            worker_obs1, worker_act, worker_rew, worker_done, additional = \
            self.replay_buffer.sample(with_additional=with_additional)

        # Update the Manager policy.
        if kwargs['update_meta']:
            # Replace the goals with the most likely goals.
            if self.off_policy_corrections:
                meta_act = self._sample_best_meta_action(
                    meta_obs0=meta_obs0,
                    meta_obs1=meta_obs1,
                    meta_action=meta_act,
                    worker_obses=additional["worker_obses"],
                    worker_actions=additional["worker_actions"],
                    k=8
                )

            if self.connected_gradients:
                # Perform the connected gradients update procedure.
                m_critic_loss, m_actor_loss = self._connected_gradients_update(
                    obs0=meta_obs0,
                    actions=meta_act,
                    rewards=meta_rew,
                    obs1=meta_obs1,
                    terminals1=meta_done,
                    update_actor=kwargs['update_meta_actor'],
                    worker_obs0=worker_obs0,
                    worker_obs1=worker_obs1,
                    worker_actions=worker_act,
                )
            else:
                # Perform the regular manager update procedure.
                m_critic_loss, m_actor_loss = self.manager.update_from_batch(
                    obs0=meta_obs0,
                    actions=meta_act,
                    rewards=meta_rew,
                    obs1=meta_obs1,
                    terminals1=meta_done,
                    update_actor=kwargs['update_meta_actor'],
                )
        else:
            m_critic_loss, m_actor_loss = [0, 0], 0

        # Update the Worker policy.
        w_critic_loss, w_actor_loss = self.worker.update_from_batch(
            obs0=worker_obs0,
            actions=worker_act,
            rewards=worker_rew,
            obs1=worker_obs1,
            terminals1=worker_done,
            update_actor=update_actor and not self.multistep_llp,
            update_target=update_actor)

        if self.multistep_llp:
            w_actor_loss += self._train_worker_model(worker_obs0,
                                                     worker_obs1,
                                                     worker_act,
                                                     additional["worker_obses"],
                                                     additional["worker_actions"])

        return (m_critic_loss, w_critic_loss), (m_actor_loss, w_actor_loss)

    def get_action(self, obs, context, apply_noise, random_actions):
        """See parent class."""
        if self._update_meta:
            # Update the meta action based on the output from the policy if the
            # time period requires is.
            self.meta_action = self.manager.get_action(
                obs, context, apply_noise, random_actions)

        else:
            # Update the meta-action in accordance with the fixed transition
            # function.
            self.meta_action = self.goal_transition_fn(
                obs0=np.asarray([self._observations[-1][self.goal_indices]]),
                goal=self.meta_action,
                obs1=obs[:, self.goal_indices]
            )

        # Return the worker action.
        worker_action = self.worker.get_action(
            obs, self.meta_action, apply_noise, random_actions)

        return worker_action

    def value(self, obs, context, action):
        """See parent class."""
        return 0, 0  # FIXME

    def store_transition(self, obs0, context0, action, reward, obs1, context1,
                         done, is_final_step, evaluate=False):
        """See parent class."""
        # Compute the worker reward and append it to the list of rewards.
        self._worker_rewards.append(
            self.worker_reward_scale *
            self.worker_reward_fn(obs0, self.meta_action.flatten(), obs1)
        )

        # Add the environmental observations and done masks, and the manager
        # and worker actions to their respective lists.
        self._worker_actions.append(action)
        self._meta_actions.append(self.meta_action.flatten())
        self._observations.append(self._get_obs(obs0, self.meta_action, 0))

        # Modify the done mask in accordance with the TD3 algorithm. Done
        # masks that correspond to the final step are set to False.
        self._dones.append(done and not is_final_step)

        # Increment the meta reward with the most recent reward.
        self.meta_reward += reward

        # Modify the previous meta observation whenever the action has changed.
        if len(self._observations) == 1:
            self.prev_meta_obs = self._get_obs(obs0, context0, 0)

        # Add a sample to the replay buffer.
        if len(self._observations) == self.meta_period or done:
            # Add the last observation.
            self._observations.append(self._get_obs(obs1, self.meta_action, 0))

            # Add the contextual observation to the most recent environmental
            # observation, if applicable.
            meta_obs1 = self._get_obs(obs1, context1, 0)

            # Avoid storing samples when performing evaluations.
            if not evaluate:
                if not self.hindsight \
                        or random.random() < self.subgoal_testing_rate:
                    # Store a sample in the replay buffer.
                    self.replay_buffer.add(
                        obs_t=self._observations,
                        goal_t=self._meta_actions[0],
                        action_t=self._worker_actions,
                        reward_t=self._worker_rewards,
                        done=self._dones,
                        meta_obs_t=(self.prev_meta_obs, meta_obs1),
                        meta_reward_t=self.meta_reward,
                    )

                if self.hindsight:
                    # Implement hindsight action and goal transitions.
                    goal, obs, rewards = self._hindsight_actions_goals(
                        meta_action=self.meta_action,
                        initial_observations=self._observations,
                        initial_rewards=self._worker_rewards
                    )

                    # Store the hindsight sample in the replay buffer.
                    self.replay_buffer.add(
                        obs_t=obs,
                        goal_t=goal,
                        action_t=self._worker_actions,
                        reward_t=rewards,
                        done=self._dones,
                        meta_obs_t=(self.prev_meta_obs, meta_obs1),
                        meta_reward_t=self.meta_reward,
                    )

            # Clear the worker rewards and actions, and the environmental
            # observation and reward.
            self.clear_memory()

    @property
    def _update_meta(self):
        """Return True if the meta-action should be updated by the policy.

        This is done by checking the length of the observation lists that are
        passed to the replay buffer, which are cleared whenever the meta-period
        has been met or the environment has been reset.
        """
        return len(self._observations) == 0

    def clear_memory(self):
        """Clear internal memory that is used by the replay buffer.

        By clearing memory, the Manager policy is then informed during the
        `get_action` procedure to update the meta-action.
        """
        self.meta_reward = 0
        self._observations = []
        self._worker_actions = []
        self._worker_rewards = []
        self._dones = []
        self._meta_actions = []

    def get_td_map(self):
        """See parent class."""
        # Not enough samples in the replay buffer.
        if not self.replay_buffer.can_sample():
            return {}

        # Get a batch.
        meta_obs0, meta_obs1, meta_act, meta_rew, meta_done, worker_obs0, \
            worker_obs1, worker_act, worker_rew, worker_done, additional = \
            self.replay_buffer.sample(with_additional=True)

        td_map = {}
        td_map.update(self.manager.get_td_map_from_batch(
            meta_obs0, meta_act, meta_rew, meta_obs1, meta_done))
        td_map.update(self.worker.get_td_map_from_batch(
            worker_obs0, worker_act, worker_rew, worker_obs1, worker_done))

        worker_obs_seq = additional['worker_obses']
        worker_act_seq = additional['worker_actions']

        goal_dim = self.manager.ac_space.shape[0]
        seq_len = worker_act_seq.shape[2]

        # TODO: time step is the last dim
        worker_obs_seq = worker_obs_seq.transpose(0, 2, 1)
        worker_act_seq = worker_act_seq.transpose(0, 2, 1)

        slots = seq_len - self.max_rollout_when_training - 1
        start = np.random.randint(0, slots, [])
        end = start + self.max_rollout_when_training

        if self.multistep_llp:
            for i in range(self.num_ensembles):

                td_map.update({
                    self.worker_obs_ph[i]: worker_obs_seq[:, start:end, :-goal_dim],
                    self.worker_obs1_ph[i]: worker_obs_seq[:, start + 1:end + 1, :-goal_dim],
                    self.worker_action_ph[i]: worker_act_seq[:, start:end, :],
                    self.rollout_worker_obs: worker_obs0
                })

        return td_map

    # ======================================================================= #
    #                       Auxiliary methods for HIRO                        #
    # ======================================================================= #

    def _sample_best_meta_action(self,
                                 meta_obs0,
                                 meta_obs1,
                                 meta_action,
                                 worker_obses,
                                 worker_actions,
                                 k=10):
        """Return meta-actions that approximately maximize low-level log-probs.

        Parameters
        ----------
        meta_obs0 : array_like
            (batch_size, m_obs_dim) matrix of Manager observations
        meta_obs1 : array_like
            (batch_size, m_obs_dim) matrix of next time step Manager
            observations
        meta_action : array_like
            (batch_size, m_ac_dim) matrix of Manager actions
        worker_obses : array_like
            (batch_size, w_obs_dim, meta_period+1) matrix of current Worker
            state observations
        worker_actions : array_like
            (batch_size, w_ac_dim, meta_period) matrix of current Worker
            environmental actions
        k : int, optional
            number of goals returned, excluding the initial goal and the mean
            value

        Returns
        -------
        array_like
            (batch_size, m_ac_dim) matrix of most likely Manager actions
        """
        batch_size, goal_dim = meta_action.shape

        # Collect several samples of potentially optimal goals.
        sampled_actions = self._sample(meta_obs0, meta_obs1, meta_action, k)
        assert sampled_actions.shape == (batch_size, goal_dim, k)

        # Compute the fitness of each candidate goal. The fitness is the sum of
        # the log-probabilities of each action for the given goal.
        fitness = self._log_probs(
            sampled_actions, worker_obses, worker_actions)
        assert fitness.shape == (batch_size, k)

        # For each sample, choose the meta action that maximizes the fitness.
        indx = np.argmax(fitness, 1)
        best_goals = np.asarray(
            [sampled_actions[i, :, indx[i]] for i in range(batch_size)])

        return best_goals

    def _sample(self, meta_obs0, meta_obs1, meta_action, num_samples, sc=0.5):
        """Sample different goals.

        The goals are sampled as follows:

        * The first num_samples-2 goals are acquired from a random Gaussian
          distribution centered at s_{t+c} - s_t.
        * The second to last goal is s_{t+c} - s_t.
        * The last goal is the originally sampled goal g_t.

        Parameters
        ----------
        meta_obs0 : array_like
            (batch_size, m_obs_dim) matrix of Manager observations
        meta_obs1 : array_like
            (batch_size, m_obs_dim) matrix of next time step Manager
            observations
        meta_action : array_like
            (batch_size, m_ac_dim) matrix of Manager actions
        num_samples : int
            number of samples
        sc : float
            scaling factor for the normal distribution.

        Returns
        -------
        array_like
            (batch_size, goal_dim, num_samples) matrix of sampled goals

        Helps
        -----
        * _sample_best_meta_action(self)
        """
        batch_size, goal_dim = meta_action.shape
        goal_space = self.manager.ac_space
        spec_range = goal_space.high - goal_space.low
        random_samples = num_samples - 2

        # Compute the mean and std for the Gaussian distribution to sample
        # from, and well as the maxima and minima.
        loc = meta_obs1[:, :goal_dim] - meta_obs0[:, :goal_dim]
        scale = [sc * spec_range / 2]
        minimum, maximum = [goal_space.low], [goal_space.high]

        new_loc = np.zeros((batch_size, goal_dim, random_samples))
        new_scale = np.zeros((batch_size, goal_dim, random_samples))
        for i in range(random_samples):
            new_loc[:, :, i] = loc
            new_scale[:, :, i] = scale

        new_minimum = np.zeros((batch_size, goal_dim, num_samples))
        new_maximum = np.zeros((batch_size, goal_dim, num_samples))
        for i in range(num_samples):
            new_minimum[:, :, i] = minimum
            new_maximum[:, :, i] = maximum

        # Generate random samples for the above distribution.
        normal_samples = np.random.normal(
            size=(random_samples * batch_size * goal_dim))
        normal_samples = normal_samples.reshape(
            (batch_size, goal_dim, random_samples))

        samples = np.zeros((batch_size, goal_dim, num_samples))
        samples[:, :, :-2] = new_loc + normal_samples * new_scale
        samples[:, :, -2] = loc
        samples[:, :, -1] = meta_action

        # Clip the values based on the Manager action space range.
        samples = np.minimum(np.maximum(samples, new_minimum), new_maximum)

        return samples

    def _log_probs(self, meta_actions, worker_obses, worker_actions):
        """Calculate the log probability of the next goal by the Manager.

        Parameters
        ----------
        meta_actions : array_like
            (batch_size, m_ac_dim, num_samples) matrix of candidate Manager
            actions
        worker_obses : array_like
            (batch_size, w_obs_dim, meta_period + 1) matrix of Worker
            observations
        worker_actions : array_like
            (batch_size, w_ac_dim, meta_period) list of Worker actions

        Returns
        -------
        array_like
            (batch_size, num_samples) fitness associated with every state /
            action / goal pair

        Helps
        -----
        * _sample_best_meta_action(self):
        """
        raise NotImplementedError

    # ======================================================================= #
    #                       Auxiliary methods for HAC                         #
    # ======================================================================= #

    def _hindsight_actions_goals(self,
                                 meta_action,
                                 initial_observations,
                                 initial_rewards):
        """Calculate hindsight goal and action transitions.

        These are then stored in the replay buffer along with the original
        (non-hindsight) sample.

        See the README at the front page of this repository for an in-depth
        description of this procedure.

        Parameters
        ----------
        meta_action : array_like
            the original Manager actions (goal)
        initial_observations : array_like
            the original worker observations with the non-hindsight goals
            appended to them
        initial_rewards : array_like
            the original worker rewards

        Returns
        -------
        array_like
            the Manager action (goal) in hindsight
        array_like
            the modified Worker observations with the hindsight goals appended
            to them
        array_like
            the modified Worker rewards taking into account the hindsight goals

        Helps
        -----
        * store_transition(self):
        """
        goal_dim = meta_action.shape[0]
        observations = deepcopy(initial_observations)
        rewards = deepcopy(initial_rewards)
        hindsight_goal = 0 if self.relative_goals \
            else observations[-1][:goal_dim]
        obs_tp1 = observations[-1]

        for i in range(1, len(observations) + 1):
            obs_t = observations[-i]

            # Calculate the hindsight goal in using relative goals.
            # If not, the hindsight goal is simply a subset of the
            # final state observation.
            if self.relative_goals:
                hindsight_goal += obs_tp1[:goal_dim] - obs_t[:goal_dim]

            # Modify the Worker intrinsic rewards based on the new
            # hindsight goal.
            if i > 1:
                rewards[-(i - 1)] = self.worker_reward_scale \
                    * self.worker_reward_fn(obs_t, hindsight_goal, obs_tp1)

            obs_tp1 = deepcopy(obs_t)

            # Replace the goal with the goal that the worker
            # actually achieved.
            observations[-i][-goal_dim:] = hindsight_goal

        return hindsight_goal, observations, rewards

    # ======================================================================= #
    #                      Auxiliary methods for HRL-CG                       #
    # ======================================================================= #

    def _setup_connected_gradients(self):
        """Create the updated manager optimization with connected gradients."""
        raise NotImplementedError

    def _connected_gradients_update(self,
                                    obs0,
                                    actions,
                                    rewards,
                                    obs1,
                                    terminals1,
                                    worker_obs0,
                                    worker_obs1,
                                    worker_actions,
                                    update_actor=True):
        """Perform the gradient update procedure for the HRL-CG algorithm.

        This procedure is similar to self.manager.update_from_batch, expect it
        runs the self.cg_optimizer operation instead of self.manager.optimizer,
        and utilizes some information from the worker samples as well.

        Parameters
        ----------
        obs0 : np.ndarray
            batch of manager observations
        actions : numpy float
            batch of manager actions executed given obs_batch
        rewards : numpy float
            manager rewards received as results of executing act_batch
        obs1 : np.ndarray
            set of next manager observations seen after executing act_batch
        terminals1 : numpy bool
            done_mask[i] = 1 if executing act_batch[i] resulted in the end of
            an episode and 0 otherwise.
        worker_obs0 : array_like
            batch of worker observations
        worker_obs1 : array_like
            batch of next worker observations
        worker_actions : array_like
            batch of worker actions
        update_actor : bool
            specifies whether to update the actor policy of the manager. The
            critic policy is still updated if this value is set to False.

        Returns
        -------
        [float, float]
            manager critic loss
        float
            manager actor loss
        """
        raise NotImplementedError

    # ======================================================================= #
    #                  Auxiliary methods for Multi-Step LLP                   #
    # ======================================================================= #

    def _setup_multistep_llp(self):
        """Create the trainable features of the multi-step LLP algorithm."""
        if self.verbose >= 2:
            print('setting up Worker dynamics model')

        # =================================================================== #
        # Part 1. Create the model.                                           #
        # =================================================================== #

        self.worker_model = []
        self.worker_model_loss = []
        self.worker_model_optimizer = []
        self.worker_obs_ph = []
        self.worker_obs1_ph = []
        self.worker_action_ph = []

        goal_dim = self.manager.ac_space.shape[0]
        ob_dim = self.worker.ob_space.shape[0]

        with tf.compat.v1.variable_scope("Worker/dynamics"):
            # Create clipping terms for the model logstd. See:
            # TODO
            self.max_logstd = tf.Variable(
                np.ones([1, ob_dim]) * 2.0,
                dtype=tf.float32,
                name="max_log_std")
            self.min_logstd = tf.Variable(
                -np.ones([1, ob_dim]) * 10.0,
                dtype=tf.float32,
                name="max_log_std")

            for i in range(self.num_ensembles):
                # Create placeholders for the model.
                self.worker_obs_ph.append(tf.compat.v1.placeholder(
                    tf.float32,
                    shape=(None, self.max_rollout_when_training) + self.worker.ob_space.shape,
                    name="worker_obs0_{}".format(i)))
                self.worker_obs1_ph.append(tf.compat.v1.placeholder(
                    tf.float32,
                    shape=(None, self.max_rollout_when_training) + self.worker.ob_space.shape,
                    name="worker_obs1_{}".format(i)))
                self.worker_action_ph.append(tf.compat.v1.placeholder(
                    tf.float32,
                    shape=(None, self.max_rollout_when_training) + self.worker.ac_space.shape,
                    name="worker_action_{}".format(i)))

                ob_pred = self.worker_obs_ph[-1][:, 0, :]

                # The additional loss term is in accordance with:
                # https://github.com/kchua/handful-of-trials
                worker_model_loss = (0.01 * tf.reduce_sum(self.max_logstd) -
                                     0.01 * tf.reduce_sum(self.min_logstd))

                # Create a trainable model of the Worker dynamics.
                for t in range(self.max_rollout_when_training):

                    worker_model, mean, logstd = self._setup_worker_model(
                        obs=ob_pred,
                        action=self.worker_action_ph[-1][:, t, :],
                        ob_space=self.worker.ob_space,
                        scope="rho_{}".format(i),
                        reuse=t > 0)

                    ob_pred = worker_model

                    # The worker model is trained to learn the change in state
                    # between two time-steps.
                    delta = (self.worker_obs1_ph[-1][:, t, :] -
                             self.worker_obs_ph[-1][:, t, :])

                    # Computes the log probability of choosing a specific output -
                    # used by the loss
                    dist = tfp.distributions.MultivariateNormalDiag(
                        loc=mean,
                        scale_diag=tf.exp(logstd))
                    rho_logp = dist.log_prob(delta)

                    # Create the model loss.
                    worker_model_loss -= tf.reduce_mean(
                        rho_logp) / self.max_rollout_when_training

                self.worker_model.append(worker_model)
                self.worker_model_loss.append(worker_model_loss)

                # Create an optimizer object.
                optimizer = tf.compat.v1.train.AdamOptimizer(self.worker_dynamics_bptt_lr)

                # Create the model optimization technique.
                worker_model_optimizer = optimizer.minimize(
                    worker_model_loss,
                    var_list=get_trainable_vars(
                        'Worker/dynamics/rho_{}'.format(i)))
                self.worker_model_optimizer.append(worker_model_optimizer)

                # Add the model loss and dynamics to the tensorboard log.
                tf.compat.v1.summary.scalar(
                    'model_{}_loss'.format(i), worker_model_loss)
                tf.compat.v1.summary.scalar(
                    'model_{}_mean'.format(i), tf.reduce_mean(worker_model))
                tf.compat.v1.summary.scalar(
                    'model_{}_std'.format(i), reduce_std(worker_model))

                # Print the shapes of the generated models.
                if self.verbose >= 2:
                    scope_name = 'Worker/dynamics/rho_{}'.format(i)
                    critic_shapes = [var.get_shape().as_list()
                                     for var in get_trainable_vars(scope_name)]
                    critic_nb_params = sum([reduce(lambda x, y: x * y, shape)
                                            for shape in critic_shapes])
                    print('  model shapes: {}'.format(critic_shapes))
                    print('  model params: {}'.format(critic_nb_params))

        # =================================================================== #
        # Part 2. Create the Worker optimization scheme.                      #
        # =================================================================== #

        # Collect the observation space of the Worker.
        ob_dim = self._get_ob_dim(self.worker.ob_space, self.worker.co_space)

        # Create a placeholder to store all worker observations for a given
        # meta-period.
        self.rollout_worker_obs = tf.compat.v1.placeholder(
            tf.float32,
            shape=(None, ob_dim[0]),
            name='rollout_worker_obs')

        # Compute the cumulative, discounted model-based loss using outputs
        # from the Worker's trainable model.
        self._multistep_llp_loss = 0

        for i in range(self.num_particles):
            # FIXME: should we choose dynamically?
            # Choose a model index to compute the trajectory over.
            model_index = i % self.num_ensembles

            # Create the initial Worker.
            with tf.compat.v1.variable_scope("Worker/model"):
                action = self.worker.make_actor(
                    obs=self.rollout_worker_obs[:, :], reuse=True)

            goal_dim = self.manager.ac_space.shape[0]

            # FIXME: goal_indices
            # Initial step observation from the perspective of the model.
            obs = self.rollout_worker_obs[:, :-goal_dim]

            # FIXME: goal_indices
            # The initial goal is provided by this placeholder.
            goal = self.rollout_worker_obs[:, -goal_dim:]

            # Collect the first step loss, and the next state and goal.
            loss, obs1, goal = self._get_step_loss(
                obs, action, goal, model_index)

            horizon = min(self.max_rollout_using_model, self.meta_period)

            # Repeat the process for the meta-period.
            for j in range(1, horizon):
                # FIXME: goal_indices
                # TODO: why is this necessary
                # Replace a subset of the next placeholder with the
                # previous step dynamic and goal.
                obs = tf.concat((obs1, goal), axis=1)

                # Create the next-step Worker actor.
                with tf.compat.v1.variable_scope("Worker/model"):
                    action = self.worker.make_actor(obs, reuse=True)

                # Collect the next loss, observation, and goal.
                next_loss, obs1, goal = self._get_step_loss(
                    obs1, action, goal, model_index)

                # Add the next loss to the discounted sum.
                loss += (self.worker.gamma ** j) * next_loss

            # Add the next loss to the multi-step LLP loss.
            self._multistep_llp_loss += loss / self.num_particles

            # calculate the q value at the end of the k step rollout
            obs = tf.concat((obs1, goal), axis=1)
            with tf.compat.v1.variable_scope("Worker/model"):
                action = self.worker.make_actor(obs, reuse=True)
                qvalue = tf.reduce_mean([
                    self.worker.make_critic(obs, action, reuse=True, scope="qf_0"),
                    self.worker.make_critic(obs, action, reuse=True, scope="qf_1")])

            # add final q value: https://openreview.net/forum?id=Skln2A4YDB
            if self.add_final_q_value:
                self._multistep_llp_loss -= (
                    self.worker.gamma ** horizon) * qvalue

        # Add the final loss for tensorboard logging.
        tf.compat.v1.summary.scalar(
            'Worker/worker_model_loss', self._multistep_llp_loss)

        # Create an optimizer object.
        optimizer = tf.compat.v1.train.AdamOptimizer(self.worker_model_bptt_lr)

        grads_and_vars = optimizer.compute_gradients(
            self._multistep_llp_loss,
            var_list=get_trainable_vars('Worker/model/pi'))

        # Log the max, min, mean, and std for each variable
        for grad, var in grads_and_vars:
            tf.compat.v1.summary.scalar(
                '{}/bptt/mean'.format(var.name), tf.reduce_mean(grad))
            tf.compat.v1.summary.scalar(
                '{}/bptt/std'.format(var.name), tf.math.reduce_std(grad))
            tf.compat.v1.summary.scalar(
                '{}/bptt/max'.format(var.name), tf.reduce_max(grad))
            tf.compat.v1.summary.scalar(
                '{}/bptt/min'.format(var.name), tf.reduce_min(grad))

        # Create the model optimization technique.
        self._multistep_llp_optimizer = optimizer.apply_gradients(
            grads_and_vars)

    def _setup_worker_model(self,
                            obs,
                            action,
                            ob_space,
                            reuse=False,
                            scope="rho"):
        """Create the trainable parameters of the Worker dynamics model.

        Parameters
        ----------
        obs : tf.compat.v1.placeholder
            the last step observation, not including the context
        action : tf.compat.v1.placeholder
            the action from the Worker policy. May be a function of the
            Manager's trainable parameters
        ob_space : gym.spaces.*
            the observation space, not including the context space
        reuse : bool
            whether or not to reuse parameters
        scope : str
            the scope name of the actor

        Returns
        -------
        tf.Variable
            the output from the Worker dynamics model
        tf.Variable
            the mean of the Worker dynamics model
        tf.Variable
            the log std of the Worker dynamics model
        """
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            # Concatenate the observations and actions.
            rho_h = tf.concat([obs, action], axis=-1)

            # Create the hidden layers.
            for i, layer_size in enumerate(self.layers):
                rho_h = self._layer(
                    rho_h,  layer_size, 'fc{}'.format(i),
                    act_fun=self.act_fun,
                    layer_norm=self.layer_norm
                )

            # Create the output mean.
            rho_mean = self._layer(rho_h, ob_space.shape[0], 'rho_mean')

            # Create the output logstd term.
            rho_logvar = self._layer(rho_h, ob_space.shape[0], 'rho_logvar')

            # Perform log-std clipping as describe in Appendix A.1 of:
            # TODO
            rho_logvar = self.max_logstd - tf.nn.softplus(
                self.max_logstd - rho_logvar)
            rho_logvar = self.min_logstd + tf.nn.softplus(
                rho_logvar - self.min_logstd)

            rho_logstd = rho_logvar / 2.

            rho_std = tf.exp(rho_logstd)

            # The model samples from its distribution.
            rho = rho_mean + tf.random.normal(tf.shape(rho_mean)) * rho_std

        return rho, rho_mean, rho_logstd

    def _get_step_loss(self, obs, action, goal, model_index):
        """Compute the next-step model-based loss.

        The inputs and outputs here are tf.Variable in order to propagate
        losses through them.

        Parameters
        ----------
        obs : tf.Variable
            the previous step observation, as relevant to the Worker model
        action : tf.Variable
            the most recent action performed by the Worker
        goal : tf.Variable
            the most recent goal
        model_index : int
            the index number of the model used to update the Worker dynamics

        Returns
        -------
        tf.Variable
            the next-step loss
        tf.Variable
            the next-step observation
        tf.Variable
            the next-step goal
        """
        goal_dim = self.manager.ac_space.shape[0]
        loss_fn = tf.compat.v1.losses.mean_squared_error

        with tf.compat.v1.variable_scope("Worker/dynamics"):
            # Compute the delta term.
            sample, mean, _ = self._setup_worker_model(
                obs=obs,
                action=action,
                ob_space=self.worker.ob_space,
                reuse=True,
                scope="rho_{}".format(model_index))

            delta = sample if self.use_sample_not_mean else mean

            # Compute the next observation.
            next_obs = obs + delta

        # FIXME: goal_indices
        # Compute the loss associated with this obs0/obs1/action tuple, as well
        # as the next goal.
        if self.relative_goals:
            loss = loss_fn(obs[:, :goal_dim] + goal, next_obs[:, :goal_dim])
            next_goal = obs[:, :goal_dim] + goal - next_obs[:, :goal_dim]
        else:
            loss = loss_fn(goal, next_obs[:, :goal_dim])
            next_goal = goal

        return loss, next_obs, next_goal

    def _train_worker_model(self,
                            worker_obs0,
                            worker_obs1,
                            worker_act,
                            worker_obs_seq,
                            worker_act_seq,
                            ):
        """Train the Worker actor and dynamics model.

        The original goal-less states and actions are used to train the model.

        Parameters
        ----------
        worker_obs0 : array_like
            worker observations at the current step
        worker_obs1 : array_like
            worker observations at the next step
        worker_act : array_like
            worker actions at the current step

        Returns
        -------
        float
            Worker loss
        """
        step_ops = [
            self._multistep_llp_loss,
            self._multistep_llp_optimizer
        ]

        feed_dict = {
            self.rollout_worker_obs: worker_obs0,
        }

        goal_dim = self.manager.ac_space.shape[0]
        seq_len = worker_obs_seq.shape[2]

        # TODO: time step is the last dim
        worker_obs_seq = worker_obs_seq.transpose(0, 2, 1)
        worker_act_seq = worker_act_seq.transpose(0, 2, 1)

        # Add the step ops and samples for each of the model training
        # operations in the ensemble.
        for i in range(self.num_ensembles):

            # Add the training operation.
            step_ops.append(self.worker_model_optimizer[i])

            slots = seq_len - self.max_rollout_when_training - 1
            start = np.random.randint(0, slots, [])
            end = start + self.max_rollout_when_training

            # Add the samples to the feed_dict.
            feed_dict.update({
                self.worker_obs_ph[i]: worker_obs_seq[:, start:end, :-goal_dim],
                self.worker_obs1_ph[i]: worker_obs_seq[:, start + 1:end + 1, :-goal_dim],
                self.worker_action_ph[i]: worker_act_seq[:, start:end, :]
            })

        # Run the training operations.
        worker_loss, *_ = self.sess.run(step_ops, feed_dict=feed_dict)

        return worker_loss
