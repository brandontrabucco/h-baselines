"""Base goal-conditioned hierarchical policy."""
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from copy import deepcopy
import random
from collections import deque
from functools import reduce

from hbaselines.fcnet.base import ActorCriticPolicy
from hbaselines.goal_conditioned.replay_buffer import HierReplayBuffer
from hbaselines.utils.reward_fns import negative_distance
from hbaselines.utils.misc import get_manager_ac_space, get_goal_indices
from hbaselines.utils.tf_util import get_trainable_vars
from hbaselines.utils.tf_util import reduce_std

from dmbrl.misc.optimizers import RandomOptimizer, CEMOptimizer
from dmbrl.modeling.models import BNN
from dmbrl.modeling.layers import FC
from dotmap import DotMap


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
            buffer_size=int(buffer_size / meta_period),
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
        self._worker_rewards_history = deque(maxlen=100)

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

        # for tensorboard logging of the worker rewards
        with tf.compat.v1.variable_scope("Train"):
            self.worker_rew_ph = tf.Variable(np.zeros([]), dtype=tf.float32)
            self.worker_rew_history_ph = tf.Variable(np.zeros([]), dtype=tf.float32)

        # for tensorboard logging of the worker rewards
        tf.compat.v1.summary.scalar("Train/worker_return", self.worker_rew_ph)
        tf.compat.v1.summary.scalar("Train/worker_return_history",
                                    self.worker_rew_history_ph)

        # for tensorboard logging of the worker rewards
        self.sess.run(tf.variables_initializer([self.worker_rew_ph]))
        self.sess.run(tf.variables_initializer([self.worker_rew_history_ph]))

        self._t = 0

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
        if self.multistep_llp:
            w_actor_loss = self._train_worker_model()

            # The Q-function is not trained.
            w_critic_loss = [0, 0]
        else:
            w_critic_loss, w_actor_loss = self.worker.update_from_batch(
                obs0=worker_obs0,
                actions=worker_act,
                rewards=worker_rew,
                obs1=worker_obs1,
                terminals1=worker_done,
                update_actor=update_actor,
            )

        self._t += 1

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

        if not self.multistep_llp:

            worker_action = self.worker.get_action(
                obs, self.meta_action, apply_noise, random_actions)

        else:

            #################################################
            # CEM planning algorithm rather than model free #
            #################################################

            # while the model is still training explore randomly
            if self._t < self.steps_before_planning:
                return np.random.uniform(
                    self.worker.ac_space.low,
                    self.worker.ac_space.high,
                    self.worker.ac_space.shape)

            # if actions are already computed, use them
            if self.ac_buf.shape[0] > 0:
                worker_action, self.ac_buf = self.ac_buf[0], self.ac_buf[1:]
                return worker_action

            # populate the current observation and the goal
            self.sy_cur_obs.load(obs[0], self.model.sess)
            self.sy_cur_goals.load(self.meta_action[0], self.model.sess)

            # run cem to obtain solutions for optimal actions
            soln = self.worker_cem.obtain_solution(self.prev_sol, self.init_var)

            # store the previous solutions using CEM
            self.prev_sol = np.concatenate([
                np.copy(soln)[self.per * self.worker.ac_space.shape[0]:],
                np.zeros(self.per * self.worker.ac_space.shape[0])])

            # store the previous solutions using CEM
            self.ac_buf = soln[:self.per * self.worker.ac_space.shape[0]].reshape(
                -1, self.worker.ac_space.shape[0])

            # pop the optimal actions off the action buffer
            worker_action, self.ac_buf = self.ac_buf[0], self.ac_buf[1:]

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

            # log the goal conditioned reward for the worker policy
            mean_worker_rew = np.mean(self._worker_rewards)
            self._worker_rewards_history.append(mean_worker_rew)
            self.worker_rew_ph.load(mean_worker_rew, self.sess)
            self.worker_rew_history_ph.load(np.mean(self._worker_rewards_history), self.sess)

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

        if self.multistep_llp:
            self.ac_buf = np.array([]).reshape(0, self.worker.ac_space.shape[0])

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
        # Part 1. Create the model and action optimization method.            #
        # =================================================================== #

        # create an ensemble of dynamics models
        self.prop_mode = 'TS1'
        self.model = BNN(DotMap(
            name="Worker/dynamics",
            num_networks=self.num_ensembles,
            sess=self.sess, ))

        # create an ensemble of dynamics models
        model_in = self.worker.ob_space.shape[0] + self.worker.ac_space.shape[0]
        model_out = self.worker.ob_space.shape[0]

        # create an ensemble of dynamics models
        self.model.add(FC(
            200, input_dim=model_in, activation="swish", weight_decay=0.000025))
        self.model.add(FC(200, activation="swish", weight_decay=0.00005))
        self.model.add(FC(200, activation="swish", weight_decay=0.000075))
        self.model.add(FC(200, activation="swish", weight_decay=0.000075))
        self.model.add(FC(model_out, weight_decay=0.0001))

        # create an ensemble of dynamics models
        self.model.finalize(tf.train.AdamOptimizer, {"learning_rate": 0.001})

        # buffer to store actions sampled from CEM
        self.ac_buf = np.array([]).reshape(0, self.worker.ac_space.shape[0])
        self.prev_sol = np.tile((
            self.worker.ac_space.high +
            self.worker.ac_space.low) / 2, [self.meta_period])
        self.init_var = np.tile(np.square(
            self.worker.ac_space.high -
            self.worker.ac_space.low) / 16, [self.meta_period])

        self.crop_to_goal = lambda g: tf.gather(
            g,
            tf.tile(tf.expand_dims(self.goal_indices, 0), [g.get_shape()[0], 1]),
            batch_dims=1, axis=1)

        # environment specific configurations for CEM
        goal_loss_fn = tf.compat.v1.losses.mean_squared_error
        if self.relative_goals:
            self.obs_cost_fn = lambda obs, goals: goal_loss_fn(
                self.crop_to_goal(obs) + goals,
                self.crop_to_goal(obs))
        else:
            self.obs_cost_fn = lambda obs, goals: goal_loss_fn(
                goals,
                self.crop_to_goal(obs))

        # environment specific configurations for CEM
        ac_loss_fn = tf.compat.v1.losses.mean_squared_error
        self.ac_cost_fn = lambda cur_ac: 0.0 * ac_loss_fn(
            cur_ac / 2.,
            -cur_ac / 2.)

        # environment specific configurations for CEM
        self.obs_preproc = lambda o: o
        self.obs_postproc = lambda o, p: o + p
        self.obs_postproc2 = lambda o: o
        self.targ_proc = lambda o, next_o: next_o - o

        # placeholders for computing forward pass using CEM
        self.sy_cur_obs = tf.Variable(
            np.zeros(self.worker.ob_space.shape), dtype=tf.float32)
        self.sy_cur_goals = tf.Variable(
            np.zeros(self.manager.ac_space.shape), dtype=tf.float32)

        # CEM optimizer for calculating actions
        self.per = 2
        self.steps_before_planning = 10000
        self.worker_cem = CEMOptimizer(
            self.meta_period * self.worker.ac_space.shape[0],
            5,
            500,
            50,
            lower_bound=np.tile(self.worker.ac_space.low, [self.meta_period]),
            upper_bound=np.tile(self.worker.ac_space.high, [self.meta_period]),
            tf_session=self.sess,
            epsilon=0.001,
            alpha=0.1)
        self.worker_cem.setup(self._compile_worker_cost, True)
        self.model.sess.run(tf.variables_initializer([self.sy_cur_obs]))
        self.model.sess.run(tf.variables_initializer([self.sy_cur_goals]))

    def _predict_next_obs(self, obs, acs, goals):
        """Compute the next-step model-based observation and goal.

        The inputs and outputs here are tf.Variable in order to propagate
        losses through them.

        Parameters
        ----------
        obs : tf.Variable
            the previous step observations, as relevant to the Worker model
        action : tf.Variable
            the most recent actions performed by the Worker
        goals : tf.Variable
            the most recent goals

        Returns
        -------
        tf.Variable
            the next-step observations
        tf.Variable
            the next-step goals
        """
        proc_obs = self.obs_preproc(obs)

        # TS Optimization: Expand so that particles are only passed through one of the networks.
        if self.prop_mode == "TS1":
            # isolate the number of particles in an independent axis
            proc_obs = tf.reshape(
                proc_obs, [-1, self.num_particles, proc_obs.get_shape()[-1]])

            # randomly sort the particles among the models
            sort_idxs = tf.nn.top_k(
                tf.random_uniform([tf.shape(proc_obs)[0], self.num_particles]),
                k=self.num_particles).indices

            # choose a model for each particle to be processed with
            tmp = tf.tile(tf.range(
                tf.shape(proc_obs)[0])[:, None],
                          [1, self.num_particles])[:, :, None]
            idxs = tf.concat([tmp, sort_idxs[:, :, None]], axis=-1)

            # select the observations that correspond to each model
            proc_obs = tf.gather_nd(proc_obs, idxs)
            proc_obs = tf.reshape(proc_obs, [-1, proc_obs.get_shape()[-1]])

        # insert a new axis into the tensor for the number of dynamics models
        if self.prop_mode == "TS1" or self.prop_mode == "TSinf":
            proc_obs = self._expand_to_ts_format(proc_obs)
            acs = self._expand_to_ts_format(acs)

        # Obtain model predictions
        inputs = tf.concat([proc_obs, acs], axis=-1)
        mean, var = self.model.create_prediction_tensors(inputs)

        # sample the next observation from the model
        if self.model.is_probabilistic:

            predictions = mean + tf.random_normal(
                shape=tf.shape(mean), mean=0, stddev=1) * tf.sqrt(var)

            if self.prop_mode == "MM":
                model_out_dim = predictions.get_shape()[-1].value

                predictions = tf.reshape(
                    predictions, [-1, self.num_particles, model_out_dim])
                prediction_mean = tf.reduce_mean(
                    predictions, axis=1, keep_dims=True)

                prediction_var = tf.reduce_mean(
                    tf.square(predictions - prediction_mean),
                    axis=1, keep_dims=True)

                z = tf.random_normal(
                    shape=tf.shape(predictions), mean=0, stddev=1)

                samples = prediction_mean + z * tf.sqrt(prediction_var)
                predictions = tf.reshape(samples, [-1, model_out_dim])
        else:
            predictions = mean

        # TS Optimization: Remove additional dimension
        if self.prop_mode == "TS1" or self.prop_mode == "TSinf":
            predictions = self._flatten_to_matrix(predictions)

        if self.prop_mode == "TS1":
            predictions = tf.reshape(
                predictions,
                [-1, self.num_particles, predictions.get_shape()[-1]])

            sort_idxs = tf.nn.top_k(
                -sort_idxs, k=self.num_particles).indices

            idxs = tf.concat([tmp, sort_idxs[:, :, None]], axis=-1)
            predictions = tf.gather_nd(predictions, idxs)
            predictions = tf.reshape(predictions, [-1, predictions.get_shape()[-1]])

        return self.obs_postproc(obs, predictions), self.goal_transition_fn(
            self.crop_to_goal(obs),
            goals,
            self.crop_to_goal(predictions))

    def _compile_worker_cost(self, ac_seqs, get_pred_trajs=False):
        """Compile the forward dynamics predictions and cost for CEM.

        See: https://github.com/kchua/handful-of-trials/blob/
             e1a62f217508a384e49ecf7d16a3249e187bcff9/dmbrl/controllers/MPC.py#L265

        Parameters
        ----------
        ac_seqs : tf.Variable
            the actions suggested by CEM at this iteration
        get_pred_trajs : boolean
            also return the predicted sequence states

        Returns
        -------
        tf.Variable
            the costs of each sequence of actions in the batch
        tf.Variable
            the sequence of states predicted by the dynamics model
        """

        # process the input tensors into the correct format
        t, nopt = tf.constant(0), tf.shape(ac_seqs)[0]
        init_costs = tf.zeros([nopt, self.num_particles])
        ac_seqs = tf.reshape(
            ac_seqs, [-1, self.meta_period, self.worker.ac_space.shape[0]])
        ac_seqs = tf.reshape(tf.tile(
            tf.transpose(ac_seqs, [1, 0, 2])[:, :, None], [1, 1, self.num_particles, 1]),
            [self.meta_period, -1, self.worker.ac_space.shape[0]])

        # get the current observation from the environment
        init_obs = tf.tile(self.sy_cur_obs[None], [nopt * self.num_particles, 1])
        init_goals = tf.tile(self.sy_cur_goals[None], [nopt * self.num_particles, 1])

        # this condition stops the planning once the meta_period is reached
        def continue_prediction(t, *args):
            return tf.less(t, self.meta_period)

        # is the predicted states should be returns
        if get_pred_trajs:
            pred_trajs = init_obs[None]

            # loop function for predicting future states using tensorflow
            def iteration(t, total_cost, cur_obs, cur_goals, pred_trajs):
                cur_acs = ac_seqs[t]
                next_obs, next_goals = self._predict_next_obs(cur_obs, cur_acs, cur_goals)
                delta_cost = tf.reshape(
                    self.obs_cost_fn(next_obs, next_goals) + self.ac_cost_fn(
                        cur_acs), [-1, self.num_particles])

                next_obs = self.obs_postproc2(next_obs)
                pred_trajs = tf.concat([pred_trajs, next_obs[None]], axis=0)
                return t + 1, total_cost + delta_cost, next_obs, next_goals, pred_trajs

            # predict into the future using static graphs
            _, costs, _, _, pred_trajs = tf.while_loop(
                cond=continue_prediction,
                body=iteration,
                loop_vars=[t, init_costs, init_obs, init_goals, pred_trajs],
                shape_invariants=[
                    t.get_shape(),
                    init_costs.get_shape(),
                    init_obs.get_shape(),
                    init_goals.get_shape(),
                    tf.TensorShape([None, None, self.worker.ob_space.shape[0]])
                ]
            )

            # Replace nan costs with very high cost
            costs = tf.reduce_mean(tf.where(
                tf.is_nan(costs),
                1e6 * tf.ones_like(costs), costs), axis=1)

            pred_trajs = tf.reshape(
                pred_trajs,
                [self.meta_period + 1, -1, self.num_particles, self.worker.ob_space.shape[0]])

            return costs, pred_trajs

        # if only the predicted actions should be returned
        else:

            # loop function for predicting future states using tensorflow
            def iteration(t, total_cost, cur_obs, cur_goals):
                cur_acs = ac_seqs[t]
                next_obs, next_goals = self._predict_next_obs(cur_obs, cur_acs, cur_goals)
                delta_cost = tf.reshape(
                    self.obs_cost_fn(next_obs, next_goals) + self.ac_cost_fn(
                        cur_acs), [-1, self.num_particles])

                next_obs = self.obs_postproc2(next_obs)
                return t + 1, total_cost + delta_cost, next_obs, next_goals

            # predict into the future using static graphs
            _, costs, _, _ = tf.while_loop(
                cond=continue_prediction,
                body=iteration,
                loop_vars=[t, init_costs, init_obs, init_goals]
            )

            # Replace nan costs with very high cost
            return tf.reduce_mean(tf.where(
                tf.is_nan(costs),
                1e6 * tf.ones_like(costs), costs), axis=1)

    def _expand_to_ts_format(self, mat):
        """Process a matrix into a format for the ensemble to predict over

        See: https://github.com/kchua/handful-of-trials/blob/
             e1a62f217508a384e49ecf7d16a3249e187bcff9/dmbrl/controllers/MPC.py#L265

        Parameters
        ----------
        mat : tf.Variable
            the tensor to be processed into the correct shape

        Returns
        -------
        tf.Variable
            the tensor with an adjusted shape
        """
        dim = mat.get_shape()[-1]
        return tf.reshape(
            tf.transpose(
                tf.reshape(
                    mat,
                    [-1, self.num_ensembles,
                     self.num_particles // self.num_ensembles, dim]),
                [1, 0, 2, 3]
            ),
            [self.num_ensembles, -1, dim]
        )

    def _flatten_to_matrix(self, ts_fmt_arr):
        """Process particles sampled for a trajectory into a matrix.

        See: https://github.com/kchua/handful-of-trials/blob/
             e1a62f217508a384e49ecf7d16a3249e187bcff9/dmbrl/controllers/MPC.py#L265

        Parameters
        ----------
        ts_fmt_arr : tf.Variable
            the tensor in the format for trajectory sampling

        Returns
        -------
        tf.Variable
            the tensor with a matrix shape
        """
        dim = ts_fmt_arr.get_shape()[-1]
        return tf.reshape(
            tf.transpose(
                tf.reshape(
                    ts_fmt_arr,
                    [self.num_ensembles, -1,
                     self.num_particles // self.num_ensembles, dim]),
                [1, 0, 2, 3]
            ),
            [-1, dim]
        )

    def _train_worker_model(self):
        """Train the dynamics model on data from the replay buffer.

        See: https://github.com/kchua/handful-of-trials/blob/
             e1a62f217508a384e49ecf7d16a3249e187bcff9/dmbrl/controllers/MPC.py#L265

        Parameters
        ----------

        Returns
        -------
        """
        worker_model_loss = 0.0

        if self._t > 0 and self._t % self.steps_before_planning == 0:
            _, _, _, _, _, worker_obs0, worker_obs1, worker_act, _, _, _ = \
                self.replay_buffer.sample(
                    batch_size=self.steps_before_planning, with_additional=False)

            goal_dim = self.manager.ac_space.shape[0]
            new_train_in = np.concatenate([
                self.obs_preproc(worker_obs0[:, :-goal_dim]), worker_act], axis=-1)
            new_train_targs = self.targ_proc(
                worker_obs0[:, :-goal_dim], worker_obs1[:, :-goal_dim])

            worker_model_loss = self.model.train(
                new_train_in, new_train_targs,
                batch_size=32, epochs=100, hide_progress=False,
                holdout_ratio=0.0, max_logging=5000).item()

        return worker_model_loss

