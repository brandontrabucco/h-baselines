import gym
import tensorflow as tf
import numpy as np

import hbaselines.trpo.tf_utils as tf_util
from hbaselines.trpo.utils import fmt_row
from hbaselines.trpo.dataset import Dataset
from hbaselines.trpo.algorithm import RLAlgorithm

import tensorflow_probability as tfp

tfco = tf.contrib.constrained_optimization
tfd = tfp.distributions


class KLProblem(tfco.ConstrainedMinimizationProblem):
    """Creat a constraint minimization problem.

    Parameters:
    -----------

    """

    def __init__(self, loss, meankl, kl_bound):
        self.loss = loss
        self.meankl = meankl
        self.kl_bound = kl_bound

    @property
    def objective(self):
        return self.loss

    @property
    def constraints(self):
        return self.meankl - self.kl_bound


class PPO(RLAlgorithm):
    """Proximal Policy Optimization.

    See: https://arxiv.org/abs/1707.06347

    Attributes
    ----------
    env : gym.Env or str
        The environment to learn from (if registered in Gym, can be str)
    policy : TODO
        The policy model to use
    timesteps_per_batch : int
        timesteps per actor per update
    clip_param : float
        clipping parameter epsilon
    entcoeff : float
        the entropy loss weight
    optim_epochs : float
        the optimizer's number of epochs
    optim_stepsize : float
        the optimizer's stepsize
    optim_batchsize : int
        the optimizer's the batch size
    gamma : float
        discount factor
    lam : float
        advantage estimation
    schedule : str
        The type of scheduler for the learning rate update ('linear',
        'constant', 'double_linear_con', 'middle_drop' or 'double_middle_drop')
    atarg : tf.placeholder
        Target advantage function (if applicable)
    ret : tf.placeholder
        placeholder for the discounted returns
    lrmult : tf.placeholder
        learning rate multiplier
    action_ph : tf.placeholder
        placeholder for the actions
    old_pi : tf.Tensor
        the output from the policy with trainable parameters from the previous
        step
    losses : TODO
        TODO
    loss_names : list of str
        TODO
    assign_old_eq_new : TODO
        TODO
    optimizer : tf.Operation
        TODO
    """

    def __init__(self,
                 policy,
                 env,
                 gamma=0.99,
                 timesteps_per_batch=2048,
                 clip_param=0.2,
                 vf_coeff=0.5,
                 entcoeff=0.01,
                 optim_epochs=4,
                 optim_stepsize=2.5e-4,
                 optim_batchsize=512,
                 lam=0.95,
                 schedule='constant',
                 verbose=0,
                 ignore_dones=False,
                 policy_kwargs=None):
        """Initialize the algorithm.

        Parameters
        ----------
        env : gym.Env or str
            The environment to learn from (if registered in Gym, can be str)
        policy : TODO
            The policy model to use
        timesteps_per_batch : int
            timesteps per actor per update
        clip_param : float
            clipping parameter epsilon
        entcoeff : float
            the entropy loss weight
        optim_epochs : float
            the optimizer's number of epochs
        optim_stepsize : float
            the optimizer's stepsize
        optim_batchsize : int
            the optimizer's the batch size
        gamma : float
            discount factor
        lam : float
            advantage estimation
        schedule : str
            The type of scheduler for the learning rate update ('linear',
            'constant', 'double_linear_con', 'middle_drop' or
            'double_middle_drop')
        verbose : int
            the verbosity level: 0 none, 1 training information, 2 tensorflow
            debug
        policy_kwargs : dict
            additional arguments to be passed to the policy on creation
        """
        super(PPO, self).__init__(policy=policy,
                                  env=env,
                                  timesteps_per_batch=timesteps_per_batch,
                                  gamma=gamma,
                                  lam=lam,
                                  ignore_dones=ignore_dones,
                                  verbose=verbose,
                                  policy_kwargs=policy_kwargs)

        self.clip_param = clip_param
        self.vf_coeff = vf_coeff
        self.entcoeff = entcoeff
        self.optim_epochs = optim_epochs
        self.optim_stepsize = optim_stepsize
        self.optim_batchsize = optim_batchsize
        self.schedule = schedule

        self.atarg = None
        self.ret = None
        self.lrmult = None
        self.action_ph = None
        self.old_pi = None
        self.losses = None
        self.loss_names = None
        self.assign_old_eq_new = None
        self.optimizer = None

        # Perform the algorithm-specific model setup procedure.
        self.setup_model()

        with self.graph.as_default():
            # Create the tensorboard summary.
            self.summary = tf.summary.merge_all()

            # Initialize the model parameters and optimizers.
            with self.sess.as_default():
                self.sess.run(tf.global_variables_initializer())

    def setup_model(self):
        """See parent class."""
        with self.graph.as_default():
            # Create relevant input placeholders.
            with tf.variable_scope("input", reuse=False):
                # Target advantage function (if applicable)
                self.atarg = tf.placeholder(
                    dtype=tf.float32, shape=[None])

                # Empirical return
                self.ret = tf.placeholder(
                    dtype=tf.float32, shape=[None])

                # learning rate multiplier, updated with schedule
                self.lrmult = tf.placeholder(
                    name='lrmult', dtype=tf.float32, shape=[])

                self.action_ph = tf.placeholder(
                    tf.float32, shape=(None, self.action_space.shape[0]))

                tf.summary.scalar('discounted_rewards',
                                  tf.reduce_mean(self.ret))
                tf.summary.scalar('learning_rate',
                                  tf.reduce_mean(self.optim_stepsize))
                tf.summary.scalar('advantage',
                                  tf.reduce_mean(self.atarg))
                tf.summary.scalar('clip_range',
                                  tf.reduce_mean(self.clip_param))

            # Network for old policy
            with tf.variable_scope("oldpi", reuse=False):
                self.old_pi = self.policy(
                    self.sess,
                    self.observation_space,
                    self.action_space,
                    reuse=False,
                    **self.policy_kwargs
                )

            # Make sure the global variables of the old and new policy match.
            assert len(tf_util.get_globals_vars("oldpi")) \
                == len(tf_util.get_globals_vars("model"))

            # A utility function that is used to assign the parameters of
            # the new policy to the old policy. Done after every update.
            self.assign_old_eq_new = [
                tf.assign(oldv, newv) for (oldv, newv) in
                zip(tf_util.get_globals_vars("oldpi"),
                    tf_util.get_globals_vars("model"))
            ]

            with tf.variable_scope("loss", reuse=False):
                # Annealed clipping parameter epsilon
                clip_param = self.clip_param * self.lrmult

                kloldnew = self.old_pi.proba_distribution.kl(
                    self.policy_pi.proba_distribution)
                ent = self.policy_pi.proba_distribution.entropy()
                meankl = tf.reduce_mean(kloldnew)
                meanent = tf.reduce_mean(ent)
                pol_entpen = -self.entcoeff * meanent

                # pnew / pold
                ratio = tf.exp(
                    self.policy_pi.proba_distribution.logp(self.action_ph) -
                    self.old_pi.proba_distribution.logp(self.action_ph))

                # surrogate from conservative policy iteration
                surr1 = ratio * self.atarg
                surr2 = tf.clip_by_value(
                    ratio, 1.0 - clip_param, 1.0 + clip_param) * self.atarg

                # PPO's pessimistic surrogate (L^CLIP)
                if self.duel_vf:
                    vf1, vf2 = self.policy_pi.value_flat
                    q1, q2 = self.policy_pi.q_fn
                    vf_loss = 0.5 * tf.reduce_mean(tf.square(vf1 - self.ret)) \
                        + tf.reduce_mean(tf.square(vf2 - self.ret))
                    q_loss = 0.5 * tf.reduce_mean(tf.square(q1 - self.ret)) \
                        + tf.reduce_mean(tf.square(q2 - self.ret))
                    pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2))
                    total_loss = pol_surr + pol_entpen

                    problem = KLProblem(total_loss, meankl, 0.01)

                    optimizer = tfco.AdditiveExternalRegretOptimizer(
                        optimizer=tf.train.AdamOptimizer(
                            learning_rate=self.optim_stepsize,
                            epsilon=1e-5,
                        )
                    )
                    self.optimizer = optimizer.minimize(
                        problem,
                        var_list=tf_util.get_trainable_vars("model")
                    )

                    # create an optimizer object
                    optimizer = tf.compat.v1.train.AdamOptimizer(
                        self.optim_stepsize * self.lrmult, epsilon=1e-5)

                    # create the optimization operation
                    self.vf_optimizer = optimizer.minimize(
                        vf_loss,
                        var_list=tf_util.get_trainable_vars("model")
                    )

                    # create the optimization operation
                    self.q_optimizer = optimizer.minimize(
                        q_loss,
                        var_list=tf_util.get_trainable_vars("model")
                    )
                else:
                    pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2))
                    vf_loss = 0.5 * tf.reduce_mean(
                        tf.square(self.policy_pi.value_flat - self.ret))
                    total_loss = \
                        pol_surr + pol_entpen + self.vf_coeff * vf_loss

                self.losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
                self.loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl",
                                   "ent"]
                if self.duel_vf:
                    self.losses.append(q_loss)
                    self.loss_names.append("q_loss")

                with tf.variable_scope("Adam_mpi", reuse=False):
                    # create an optimizer object
                    optimizer = tf.compat.v1.train.AdamOptimizer(
                        self.optim_stepsize * self.lrmult, epsilon=1e-5)

                    # create the optimization operation
                    self.optimizer = optimizer.minimize(
                        total_loss,
                        var_list=tf_util.get_trainable_vars("model")
                    )

                tf.summary.scalar('entropy_loss', pol_entpen)
                tf.summary.scalar('policy_gradient_loss', pol_surr)
                tf.summary.scalar('value_function_loss', vf_loss)
                tf.summary.scalar('approximate_kullback-leibler', meankl)
                tf.summary.scalar('clip_factor', clip_param)
                tf.summary.scalar('loss', total_loss)

        return tf_util.get_trainable_vars("model")

    def _train(self, seg, writer, total_timesteps):
        """See parent class.

        TODO: describe
        """
        if self.schedule == 'constant':
            cur_lrmult = 1.0
        elif self.schedule == 'linear':
            cur_lrmult = max(
                1.0 - float(self.timesteps_so_far) / total_timesteps, 0)
        else:
            raise NotImplementedError

        observations, actions = seg["observations"], seg["actions"]
        atarg, tdlamret = seg["adv"], seg["tdlamret"]

        # predicted value function before update
        vpredbefore = seg["vpred"]

        # standardized advantage function estimate
        atarg = (atarg - atarg.mean()) / atarg.std()
        dataset = Dataset(dict(ob=observations, ac=actions,
                               atarg=atarg, vtarg=tdlamret),
                          shuffle=True)
        optim_batchsize = self.optim_batchsize or observations.shape[0]

        # set old parameter values to new parameter values
        self.sess.run(self.assign_old_eq_new)

        print("Optimizing...")
        print(fmt_row(13, self.loss_names))

        # Here we do a bunch of optimization epochs over the data.
        for k in range(int(self.optim_epochs)):
            # list of tuples, each of which gives the loss for a minibatch
            losses = []
            for i, batch in enumerate(dataset.iterate_once(optim_batchsize)):
                # TODO: remove?
                steps = (self.timesteps_so_far +
                         k * optim_batchsize +
                         int(i * (optim_batchsize / len(dataset.data_map))))
                # Run loss backprop with summary.
                if self.duel_vf:
                    summary, vf_grad, q_grad, grad, *newlosses = self.sess.run(
                        [self.summary, self.vf_optimizer, self.q_optimizer,
                         self.optimizer] + self.losses,
                        feed_dict={
                            self.policy_pi.obs_ph: batch["ob"],
                            self.old_pi.obs_ph: batch["ob"],
                            self.action_ph: batch["ac"],
                            # self.policy_pi.ac_ph: batch["ac"],
                            self.atarg: batch["atarg"],
                            self.ret: batch["vtarg"],
                            self.lrmult: cur_lrmult,
                        }
                    )
                else:
                    summary, grad, *newlosses = self.sess.run(
                        [self.summary,  self.optimizer] + self.losses,
                        feed_dict={
                            self.policy_pi.obs_ph: batch["ob"],
                            self.old_pi.obs_ph: batch["ob"],
                            self.action_ph: batch["ac"],
                            self.atarg: batch["atarg"],
                            self.ret: batch["vtarg"],
                            self.lrmult: cur_lrmult,
                        }
                    )
                writer.add_summary(summary, steps)  # TODO: out of loop?
                losses.append(newlosses)
            print(fmt_row(13, np.mean(losses, axis=0)))

        # TODO: remove?
        print("Evaluating losses...")
        losses = []
        for batch in dataset.iterate_once(optim_batchsize):
            newlosses = self.sess.run(
                self.losses,
                feed_dict={
                    self.policy_pi.obs_ph: batch["ob"],
                    self.old_pi.obs_ph: batch["ob"],
                    self.action_ph: batch["ac"],
                    # self.policy_pi.ac_ph: batch["ac"],
                    self.atarg: batch["atarg"],
                    self.ret: batch["vtarg"],
                    self.lrmult: cur_lrmult,
                }
            )
            losses.append(newlosses)
        mean_losses = np.mean(losses, axis=0)

        return mean_losses, vpredbefore, tdlamret
