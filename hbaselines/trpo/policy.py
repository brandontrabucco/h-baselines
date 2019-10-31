import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np


class FeedForwardPolicy(object):
    """Actor critic feed forward neural network.

    Attributes
    ----------
    sess : tf.Session
        The current TensorFlow session
    ob_space : gym.spaces.*
        The observation space of the environment
    ac_space : gym.spaces.*
        The action space of the environment
    reuse : bool
        If the policy is reusable or not
    layers : list of int
        The size of the Neural network for the policy (if None, default to
        [64, 64])
    act_fun : tf.nn.*
        the activation function to use in the neural network.
    shared : bool
        specifies whether to shared the hidden layers between the actor and
        critic
    duel_vf : bool
        specifies whether to create two value functions. If not, only one is
        created.
    obs_ph : tf.compat.v1.placeholder
        observation placeholder
    value_fn : tf.Tensor
        value estimate, of shape (self.n_batch, 1)
    value_flat : tf.Tensor
        value estimate, of shape (self.n_batch, )
    proba_distribution : ProbabilityDistribution
        distribution of stochastic actions.
    action : tf.Tensor
        stochastic action, of shape (self.n_batch, ) + self.ac_space.shape.
    deterministic_action : tf.Tensor
        deterministic action, of shape (self.n_batch, ) + self.ac_space.shape.
    neglogp : tf.Tensor
        negative log likelihood of the action sampled by self.action.
    policy_proba : tf.Tensor
        parameters of the probability distribution. Depends on pdtype.
    """

    def __init__(self,
                 sess,
                 ob_space,
                 ac_space,
                 layers,
                 act_fun,
                 shared,
                 duel_vf,
                 reuse=False):
        """Instantiate the policy.

        Parameters
        ----------
        sess : tf.Session
            The current TensorFlow session
        ob_space : gym.spaces.*
            The observation space of the environment
        ac_space : gym.spaces.*
            The action space of the environment
        reuse : bool
            If the policy is reusable or not
        layers : list of int
            The size of the Neural network for the policy
        act_fun : tf.nn.*
            the activation function to use in the neural network.
        shared : bool
            specifies whether to shared the hidden layers between the actor and
            critic
        duel_vf : bool
            specifies whether to create two value functions. If not, only one
            is created.
        """
        self.sess = sess
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.reuse = reuse
        self.layers = layers
        self.act_fun = act_fun
        self.shared = shared
        self.duel_vf = duel_vf

        # Create the observation placeholder
        with tf.variable_scope("input", reuse=False):
            self.obs_ph = tf.compat.v1.placeholder(
                tf.float32, shape=(None,) + ob_space.shape, name="obs_ph")
            self.ac_ph = tf.compat.v1.placeholder(
                tf.float32, shape=(None,) + ac_space.shape, name="ac_ph")

        with tf.variable_scope("model", reuse=reuse):
            # # Flatten the observations before passing to the policy.
            # flat_obs = tf.layers.flatten(self.obs_ph)
            # flat_ac = tf.layers.flatten(self.ac_ph)

            # Create the actor mean and logstd, and the value function(s).
            if shared:
                mean, logstd, value_fn, q_fn = self._shared_mlp(
                    self.obs_ph, self.layers, self.act_fun)
            else:
                mean, logstd, value_fn, q_fn = self._non_shared_mlp(
                    self.obs_ph, self.ac_ph, self.layers, self.act_fun, duel_vf)

            self._mean = mean
            self._logstd = logstd
            self._std = tf.exp(self._logstd)
            self.value_fn = value_fn
            self.q_fn = q_fn
            if duel_vf:
                vf1, vf2 = self.value_fn
                self.value_flat = (vf1[:, 0], vf2[:, 0])
            else:
                self.value_flat = self.value_fn[:, 0]

            # Create a probability distribution from the
            self.proba_distribution = DiagGaussianProbabilityDistribution(
                self._mean, self._logstd
            )

        # Compute an action from the probability distribution.
        self.action = self._mean + self._std * tf.random_normal(
            tf.shape(self._mean), dtype=self._mean.dtype)

        # Set up the distributions, actions, and value.
        self.deterministic_action = self._mean
        self.neglogp = self.proba_distribution.neglogp(self.action)
        self.policy_proba = [self._mean, self._std]

    def compute_action(self, obs, deterministic=False):
        """Return the policy for a single step.

        Parameters
        ----------
        obs : array_like
            The current observation of the environment
        deterministic : bool
            Whether or not to return deterministic actions.

        Returns
        -------
        (array_like, array_like, array_like)
            actions, values, neglogp
        """
        if deterministic:
            action = self.sess.run(self.deterministic_action,
                                   feed_dict={self.obs_ph: obs})
        else:
            action = self.sess.run(self.action,
                                   feed_dict={self.obs_ph: obs})

        return action

    def proba_step(self, obs):
        """Return the action probability for a single step.

        Parameters
        ----------
        obs : array_like
            The current observation of the environment

        Returns
        -------
        array_like
            the action probability
        """
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, use_q=False):
        """Return the value for a single step.

        Parameters
        ----------
        obs : array_like
            The current observation of the environment
        use_q : bool
            specifies whether to use the Q-function to estimate the value

        Returns
        -------
        array_like
            The associated value of the action
        """
        if use_q:
            actions = [self.compute_action(obs) for _ in range(1)]
            q_values = [
                self.sess.run(self.q_fn,
                              feed_dict={self.obs_ph: obs, self.ac_ph: ac})
                for ac in actions
            ]
            return (np.mean([q for q, _ in q_values]),
                    np.mean([q for _, q in q_values]))
        else:
            return self.sess.run(self.value_flat, {self.obs_ph: obs})

    def _shared_mlp(self, flat_obs, layers, act_fun):
        """Create a shared representation of the policy.

        The hidden layers are shared by both the actor and value function,
        following by separate linear layers to compute the outputs of each. For
        the actor, this output acts as the mean; the logstd is computed by a
        single trainable variable.

        **Note**: Duel value functions does not work in this setting.

        Parameters
        ----------
        flat_obs : tf.Tensor
            The observations to base policy and value function on.
        layers : list of int
            The number of units per layer.
        act_fun : tf.nn.*
            The activation function to use for the networks.

        Returns
        -------
        tf.Tensor
            mean value from the policy's actor
        tf.Tensor
            logstd value from the policy's actor
        tf.Tensor
            output from the policy's critic / value function
        """
        latent = flat_obs

        # Iterate through the shared layers and build the shared parts of the
        # network
        for idx, layer in enumerate(layers):
            layer_size = layer
            latent = tf.compat.v1.layers.dense(
                latent,
                layer_size,
                activation=act_fun,
                name="shared_{}".format(idx),
                kernel_initializer=slim.variance_scaling_initializer(
                    factor=1.0 / 3.0, mode='FAN_IN', uniform=True),
            )

        return self._gen_output_layers(latent, latent, None)

    def _non_shared_mlp(self, obs, action, layers, act_fun, duel_vf):
        """Create a non-shared representation of the policy.

        In this case, the actor and critic policies do not share any hidden
        layers.

        Parameters
        ----------
        obs : tf.Tensor
            The observations to base policy and value function on.
        action : tf.Tensor
            the action to base the Q-function in part on
        layers : list of int
            The number of units per layer.
        act_fun : tf.nn.*
            The activation function to use for the networks.
        duel_vf : bool
            specifies whether to create two value functions. If not, only one
            is created.

        Returns
        -------
        tf.Tensor
            mean value from the policy's actor
        tf.Tensor
            logstd value from the policy's actor
        tf.Tensor or (tf.Tensor, tf.Tensor)
            output from the policy's critic / value function. Tuple if duel_vf
            is set to True
        """
        vf_latent = (obs, obs) if duel_vf else obs
        pi_latent = obs
        q_latent = (tf.concat([obs, action], axis=-1),
                    tf.concat([obs, action], axis=-1))

        # Iterate through and build separate hidden layers for the actor and
        # the critic.
        for idx, layer in enumerate(layers):
            if duel_vf:
                vf1, vf2 = vf_latent
                vf1 = self._layer(vf1, layer, "vf1_{}".format(idx), act_fun)
                vf2 = self._layer(vf2, layer, "vf2_{}".format(idx), act_fun)
                vf_latent = (vf1, vf2)

                q1, q2 = q_latent
                q1 = self._layer(q1, layer, "q1_{}".format(idx), act_fun)
                q2 = self._layer(q2, layer, "q2_{}".format(idx), act_fun)
                q_latent = (q1, q2)
            else:
                vf_latent = self._layer(
                    vf_latent, layer, "vf_{}".format(idx), act_fun)

                q_latent = None

            pi_latent = self._layer(
                pi_latent, layer, "pi_{}".format(idx), act_fun)

        return self._gen_output_layers(pi_latent, vf_latent, q_latent)

    def _gen_output_layers(self, pi_latent, vf_latent, q_latent):
        """Create the necessary output layers.

        Parameters
        ----------
        pi_latent : tf.Tensor
            the output from the hidden layers of the policy
        vf_latent : tf.Tensor or (tf.Tensor, tf.Tensor)
            the output from the hidden layers of the value function(s). If a
            tuple of tensors of tensors are provided, then this implies that
            duel value functions are being created.

        Returns
        -------
        tf.Tensor
            mean value from the policy's actor
        tf.Tensor
            logstd value from the policy's actor
        tf.Tensor or (tf.Tensor, tf.Tensor)
            output from the policy's critic / value function
        """
        if isinstance(vf_latent, tuple):
            # Separate the tuple of vf latents and use these as the inputs to
            # final layer for each of the two value functions.
            vf1_latent, vf2_latent = vf_latent
            vf1 = self._layer(vf1_latent, 1, 'vf1_out', None)
            vf2 = self._layer(vf2_latent, 1, 'vf2_out', None)

            # The output value function is a tuple of both value functions.
            value_fn = (vf1, vf2)

            # Separate the tuple of Q-functions latents and use these as the
            # inputs to final layer for each of the two value functions.
            q1_latent, q2_latent = q_latent
            q1 = self._layer(q1_latent, 1, 'q1_out', None)
            q2 = self._layer(q2_latent, 1, 'q2_out', None)

            # The output Q-function is a tuple of both Q-functions.
            q_fn = (q1, q2)
        else:
            # Add an extra layer to produce the output from the value function.
            value_fn = self._layer(vf_latent, 1, 'vf_out', None)

            # No Q-function in this case for now.
            q_fn = None

        # Add an extra layer after the shared layers to produce the mean action
        # by the actor.
        mean = self._layer(pi_latent, self.ac_space.shape[0], 'pi_mean', None)

        # The logstd is a single trainable variable.
        logstd = tf.get_variable(
            name='pi_logstd',
            shape=(1, self.ac_space.shape[0]),
            initializer=tf.zeros_initializer()
        )

        return mean, logstd, value_fn, q_fn

    @staticmethod
    def _layer(in_val, num_output, name, act_fun):
        """Create a fully connected layer.

        This is here to reduce code duplication.

        Parameters
        ----------
        in_val : tf.Tensor
            input tensor
        num_output : int
            number of outputs from the layer
        name : str
            the name of the layer
        act_fun : tf.nn.* or None
            activation function. None refers to a linear combination

        Returns
        -------
        tf.Tensor
            the output from the new layer
        """
        return tf.compat.v1.layers.dense(
            in_val,
            num_output,
            activation=act_fun,
            name=name,
            kernel_initializer=slim.variance_scaling_initializer(
                factor=1.0 / 3.0,
                mode='FAN_IN',
                uniform=True
            )
        )


class DiagGaussianProbabilityDistribution(object):
    """TODO

    """

    def __init__(self, mean, logstd):
        """
        Probability distributions from multivariate gaussian input

        :param flat: ([float]) the multivariate gaussian input data
        """
        self.mean = mean
        self.logstd = logstd
        self.std = tf.exp(logstd)
        self.flat = tf.concat([mean, mean * 0.0 + logstd], axis=1)

    def flatparam(self):
        """Return the direct probabilities."""
        return self.flat

    def mode(self):
        """Return the probability (deterministic action)."""
        return self.mean

    def neglogp(self, x):
        """Return the of the negative log likelihood.

        Parameters
        ----------
        x : str
            the labels of each index

        Returns
        -------
        array_like
            The negative log likelihood of the distribution
        """
        return 0.5 * tf.reduce_sum(tf.square((x - self.mean) / self.std),
                                   axis=-1) \
            + 0.5 * np.log(2. * np.pi) * tf.cast(tf.shape(x)[-1], tf.float32) \
            + tf.reduce_sum(self.logstd, axis=-1)

    def kl(self, other):
        """Calculate the Kullback-Leibler divergence of the distribution.

        Parameters
        ----------
        other : list of float
            the distribution to compare with

        Returns
        -------
        float
            the KL divergence of the two distributions
        """
        assert isinstance(other, DiagGaussianProbabilityDistribution)
        return tf.reduce_sum(
            other.logstd - self.logstd +
            (tf.square(self.std) + tf.square(self.mean - other.mean)) /
            (2.0 * tf.square(other.std)) - 0.5,
            axis=-1)

    def entropy(self):
        """Return shannon's entropy of the probability."""
        return tf.reduce_sum(self.logstd + .5 * np.log(2.0 * np.pi * np.e),
                             axis=-1)

    def sample(self):
        """Return a sample from the probability distribution.

        Note: Bounds are taken into account outside this class (during training
        only).

        Returns
        -------
        tf.Tensor
            the stochastic action
        """
        return self.mean + self.std * tf.random_normal(tf.shape(self.mean),
                                                       dtype=self.mean.dtype)

    def logp(self, x):
        """Return the of the log likelihood.

        Parameters
        ----------
        x : str
            the labels of each index

        Returns
        -------
        list of float
            the log likelihood of the distribution
        """
        return - self.neglogp(x)
