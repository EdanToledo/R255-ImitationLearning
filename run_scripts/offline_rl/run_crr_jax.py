# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""An example CRR running on locomotion datasets (mujoco) from D4rl."""

from absl import app
from absl import flags
import acme
from acme import specs
from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax import actors
from acme.agents.jax import crr
from acme.datasets import tfds
import run_scripts.imitation_learning.helpers as helpers
from acme.jax import variable_utils
from acme.types import Transition
from acme.utils import loggers
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import rlds

# Agent flags
flags.DEFINE_integer("batch_size", 64, "Batch size.")
flags.DEFINE_integer("evaluate_every", 20, "Evaluation period.")
flags.DEFINE_integer("evaluation_episodes", 10, "Evaluation episodes.")
flags.DEFINE_integer(
    "num_demonstrations",
    10,
    "Number of demonstration episodes to load from the dataset. If None, loads the full dataset.",
)
flags.DEFINE_integer("seed", 0, "Random seed for learner and evaluator.")
# CQL specific flags.
flags.DEFINE_float("policy_learning_rate", 3e-5, "Policy learning rate.")
flags.DEFINE_float("critic_learning_rate", 3e-4, "Critic learning rate.")
flags.DEFINE_float("discount", 0.99, "Discount.")
flags.DEFINE_integer("target_update_period", 100, "Target update periode.")
flags.DEFINE_integer("grad_updates_per_batch", 1, "Grad updates per batch.")
flags.DEFINE_bool(
    "use_sarsa_target",
    True,
    "Compute on-policy target using iterator actions rather than sampled " "actions.",
)
# Environment flags.
flags.DEFINE_string("env_name", "HalfCheetah-v2", "What environment to run")
flags.DEFINE_integer("num_demonstrations", 11, "Number of demonstration trajectories.")

FLAGS = flags.FLAGS


def _add_next_action_extras(double_transitions: Transition) -> Transition:
    return Transition(
        observation=double_transitions.observation[0],
        action=double_transitions.action[0],
        reward=double_transitions.reward[0],
        discount=double_transitions.discount[0],
        next_observation=double_transitions.next_observation[0],
        extras={"next_action": double_transitions.action[1]},
    )


def main(_):
    key = jax.random.PRNGKey(FLAGS.seed)
    key_demonstrations, key_learner = jax.random.split(key, 2)

    # Create an environment and grab the spec.
    environment = helpers.make_environment(task=FLAGS.env_name)
    environment_spec = specs.make_environment_spec(environment)

    dataset_name = helpers.get_dataset_name(FLAGS.env_name)

    # Get a demonstrations dataset with next_actions extra.
    transitions = tfds.get_tfds_dataset(
        dataset_name, FLAGS.num_demonstrations, env_spec=environment_spec
    )

    double_transitions = rlds.transformations.batch(
        transitions, size=2, shift=1, drop_remainder=True
    )
    transitions = double_transitions.map(_add_next_action_extras)
    demonstrations = tfds.JaxInMemoryRandomSampleIterator(
        transitions, key=key_demonstrations, batch_size=FLAGS.batch_size
    )

    # Create the networks to optimize.
    networks = crr.make_networks(environment_spec)

    # CRR policy loss function.
    policy_loss_coeff_fn = crr.policy_loss_coeff_advantage_exp

    # Create the learner.
    learner = crr.CRRLearner(
        networks=networks,
        random_key=key_learner,
        discount=FLAGS.discount,
        target_update_period=FLAGS.target_update_period,
        policy_loss_coeff_fn=policy_loss_coeff_fn,
        iterator=demonstrations,
        policy_optimizer=optax.adam(FLAGS.policy_learning_rate),
        critic_optimizer=optax.adam(FLAGS.critic_learning_rate),
        grad_updates_per_batch=FLAGS.grad_updates_per_batch,
        use_sarsa_target=FLAGS.use_sarsa_target,
    )

    def evaluator_network(
        params: hk.Params, key: jnp.DeviceArray, observation: jnp.DeviceArray
    ) -> jnp.DeviceArray:
        dist_params = networks.policy_network.apply(params, observation)
        return networks.sample_eval(dist_params, key)

    actor_core = actor_core_lib.batched_feed_forward_to_actor_core(evaluator_network)
    variable_client = variable_utils.VariableClient(learner, "policy", device="cpu")
    evaluator = actors.GenericActor(actor_core, key, variable_client, backend="cpu")

    eval_loop = acme.EnvironmentLoop(
        environment=environment,
        actor=evaluator,
        logger=loggers.TerminalLogger("evaluation", time_delta=0.0),
    )

    # Run the environment loop.
    while True:
        for _ in range(FLAGS.evaluate_every):
            learner.step()
        eval_loop.run(FLAGS.evaluation_episodes)


if __name__ == "__main__":
    app.run(main)
