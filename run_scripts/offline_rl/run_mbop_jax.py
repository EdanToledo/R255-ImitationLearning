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

"""An example running MBOP on D4RL dataset."""

import functools

from absl import app
from absl import flags
import acme
from acme import specs
from acme.agents.jax import mbop
from acme.datasets import tfds
import helpers
from acme.jax import running_statistics
from acme.utils import loggers
import jax
import optax
import tensorflow_datasets

# Training flags.
FLAGS = flags.FLAGS

flags.DEFINE_integer("num_networks", 10, "Number of ensemble networks.")
flags.DEFINE_float("learning_rate", 1e-3, "Learning rate.")
flags.DEFINE_integer("batch_size", 64, "Batch size.")
flags.DEFINE_multi_integer(
    "hidden_layer_sizes", [64, 64], "Sizes of the hidden layers."
)
flags.DEFINE_integer(
    "num_sgd_steps_per_step",
    1,
    "Denotes how many gradient updates perform per one learner step.",
)
flags.DEFINE_integer(
    "num_normalization_batches",
    50,
    "Number of batches used for calculating the normalization statistics.",
)
flags.DEFINE_integer("evaluate_every", 20, "Evaluation period.")
flags.DEFINE_integer(
    "evaluation_episodes", 10, "Evaluation episodes."
)
flags.DEFINE_integer("seed", 0, "Random seed for learner and evaluator.")

# Environment flags.
flags.DEFINE_string(
    "env_name", "HalfCheetah-v2", "Gym mujoco environment name."
)


def main(_):
    # Create an environment and grab the spec.
    environment = helpers.make_environment(task=FLAGS.env_name)
    spec = specs.make_environment_spec(environment)

    key = jax.random.PRNGKey(FLAGS.seed)
    key, dataset_key, evaluator_key = jax.random.split(key, 3)

    dataset_name = helpers.get_dataset_name(FLAGS.env_name)
    
    # Load the dataset.
    dataset = tensorflow_datasets.load(dataset_name)["train"]
    # Unwrap the environment to get the demonstrations.
    dataset = mbop.episodes_to_timestep_batched_transitions(dataset, return_horizon=10)
    dataset = tfds.JaxInMemoryRandomSampleIterator(
        dataset, key=dataset_key, batch_size=FLAGS.batch_size
    )

    # Apply normalization to the dataset.
    mean_std = mbop.get_normalization_stats(dataset, FLAGS.num_normalization_batches)
    apply_normalization = jax.jit(
        functools.partial(running_statistics.normalize, mean_std=mean_std)
    )
    dataset = (apply_normalization(sample) for sample in dataset)

    # Create the networks.
    networks = mbop.make_networks(
        spec, hidden_layer_sizes=tuple(FLAGS.hidden_layer_sizes)
    )

    # Use the default losses.
    losses = mbop.MBOPLosses()

    def logger_fn(label: str, steps_key: str):
        return loggers.make_default_logger(label, steps_key=steps_key)

    def make_learner(name, logger_fn, counter, rng_key, dataset, network, loss):
        return mbop.make_ensemble_regressor_learner(
            name,
            FLAGS.num_networks,
            logger_fn,
            counter,
            rng_key,
            dataset,
            network,
            loss,
            optax.adam(FLAGS.learning_rate),
            FLAGS.num_sgd_steps_per_step,
        )

    learner = mbop.MBOPLearner(
        networks,
        losses,
        dataset,
        key,
        logger_fn,
        functools.partial(make_learner, "world_model"),
        functools.partial(make_learner, "policy_prior"),
        functools.partial(make_learner, "n_step_return"),
    )

    planning_config = mbop.MPPIConfig()

    assert (
        planning_config.n_trajectories % FLAGS.num_networks == 0
    ), "Number of trajectories must be a multiple of the number of networks."

    actor_core = mbop.make_ensemble_actor_core(
        networks, planning_config, spec, mean_std, use_round_robin=False
    )
    evaluator = mbop.make_actor(actor_core, evaluator_key, learner)

    eval_loop = acme.EnvironmentLoop(
        environment=environment,
        actor=evaluator,
        logger=loggers.TerminalLogger("evaluation", time_delta=0.0),
    )

    # Train the agent.
    while True:
        for _ in range(FLAGS.evaluate_every):
            learner.step()
        eval_loop.run(FLAGS.evaluation_episodes)


if __name__ == "__main__":
    app.run(main)
