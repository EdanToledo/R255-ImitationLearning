## Continuous control

We include a number of agents running on continuous control tasks. These agents
are representative examples, but any continuous control algorithm implemented in
Acme should be able to be swapped in.

Note that many of the examples, particularly those based on the DeepMind Control
Suite, will require a [MuJoCo license](https://www.roboti.us/license.html) in
order to run. See our [tutorial] for more details or see refer to the
[dm_control] repository for further information.

-   [D4PG](https://colab.research.google.com/github/deepmind/acme/blob/master/examples/baselines/rl_continuous/run_d4pg.py): a deterministic policy gradient (D4PG) agent which includes a determinstic
 policy and a distributional critic running on the DeepMind Control Suite or
 the [OpenAI Gym]. By default it runs on the "half cheetah" environment from the
 OpenAI Gym.
-   [MPO](https://colab.research.google.com/github/deepmind/acme/blob/master/examples/baselines/rl_continuous/run_mpo.py): a maximum-a-posterior policy optimization agent which combines both a distributional critic and a stochastic policy.

[dm_control]: https://github.com/deepmind/dm_control
[OpenAI Gym]: https://github.com/openai/gym


