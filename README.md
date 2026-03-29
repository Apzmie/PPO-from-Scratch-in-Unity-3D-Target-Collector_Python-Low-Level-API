# PPO-from-Scratch-in-Unity-3D-Target-Collector (Python-Low-Level-API)
Because directly maximizing rewards is difficult to optimize due to high variance, Proximal Policy Optimization (PPO) uses a surrogate objective function that increases the probability of good actions while decreasing it for bad ones. PPO maximizes this function with an Actor-Critic method, where the critic evaluates the state values to help the actor choose better actions through advantages that measure how much better actions are than the expectation. PPO is not only applicable to continuous action spaces by sampling actions from probability distributions, but also clips the probability ratio between the new and old policies to prevent large updates.

In this project, PPO is used with an MLP, taking the agent’s velocities, configurable joint velocities, and the relative target position as inputs, with the target angular velocities for each joint as outputs. PPO is implemented in PyTorch, based on the paper [*Proximal Policy Optimization Algorithms*](https://arxiv.org/pdf/1707.06347).

You can play against the target collector AI directly in your browser [Play on itch.io](https://apzmie.itch.io/target-collector-ai), on both PC and mobile devices.

<img src="images/target_collector.png" alt="target_collector" width="40%">

## Environment
### Unity
- Unity Editor: 6000.3.0f1
- ML Agents: 4.0.2
- Sentis: 2.5.0

### Python
- Python 3.10.12

## PPO Diagram
![dqn_diagram](images/ppo_diagram.png)
### GAE (Generalized Advantage Estimation)
The TD error ($\delta$) evaluates whether the action taken in the current state was good or bad, representing the basic advantage. To estimate the advantage more accurately, the next state's advantage is added to the current one recursively to include future information. This means that the advantage added to the first one incorporates the entire future information of a trajectory, which is enabled by calculating GAE backwards when implementing.
