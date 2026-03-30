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
The TD error ($\delta$) evaluates whether the action taken in the current state was good or bad, representing the basic advantage. To estimate the advantage more accurately, the next state's advantage is added to the current one recursively to include future information. This means that the advantage added to the first one incorporates the entire future information of a trajectory, which is enabled by calculating GAE backwards when implementing. Gamma ($\gamma$) 0.99 is multiplied to make the current more important than the future, and lambda ($\lambda$) 0.95 is multiplied instead of using 1, which trusts the whole trajectory completely.

### Loss
The objective function has to be maximized to properly adjust action probabilities, and entropy has to be maximized to encourage exploration. However, PyTorch's autograd is designed to minimize the loss when calculating gradients. The solution is to apply a negative sign to maximize them. Value loss is multiplied by a coefficient of 0.5 to prioritize policy loss, and entropy loss is multiplied by a coefficient like 0.001 to allow for small exploration.

### Iterative Update
Once all transitions are collected, the parameters are updated using mini-batches to adjust the gradient direction little by little, with multiple epochs to maximize learning from a limited amount of data. This iterative update does not change the policy dramatically due to clipping mechanism.

### Zero Gradient
![zero_gradient](images/zero_gradient.png)

## Training Progress
![plot](images/plot.png)

In the policy loss plot, values above 0 indicate reducing the probability of bad actions, while values below 0 indicate increasing the probability of good actions. Oscillation around zero indicates learning well, rather than the loss staying only on one side throughout training.

In the entropy loss plot, a decrease in value indicates increased exploration, while an increase in value indicates reduced exploration. The entropy loss should rise gradually for successful learning, indicating that the agent is increasing its confidence in its actions.

## Conclusion
The training was completed quickly with high performance. As expected, the agent reduces the speed of one wheel or applies counter-force to it when rotating.

The real challenge is to make the agent walk with legs instead of wheels. Despite several attempts using both Configurable Joint and Articulation Body, it did not progress well as expected. The Articulation Body version showed almost no movement during deterministic actions. The Configurable Joint version was able to move forward with knee-less legs during deterministic actions, but the results were highly inconsistent depending on minor factors like body size or leg spacing. When knees were added, the training failed to converge entirely. 
