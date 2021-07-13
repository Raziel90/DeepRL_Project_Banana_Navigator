# Report BananaBrain Submission

## **Model Description**
The code in this repository implements processes to train and execute autonomous agents to navigate in the BananaBrain Unity environment.
The code makes available 3 classes for agents (`Agent`, `ReplayDDQNAgent`, `PriorityReplayDDQNAgent`) using 2 different types of Neural Network (`DQN`, `Dueling_DQN`). Thus, it implements the following Deep Reinforcement Mechanisms:
- `DQN` is an approach to train agents using reinforcement learning using a Neural Network. It is used to estimate the `Q(s,a)` advantage of using a certain action `a` in a specific state `s`. The neural network is trained using the following loss.
    <!-- $$ 
    L_i(\Theta_i) = \mathbb{E}_{s,a,r,s' \sim \mathbb{U}(D)} [(r + \gamma\max_{a'}Q(s,a';Θ_i^{-}) - Q(s,a; Θ_i))^2]
    $$ --> 
    <div align="center"><img style="background: white;" src="../svg/gxiczXoKT3.svg"></div> in which:
  - <code>γ</code> is the discount factor determining the agent’s horizon;
  - <code>Θᵢ</code> are the parameters of the Q-network at iteration i;
  - <code>Θ⁻ᵢ</code> are the network parameters used to compute the target at iteration i;
  - <code>D</code> is the dataset <code>D = {e₁, e₂, ..., eₜ}</code> and <code>eₖ = (sₖ, aₖ, rₖ, sₖ₊₁)</code>
  - <code>r</code> is the reward
  -  The target network parameters <code>Θ⁻ᵢ</code> are only updated with the Q-network parameters <code>Θᵢ</code> every <code>C</code> steps and are held fixed between individual updates.
    
    In practice however, as a single network is used, the target are computed with the following following formula:

    <!-- $
    r_{t+1} + γQ(S_{t+1}, \underset{a} {\mathrm{argmax}} Q(s_{t+1},a;Θ_i);Θ_i)
    $ -->
    <div align="center"><img style="background: white;" src="../svg/ClrjitGl3B.svg"></div>
    Which often results in overoptimistic values.

- `Double DQN (DDQN)` is a modification of the DQN trained agents in which the target and local network parameters of the formula above are extrapolated by 2 separate networks occasionally synchornized when perfoming the learning phase. Using a single network for considering both parameters causes an overoptimistic estimation of the Q function. 
  In this case the target are computed as following:
<!-- $
r_{t+1} + γQ(S_{t+1}, \underset{a} {\mathrm{argmax}} Q(s_{t+1},a;Θ_i);Θ^{-}_i)
$ --> 
<div align="center"><img style="background: white;" src="../svg/yUOfwqdxnl.svg"></div>

- `Memory Replay`
- `Priority Memory Replay`
- `Dueling Neural Network`

## **Training Performance**
![Training Score averaged](./assets/training.gif)
![Training Score averaged](./assets/figs/trained_model.svg)
## **Example Run**
![Example execution of the trained agent](./assets/example-run.gif)

## **Future Improvements**

The agents performs already quite well during the game. However, its average score seems to saturate between 15 and 16. Thus it is still improvable. In particularm the following misbehaviours have been found.
1. In certain situations, the agent seems looping without knowing how to proceed, wasting precious steps until the end of the episode.  
2. When a yellow and a blue banana are very close to each other the agents approaches losing the game.


In the case of (1.) one could think to evaluate differt variations of the reward function. For example, introducing mechanisms to make the agent more efficient in its decisions like lowering the reward based on the number of steps between consecutive positive rewards. Such mechanism should be used very carefully, because if the penalty is too high the agent may prefer to take a blue banana nearby rather than a far away yellow banana.
Similarly, for reducing (2.) the negative reward for getting a blue banana could be amplified to avoid selecting yellow bananas close to blue ones.

More importantly, the agent should be trained on the image data from the environment, which may make the agent play with the same baseline as a human would.

