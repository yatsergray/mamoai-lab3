# Reinforcement Learning in GridWorld

## Project Overview

This project explores reinforcement learning (RL) techniques, particularly **Q-learning**, in a stochastic environment. The environment consists of **rooms with narrow doorways**, **gold** (reward), and **bombs** (negative reward). The main goal of the agent is to **find the optimal strategy to navigate the environment** while avoiding bombs and maximizing rewards.

## Problem Statement

- **Environment Type:** Grid-based world with obstacles, rewards, and stochasticity.
- **Grid Size:** 17x17 (scalable).
- **State Types:**
  - **Passage:** -1 reward (movement cost).
  - **Wall:** 0 reward (impassable).
  - **Terminal States:**
    - **Gold:** +50 or +25 reward.
    - **Bomb:** -50 penalty.
- **Stochasticity:**
  - The environment is **partially random**, meaning the agent can sometimes execute an unintended action due to a **slip probability (ε in ε-greedy strategy)**.

![Generated Environment Example](/images/grid1.png)

## Implementation Details

Google Colab: [Link](https://colab.research.google.com/drive/13ttsI_p37HaKKweeJGrOAQzZ0ZrZuPRU?usp=sharing)

- **Algorithm:** Q-learning.
- **Learning Type:** Online learning — Q-table values are updated dynamically after each action.
- **Exploration Strategy:** ε-greedy policy:
  - With probability ε, the agent takes a **random** action.
  - With probability **1 - ε**, the agent follows the **best** known action from the Q-table.

## Code Overview

```python
import pandas as pd

from classes.Agent import Agent
from classes.Converter import Converter
from classes.GridWorld import GridWorld
from classes.QLearning import QLearning

def run():
    g = GridWorld(room_depth=3, room_width=3, rooms_number_in_depth=4, rooms_number_in_width=4)
    
    g.build()
    
    g.put_gold(gold_number=1)
    g.put_gold(25, 2)
    g.put_bombs(bombs_number=15)
    
    a = Agent(g)
    q = QLearning(a)
    c = Converter(g)
    
    rewards_per_episode = q.learn(episodes_number=10000)
    
    first_path = q.get_shortest_path()
    second_path = q.get_shortest_path()
    third_path = q.get_shortest_path()
    
    c.convert_grid_into_plot(first_path)
    c.convert_grid_into_plot(second_path)
    c.convert_grid_into_plot(third_path)
    
    c.convert_learning_data_into_plot(rewards_per_episode)

if __name__ == '__main__':
    run()
```

## Results

### Training Performance

- The **total reward per episode** fluctuates, indicating both successful gold retrieval and bomb encounters.
- As training progresses, the agent **learns to avoid bombs** and **maximize rewards**.

![Training Performance](/images/graph1.png)
![Training Performance](/images/grid2.png)

### Optimal Strategy

- The **final learned policy** directs the agent toward **high-reward gold (50 points)**.
- The strategy is **optimal** as the environment follows **Markov Decision Process (MDP) properties**.
- Training was conducted with **10,000 episodes** at a **low slip probability (0.1)**.

![Optimal Strategy](/images/grid3.png)

## Effect of Slip Probability

- **High slip probability (ε = 1.0):** The agent moves randomly, leading to many **negative rewards**.
- **Moderate slip probability (ε = 0.5):** Some randomness, but the agent can still find **optimal paths**.
- **Low slip probability (ε = 0.01):** The agent **rarely deviates** from the best path but may **miss better rewards**.

![Plots Comparison](/images/general.png)

## Conclusion

- **Q-learning is effective** in training an agent to navigate a stochastic environment.
- **Exploration-exploitation balance** (ε-greedy strategy) significantly impacts performance.
- **Optimized policy ensures** the agent finds the **best reward** while avoiding obstacles.

## Future Work

- Implement **deep Q-learning** (DQN) for large state spaces.
- Introduce **dynamic obstacles** for added complexity.
- Extend to **multi-agent RL** scenarios.
