## About
This is the code for experiments of value-based contractual q-learning for multi-agent reinforcement learning, which explores how non-cooperative agents behave when rewards shares can be traded against non-optimal policies.

## Setup
1. Set up a virtualenv with 
2. Run ```pip install -r requirements.txt``` to get requirements
3. We used neptune.ai for experiment tracking. In [neptune.ai](neptune.ai), create a Neptune project following the [documentation](https://docs.neptune.ai/setup/creating_project/). Add you project name and api_token in [main.py](main.py). 

## Remarks
1. We use the config file [params](params-0.json) to configure the environment and the RL algorithm for our experiments.
2. Results of the experiments are stored in the [experiments](experiments) folder and within the neptune.ai project.
3. The code for the environment can be found in [envs](envs) folder. The size of the environment and number/behavior of machines can be configured in [params-0.json](params-0.json).
4. The code for the A2C and DQN agents can be found in the [agents](agents) folder.
5. The code for the contractual behavior can be found in the [contracting](contracting) folder.
6. Visualization is disabled.
