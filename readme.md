-# OptSFC
# OptSFC - Optimizer for security functions

-----------------------------------------

## :page_with_curl: What is OptSFC?


OptSFC is a ML framework to optimize Moving Target Defense (MTD) strategies in Telco Cloud networks using deep Reinforcement Learning (deep-RL). It uses both legacy mono-objective algorithms, such as PPO, A2C, and DQN; as well as multi-objective RL (MORL), such as EUPG and MORL Envelope.

## :clipboard: Features

- Modeling into a Multi-Objective Markov Decision Process (MOMDP) a 5G testbed identifying three concurrent optimization objectives: improve security, reduce the overhead on services' performance, and reduce the operational cost of the MTD mechanisms
- Empirical data on the cost of virtual resources based on RAM, CPU and hot-storage, as well as the network metrics of an actual 5G testbed used for the simulated environment for the deep-RL agents training
- A framework allowing to benchmark different state-of-the-art deep-RL algorithms using the OpenAI Gym interface

## :hammer_and_pick: Quick Start

**REQUIREMENTS:**
- Operating System: Ubuntu 18.04
- Python3.8 (```sudo apt install python3.8```)
- Python3-pip (```sudo apt install python3-pip```)
- ML modules such as Torch, OpenAI Gym, Stable-Baselines, MORL-baselines, and other Python-modules to install from the requirements.txt file in  (```pip install -r requirements.txt```)
