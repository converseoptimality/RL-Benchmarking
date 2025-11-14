# Benchmarking RL via Converse Optimality

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)

Official implementation of **"Benchmarking RL via Converse Optimality: Generating Systems with Known Optimal Policies"**. A novel framework for creating reinforcement learning benchmarks with **provably known optimal policies** and **analytical value functions**, enabling rigorous evaluation against ground truth.




## Motivations and objectives

Traditional RL benchmarking suffers from not knowing the true optimal performance. Our method solves the *converse optimality problem* to generate environments where the optimal policy `π*` and value function `V*` are known exactly, allowing precise measurement of:
- **Optimality gaps**
- **True regret**
- **Sample efficiency**
- **Generalization performance**

### NVDEx system
### Arm link system

## Main features

- **Ground-truth evaluation**: Compare algorithms against certified optimal policies
- **Stochastic control-affine systems**: Discrete-time, nonlinear systems with additive Gaussian noise
- **Tunable difficulty**: Homotopy parameters control problem complexity while maintaining analytical optimality
- **Multiple benchmark families**:
  - `ConverseArm-v0`: n-link planar robotic arm with realistic dynamics
  - `NUDEx`: Nonholonomic vehicle with dynamic extension (open-loop unstable)
  - Multi-agent coordination tasks
- **Certified optimality**: Every benchmark comes with mathematically proven `(π*, V*)`

## NVDEx system


Here we can see a few tests on the performance of the Vehicle system, with different starting positions and different hyper-parameters in the difficulty ladder

<img width="1489" height="1589" alt="image" src="https://github.com/user-attachments/assets/984c3668-c465-48db-ab64-5a614825ae01" />

## Arm link system

Optimal control vs Uncontrolled
<img width="1044" height="759" alt="image" src="https://github.com/user-attachments/assets/2087eeee-0373-4d55-ac26-494f148da720" />




