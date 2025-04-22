# Tetris

This repository contains an implementation of a Q-learning based Tetris agent, TetrisQAgent.java, designed to learn and play the game of Tetris using reinforcement learning. This agent is built within a custom Tetris framework provided by Boston University (edu.bu.tetris).

# Overview 

TetrisQAgent extends QAgent, and implements a Q-learning algorithm with a neural network as the Q-function approximator. It interacts with the game environment, extracts features from the board state, predicts Q-values for potential moves, and learns from experience using a customizable reward signal.

# Core Components

initQFunction()
Defines a simple feedforward neural network:
Input: 5-dimensional feature vector
Architecture: Input → Dense → ReLU → Dense → Output (Q-value)

getQFunctionInput(GameView, Mino)
Extracts hand-crafted features from the board and the current Mino:
Base height of the structure
Column bumpiness
Empty cells beneath blocks
Number of full rows
Mino type

shouldExplore(GameView, GameCounter)
Controls the agent’s exploration vs exploitation behavior using a decaying epsilon strategy.

getExplorationMove(GameView)
Selects a move probabilistically based on softmax-scaled Q-values, introducing guided randomness during exploration.

trainQFunction(Dataset, LossFunction, Optimizer, long)
Performs gradient-based updates to the Q-function using a replay buffer and minibatches for training stability.

getReward(GameView)
Custom reward function based on:
Score earned this turn
Number of full rows
Gaps and column irregularity
Height of the top block

#Features 

Hand-crafted state representation for faster learning
Adjustable exploration rate via exponential decay
Softmax-based exploration (rather than purely random)
Highly customizable reward design
Integration with BU’s TrainerAgent, Model, Dataset, and Matrix libraries

#Usage 

Place the file in the appropriate agents directory under your BU Tetris framework.
Run training via the TrainerAgent using this agent.
Monitor performance over episodes and adjust features or reward shaping as needed.

#Dependencies 
BU Tetris environment
Custom edu.bu.tetris libraries: nn, game, agents, utils, training

#How to Run 
java -cp "./lib/*:." edu.bu.tetris.Main -q src.pas.tetris.agents.TetrisQAgent | tee my_logfile.log 






