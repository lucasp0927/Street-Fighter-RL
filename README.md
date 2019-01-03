# StreetFighterRL

In this project, I would like to develop an AI to play the fighting video game, Street Fighter I. The AI learns its gaming strategy through reinforcement learning.
A modified Q-learning algorithm, double deep Q network (DDQN) is applied. Given information such as two players’ positions and postures as states, with changes in
Health Points(HP) as reward, the AI should learn to make actions which optimize its rewards.

Today we can play Street Fighter I on personal computers with Multiple Arcade Machine Emulator(MAME). MAME’s debugger mode helps reverse engineering random
access memory of the game and lets us access useful inforamtion on current game
status. The new [MAME LUA API](http://docs.mamedev.org/techspecs/luaengine.html), 
which was just released on April this year, offers
ioport functions for manipulating I/O input of the game. This makes the implementation 
of automated game control a lot more efficient and easier.
As for machine learning methods in the project, a three-layered, double deep Q network is used to approximate the Q values in Q-learning. Given current state of game
play, the AI would act the action with highest Q value. To make the training process
feasible and more efficient, I use rectified linear units(ReLU) as activation function
and RMSProp as optimizer(update rules) in the neural network, some other machine
learning techiniques are also applied.

Here is a [project report](https://github.com/rpedsel/StreetFighterRL/blob/master/project_report.pdf) for previous progress.
