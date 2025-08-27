# VDInM
A versatile decision-making agent for infrastructure life-cycle management
<img src="https://github.com/LAILI-civil/VDBrM/blob/main/logo.jpg" width="200px">  
Introduction 
-------
VDbrM is an intelligent decision-making software for infrastructural inspection and maintenance.<br>
The software is developed using Python, with the Markov matrix optimization part utilizing functions from Python's SciPy library. For training the neural network of the decision-making agent, TensorFlow 2.17 and the A3C algorithm are used, and the GUI interactions are developed with PyQt5.<br>

.PY function 
-------
Markov_state_transition.py receives the four state duration parameters and outputs the state transition matrix.<br>
Observation.py receives the inspection type and accuracy and outputs the observation error matrix.<br>
general_decision_framework.py constructs the component maintenance problem into POMDPs. Then, POMDPs are set as environments in Deep reinforcement learning. This function is for decision-making agent training.<br>
UI_design.py develops the GUI interface for users.<br>

GUI 
-------
This operational interface enables users to (1) utilize historical inspection data to fit the Markov deteriorated process.<br>
<img src="https://github.com/LAILI-civil/VDBrM/blob/main/Module1.jpg" width="500px">  <br>
(2) Select target infrastructure components and input management information, including the current state parameters, corrosion resistance duration, and state transition probability matrices.<br>
<img src="https://github.com/LAILI-civil/VDBrM/blob/main/Module2.jpg" width="500px">  <br>
(3) Design the optimized maintenance scheme from the AI agent. The system additionally generates prognostic visualizations depicting possible deterioration trajectories and corresponding intervention strategies over a 20-year horizon, providing maintenance planners with an anticipatory decision-making policy. <br>
<img src="https://github.com/LAILI-civil/VDBrM/blob/main/Module3.jpg" width="500px">  <br>

Authors
-------
Copyright (c) 2025-2026 Li LAI <civil-li.lai@connect.polyu.hk> and You DONG <you.dong@polyu.edu.hk> <br>
Li LAI <br>
A PhD graduate from Hong Kong Polytechnic University <br>
You DONG <br>
Associate Professor at Hong Kong Polytechnic University <br>
Research in infrastructure management, life cycle risk control, climate change, city resilience, digital twins, and sustainability. <br>
[Research interest](https://youdongpolyu.weebly.com/)

