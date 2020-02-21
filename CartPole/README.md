
<h3 align="center">
  <img src="assets/cartpole.png" width=300px>
</h3>


<div align="center">

# CartPole-OpenAI

[![made-with-python](https://forthebadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

</div>

Reinforcement Learning solution of the [OpenAI's Cartpole](https://gym.openai.com/envs/CartPole-v0/).


## About

> Cartpole - known also as an Inverted Pendulum is a pendulum with a center of gravity above its pivot point. It’s unstable, but can be controlled by moving the pivot point under the center of mass. The goal is to keep the cartpole balanced by applying appropriate forces to a pivot point. [source](https://gym.openai.com/envs/CartPole-v0/)
------------------------------------------

- The OpenAI gym python toolkit was used to simulate the environment
- Training and Test examples were also generated using the same
- Neural Network Model was trained on the datasets using the Tensorflow Keras API
- The top score achieved by the model was 10000  which was the limit set by the environment 😁 
- **Training Accuracy** : 64%
- **Test Accuracy** : 61%
- To Try out:
  - Clone the Repo
  - `python3 cartpool.py`
## Demo
<div align="center">
  <img src="assets/cartpole.gif" width=600px>
</div>

## Future Plans

- [ ] Try to improve the accuracy of the model
- [ ] Increasing the value of `score_req`(Min score considered for training set examples) for better quality of examples
- [ ] Check the maximum score the model can achieve 😜

------------------------------------------
## Contributing
Open to `enhancements` & `bug-fixes`
## Note
The project was made just to try out the OpenAI Gym and I hope to solve many more OpenAI Gym Environments.
