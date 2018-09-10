import numpy as np
from cost_functions import trajectory_cost_fn
import time

class Controller():
	def __init__(self):
		pass

	# Get the appropriate action(s) for this state(s)
	def get_action(self, state):
		pass


class RandomController(Controller):
	def __init__(self, env):
		""" YOUR CODE HERE """
		self.env = env

	def get_action(self, state):
		""" YOUR CODE HERE """
		""" Your code should randomly sample an action uniformly from the action space """
		return self.env.action_space.sample()


class MPCcontroller(Controller):
	""" Controller built using the MPC method outlined in https://arxiv.org/abs/1708.02596 """
	def __init__(self, 
				 env, 
				 dyn_model, 
				 horizon=5, 
				 cost_fn=None, 
				 num_simulated_paths=10,
				 ):
		self.env = env
		self.dyn_model = dyn_model
		self.horizon = horizon
		self.cost_fn = cost_fn
		self.num_simulated_paths = num_simulated_paths

	def get_action(self, state):
		""" YOUR CODE HERE """
		""" Note: be careful to batch your simulations through the model for speed """
		state = np.expand_dims(state, axis = 0)
		states = state
		for i in range(self.num_simulated_paths - 1):
			states = np.concatenate((states, state), axis = 0)
		action_list = []
		states_list = [states]
		for i in range(self.horizon):
			actions = np.expand_dims(self.env.action_space.sample(), axis = 0)
			for j in range(self.num_simulated_paths - 1):
				actions = np.concatenate((actions, np.expand_dims(self.env.action_space.sample(), axis = 0)), axis = 0)
			action_list.append(actions)
			next_states = self.dyn_model.predict(states_list[i], actions)
			states_list.append(next_states)
		trajectory_costs = []
		for i in range(self.num_simulated_paths):
			trajectory_costs.append(trajectory_cost_fn(self.cost_fn, np.array(states_list[:-1])[:, i, :], np.array(action_list)[:, i, :], np.array(states_list[1:])[:, i, :]))
		best_trajectory = np.argmax(trajectory_costs)
		return action_list[0][best_trajectory, :]

