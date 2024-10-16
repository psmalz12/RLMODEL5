import random

# copied from GRIDWORLD22
# SARSA on policy
# In SARSA the Q-value update relies on the Q-value of the next state-action pair that the agent actually will take
#  rather than selecting the maximum Q-value among all possible actions in the next state (Q-learning).
#  This approach results in a more cearful update compared to Q-learning as it directly integrates the agent's policy into the Q-value update

class RL:
    def __init__(self, epsilon, learning_rate, epsilon_decay_rate, min_epsilon, epsilon_point):
        self.epsilon = epsilon  # high mean more explore new action & low exploit the action with the highest q-value
        self.learning_rate = learning_rate  # Learning rate for Q-value updates
        self.q_values = {}  # Q-value table (state-action pairs)
        self.st_epsilon = epsilon  # Store init epsilon for reset
        self.epsilon_decay_rate = epsilon_decay_rate  # Rate of epsilon decay
        self.epsilon_point = epsilon_point  # Point where epsilon decays
        self.min_epsilon = min_epsilon  # Min value for epsilon

    def action_policy(self, state, available_actions):
        """
        Choose an action (phase) based on the current state
        """


        if random.random() < self.epsilon:
            # Exploration: choose a random action (phase + duration)
            return random.choice(available_actions)
        else:
            # Exploitation: choose the best action with the highest Q-value
            if state in self.q_values:
                return max(self.q_values[state], key=self.q_values[state].get)  # Return action with max Q-value
            else:
                # If state is not in Q-values, choose a random action
                return random.choice(available_actions)

    def update_q_values(self, state, action, reward, next_state, gamma):
        """
        Update Q-values using the Q-learning update rule.
        """
        # to ensure reward is a float
        if isinstance(reward, dict):
            raise TypeError(f"Reward cannot be a dict: {reward}")

        if state not in self.q_values:
            self.q_values[state] = {}
        if action not in self.q_values[state]:
            self.q_values[state][action] = 0

        # Get the maximum Q-value for the next state
        max_next_q_value = max(self.q_values.get(next_state, {}).values(), default=0)

        # Q-learning update rule
        #self.q_values[state][action] += self.learning_rate * (reward + gamma * max(self.q_values.get(next_state, {}).values(), default=0) - self.q_values[state][action])

        self.q_values[state][action] += self.learning_rate * (reward + gamma * max_next_q_value - self.q_values[state][action])
        # Q(s, a) = Q(s, a) + alpha * (reward + gamma * max(Q(s', a')) - Q(s, a))
        # the Q-value update uses the maximum possible Q-value of the next state regardless of the action the agent actually takes.
        # this means that Q-Learning aims to learn the optimal policy directly by considering the best action in the next state
        # Return the updated q-value
        return self.q_values[state][action]

    def decay_epsilon(self, episode):
        """
        Decay epsilon over time to encourage exploitation as learning progresses
        """
        if episode < self.epsilon_point:
            self.epsilon = max(self.min_epsilon, self.st_epsilon - self.epsilon_decay_rate * episode / self.epsilon_point * self.st_epsilon)
        else:
            self.epsilon = self.min_epsilon

    def reset_epsilon(self):
        """
        Reset epsilon to its init value
        """
        self.epsilon = self.st_epsilon
