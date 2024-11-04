import random

class RL:
    def __init__(self, epsilon, learning_rate, epsilon_decay_rate, min_epsilon, epsilon_point):
        self.epsilon = epsilon  # High means more exploration; low means exploitation of highest Q-value action
        self.learning_rate = learning_rate  # Learning rate for Q-value updates
        self.q_values = {}  # Q-value table for (state-action pairs)
        self.st_epsilon = epsilon  # Store initial epsilon for reset
        self.epsilon_decay_rate = epsilon_decay_rate  # Rate of epsilon decay
        self.epsilon_point = epsilon_point  # Point where epsilon decays
        self.min_epsilon = min_epsilon  # Minimum epsilon for exploration
        self.action_state_pairs = {}  # Store action-state pairs with Q-values

    phases = {
        0: "West-East Green - E0",
        3: "South-North Green - E2",
        6: "East-West Green - E1",
        9: "North-South Green - E3",
    }

    def action_policy(self, state, steps_in_episode, available_actions):
        """
        Choose action based on epsilon-greedy policy.
        """
        print(f"State {state}Available actions at step {steps_in_episode}: {available_actions}")

        if state not in self.q_values:
            self.q_values[state] = {action: 0 for action in available_actions} # Actions: Q-values {0: 0.0, 3: 0.0, 6: 0.0, 9: 0.0}
        if state not in self.action_state_pairs:
            self.action_state_pairs[state] = {}

        print(f"(in RL_agent) State: {state} -> Actions: {self.q_values[state]}")

        if random.random() < self.epsilon: # generates random number between 0 and 1 then comp it to epsilon
            # Exploration: choose an action based on modulo number of actions
            # use remainder dividing the episode step count by the number of available actions to pick an action index
            # step in episode we get it from main dont need too gnreat t++ here
            action_index = steps_in_episode % len(available_actions)
            action = available_actions[action_index]
            action_index = steps_in_episode % len(available_actions)
            print(f"Computed action index: {action_index}")

            print(f"Exploration chosen action: {action} -> Phase ({self.phases[action]})")
        else:
            # Exploitation: choose the action with the highest Q-value for this state
            action = max(self.q_values[state], key=self.q_values[state].get)
            print(f"Exploitation chosen action: {action} -> Phase ({self.phases[action]})")

        return action

    def update_q_values(self, state, action, reward, next_state, gamma):
        """
        update Q-values using the Q-learning update rule
        """

        # Init Q-values for the next_state if not already gnerated
        if next_state not in self.q_values:
            self.q_values[next_state] = {action: 0 for action in self.q_values[state].keys()}

        # Get the maximum Q-value for the next state if not is def to 0
        max_next_q_value = max(self.q_values.get(state, {}).values(), default=0)

        # Q-learning update rule
        prev_q_value = self.q_values[state][action]

        new_q_value = prev_q_value + self.learning_rate * (reward + gamma * max_next_q_value - prev_q_value)
        # Q(s, a) = Q(s, a) + alpha * (reward + gamma * max(Q(s', a')) - Q(s, a))
        # the Q-value update uses the maximum possible Q-value of the next state regardless of the action the agent actually takes
        # this means that Q-Learning aims to learn the optimal policy directly by considering the best action in the next state
        # Return the updated q-value
        self.q_values[state][action] = new_q_value

        # record the next state and updated Q-value for each state-action pair
        self.action_state_pairs[state][action] = (next_state, new_q_value)

        # print out change in Q-value for convergence check
        # compute the dif between the new and previous Q-value for state-action pair not nessery just see in some other code but did not change anything
       # q_value_delta = abs(new_q_value - prev_q_value)
        #print(f"Q-value delta for state {state} and action {action}: {q_value_delta}")
       # print(f"Updated Q-value for (state, action): ({state}, {action}): {new_q_value}")

        return new_q_value

    def decay_epsilon(self, episode):
        """
        Decay epsilon over time to encourage exploitation as learning progresses.
        """
        if episode < self.epsilon_point:
            # decayed epsilon based on episode number
            self.epsilon = max(self.min_epsilon,self.st_epsilon - self.epsilon_decay_rate * episode / self.epsilon_point * self.st_epsilon)
        else:
            # when reaching the epsilon_point episode remains at min_epsilon
            self.epsilon = self.min_epsilon

        print(f"Epsilon after episode {episode}: {self.epsilon}")

    def reset_epsilon(self):
        """
        reset epsilon to its initial value.
        """
        self.epsilon = self.st_epsilon

    def print_action_state_pairs(self): # not in use 
        """
        print the action-state pairs with Q-values.
        """
        print(" (in RL_agent3.py) Action-State Pairs with Q-values:")
        for state, actions in self.action_state_pairs.items():
            print(f"State: {state}")
            for action, (next_state, q_value) in actions.items():
                action_name = self.phases[action]
                print(f"  Action: {action_name} ({action}), Next State: {next_state}, Q-value: {q_value:.4g}")
