

The agent will add time step +(-1) each time he is seen in the env (window of 30 m) 


![snip](https://github.com/user-attachments/assets/7b7ef096-d9dd-44a7-980c-05f80babfb51)


this part is from ENV.py :

             def time_step_count2(self, state):
                    """
                    Calculate the highest (most negative) time step for all vehicles in each lane
                    key is the lane and the value is the most negative time step
                    """
                    lane_max_time_step = {}
            
                    # Iterate over each lane in the state (skip the current_phase)
                    for lane, vehicles in state:
                        if lane == "current_phase":
                            continue  # Skiping traffic light phase info
            
                        # find the most negative (smallest) time step for all veh in this lane
                        if vehicles:
                            max_time_step = min(vehicle[1] for vehicle in vehicles)  # Find the most negative time step
                        else:
                            max_time_step = 0  # If no vehicles in the lane set to 0
            
                        lane_max_time_step[lane] = max_time_step
            
                    return lane_max_time_step


Then the reward depends on this highest time step for each lane:

                            def calculate_reward4(self, previous_state, current_state):
                                """
                                Calculate the reward based on the difference between the highest time-step (most negative) in each lane before the action and after the action )
                                """
                                # Get the highest time steps for each lane in the previous and current_states
                                pre_highest_time_steps = self.time_step_count2(previous_state)  # Previous highest time steps per lane
                                #print(f"calculate_reward4 fun :")
                                #print(f"previous_highest_time_steps: {pre_highest_time_steps}")
                        
                                cur_highest_time_steps = self.time_step_count2(current_state)  # Current highest time steps per lane
                                #print(f"current_highest_time_steps: {cur_highest_time_steps}")
                        
                                #  The difference in the highest time steps for each lane
                                highest_time_step_diffs = {lane:  cur_highest_time_steps[lane] - pre_highest_time_steps[lane] for lane in pre_highest_time_steps}
                                print(f"difference in the highest time steps for each lane (current-previous) : {highest_time_step_diffs}")
                        
                                # The reward is the sum of the differences in the highest time steps for each lane
                                total_reward = sum(highest_time_step_diffs.values())
                        
                                return total_reward


                        For Example:
                        
                              previous_highest_time_steps: {'-E1_0': -13, '-E2_0': 0, '-E3_0': 0, 'E0_0': -12}
                              current_highest_time_steps: {'-E1_0': -7, '-E2_0': 0, '-E3_0': 0, 'E0_0': -13}
                              difference in the highest time steps for each lane (current-previous) : {'-E1_0': 6, '-E2_0': 0, '-E3_0': 0, 'E0_0': -1}
                              Cumulative reward based on calculate_reward4: 5




![RLMODEL5 - 500 ep - reward4](https://github.com/user-attachments/assets/f616f896-ba87-4ad4-bb8d-5ffe999fc1f1)

![RLMODEL5 -2 - 500 ep - reward4](https://github.com/user-attachments/assets/c2ad3a34-8132-4ace-bd49-f64b14ac5cff)


Here, the Q-values have converged, which indicates that the agent has learned a stable and consistent policy for choosing actions. The agent now knows, for a given state, which actions tend to gain the best long-term rewards.

Fluctuations in rewards might indicate that the environment is not deterministic. This could be due to traffic variability, random actions taken for exploration, or other dynamic factors affecting the agent's immediate rewards. see sutton p.545


Q-learning update rule
                self.q_values[state][action] += self.learning_rate * (reward + gamma * max_next_q_value - self.q_values[state][action])


