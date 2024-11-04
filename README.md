

The agent will add time step +(-1) each time he is seen in the env (window of 30 m) 


![snip](https://github.com/user-attachments/assets/7b7ef096-d9dd-44a7-980c-05f80babfb51)


This part is from ENV.py :

            


Then, the reward depends on this highest time step for each lane:

                                    def calculate_reward(self, previous_state, new_state):
                                            """
                                            Calculate the reward based on the time step changes in all lanes
                                            """
                                            reward = 0
                                            # compare previous and new states
                                            for lane_index in range(len(previous_state)):
                                                prev_time_step = previous_state[lane_index]
                                                new_time_step = new_state[lane_index]
                                    
                                                if new_time_step > prev_time_step:
                                                    reward -= 1  # increasing time steps
                                                elif new_time_step < prev_time_step:
                                                    reward += 1  # decreas time steps
                                    
                                            return reward

                        For Example:
                        
![image](https://github.com/user-attachments/assets/8279417f-7e43-4623-9760-7d344039479e)

and this is how store state action pairs: 
where we have 16,384 possible state action pairs 
![image](https://github.com/user-attachments/assets/49e7a7e9-789f-4289-8c39-3a925112fe18)





Here plotting the reward to episodes: 
3000 episodes - epsilon reaches its minimum value(0.05) at 1000 

1- a snip at each 100-episodes 


![RLMODEL 7-dif flow- 3000ep at 100 ep- reward](https://github.com/user-attachments/assets/3c7458b2-e249-49e5-b655-5e912b8e7e14)

2- a snip at each 20-episodes 

![RLMODEL 7-dif flow- 3000ep 2- reward](https://github.com/user-attachments/assets/0f77eff1-c267-46db-8fa5-b2dd642eff37)



Here, the Q-values have converged, which indicates that the agent has learned a stable and consistent policy for choosing actions. The agent now knows, for a given state, which actions tend to gain the best long-term rewards.




Q-learning update rule
                  
                  
                  
                  self.q_values[state][action] += self.learning_rate * (reward + gamma * max_next_q_value - self.q_values[state][action])


