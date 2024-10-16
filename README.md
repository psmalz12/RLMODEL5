

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
