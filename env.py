import traci
import random

class Env:
    def __init__(self):
        self.state = None  # Init the state of the env
        self.rewards = {}  # DIC to store rewards for dif state-action pairs
        self.action = None  # Init the state of the env
        self.action_state_pairs = {}  # track state-action pairs
        self.current_phase_duration = 0  # track the duration of the current phas
        self.duration = 6 # Set duration for TLP (sec) Gren L
        self.yellow_duration= 3
        self.total_steps = 0  # Track the total number of steps taken
        self.vehicle_time_steps = {}  # Init dictionary to track vehicle time steps

    def extract_state(self, junction_id):
        """
        Extract the current state of the junction (within 35 meters from the junction)
        Track vehicle time steps, and maintain their ID, waiting time, and type
        """
        state = {}

        # get all lanes controlled by the traffic light at this jun
        lanes = traci.trafficlight.getControlledLanes(junction_id)
        unique_lanes = list(set(lanes))  # Remove duplicates

        # set a limit for counting vehicles
        distance_limit = 30
        for lane in unique_lanes:
            # Get the total length of the lane
            lane_length = traci.lane.getLength(lane)

            # get all vehicles in this lane
            vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)

            # only consider vehicles within the distance_limit m from junction ((backword))
            vehicles_within_limit = [veh_id for veh_id in vehicle_ids if
                                     (lane_length - traci.vehicle.getLanePosition(veh_id)) <= distance_limit]

            # Init a list for the lane to store vehicle-specific info and time steps
            lane_info = []

            for veh_id in vehicles_within_limit:
                # get the waiting time of each vehicle not needed (each step = 9 sec)
               # waiting_time = traci.vehicle.getWaitingTime(veh_id)

                # get the vehicle type   - onto
                vehicle_type = traci.vehicle.getTypeID(veh_id)

                # update time step: if the veh was seen before add time step, otherwise init it by -1
                if veh_id not in self.vehicle_time_steps:
                    self.vehicle_time_steps[veh_id] = -1  # 1st time seeing this veh
                else:
                    self.vehicle_time_steps[veh_id] -= 1  # Seen before + time step

                # store the information as a tuple (vehicle ID, time step, waiting time, vehicle type)
                #lane_info.append((veh_id, self.vehicle_time_steps[veh_id], waiting_time, vehicle_type))
                lane_info.append((veh_id, self.vehicle_time_steps[veh_id],  vehicle_type)) # without actual timing

            # add lane vehicle info to the state as a tuple
            state[lane] = tuple(lane_info)

        # add the current traffic light phase to the state
        current_phase = traci.trafficlight.getPhase(junction_id)
        state["current_phase"] = current_phase
        # ====== State Before taken Action (lane(vehicle  ID, vehicle time step "-" , vehicle type) current_phase):
        # (('-E1_0', (('vehicle_3.10', -1, 2.0, 'veh_passengerG'), ('vehicle_3.9', -2, 3.0, 'veh_passengerG'), ('vehicle_4.16', -4, 5.0, 'veh_passengerW'), ('vehicle_3.8', -5, 5.0, 'veh_passengerG'))), ('-E2_0', (('vehicle_5.13', -9, 0.0, 'veh_passengerW'), ('vehicle_5.12', -10, 0.0, 'veh_passengerW'))), ('-E3_0', ()), ('E0_0', (('vehicle_1.15', -2, 9.0, 'veh_passengerG'), ('vehicle_1.14', -3, 16.0, 'veh_passengerG'), ('vehicle_1.13', -4, 15.0, 'veh_passengerG'), ('vehicle_2.8', -5, 22.0, 'Ambulance'))), ('current_phase', 4)) for Q-Learning =====****=====

        # Convert the state dictionary into a tuple (sorted by lane ID to ensure order)
        state_tuple = tuple(sorted(state.items()))
        return state_tuple

    def take_action(self, action, junction_id):
        """
        Apply the selected action which includes both the phase and duration to the TL
        """
        # save the current state as the previous state
        previous_state = self.state

        # If this is  1st step - init the previous state
        if previous_state is None:
            print("Previous state is None because it is the first step in the Episode")
            previous_state = self.extract_state(junction_id)

        # +  total step counter in the episode
        self.total_steps += 1
        print(f"Total steps: {self.total_steps}")

        # apply the action (phase ) and get the new state after applying it
        new_state = self.agentmove(junction_id, action)

        # Calculate total time steps for each lane --- need to check because already did veh time step
        total_time_steps = self.time_step_count1(new_state)
        print(f"Total time steps in each lane after action: {total_time_steps}")

        # add the highest time step for each lane
        highest_time_steps = self.time_step_count2(new_state)

        print(f"Highest time steps in each lane after action: {highest_time_steps}")

        # Sum the highest time steps across all lanes as the reward
        reward = sum(highest_time_steps.values())

        #print(f"Lane time steps after action: {highest_time_steps}")
        print(f"Cumulative reward based on highest time steps: {reward}")

        # Update the state to the new one after taking the action
        self.state = new_state

        # Return the new state, the reward, the previous state, the done flag, and the total reward
        total_reward = reward
        return new_state, reward, previous_state, False, total_reward


    def agentmove(self, junction_id, action):
        """
        move the traffic light to the chosen phase
        """
        phase = action  # unpack the action into phase and duration

        # Define the green and yellow phas
        phases = {
            0: "West-East Green - E0",
            1: "West-East Yellow - E0",
            3: "South-North Green - E2",
            4: "South-North Yellow - E2",
            6: "East-West Green - E1",
            7: "East-West Yellow - E1" ,
            9: "North-South Green - E3",
            10: "North-South Yellow - E3"
        }

        # get the current traffic light phase
        current_phase = traci.trafficlight.getPhase(junction_id)

        # if the current phase is not the same as the new phase, change to the new phase
        if current_phase != phase:
            print(f"Setting new phase: {phases[phase]} at junction {junction_id}.")
            traci.trafficlight.setPhase(junction_id, phase)
        # Set the phase duration to the chosen value
        print(f"Setting phase duration: {self.duration} seconds for phase {phases[phase]}.")
        traci.trafficlight.setPhaseDuration(junction_id, self.duration)

        # advance the simulation by the chosen phase duration
        for _ in range(self.duration):
            traci.simulationStep()

        # set amber phase after the green phase completes
        yellow_phase = phase + 1
        print(f"Switching to amber phase: {phases[yellow_phase]} to clear the jun")
        traci.trafficlight.setPhase(junction_id, yellow_phase)

        # amber duration set to 3 seconds 1/3 of total duration
        for _ in range(self.yellow_duration):
            traci.simulationStep()

        # after completing the action (extract_state) the new state
        new_state = self.extract_state(junction_id)
        return new_state

    def reset(self):
        """
        Reset the env to its ini state
        This Only clears any previous state-action pairs

        """
        self.action_state_pairs = {}
        return self.state

    def reset2(self):
        """
        Reset the env to its init state
        """
        self.action_state_pairs = {}  # clear state-action pairs memory
        self.total_steps = 0  # Reset total steps at the beginning of each episode
        self.state = None  # reset state to None
        return self.state

    def calculate_reward(self, previous_state, current_state):
        """
        Calculate the reward based on the change in waiting times for vehicles across all lanes
        """
        prev_waiting_times = [lane[2] for lane in previous_state[:-1]] # -1 dont include phase -  3rd element (index 2) represents the total waiting time for that lane.
        curr_waiting_times = [lane[2] for lane in current_state[:-1]]

        reward = 0
        for prev_wait, curr_wait in zip(prev_waiting_times, curr_waiting_times):
            if curr_wait < prev_wait:
                reward += 1  #
            elif curr_wait > prev_wait:
                reward -= 1  # do need to add if =?

        return reward

    def calculate_reward2(self, previous_state, current_state):
        """
        Calculate the reward based on the difference in total waiting time between the previous and current state.
        """
        prev_waiting_times = [lane[2] for lane in previous_state[:-1]]
        curr_waiting_times = [lane[2] for lane in current_state[:-1]]
        print(f"Previous waiting times: {prev_waiting_times}, Current waiting times: {curr_waiting_times}")

        # sum total waiting time in the previous and current states
        total_prev_waiting_time = sum(prev_waiting_times)
        total_curr_waiting_time = sum(curr_waiting_times)

        # The reward is the subtraction of waiting time
        reward = total_prev_waiting_time - total_curr_waiting_time

        return reward

    def time_step_count1(self, state):
        """
        Calculate the total time step for all vehicles in each lane
        key is the lane and the value is the total time step
        """
        lane_time_steps = {}

        # Iterate over each lane in the state (skip the "current_phase" entry)
        for lane, vehicles in state:
            if lane == "current_phase":
                continue  # Skip the traffic light phase info

            # Sum the time steps for all vehicles in the lane
            total_time_step = sum(vehicle[1] for vehicle in vehicles) if vehicles else 0
            lane_time_steps[lane] = total_time_step

        return lane_time_steps

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


    def time_step_for_lanes(self, action, previous_state, current_state): # not in use
        """
        Calculate the total time step for each lane, ignoring the lane with the green light (action).
        Returns a numeric reward based on waiting times.
        """
        lane_time_steps = {}

        # process each lane in the state (skip the current_phase)
        for lane, vehicles in current_state:
            if lane == "current_phase":
                continue  # Skiping the phase key

            # ignore the lane with current green light
            if action in (0, 3, 6, 9) and lane in dict(previous_state):
                continue

            # Sum the time steps for all vehicles in the lane
            total_time_step = sum(
                vehicle[1] for vehicle in vehicles)  # use the second element in the tuple for time step
                    # (('-E1_0', (('vehicle_3.10', ((((-1)))), 2.0, 'veh_passengerG'), ('vehicle_3.9', -2, 3.0, 'veh_passengerG'), ('vehicle_4.16', -4, 5.0, 'veh_passengerW'), ('vehicle_3.8', -5, 5.0, 'veh_passengerG'))), ('-E2_0', (('vehicle_5.13', -9, 0.0, 'veh_passengerW'), ('vehicle_5.12', -10, 0.0, 'veh_passengerW'))), ('-E3_0', ()), ('E0_0', (('vehicle_1.15', -2, 9.0, 'veh_passengerG'), ('vehicle_1.14', -3, 16.0, 'veh_passengerG'), ('vehicle_1.13', -4, 15.0, 'veh_passengerG'), ('vehicle_2.8', -5, 22.0, 'Ambulance'))), ('current_phase', 4)) for Q-Learning =====****=====

            lane_time_steps[lane] = total_time_step

        # the reward can be sum of all lane
        total_reward = sum(lane_time_steps.values())

        return total_reward

#https://sumo.dlr.de/pydoc/traci._trafficlight.html
#setPhase(self, tlsID, index)
#setPhase(string, integer) -> None
#Switches to the phase with the given index in the list of all phases for the current program.

#setPhaseDuration(self, tlsID, phaseDuration)
#setPhaseDuration(string, double) -> None
# also look at @How to use the traci.trafficlight function in traci
# https://snyk.io/advisor/python/traci/functions/traci.trafficlight

#Set the remaining phase duration of the current phase in seconds.
#This value has no effect on subsquent repetitions of this phase.