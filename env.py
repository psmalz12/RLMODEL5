import traci

class Env:
    def __init__(self):
        self.state = None  # Init the state of the env
        self.rewards = {}  # dictionary to store rewards for dif state-action pairs
        self.action_state_pairs = {}  # track state-action pairs
        self.total_steps = 0  # track the total number of steps taken (in each eps)
        self.duration = 15  # set duration for green traffic light phase (seconds) this changed because the veh speed to leave the jun
        self.yellow_duration = 5  # set duration for amber traffic light phase (seconds)
        self.vehicle_time_steps = {}  # Init dictionary to track veh time steps

    def extract_state(self, junction_id):
        """
        Extract the highest waiting times for vehicles in each lane within a 30-meter limit
        """
        # 1- extract lanes in the env
        lanes = traci.trafficlight.getControlledLanes(junction_id)
        incoming_lanes = list(set(lanes))  # remove duplicates only show the incoming lanes
        distance_limit = 30
        lane_count = 4  # Four lanes: E0, E1, E2, E3

        # dictionary for lane names
        lane_mapping = {"E0_0": 0, "-E1_0": 1, "-E2_0": 2, "-E3_0": 3}
        # Init list to store max waiting time for each lane default 0
        highest_waiting_times = [0] * lane_count
        # Sort lanes based on lane_mapping order to ensure order in printing
        incoming_lanes.sort(key=lambda lane: lane_mapping.get(lane, float('inf')))

        # Process each lane individual
        for lane in incoming_lanes:
            lane_length = traci.lane.getLength(lane)
            vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)
            lane_index = lane_mapping.get(lane, None)
            # Only consider vehicles within the distance_limit meters from the junction (backward)
            vehicles_within_limit = [veh_id for veh_id in vehicle_ids if (lane_length - traci.vehicle.getLanePosition(veh_id)) <= distance_limit]

            if lane_index is not None:
                print(f"vehicle_ids for lane {lane} (mapped to index {lane_index}): {vehicles_within_limit}")


                for veh_id in vehicles_within_limit:
                    # Update time step for the vehicle
                    if veh_id not in self.vehicle_time_steps:
                        self.vehicle_time_steps[veh_id] = 1  # First time seeing this vehicle
                    else:
                        self.vehicle_time_steps[veh_id] += 1  # Increment

                    # Track the highest waiting time per lane
                    highest_waiting_times[lane_index] = max(highest_waiting_times[lane_index],
                                                            self.vehicle_time_steps[veh_id])

        # Print the highest waiting times per lane for debugging
        print(f" (In Env3.py) Highest waiting times per lane: {highest_waiting_times}")

        # Store and return the state as a tuple in the order [E0, E1, E2, E3]
        self.state = tuple(highest_waiting_times)
        print(f"(In Env3.py) Extracted state for junction {junction_id}: {self.state}")
        return self.state

    def get_highest_time_steps(self, vehicle_time_steps_per_lane): # not in use in env3
        """
        extract the highest time step for each lane from the total vehicle time steps
        """
        highest_time_steps = []
        for lane_time_steps in vehicle_time_steps_per_lane:
            # get the maximum time step for the lane - default to 0 if there are no veh
            highest_time_step = max(lane_time_steps, default=0)
            highest_time_steps.append(highest_time_step)

        return highest_time_steps

    def take_action(self, action, junction_id):
        """
        Apply the selected action to the traffic light and return the results of the action
        """
        action_ac = {0: "West-East Green - E0", 3: "South-North Green - E2", 6: "East-West Green - E1",
                     9: "North-South Green - E3"}

        if self.state is None:
            # Extract init state if not yet set
            self.state = self.extract_state(junction_id)

        # Capture the current state as the previous state
        previous_state = tuple(self.state)
        self.total_steps += 1  # Increment the total steps

        # Execute the chosen action, observe new state and calculate reward
        new_state = self.agentmove(junction_id, action)
        reward = self.calculate_reward(previous_state, new_state)

        # Track total cumulative reward
        total_reward = reward  # Init with the immediate reward

        # Record rewards for state-action pairs
        if previous_state not in self.rewards:
            self.rewards[previous_state] = {}
        self.rewards[previous_state][action] = self.rewards[previous_state].get(action, 0) + reward

        # Update state-action pairs for each state and actions
        if previous_state not in self.action_state_pairs:
            self.action_state_pairs[previous_state] = {}  # Initi action dict for this state
        self.action_state_pairs[previous_state][action] = new_state  # Store action and resulting next state
        print(f" (In Env3.py) action_state_pairs: {self.action_state_pairs}")

        # Print the state-action pairs for debug
        #print(f"self.action_state_pairs: {self.action_state_pairs}")
        print(" (In Env3.py) State-Action Pairs:")
        for state, actions in self.action_state_pairs.items():
            print(f"Current State: {state}")
            for action, resulting_state in actions.items():
                action_name = action_ac.get(action, f"Unknown Action ({action})")
                print(f"  Action: {action_name} ({action}) -> Resulting State: {resulting_state}")
        # Update the env state to the new state after taking the action
        self.state = new_state


        return new_state, reward, previous_state, False, total_reward

    def agentmove(self, junction_id, action):
        """
        move the traffic light to the chosen phase
        """
        phase = action  # unpack the action into phase

        # Define the green and yellow phas
        phases = {
            0: "West-East Green - E0",
            1: "West-East Yellow - E0",
            3: "South-North Green - E2",
            4: "South-North Yellow - E2",
            6: "East-West Green - E1",
            7: "East-West Yellow - E1",
            9: "North-South Green - E3",
            10: "North-South Yellow - E3"
        }

        # get the current traffic light phase
        current_phase = traci.trafficlight.getPhase(junction_id)

        # if the current phase is not the same as the new phase, change to the new phase
        if current_phase != phase:
            print(f"Setting new phase: {phases[phase]} at junction {junction_id}.")
            traci.trafficlight.setPhase(junction_id, phase)
        else:
            print(f"Phase remains the same: {phases[phase]} at junction {junction_id} No change applied.")

        #
        #print(f"Setting phase duration: {self.duration} seconds for phase {phases[phase]}.")
        traci.trafficlight.setPhaseDuration(junction_id, self.duration)

        # advance the simulation by the chosen phase duration
        for _ in range(self.duration):
            traci.simulationStep()

        # set amber phase after the green phase completes
        yellow_phase = phase + 1
        print(f"Switching to amber phase: {phases[yellow_phase]} to clear the jun")
        traci.trafficlight.setPhase(junction_id, yellow_phase)

        # amber duration set to 5 seconds 1/3 of total duration
        for _ in range(self.yellow_duration):
            traci.simulationStep()

        # after completing the action (extract_state) the new state
        new_state = self.extract_state(junction_id)
        return new_state

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

    def reset(self, junction_id):
        """
        Reset the environment to its initial state.
        """
        self.rewards = {}
        self.action_state_pairs = {}
        self.total_steps = 0
        self.state = self.extract_state(junction_id)
        return self.state
