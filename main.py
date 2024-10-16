import traci
from RL_agent import RL
from env import Env
import time
from metric import plot_rewards2


class RLRunnerSUMO:
    def __init__(self, epsilon_point, RL_Agent, algo_name, junction_id, episodes):
        # Init RL agent hyperparameters
        self.epsilon = 1  #  high mean more explore new action & low exploit the action with the highest q-value
        self.learning_rate = 0.2  # low slower update the q-value and higher opposite
        self.gamma = 0.9  # high long term reward & low immediate reward (short term)
        self.epsilon_decay_rate = 0.4  # decay rate for epsilon high will decrease epsilon faster overtime and lead to exploit (depend on q-value of A) and the opposite correct
        self.min_epsilon = 0.05  # Min epsilon value
        self.epsilon_point = epsilon_point  # epsilon decay point
        self.junction_id = junction_id  # Traffic light junction to control
        self.episodes = episodes  # Total episodes to run

        # Init RL agent
        self.algo_name = algo_name  # algo
        self.rl_agent = RL_Agent(self.epsilon, self.learning_rate, self.epsilon_decay_rate, self.min_epsilon, self.epsilon_point)
        self.env = Env()  # Init SUMO environment
        self.total_rewards = []  # Store total rewards per episode
        self.total_episode_reward = []  # Store total rewards for each episode
        self.total_episode_reward_sn = []  # Store total rewards every 100 episodes sn = snip

    def run(self):
        for episode in range(self.episodes):
            print(f"\n====== Starting Episode {episode + 1}/{self.episodes} ======")

            # Reset env and get the init state
            self.env.reset2()
            state = self.env.state
            total_episode_reward = 0  # total reward for this episode
            done = False
            steps_in_episode = 0
            # Decay epsilon for exploration-exploitation
            self.rl_agent.decay_epsilon(episode)

            while not done and steps_in_episode < 50: # 2 cond just to make sure
                print(f"\n====== New Time step Count {steps_in_episode +1} in episode: {episode}==========")

                print(f"\n====== State Before taken Action (lane(vehicle  ID, vehicle time step  , vehicle type) current_phase): {state} for {self.algo_name} =====****=====")


                # choose an action: phase + duration
                action = self.rl_agent.action_policy(state, [0, 3, 6, 9])
                # action =


                # Take action in the env (execute phase ) and extract state (inside take action)
                next_state, reward, current_state, done_flag, total_reward = self.env.take_action(action, self.junction_id)

                # update Q-values using state from the env
                self.rl_agent.update_q_values(current_state, action, reward, next_state, self.gamma)

                # Update state and + rewards
                state = next_state
                total_episode_reward += reward  # Add step reward to total episode reward
                #print(f"Action taken: {action}, Current State: {current_state}, Next State: {next_state}")

                print(f"Action taken: {action},  New State AFTER action : {next_state}")
                print(f"Reward: {reward} ")

                done = done_flag  # Check if the episode is done
                steps_in_episode += 1
                done = done_flag or steps_in_episode >= 100

                # Advance the simulation
                traci.simulationStep()

            # Store the total reward for this episode
            #self.total_rewards.append(total_episode_reward)
            # Store the total reward for the episode
            self.total_episode_reward.append(total_episode_reward)  # Total reward for this episode
            # Compute average reward for the episode
            total_reward = total_episode_reward / steps_in_episode if steps_in_episode > 0 else 0
            print(f"Episode {episode + 1} completed. Total Reward: {total_episode_reward}, Average Reward: {total_reward}") # test the avg

            #print(f"Rewa//rd: {reward}, Total Reward (for this step): {total_reward}, Done: {done_flag}")

            # not working proberly can use it for sampling in each 50-100 eps
           # if (episode + 1) % 100 == 0:
            #    print(f"Episode {episode + 1} completed with total reward: {total_episode_reward}")
            # store cumulative reward every 100 episodes
            if (episode + 1) % 100 == 0:
                avg_reward_sn = sum(self.total_episode_reward[-100:]) / 100  # Average reward over the last 100 episodes
                self.total_episode_reward_sn.append(avg_reward_sn)
                print(f"Cumulative reward for the last 100 episodes: {avg_reward_sn}")


        # End simulation
        traci.close()

    def display_results(self):  # not updated yet
        action_ac = {
            0: "West-East Green",
            3: "South-North Green",
            6: "East-West Green",
            9: "North-South Green"
        }

        print(f"Total rewards for every 100 episodes ({self.algo_name}):")
        for i, total_reward_episodes in enumerate(self.total_rewards):
            end_episode = min((i + 1) * 100, self.episodes)
            print(f"At Episode Num. {end_episode}: Reward: {total_reward_episodes:.2f}")

        print(f"Total reward for the last episode ({self.algo_name}): {self.total_rewards[-1] if self.total_rewards else 0}")

        print(f"All episodes completed ({self.algo_name}).")
        print(f"Updated Q-values ({self.algo_name}):")
        for state, actions in self.rl_agent.q_values.items():
            print(f"State: {state}")
            for action, q_value in actions.items():
                phase, duration = action
                action_a = action_ac.get(phase, "other than what in the desplay dictionary")
                print(f"  Phase: {action_a} ({phase}), Duration: {duration}, Q-value: {q_value:.4g}")


if __name__ == "__main__":
    # Start SUMO
    sumo_cmd = ['sumo-gui', '-c', r'C:\Users\psmalz12\OneDrive\PGR\pycharm\RLMODEL5\sumo.sumocfg']
    traci.start(sumo_cmd)

    epsilon_point = 50
    episodes = 1000
    rl_runner_sumo = RLRunnerSUMO(epsilon_point, RL, "Q-Learning", "J1", episodes)

    # Run the RL simulation
    rl_runner_sumo.run()


    plot_rewards2(rl_runner_sumo.total_rewards)  # Plt rewards
    rl_runner_sumo.display_results()  # Show results
