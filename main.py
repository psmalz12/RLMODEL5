import traci
from RL_agent3 import RL  
from RL_agent2 import RL as Baseline  
from env3 import Env
from metric import plt_total_reward_steps_per_episode33


# here without GUI and run both RL and Baseline
# it show the avg reward for the whole episode

class RLRunnerSUMO:
    def __init__(self, epsilon_point, RL_Agent, algo_name, junction_id, episodes):
        # Initialize RL agent hyperparameters
        self.epsilon = 1 # high mean more explore new action & low exploit the action with the highest q-value
        self.learning_rate = 0.2 # low slower update the q-value and higher opposite
        self.gamma = 0.9 # high long term reward & low immediate reward (short term)
        self.epsilon_decay_rate = 0.4  # decay rate for epsilon high will decrease epsilon faster overtime and lead to exploit (depend on q-value of A) and the opposite correct
        self.min_epsilon = 0.05
        self.epsilon_point = epsilon_point # episode number where epsilon reaches its min value
        self.junction_id = junction_id
        self.episodes = episodes

        # Init RL agent or Baseline agent
        self.algo_name = algo_name
        self.rl_agent = RL_Agent(self.epsilon, self.learning_rate, self.epsilon_decay_rate, self.min_epsilon,
                                 self.epsilon_point)
        self.env = Env()
        self.total_episode_reward = []
        self.total_episode_reward_sn = []
        self.avg_episode_rewards = []  # store avg rewards per episode /50 steps
        self.sn_reward = []  # store rewards every 10 episodes
        self.sn_avg_reward = []  # store avg rewards every 10 episodes

    def run(self):
        for episode in range(self.episodes):
            print(f"\n====== Starting Episode {episode + 1}/{self.episodes} ======")
            self.env.reset(self.junction_id)  # reset and set state
            state = self.env.state
            total_episode_reward = 0
            done = False
            steps_in_episode = 0
            self.rl_agent.decay_epsilon(episode)

            while not done and steps_in_episode < 50:
                print(f"\n====== Time step {steps_in_episode + 1} in episode {episode + 1} ({self.algo_name})=======")
                steps_in_episode += 1

                # ask for action in RL_agent to excute in env based on fixed policy,
                # need the state to store the Q-values and episode num to use it in remainder
                action = self.rl_agent.action_policy(state, steps_in_episode, [0, 3, 6, 9])
                print(f"Chosen action for this episode is (in main)  {action}")

                next_state, reward, current_state, done_flag, total_reward = self.env.take_action(action, self.junction_id)

               # update Q-values
                updated_q_value = self.rl_agent.update_q_values(state, action, reward, next_state, self.gamma)

                # print detailed state, action, reward, Q-value (just for track not needed)
                print(f"In main2.py ")
                print(f"State: {current_state}")
                print(f"Action taken: {action}")
                print(f"Reward: {reward}")
                print(f"total_reward: {total_reward}")
                print(f"Updated Q-value for (state, action): ({current_state}, {action}): {updated_q_value}")
                state = next_state
                total_episode_reward += reward
                print(f"Total reward for episode {episode + 1}: {total_episode_reward}")

                done = done_flag

                traci.simulationStep()
            #avg reward per episode - Bei: not commen practice
            avg_episode_reward = total_episode_reward / steps_in_episode
            self.total_episode_reward.append(total_episode_reward)
            self.avg_episode_rewards.append(avg_episode_reward)

            # Store a snippet of rewards every 10 episodes xxx
            if (episode + 1) % 100 == 0:
                self.sn_reward.append(total_episode_reward)
                self.sn_avg_reward.append(avg_episode_reward)

            print(f" (In main2.py) Total reward for episode {episode + 1}: {total_episode_reward}")
            print(f" (In main2.py) Average reward for episode {episode + 1}: {avg_episode_reward}")

        self.display_results()

        traci.close()

    def display_results(self):
        phases = {
            0: "E0",
            3: "E2",
            6: "E1",
            9: "E3",
        }

        print(f"(In main2.py) Total rewards every 100 episodes ({self.algo_name}):")
        for i, total_reward in enumerate(self.total_episode_reward_sn):
            print(f"At episode {i * 100 + 100}: Reward: {total_reward:.2f}")

        print(f"All episodes completed ({self.algo_name}).")

        print(f" (In main2.py) Updated Q-values ({self.algo_name}):")
        for state, actions in self.rl_agent.q_values.items():
            print(f"State: {state}")
            for action, q_value in actions.items():
                action_a = phases[action]
                print(f"  Action: {action_a} ({action}), Q-value: {q_value:.4g}")


if __name__ == "__main__":
    sumo_cmd = ['sumo', '-c', r'C:\Users\psmalz12\OneDrive\PGR\pycharm\RLMODEL7\sumo.sumocfg']
    traci.start(sumo_cmd)

    epsilon_point = 1000
    episodes = 3000
    episode_numbers = list(range(1, episodes + 1, 100))  # snip only every 10 episodes

    # Init result dictionaries
    results_q_learning = {}
    results_baseline = {}

    # Run Q-Learning agent
    q_learning_runner = RLRunnerSUMO(epsilon_point, RL, "Q-Learning", "J1", episodes)
    q_learning_runner.run()

    # Store Q-Learning snip result
    results_q_learning[epsilon_point] = {
        'sn_rewards': q_learning_runner.sn_reward,
        'sn_avg_rewards': q_learning_runner.sn_avg_reward
    }

    # restart SUMO simulation for Baseline
    traci.start(sumo_cmd)

    # run Baseline agent
    baseline_runner = RLRunnerSUMO(epsilon_point, Baseline, "Baseline", "J1", episodes)
    baseline_runner.run()

    # Store Baseline snip results
    results_baseline[epsilon_point] = {
        'sn_rewards': baseline_runner.sn_reward,
        'sn_avg_rewards': baseline_runner.sn_avg_reward
    }

    plt_title = "Comparison: Q-Learning vs Baseline Performance in SUMO Traffic Control System"
    plt_total_reward_steps_per_episode33(
        results_q_learning,
        results_baseline,
        episode_numbers,
        plt_title,
    )
