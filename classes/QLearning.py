import numpy as np


class QLearning:
    def __init__(self, agent):
        self.agent = agent

        self.q_values = np.zeros((agent.grid_world.depth, agent.grid_world.width, len(agent.actions)))

    def get_next_action(self, epsilon):
        return np.argmax(
            self.q_values[self.agent.current_state]) if np.random.random() < epsilon else np.random.randint(
            len(self.agent.actions))

    def learn(self, epsilon=0.9, discount_factor=0.9, learning_rate=0.9, episodes_number=1000):
        rewards_per_episode = []

        for _ in range(episodes_number):
            total_reward = 0

            while not self.agent.in_terminal_state():
                action_index = self.get_next_action(epsilon)

                old_state = self.agent.current_state
                self.agent.move_to_next_state(action_index)

                reward = self.agent.get_reward()

                total_reward += reward

                old_q_value = self.q_values[old_state[0], old_state[1], action_index]
                temporal_difference = reward + (
                        discount_factor * np.max(self.q_values[self.agent.current_state])) - old_q_value

                new_q_value = old_q_value + (learning_rate * temporal_difference)
                self.q_values[old_state[0], old_state[1], action_index] = new_q_value

            rewards_per_episode.append(total_reward)

            self.agent.update_initial_state()

        return rewards_per_episode

    def get_shortest_path(self):
        self.agent.update_initial_state()

        if self.agent.in_terminal_state():
            return []

        shortest_path = [self.agent.current_state]

        while not self.agent.in_terminal_state():
            action_index = self.get_next_action(1)

            self.agent.move_to_next_state(action_index)

            shortest_path.append(self.agent.current_state)

        return shortest_path
