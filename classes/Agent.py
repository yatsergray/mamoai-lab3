import numpy as np


class Agent:
    def __init__(self, grid_world):
        self.grid_world = grid_world

        self.current_state = self.__generate_initial_state()

        self.actions = ["UP", "RIGHT", "DOWN", "LEFT"]

    def in_terminal_state(self):
        return self.grid_world.grid[self.current_state] != self.grid_world.aisle_marker

    def update_initial_state(self):
        self.current_state = self.__generate_initial_state()

    def move_to_next_state(self, action_index):
        new_state_depth_index = self.current_state[0]
        new_state_width_index = self.current_state[1]

        match self.actions[action_index]:
            case "UP":
                if not (new_state_depth_index - 1, new_state_width_index) in self.grid_world.wall_states:
                    new_state_depth_index -= 1
            case "RIGHT":
                if not (new_state_depth_index, new_state_width_index + 1) in self.grid_world.wall_states:
                    new_state_width_index += 1
            case "DOWN":
                if not (new_state_depth_index + 1, new_state_width_index) in self.grid_world.wall_states:
                    new_state_depth_index += 1
            case "LEFT":
                if not (new_state_depth_index, new_state_width_index - 1) in self.grid_world.wall_states:
                    new_state_width_index -= 1

        self.current_state = (new_state_depth_index, new_state_width_index)

    def get_reward(self):
        return self.grid_world.grid[self.current_state]

    def __generate_initial_state(self):
        available_states = self.grid_world.get_available_states()

        return available_states[np.random.randint(len(available_states))] if len(available_states) > 0 else (-1, -1)
