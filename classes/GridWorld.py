import numpy as np


class GridWorld:
    def __init__(self, room_depth=3, room_width=3, rooms_number_in_depth=3, rooms_number_in_width=3):
        self.room_depth = room_depth
        self.room_width = room_width
        self.rooms_number_in_depth = rooms_number_in_depth
        self.rooms_number_in_width = rooms_number_in_width

        self.depth = 0
        self.width = 0

        self.aisle_marker = -1
        self.wall_marker = 0

        self.grid = None
        self.wall_states = None
        self.gold_states = None
        self.bombs_states = None

    def build(self):
        wall_depth_frequency = self.room_depth + 1
        wall_width_frequency = self.room_width + 1

        environment_depth = self.rooms_number_in_depth * self.room_depth + (self.rooms_number_in_depth + 1)
        environment_width = self.rooms_number_in_width * self.room_width + (self.rooms_number_in_width + 1)

        environment_grid = np.full((environment_depth, environment_width), self.aisle_marker)

        wall_depth_indexes = np.arange(0, environment_depth, wall_depth_frequency)
        wall_width_indexes = np.arange(0, environment_width, wall_width_frequency)

        wall_aisle_depth_first_index = int(self.room_depth / 2) if self.room_depth % 2 == 0 else int(
            self.room_depth / 2) + 1
        wall_aisle_width_first_index = int(self.room_width / 2) if self.room_width % 2 == 0 else int(
            self.room_width / 2) + 1

        wall_with_aisle_depth_indexes = np.arange(wall_depth_frequency, environment_depth - wall_depth_frequency,
                                                  wall_depth_frequency)
        wall_with_aisle_width_indexes = np.arange(wall_width_frequency, environment_width - wall_width_frequency,
                                                  wall_width_frequency)

        wall_aisle_depth_indexes = np.arange(
            wall_aisle_depth_first_index, environment_depth,
            self.room_depth * 2 - 1 if self.room_depth == 2 else wall_aisle_depth_first_index * 2)
        wall_aisle_width_indexes = np.arange(
            wall_aisle_width_first_index, environment_width,
            self.room_width * 2 - 1 if self.room_width == 2 else wall_aisle_width_first_index * 2)

        environment_grid[wall_depth_indexes] = self.wall_marker
        environment_grid[:, wall_width_indexes] = self.wall_marker

        for i in wall_aisle_depth_indexes:
            environment_grid[i][wall_with_aisle_width_indexes] = self.aisle_marker

        for i in wall_with_aisle_depth_indexes:
            environment_grid[i][wall_aisle_width_indexes] = self.aisle_marker

        self.depth = environment_depth
        self.width = environment_width

        self.grid = environment_grid
        self.wall_states = self.__get_states_by_marker(self.wall_marker)

    def put_gold(self, gold_marker=50, gold_number=2):
        gold_states = self.__put_marker(gold_marker, gold_number)

        self.gold_states = gold_states if self.gold_states is None else self.__convert_to_tuple_array(
            np.concatenate([self.gold_states, gold_states]))

    def put_bombs(self, bombs_marker=-50, bombs_number=5):
        bombs_states = self.__put_marker(bombs_marker, bombs_number)

        self.bombs_states = bombs_states if self.bombs_states is None else self.__convert_to_tuple_array(
            np.concatenate([self.bombs_states, bombs_states]))

    def get_available_states(self):
        return self.__get_states_by_marker(self.aisle_marker)

    @staticmethod
    def __convert_to_tuple_array(array):
        return [tuple(elem) for elem in array]

    def __get_states_by_marker(self, marker):
        return self.__convert_to_tuple_array(np.array(np.where(self.grid == marker)).transpose())

    def __check_number(self, number):
        return number if 0 <= number <= len(self.get_available_states()) else 0

    def __put_marker(self, marker, marker_number=1):
        available_states = self.get_available_states()
        states_indexes = np.random.randint(0, len(available_states), self.__check_number(marker_number))

        states = []

        for state_indexes in states_indexes:
            position = available_states[state_indexes]

            self.grid[position] = marker

            states.append(position)

        return states
