import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


class Converter:
    def __init__(self, grid_world):
        self.grid_world = grid_world

        self.markers_colors = ["red", "white", "black", "yellow"]

    def convert_grid_into_plot(self, path_to_gold):
        cmap = colors.ListedColormap(self.markers_colors)

        fig, ax = plt.subplots()
        ax.imshow(self.grid_world.grid, cmap=cmap)

        ax.set_xticks(np.arange(0, self.grid_world.width, 1))
        ax.set_yticks(np.arange(self.grid_world.depth - 1, -1, -1))

        plt.xlabel("width")
        plt.ylabel("depth")

        for i in range(len(self.grid_world.grid)):
            for j in range(len(self.grid_world.grid[i])):
                state = (i, j)

                color = "white"

                if self.grid_world.grid[state] == self.grid_world.aisle_marker or state in self.grid_world.gold_states:
                    color = "black"

                    if state in path_to_gold:
                        if state == path_to_gold[0]:
                            color = "blue"
                        else:
                            color = "green"

                plt.text(j, i, self.grid_world.grid[state], color=color, ha='center', va='center', size=8)

        plt.show()

    @staticmethod
    def convert_learning_data_into_plot(rewards_per_episode):
        x = np.arange(len(rewards_per_episode))

        plt.plot(x, rewards_per_episode)

        plt.xlabel('episodes')
        plt.ylabel('rewards')
        plt.title('Learning data')

        plt.show()
