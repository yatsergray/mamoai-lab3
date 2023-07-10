import pandas as pd

from classes.Agent import Agent
from classes.Converter import Converter
from classes.GridWorld import GridWorld
from classes.QLearning import QLearning


def run():
    g = GridWorld(room_depth=3, room_width=3, rooms_number_in_depth=4, rooms_number_in_width=4)

    g.build()

    g.put_gold(gold_number=1)
    g.put_gold(25, 2)
    g.put_bombs(bombs_number=15)

    print(pd.DataFrame(g.grid))

    a = Agent(g)
    q = QLearning(a)
    c = Converter(g)

    rewards_per_episode = q.learn(episodes_number=10000)

    print(rewards_per_episode)

    print("Learning is finished!")

    first_path = q.get_shortest_path()
    second_path = q.get_shortest_path()
    third_path = q.get_shortest_path()

    print(first_path)
    print(second_path)
    print(third_path)

    c.convert_grid_into_plot(first_path)
    c.convert_grid_into_plot(second_path)
    c.convert_grid_into_plot(third_path)

    c.convert_learning_data_into_plot(rewards_per_episode)


if __name__ == '__main__':
    run()
