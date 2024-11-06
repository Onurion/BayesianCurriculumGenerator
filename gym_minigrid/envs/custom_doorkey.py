from gym_minigrid.minigrid import *
from gym_minigrid.custom_minigrid import *
from gym_minigrid.register import register
# from ray.rllib.env.env_context import EnvContext
import gym


class CustomDoorKeyEnv(CustomMiniGridEnv):
    """
    Environment with a door and key, sparse reward
    """

    def __init__(self, 
        seed=None,
        visualization=False,
        env_map = None,
        frame_file = "frames"):

        self.seed_no = seed
        self.size = env_map.shape[0] + 2
        self.env_map = env_map
        self.visualization = visualization

        super().__init__(
            grid_size=self.size,
            max_steps=10*self.size*self.size,
            visualization = self.visualization,
            frame_file = frame_file
        )

    def _gen_grid(self, width, height):
        self.seed(seed=self.seed_no)
        if self.env_map is None:
            
            # Create an empty grid
            self.grid = Grid(width, height)

            # Generate the surrounding walls
            self.grid.wall_rect(0, 0, width, height)

            # Place a goal in the bottom-right corner
            self.put_obj(Goal(), width - 2, height - 2)

            # Create a vertical splitting wall
            splitIdx = self._rand_int(2, width-2)
            self.grid.vert_wall(splitIdx, 0)

            # Place the agent at a random position and orientation
            # on the left side of the splitting wall
            self.place_agent(size=(splitIdx, height))

            # Place a door in the wall
            doorIdx = self._rand_int(1, width-2)
            self.put_obj(Door('yellow', is_locked=True), splitIdx, doorIdx)

            # Place a yellow key on the left side
            self.place_obj(
                obj=Key('yellow'),
                top=(0, 0),
                size=(splitIdx, height)
            )
        else:
            # print ("env_map: ", self.env_map)
            # Create an empty grid
            self.grid = Grid(width, height)

            # Generate the surrounding walls
            self.grid.wall_rect(0, 0, width, height)

            # 0-Empty, 1-Wall, 2-Key, 3-Door, 4-Goal, 5-Start

            for r in range(len(self.env_map)):
                for c in range(len(self.env_map[0])):
                    if self.env_map[r][c] == 1:
                        self.put_obj(Wall(), c+1, r+1)
                    elif self.env_map[r][c] == 2:
                        self.put_obj(Key('yellow'), c+1, r+1)
                    elif self.env_map[r][c] == 3:
                        self.put_obj(Door('yellow', is_locked=True), c+1, r+1)
                    elif self.env_map[r][c] == 4:
                        self.put_obj(Goal(), c+1, r+1)
                    elif self.env_map[r][c] == 5:
                        self.agent_pos = np.array((c+1, r+1))

            self.agent_dir = 0


        self.mission = "use the key to open the door and then get to the goal"

class CustomDoorKeyEnv5x5(CustomDoorKeyEnv):
    def __init__(self):
        super().__init__(size=5)

class CustomDoorKeyEnv6x6(CustomDoorKeyEnv):
    def __init__(self):
        super().__init__(size=6)

class CustomDoorKeyEnv16x16(CustomDoorKeyEnv):
    def __init__(self):
        super().__init__(size=16)

register(
    id='MiniGrid-CustomDoorKey-v0',
    entry_point='gym_minigrid.envs:CustomDoorKeyEnv'
)

register(
    id='MiniGrid-CustomDoorKey-5x5-v0',
    entry_point='gym_minigrid.envs:CustomDoorKeyEnv5x5'
)

register(
    id='MiniGrid-CustomDoorKey-6x6-v0',
    entry_point='gym_minigrid.envs:CustomDoorKeyEnv6x6'
)

register(
    id='MiniGrid-CustomDoorKey-8x8-v0',
    entry_point='gym_minigrid.envs:CustomDoorKeyEnv'
)

register(
    id='MiniGrid-CustomDoorKey-16x16-v0',
    entry_point='gym_minigrid.envs:CustomDoorKeyEnv16x16'
)
