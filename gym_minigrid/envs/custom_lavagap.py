from gym_minigrid.custom_minigrid import *
from gym_minigrid.register import register

class CustomLavaGapEnv(CustomMiniGridEnv):
    """
    Environment with one wall of lava with a small gap to cross through
    This environment is similar to LavaCrossing but simpler in structure.
    """

    def __init__(self, size=5, obstacle_type=Lava, seed=None, visualization=False, env_map = None):
        self.obstacle_type = obstacle_type
        self.env_map = env_map
        if env_map is not None:
            size = env_map.shape[0] + 2
            
        super().__init__(
            grid_size=size,
            max_steps=4*size*size,
            # Set this to True for maximum speed
            see_through_walls=False,
            seed=None, 
            visualization = visualization
        )

    def _gen_grid(self, width, height):
        assert width >= 5 and height >= 5

        if self.env_map is None:
            # Create an empty grid
            self.grid = Grid(width, height)

            # Generate the surrounding walls
            self.grid.wall_rect(0, 0, width, height)

            # Place the agent in the top-left corner
            self.agent_pos = (1, 1)
            self.agent_dir = 0

            # Place a goal square in the bottom-right corner
            self.goal_pos = np.array((width - 2, height - 2))
            self.put_obj(Goal(), *self.goal_pos)

            # Generate and store random gap position
            self.gap_pos = np.array((
                self._rand_int(2, width - 2),
                self._rand_int(1, height - 1),
            ))

            # Place the obstacle wall
            self.grid.vert_wall(self.gap_pos[0], 1, height - 2, self.obstacle_type)

            # Put a hole in the wall
            self.grid.set(*self.gap_pos, None)
        else:
            # print ("env_map: ", self.env_map)
            # Create an empty grid
            self.grid = Grid(self.width, self.height)

            # Generate the surrounding walls
            self.grid.wall_rect(0, 0, self.width, self.height)

            # 0-Empty, 1-Wall, 2-Key, 3-Door, 4-Goal, 5-Start, 6-Grey door, 7-Blue door

            for r in range(len(self.env_map)):
                for c in range(len(self.env_map[0])):
                    if self.env_map[r][c] == 1:
                        self.put_obj(Lava(), c+1, r+1)
                    elif self.env_map[r][c] == 4:
                        self.put_obj(Goal(), c+1, r+1)
                    elif self.env_map[r][c] == 5:
                        self.agent_pos = np.array((c+1, r+1))

                self.agent_dir = 0
        

        self.mission = (
                "avoid the lava and get to the green goal square"
                if self.obstacle_type == Lava
                else "find the opening and get to the green goal square"
            )
            

class CustomLavaGapS5Env(CustomLavaGapEnv):
    def __init__(self, visualization=False):
        super().__init__(size=5, visualization=visualization)

class CustomLavaGapS6Env(CustomLavaGapEnv):
    def __init__(self, visualization=False):
        super().__init__(size=6, visualization=visualization)

class CustomLavaGapS7Env(CustomLavaGapEnv):
    def __init__(self, visualization=False):
        super().__init__(size=7, visualization=visualization)

register(
    id='MiniGrid-CustomLavaGap-v0',
    entry_point='gym_minigrid.envs:CustomLavaGapEnv'
)

register(
    id='MiniGrid-CustomLavaGapS5-v0',
    entry_point='gym_minigrid.envs:CustomLavaGapS5Env'
)

register(
    id='MiniGrid-CustomLavaGapS6-v0',
    entry_point='gym_minigrid.envs:CustomLavaGapS6Env'
)

register(
    id='MiniGrid-CustomLavaGapS7-v0',
    entry_point='gym_minigrid.envs:CustomLavaGapS7Env'
)
