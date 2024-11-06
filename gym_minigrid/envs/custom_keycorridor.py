from gym_minigrid.register import register
from gym_minigrid.custom_roomgrid import CustomRoomGrid
from gym_minigrid.custom_minigrid import *




class CustomKeyCorridor(CustomRoomGrid):
    """
    A ball is behind a locked door, the key is placed in a
    random room.
    """

    def __init__(
        self,
        num_rows=3,
        obj_type="ball",
        room_size=6,
        seed=None,
        visualization=False,
        env_map = None,
        frame_file = "frames_keycorridor"
    ):

        self.obj_type = obj_type
        self.env_map = env_map

        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            max_steps=30*room_size**2,
            seed=seed,
            visualization=visualization,
            env_map=env_map,
            frame_file=frame_file
        )

    def _gen_grid(self, width, height):

        if self.env_map is not None:
            # print ("env_map: ", self.env_map)
            # Create an empty grid
            self.grid = Grid(self.width, self.height)

            # Generate the surrounding walls
            self.grid.wall_rect(0, 0, self.width, self.height)

            # 0-Empty, 1-Wall, 2-Key, 3-Door, 4-Goal, 5-Start, 6-Grey door, 7-Blue door

            for r in range(len(self.env_map)):
                for c in range(len(self.env_map[0])):
                    if self.env_map[r][c] == 1:
                        self.put_obj(Wall(), c+1, r+1)
                    elif self.env_map[r][c] == 2:
                        self.put_obj(Key('red'), c+1, r+1)
                    elif self.env_map[r][c] == 3:
                        self.put_obj(Door('red', is_locked=True), c+1, r+1)
                    elif self.env_map[r][c] == 4:
                        # self.put_obj(Goal(), c+1, r+1)
                        self.obj = Ball(color="yellow")
                        self.put_obj(self.obj, c+1, r+1)
                    elif self.env_map[r][c] == 5:
                        self.agent_pos = np.array((c+1, r+1))
                    elif self.env_map[r][c] == 6:
                        self.put_obj(Door('grey', is_locked=False), c+1, r+1)
                    elif self.env_map[r][c] == 7:
                        self.put_obj(Door('blue', is_locked=False), c+1, r+1)

            self.mission = "pick up the %s %s" % (self.obj.color, self.obj.type)
            self.agent_dir = 0


        else:
            super()._gen_grid(width, height)
            # Connect the middle column rooms into a hallway
            for j in range(1, self.num_rows):
                self.remove_wall(1, j, 3)

            # Add a locked door on the bottom right
            # Add an object behind the locked door
            room_idx = self._rand_int(0, self.num_rows)
            door, _ = self.add_door(2, room_idx, 2, locked=True)
            obj, _ = self.add_object(2, room_idx, kind=self.obj_type)

            # Add a key in a random room on the left side
            self.add_object(0, self._rand_int(0, self.num_rows), 'key', door.color)

            # Place the agent in the middle
            self.place_agent(1, self.num_rows // 2)

            # Make sure all rooms are accessible
            self.connect_all()

            self.obj = obj
            self.mission = "pick up the %s %s" % (obj.color, obj.type)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        if action == self.actions.pickup:
            if self.carrying and self.carrying == self.obj:
                reward = self._reward()
                done = True
        return obs, reward, done, info

class CustomKeyCorridorS3R1(CustomKeyCorridor):
    def __init__(self, seed=None, visualization=False):
        super().__init__(
            room_size=3,
            num_rows=1,
            seed=seed,
            visualization=visualization
        )

class CustomKeyCorridorS3R2(CustomKeyCorridor):
    def __init__(self, seed=None, visualization=False):
        super().__init__(
            room_size=3,
            num_rows=2,
            seed=seed,
            visualization=visualization
        )

class CustomKeyCorridorS3R3(CustomKeyCorridor):
    def __init__(self, seed=None, visualization=False):
        super().__init__(
            room_size=3,
            num_rows=3,
            seed=seed,
            visualization=visualization
        )

class CustomKeyCorridorS4R3(CustomKeyCorridor):
    def __init__(self, seed=None, visualization=False):
        super().__init__(
            room_size=4,
            num_rows=3,
            seed=seed,
            visualization=visualization
        )

class CustomKeyCorridorS5R3(CustomKeyCorridor):
    def __init__(self, seed=None, visualization=False):
        super().__init__(
            room_size=5,
            num_rows=3,
            seed=seed,
            visualization=visualization
        )

class CustomKeyCorridorS6R3(CustomKeyCorridor):
    def __init__(self, seed=None, visualization=False):
        super().__init__(
            room_size=6,
            num_rows=3,
            seed=seed,
            visualization=visualization
        )

register(
    id='MiniGrid-CustomKeyCorridor-v0',
    entry_point='gym_minigrid.envs:CustomKeyCorridor'
)

register(
    id='MiniGrid-CustomKeyCorridorS3R1-v0',
    entry_point='gym_minigrid.envs:CustomKeyCorridorS3R1'
)

register(
    id='MiniGrid-CustomKeyCorridorS3R2-v0',
    entry_point='gym_minigrid.envs:CustomKeyCorridorS3R2'
)

register(
    id='MiniGrid-CustomKeyCorridorS3R3-v0',
    entry_point='gym_minigrid.envs:CustomKeyCorridorS3R3'
)

register(
    id='MiniGrid-CustomKeyCorridorS4R3-v0',
    entry_point='gym_minigrid.envs:CustomKeyCorridorS4R3'
)

register(
    id='MiniGrid-CustomKeyCorridorS5R3-v0',
    entry_point='gym_minigrid.envs:CustomKeyCorridorS5R3'
)

register(
    id='MiniGrid-CustomKeyCorridorS6R3-v0',
    entry_point='gym_minigrid.envs:CustomKeyCorridorS6R3'
)
