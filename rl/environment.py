import random

from copy import deepcopy
from utils.minihack_rect import get_rect
from utils.a_star import has_path

from minihack.envs.room import (
    register,
    MiniHackRoom,
    LevelGenerator,
    MiniHackNavigation,
)
from minihack.base import MiniHack
from nle import nethack


def chars_to_ascii(chars):
    s = ""
    for row in chars:
        line = "".join(chr(c) for c in row) + "\n"
        s += line
    return s


class MiniHackRoom9x9Random(MiniHackRoom):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, size=9, random=True, **kwargs)


class MiniHackDangerousMonster5x5Room(MiniHackNavigation):
    def __init__(self, *args, size=5, random=True, n_monster=1, lit=True, **kwargs):
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", size * 20)

        lvl_gen = LevelGenerator(w=size, h=size, lit=lit)
        if random:
            lvl_gen.add_goal_pos()
        else:
            lvl_gen.add_goal_pos((size - 1, size - 1))
            lvl_gen.set_start_pos((0, 0))

        for _ in range(n_monster):
            lvl_gen.add_monster(name="baby gray dragon", symbol="D", args=("hostile",))

        super().__init__(*args, des_file=lvl_gen.get_des(), **kwargs)


class MiniHackNavigationCardinal(MiniHack):
    """
    Updated base class for Navigation environments, replaces inheriting from MiniHackNavigation.
    Allows only 4-way (Cardinal directions) not 8-way (Cardinal & inter-Cardinal directions).
    """

    KEY_ACTION_SPACE = "action-space"
    ACTIONS_4_WAY = 4
    ACTIONS_8_WAY = 8

    def __init__(self, *args, des_file, **kwargs):
        action_space = kwargs.get(MiniHackNavigationCardinal.KEY_ACTION_SPACE)
        kwargs.pop(MiniHackNavigationCardinal.KEY_ACTION_SPACE)
        # print(f"MiniHackNavigationCardinal action space: {action_space}")
        if (action_space is not None) and (
            action_space == MiniHackNavigationCardinal.ACTIONS_4_WAY
        ):
            kwargs["actions"] = kwargs.pop(
                "actions", tuple(nethack.CompassCardinalDirection)
            )
        else:  # default
            # Actions space - move only by default
            kwargs["actions"] = kwargs.pop("actions", tuple(nethack.CompassDirection))

        # Disallowing one-letter menu questions
        kwargs["allow_all_yn_questions"] = kwargs.pop("allow_all_yn_questions", False)
        # Perform known steps
        kwargs["allow_all_modes"] = kwargs.pop("allow_all_modes", False)
        # Play with Rogue character by default
        kwargs["character"] = kwargs.pop("character", "rog-hum-cha-mal")
        # Default episode limit
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 100)

        super().__init__(*args, des_file=des_file, **kwargs)


class MiniHackDynamicMaze(MiniHackNavigationCardinal):
    """
    A fixed 4-rooms environment from:
        R. S. Sutton, D. Precup, and S. Singh. Between MDPs and semi-MDPs: A framework for temporal
        abstraction in reinforcement learning. Artificial intelligence, 112(1-2):181–211, 1999b.

    ... with optional "doors" placed between rooms to complicate navigation.

    Since the doors are opened/closed dynamically, they're hacked into the environment and not present in NLE/MiniHack.
    Instead, doors must be in a doorway of two wall-like cells. The action to enter the doorway cell is replaced with
    an action to walk into the wall, resulting in no movement.

    Chars:
        124 | (wall)
        60 < (start)
        62 > (goal)
        64 @ (agent)
        45 - (wall)
        46 . (space)
        32   (off map)
    """

    KEY_P_CHANGE_DOORS = "p-change-doors"
    KEY_NUMBER_CLOSED_DOORS = "closed-doors"

    def __init__(self, *args, **kwargs):
        self.p_change = MiniHackDynamicMaze._get_kwarg_float(
            kw_args=kwargs,
            kw_key=MiniHackDynamicMaze.KEY_P_CHANGE_DOORS,
            default_value=0,
        )
        self.num_closed_doors = int(
            MiniHackDynamicMaze._get_kwarg_float(
                kw_args=kwargs,
                kw_key=MiniHackDynamicMaze.KEY_NUMBER_CLOSED_DOORS,
                default_value=1,
            )
        )
        # print(f"MiniHackDynamicMaze p(Change doors): {self.p_change} Dynamic? {self.is_dynamic} #doors: {self.num_closed_doors}")

        self.agent_char = 64
        self.start_char = 60
        self.goal_char = 62
        self.door_char = "d"
        self.space_char = "."
        self.wall_chars = "|-"
        self.obs_door_char_closed = 124  # How doors appear in obs when closed
        self.obs_wall_chars = [45, 124]

        self.action_offsets = {
            0: [-1, 0],  # N
            1: [0, 1],  # E
            2: [1, 0],  # S
            3: [0, -1],  # W
            4: [-1, 1],  # NE
            5: [1, 1],  # SE
            6: [1, -1],  # SW
            7: [-1, -1],  # NW
        }  # action: [row, col]

        self.setup_map()
        lvl_gen = self.create_level_generator(self.map)
        des_file = lvl_gen.get_des()
        # print(self._des_file)

        super().__init__(*args, des_file=des_file, **kwargs)

    @staticmethod
    def _get_kwarg_float(kw_args: dict, kw_key: str, default_value: float):
        assert isinstance(default_value, float) or isinstance(default_value, int)
        return float(kw_args.pop(kw_key, default_value))

    @property
    def is_dynamic(self) -> bool:
        return self.p_change > 0

    def get_map_definition(self) -> str:
        """
        Return the map with dynamic elements defined in it.
        d = Door
        """
        map = """
            |||||||||||||
            |.....|.....|
            |.....|.....|
            |.....d.....|
            |.....|.....|
            |.....|.....|
            ||d||||.....|
            |.....|||d|||
            |.....|.....|
            |.....|.....|
            |.....d.....|
            |.....|.....|
            |||||||||||||
        """
        return map

    def get_map(self, map_definition: str) -> str:
        return map_definition.replace(self.door_char, self.space_char)

    def create_level_generator(self, map: str):
        lvl_gen = LevelGenerator(
            flags=("premapped",),
            map=map,
            w=None,
            h=None,
            lit=True,
        )
        lvl_gen.add_goal_pos()
        return lvl_gen

    def setup_map(self):
        """
        Initializes info about the map for speed.
        """
        # Create ASCII map for NLE and replace doors 'd' with open space '.'
        map_definition = self.get_map_definition()
        self.map = self.get_map(map_definition)

        # Init some state
        self.door_pos = []  # Allowed in any episode
        self.door_pos_allowed = []  # Allowed this episode ie not start, goal
        self.door_pos_closed = []  # Currently losed door positions

        lines = map_definition.splitlines()
        min_row = max_row = min_col = max_col = None

        def fn_or_none(fn, value: int, new_value: int) -> int:
            if value is None:
                return new_value
            return fn(value, new_value)

        def update_bounding_box(row, col):
            nonlocal min_row
            nonlocal max_row
            nonlocal min_col
            nonlocal max_col
            min_row = fn_or_none(min, min_row, row)
            max_row = fn_or_none(max, max_row, row)
            min_col = fn_or_none(min, min_col, col)
            max_col = fn_or_none(max, max_col, col)

        # Examine each cell in the map to find potential walls
        # Also, build a quick to index 2d map for pathfinding
        map_origin = None
        self.map_2d = []
        for row, line in enumerate(lines):
            map_2d_row = []
            for col, value in enumerate(line):
                # BBox defined by outer wall
                if value in self.wall_chars:
                    update_bounding_box(row, col)

                if map_origin is None:
                    if value in self.wall_chars:
                        map_origin = [row, col]

                if value == self.door_char:
                    relative_row = row - map_origin[0]
                    relative_col = col - map_origin[1]
                    door_pos = [relative_row, relative_col]
                    self.door_pos.append(door_pos)

                # Creating pathfinding map
                if value in self.wall_chars:
                    map_2d_row.append(self.wall_chars[0])
                elif (value == self.space_char) or (value == self.door_char):
                    map_2d_row.append(self.space_char)

            if len(map_2d_row) > 0:  # Discard empty lines
                self.map_2d.append(map_2d_row)

        # print("Pathfinding map:")
        # for row in self.map_2d:
        #    print(row)

        self.size = max(
            max_row - min_row - 1,
            max_col - min_col - 1,
        )

        # print(f"Map origin: r{self.map_origin[0]},c{self.map_origin[1]} size: {self.size}")
        # print(f"Possible door pos: {self.door_pos}")
        self.rect = get_rect(self.size, self.size)

    def reset(self, *args, **kwargs):
        v = super().reset(*args, **kwargs)
        self.o = v[0]  # Keep obs to find agent
        self.find_start_and_goal_pos(self.o)
        self.set_allowed_doors()
        # We don't know where agent is, and need to search for it again
        self.agent_pos = self.find_agent_pos(self.o, near_pos=None)
        self.update_closed_doors()  # Initially closed
        return v

    def set_allowed_doors(self):
        self.door_pos_allowed = []
        self.door_pos_closed = []  # Clear list, set by update_closed_doors
        for door_pos in self.door_pos:
            if self.goal_pos == door_pos:
                continue
            if self.start_pos == door_pos:
                continue
            self.door_pos_allowed.append(door_pos)

    def step(self, action):
        # Prevent walking into doors
        # if agent plus move would hit door, then replace action with one which walks into walls.

        # print(f"BEFORE step:{action} Offsets:{action_offsets}")

        # Should already have done this on reset or last step.
        # self.agent_pos = self.find_agent_pos(self.o, near_pos=self.agent_pos)
        # print(f"Agent would move from {agent_pos} to {agent_pos_next}")"""
        if self.skip_env_step(action):
            # NOTE: If it would hit the wall, Agent "sees" that action but env sees the "wall-impacting" action.
            # This allows agent to learn the effect of that action in this context (ie doesn't work).
            # Skip env update. Some consequences:
            # - Reward will not be accumulated correctly? Attempting to compensate.
            # - Max steps limit won't be counted correctly for these steps.
            reward = self.penalty_step  # From NetHackScore/NetHackStaircase
            terminated = False
            truncated = False
            info = None
            # Obs self.o unchanged
            o = deepcopy(self.o)  # because we might draw a door in it
            # print(f"skip R={reward}")
        else:
            o, reward, terminated, truncated, info = super().step(action)
            # print(f"update R={reward}")
            self.o = deepcopy(o)
            self.agent_pos = self.find_agent_pos(o, near_pos=self.agent_pos)
            # print(f"AFTER step: Found agent at: {self.agent_pos}")

        if self.is_dynamic:  # Save compute when static
            # Maybe update doors
            r = random.random()
            if (not self.door_pos_closed) or (r < self.p_change):
                self.update_closed_doors()

            # Draw doors in observation
            self.add_doors_to_observation(o)

        return o, reward, terminated, truncated, info

    def skip_env_step(self, action: int) -> bool:
        if self.is_dynamic and self.would_hit_door(action):
            # print(f"Agent would hit door. Changing action...")
            return True
        return False

    def would_hit_door(self, action: int) -> bool:
        action_offsets = self.action_offsets[action]
        agent_pos_next = [
            self.agent_pos[0] + action_offsets[0],
            self.agent_pos[1] + action_offsets[1],
        ]  # Where it would be after the action

        for door_pos in self.door_pos_closed:
            if agent_pos_next == door_pos:
                return True
        return False

    def update_closed_doors(self):
        self.door_pos_closed = self.select_closed_doors(
            num_closed=self.num_closed_doors,
            door_pos_allowed=self.door_pos_allowed,
        )

    def select_closed_doors(
        self,
        num_closed,
        door_pos_allowed: list,
        transform_to_observation_pos: bool = True,
    ):
        """
        Randomly selects `num_closed` doors from `door_pos_allowed`
        without replacement. It is efficient if the number of doors
        selected is much less than the number of possible door positions.
        """
        selected_door_pos = []
        # Select random doors, but don't put a door where Agent is.
        # Selection is also with replacement meaning that the same door can be picked more than once.
        # print(f"num_closed:{num_closed} +1 < len(door_pos_allowed):{len(door_pos_allowed)}")
        assert (num_closed + 1) < len(
            door_pos_allowed
        )  # +1 because maybe one disallowed due to agent pos

        # print(f"door_pos_allowed:{door_pos_allowed}")

        for _ in range(num_closed):
            while True:
                door_pos = self.select_random_pos(door_pos_allowed)
                if door_pos in selected_door_pos:
                    continue
                if door_pos is not None:
                    if self.agent_pos is not None:  # can be null when terminated
                        if self.agent_pos == door_pos:
                            continue
                selected_door_pos.append(door_pos)
                break

        # print(f"selected_door_pos pre tx:{selected_door_pos}")

        if transform_to_observation_pos:
            selected_door_pos = self.list_to_observation_pos(selected_door_pos)
        return selected_door_pos

    def select_closed_doors_with_path_check(
        self,
        num_closed,
        door_pos_allowed: list,
        transform_to_observation_pos: bool = True,
    ):
        """
        Randomly selects `num_closed` doors from `door_pos_allowed`
        without replacement. This implementation checks for the existence of a path from Agent to goal.
        """
        selected_door_pos = []
        # Select random doors, but don't put a door where Agent is.
        # Selection is also with replacement meaning that the same door can be picked more than once.
        # print(f"num_closed:{num_closed} +1 < len(door_pos_allowed):{len(door_pos_allowed)}")
        assert (num_closed + 1) < len(
            door_pos_allowed
        )  # +1 because maybe one disallowed due to agent pos

        if self.agent_pos is not None:
            agent_pos_map = self.to_map_pos(self.agent_pos)
        else:
            agent_pos_map = None
        # print(f"Agent pos (map): {agent_pos_map}")
        # print(f"Goal  pos (map): {self.goal_pos}")

        map_2d = deepcopy(self.map_2d)
        obstacle = self.wall_chars[0]
        no_obstacle = self.space_char
        max_iterations = 10
        for n in range(num_closed):
            # print(f"Placing obstacle {n} of {num_closed}")
            i = 0
            while True and i < max_iterations:
                i += 1  # failsafe for infinite loop
                door_pos = self.select_random_pos(door_pos_allowed)

                # Don't select duplicates
                if door_pos in selected_door_pos:
                    continue

                # Don't select agent pos
                if agent_pos_map is not None:  # can be null when terminated
                    if agent_pos_map == door_pos:
                        continue

                    # Draw doors in map_2d
                    # print(f"Adding obstacle at door pos: {door_pos}")
                    map_2d[door_pos[0]][door_pos[1]] = obstacle

                    # Confirm existence of path from Agent to goal
                    if not has_path(
                        grid=map_2d,
                        start=agent_pos_map,
                        goal=self.goal_pos,
                        obstacle=obstacle,
                    ):
                        # print(f"Removing obstacle at door pos: {door_pos}")
                        map_2d[door_pos[0]][door_pos[1]] = no_obstacle  # revert edit
                        continue
                    # Else: Leave obstacle for pathfinding

                selected_door_pos.append(door_pos)
                break

        # print("Pathfinding map:")
        # for row in map_2d:
        #    print(row)

        # print(f"selected_door_pos pre tx:{selected_door_pos}")

        if transform_to_observation_pos:
            selected_door_pos = self.list_to_observation_pos(selected_door_pos)
        return selected_door_pos

    def list_to_observation_pos(self, pos: list) -> list:
        transformed = [self.to_observation_pos(door_pos) for door_pos in pos]
        return transformed

    def add_doors_to_observation(self, o):
        self.set_char_at_pos_in_observation(
            o, self.door_pos_closed, self.obs_door_char_closed
        )

    def set_char_at_pos_in_observation(self, o, pos: list, char_value):
        if self.agent_pos is None:  # can be null when terminated
            return  # don't draw anything

        for door_pos in pos:
            row_door = door_pos[0]
            col_door = door_pos[1]

            # Door can't be where agent is
            if door_pos == self.agent_pos:
                continue

            o_chars = o["chars"]
            o_chars[row_door][col_door] = char_value

    def find_relative_action(self, relative_pos) -> int:
        # TODO potentially could make a faster lookup
        for k, v in self.action_offsets.items():
            if v == relative_pos:
                return k
        return None

    def get_relative_pos(self, origin, other) -> list[int]:
        return [
            other[0] - origin[0],
            other[1] - origin[1],
        ]

    def find_agent_pos(self, observation, near_pos=None) -> list[int]:
        """
        Returns coordinates of Agent in observation
        """
        o_chars = observation["chars"]

        # Quick look near last observed position
        if near_pos is not None:
            near_row = near_pos[0]
            near_col = near_pos[1]
            for y in range(3):
                row = near_row - 1 + y
                cols = o_chars[row]
                for x in range(3):
                    col = near_col - 1 + x
                    value = cols[col]
                    if value == self.agent_char:
                        return [row, col]  # Early return mean O(4.5)

        # Full search (eg after reset)
        for row, cols in enumerate(o_chars):
            for col, value in enumerate(cols):
                if value == self.agent_char:
                    return [row, col]

        # print(o_chars)
        # raise ValueError("Agent not found AT ALL") Legit happens at end of episode
        return None

    def find_start_and_goal_pos(self, observation) -> list[int]:
        """
        Returns coordinates of Agent in observation
        """
        o_chars = observation["chars"]

        self.start_pos = None
        self.goal_pos = None

        # Full search
        for row, cols in enumerate(o_chars):
            for col, value in enumerate(cols):
                if value == self.goal_char:
                    self.goal_pos = self.to_map_pos([row, col])
                    if self.goal_pos is not None and self.start_pos is not None:
                        return  # Early stopping when found something
                if value == self.start_char:
                    self.start_pos = self.to_map_pos([row, col])
                    if self.goal_pos is not None and self.start_pos is not None:
                        return  # Early stopping when found something

    def get_door_pos(self, door_index: int) -> list[int]:
        """
        Returns coordinates of a Door in observation
        """
        if door_index is None:
            return None
        row_door = self.rect.first_row + self.door_pos[door_index][0]
        col_door = self.rect.first_col + self.door_pos[door_index][1]
        return [row_door, col_door]

    def to_observation_pos(self, map_pos) -> list[int]:
        """
        Returns coordinates in observation space
        """
        row = self.rect.first_row + map_pos[0]
        col = self.rect.first_col + map_pos[1]
        return [row, col]

    def to_map_pos(self, observation_pos) -> list[int]:
        """
        Returns coordinates in map
        """
        row = observation_pos[0] - self.rect.first_row
        col = observation_pos[1] - self.rect.first_col
        return [row, col]

    def select_random_pos(self, values: list) -> int:
        N = len(values)
        index = random.randint(0, N - 1)
        return values[index]


class MiniHackRandomCorridors(MiniHackDynamicMaze):
    """
    Random corridor-style mazes with dynamic doors feature.
    """

    KEY_NUMBER_FIXED_WALLS = "fixed-walls"

    def __init__(self, *args, **kwargs):
        self.num_fixed_walls = int(
            MiniHackDynamicMaze._get_kwarg_float(
                kw_args=kwargs,
                kw_key=MiniHackRandomCorridors.KEY_NUMBER_FIXED_WALLS,
                default_value=5,
            )
        )

        super().__init__(*args, **kwargs)

        self.obs_random_wall_char = self.obs_door_char_closed
        self.obs_dynamic_closed_door_char = 35  # # (hash) character

    def get_map_definition(self) -> str:
        """
        Grid of random doors to make a maze.
        """
        map = """
            |||||||||||||
            |.d.d.d.d.d.|
            |d|d|d|d|d|d|
            |.d.d.d.d.d.|
            |d|d|d|d|d|d|
            |.d.d.d.d.d.|
            |d|d|d|d|d|d|
            |.d.d.d.d.d.|
            |d|d|d|d|d|d|
            |.d.d.d.d.d.|
            |||||||||||||
        """
        return map

    def get_map_definition_small(self) -> str:
        """
        A very small map
        """
        map = """
            |||||||||||||
            |.d.d.d.d.|||
            |d|d|d|d|d|||
            |.d.d.d.d.|||
            |||||||||||||
            |||||||||||||
            |||||||||||||
            |||||||||||||
            |||||||||||||
            |||||||||||||
            |||||||||||||
        """
        return map

    def reset(self, *args, **kwargs):
        self.reset_fixed_walls = True
        v = super().reset(*args, **kwargs)
        return v

    def update_closed_doors(self):
        if self.reset_fixed_walls:
            # print("Setting fixed walls...")
            self.reset_fixed_walls = False

            # On reset, randomly close some doors.
            # self.fixed_walls_pos = self.select_closed_doors(
            self.fixed_walls_pos = self.select_closed_doors_with_path_check(
                num_closed=self.num_fixed_walls,
                door_pos_allowed=self.door_pos_allowed,
                transform_to_observation_pos=False,
            )
            # print(f"Allowed (1): {self.door_pos_allowed}")
            # print(f"Fixed: {self.fixed_walls_pos}")

            # Reduce allowed dynamic door pos to those not set by fixed walls
            # so they can't be selected twice
            self.door_pos_allowed_excluding_fixed_walls = []
            for door_pos in self.door_pos_allowed:
                if door_pos in self.fixed_walls_pos:
                    continue
                self.door_pos_allowed_excluding_fixed_walls.append(door_pos)
            # print(f"Allowed (2): {self.door_pos_allowed_excluding_fixed_walls}")

            # Was generated in maze coords, not transform to obs. coords.
            self.fixed_walls_pos = self.list_to_observation_pos(self.fixed_walls_pos)

        self.dynamic_door_pos = self.select_closed_doors(
            num_closed=self.num_closed_doors,
            door_pos_allowed=self.door_pos_allowed_excluding_fixed_walls,
        )
        self.door_pos_closed = self.fixed_walls_pos + self.dynamic_door_pos
        # print(f"Dynamic: {self.dynamic_door_pos}")
        # print(f"Closed: {self.door_pos_closed}")

    def add_doors_to_observation(self, o):
        self.set_char_at_pos_in_observation(
            o, self.fixed_walls_pos, self.obs_random_wall_char
        )
        self.set_char_at_pos_in_observation(
            o, self.dynamic_door_pos, self.obs_dynamic_closed_door_char
        )


class MiniHackRandomCorridorsFourWay(MiniHackRandomCorridors):
    """
    Random corridors
    4 Way
    Dynamic
    """

    def __init__(self, *args, **kwargs):
        if MiniHackNavigationCardinal.KEY_ACTION_SPACE not in kwargs:
            kwargs[MiniHackNavigationCardinal.KEY_ACTION_SPACE] = 4

        if MiniHackDynamicMaze.KEY_P_CHANGE_DOORS not in kwargs:
            kwargs[MiniHackDynamicMaze.KEY_P_CHANGE_DOORS] = 0.05

        if MiniHackDynamicMaze.KEY_NUMBER_CLOSED_DOORS not in kwargs:
            kwargs[MiniHackDynamicMaze.KEY_NUMBER_CLOSED_DOORS] = 5

        if MiniHackRandomCorridors.KEY_NUMBER_FIXED_WALLS not in kwargs:
            kwargs[MiniHackRandomCorridors.KEY_NUMBER_FIXED_WALLS] = 10

        super().__init__(*args, **kwargs)


class MiniHack4Room(MiniHackDynamicMaze):
    KEY_MIN_ROOM_DISTANCE_BETWEEN_START_END = "min-room-distance-between-start-end"

    def __init__(self, *args, **kwargs):
        self.min_distance_between_start_end = kwargs.pop(
            self.KEY_MIN_ROOM_DISTANCE_BETWEEN_START_END, None
        )
        super().__init__(*args, **kwargs)

    def create_level_generator(self, map):
        lvl_gen = LevelGenerator(
            flags=("premapped",),
            map=map,
            w=None,
            h=None,
            lit=True,
        )

        if self.min_distance_between_start_end is None:
            lvl_gen.add_goal_pos()
            return lvl_gen

        # Force start and end to be at least a certain distance apart by first choosing a random start and then finding a random goal far enough
        # room start is top-left, bottom-left, top-right, bottom-right
        top_left_room_rect = (
            1,
            5,
            1,
            5,
        )  # first_row, last_row, first_col, last_col, all inclusive
        bottom_left_room_rect = (7, 11, 1, 5)
        top_right_room_rect = (1, 6, 7, 11)
        bottom_right_room_rect = (8, 11, 7, 11)

        adjacent_pairs = [
            (top_left_room_rect, bottom_left_room_rect),
            (top_left_room_rect, top_right_room_rect),
            (bottom_left_room_rect, bottom_right_room_rect),
            (top_right_room_rect, bottom_right_room_rect),
        ]
        opposite_pairs = [
            (top_left_room_rect, bottom_right_room_rect),
            (bottom_left_room_rect, top_right_room_rect),
        ]

        if self.min_distance_between_start_end == 0:
            room_start = random.choice(
                [
                    top_left_room_rect,
                    bottom_left_room_rect,
                    top_right_room_rect,
                    bottom_right_room_rect,
                ]
            )
            room_end = room_start
        elif self.min_distance_between_start_end == 1:
            room_start, room_end = random.choice(adjacent_pairs)
        else:  # 2 or more
            room_start, room_end = random.choice(opposite_pairs)

        if random.random() > 0.5:
            room_start, room_end = room_end, room_start

        OFFSET_BY_MAP_STRING_FORMATTING = 12  # string formatting in get_map actually bumps the map right by this much. No down affected.
        start = (
            random.randint(room_start[0], room_start[1]),
            random.randint(room_start[2], room_start[3])
            + OFFSET_BY_MAP_STRING_FORMATTING,
        )
        end = (
            random.randint(room_end[0], room_end[1]),
            random.randint(room_end[2], room_end[3]) + OFFSET_BY_MAP_STRING_FORMATTING,
        )
        lvl_gen.set_start_pos((start[1], start[0]))
        lvl_gen.add_goal_pos((end[1], end[0]))
        return lvl_gen


register(
    id="MiniHack-Room-Random-9x9-v0", entry_point="rl.environment:MiniHackRoom9x9Random"
)

register(
    id="MiniHack-Room-DangerousMonster-5x5-v0",
    entry_point="rl.environment:MiniHackDangerousMonster5x5Room",
)

register(id="MiniHack-4-Rooms", entry_point="rl.environment:MiniHack4Room")

register(
    id="MiniHack-Corridor-Maze-4-Way-Dynamic",
    entry_point="rl.environment:MiniHackRandomCorridorsFourWay",
)


REGISTER_ALL_IMPORT = True  # a value that can be imported to make sure that all the registration works... kinda hacks but whatever
