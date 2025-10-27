import heapq


def heuristic(a, b):
    """
    Manhattan distance for 4 way movement
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def a_star(grid, start, goal, obstacle):
    """
    Finds and returns shortest path from start to goal.
    """
    rows, cols = len(grid), len(grid[0])
    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start))  # (f, g, position)

    came_from = {}  # path reconstruction
    g_score = {tuple(start): 0}

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right

    while open_set:
        _, current_g, current = heapq.heappop(open_set)

        if current == goal:
            # print("Found goal")
            # Reconstruct path
            path = []
            while tuple(current) in came_from:
                path.append(current)
                current = came_from[tuple(current)]
            path.append(start)
            path.reverse()
            return path  # List of [row, col] from start to goal

        for dx, dy in directions:
            neighbor = [current[0] + dx, current[1] + dy]
            r, c = neighbor
            if 0 <= r < rows and 0 <= c < cols and grid[r][c] != obstacle:
                tentative_g = current_g + 1
                # print(f"move to neighbor {neighbor} score: {tentative_g}")
                if (
                    tuple(neighbor) not in g_score
                    or tentative_g < g_score[tuple(neighbor)]
                ):
                    g_score[tuple(neighbor)] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))
                    came_from[tuple(neighbor)] = current

    # print("No path")
    return None  # No path found


def has_path(grid, start, goal, obstacle):
    # print("Grid:")
    # for row in grid:
    #    print(row)
    path = a_star(grid, start, goal, obstacle)
    # print(f"Path:\n{path}")
    return path is not None
