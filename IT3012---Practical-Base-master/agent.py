from collections import deque
import heapq
import random


class SimpleReflexAgent:
    """Simple Reflex Agent using strict Condition-Action rules without memory."""
    def sense_and_act(self, percept: dict) -> str:
        wall = percept['wall_ahead']
        food = percept['food_nearby']

        if food['Up'] and not wall['Up']:
            return 'Up'
        elif food['Right'] and not wall['Right']:
            return 'Right'
        elif food['Down'] and not wall['Down']:
            return 'Down'
        elif food['Left'] and not wall['Left']:
            return 'Left'
        
        if not wall['Up']:
            return 'Up'
        elif not wall['Right']:
            return 'Right'
        elif not wall['Down']:
            return 'Down'
        else:
            return 'Left'


class ModelBasedAgent:
    """Model-Based Agent with internal memory tracking visited cells."""
    def __init__(self):
        self.visited_cells = set()
        self.current_pos = (0, 0)

    def sense_and_act(self, percept: dict) -> str:
        self.visited_cells.add(self.current_pos)
        wall = percept['wall_ahead']
        x, y = self.current_pos

        dirs = {
            'Up': (x, y + 1),
            'Right': (x + 1, y),
            'Down': (x, y - 1),
            'Left': (x - 1, y)
        }

        unvisited = [d for d in ['Up', 'Right', 'Down', 'Left'] if not wall[d] and dirs[d] not in self.visited_cells]

        if unvisited:
            chosen = unvisited[0]
        else:
            valid_moves = [d for d in ['Up', 'Right', 'Down', 'Left'] if not wall[d]]
            chosen = valid_moves[0] if valid_moves else 'Up'

        self.current_pos = dirs[chosen]
        return chosen


class SearchAgent:
    """Step 1.2 & 1.3: Goal-Based Agent implementing BFS, DFS, and UCS search."""
    def __init__(self, active_algo='BFS'):
        self.plan = []
        self.active_algo = active_algo  # 'BFS', 'DFS', or 'UCS'

    def get_neighbors(self, state, grid_size, walls):
        x, y = state
        width, height = grid_size
        moves = [
            ('Up', (x, y + 1)),
            ('Down', (x, y - 1)),
            ('Left', (x - 1, y)),
            ('Right', (x + 1, y))
        ]
        valid_neighbors = []
        for action, (nx, ny) in moves:
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in walls:
                valid_neighbors.append((action, (nx, ny)))
        return valid_neighbors

    def bfs_search(self, start, goal, grid_size, walls):
        """Breadth-First Search using FIFO Queue."""
        frontier = deque([(start, [])])
        reached = {start}

        while frontier:
            state, path = frontier.popleft()
            if state == goal:
                return path

            for action, neighbor in self.get_neighbors(state, grid_size, walls):
                if neighbor not in reached:
                    reached.add(neighbor)
                    frontier.append((neighbor, path + [action]))
        return []

    def dfs_search(self, start, goal, grid_size, walls):
        """Depth-First Search using LIFO Stack."""
        frontier = [(start, [])]
        reached = {start}

        while frontier:
            state, path = frontier.pop()
            if state == goal:
                return path

            for action, neighbor in self.get_neighbors(state, grid_size, walls):
                if neighbor not in reached:
                    reached.add(neighbor)
                    frontier.append((neighbor, path + [action]))
        return []

    def ucs_search(self, start, goal, grid_size, walls):
        """Uniform-Cost Search using Priority Queue."""
        counter = 0
        frontier = [(0, counter, start, [])]
        reached = {start: 0}

        while frontier:
            cost, _, state, path = heapq.heappop(frontier)

            if state == goal:
                return path

            for action, neighbor in self.get_neighbors(state, grid_size, walls):
                new_cost = cost + 1
                if neighbor not in reached or new_cost < reached[neighbor]:
                    reached[neighbor] = new_cost
                    counter += 1
                    heapq.heappush(frontier, (new_cost, counter, neighbor, path + [action]))
        return []

    def sense_and_act(self, percept: dict) -> str:
        if not self.plan:
            start = percept['agent_pos']
            all_food = percept['all_food']

            if not all_food:
                return 'Up'

            # Target nearest food using Manhattan distance
            closest_food = min(all_food, key=lambda f: abs(f[0] - start[0]) + abs(f[1] - start[1]))

            grid_size = percept['grid_size']
            walls = set(percept['walls'])

            if self.active_algo == 'BFS':
                self.plan = self.bfs_search(start, closest_food, grid_size, walls)
            elif self.active_algo == 'DFS':
                self.plan = self.dfs_search(start, closest_food, grid_size, walls)
            elif self.active_algo == 'UCS':
                self.plan = self.ucs_search(start, closest_food, grid_size, walls)

        return self.plan.pop(0) if self.plan else 'Up'