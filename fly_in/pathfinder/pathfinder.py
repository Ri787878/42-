from __future__ import annotations

from heapq import heappop, heappush

from models import Hub, Zone_Network


class Pathfinder():
    def _heuristic(self, current: Hub, goal: Hub) -> int:
        return (abs(current.x_coord - goal.x_coord) +
                abs(current.y_coord - goal.y_coord))

    def _reconstruct_path(
        self,
        came_from: dict[str, str],
        hub_map: dict[str, Hub],
        current_name: str,
    ) -> list[Hub]:
        path = [hub_map[current_name]]

        while current_name in came_from:
            current_name = came_from[current_name]
            path.append(hub_map[current_name])

        path.reverse()
        return path

    def next_step(
        self,
        network: Zone_Network,
        current_hub: Hub,
        occupied_hubs: dict[str, int] | None = None,
        used_links: dict[tuple[str, str], int] | None = None,
    ) -> Hub | None:
        occupied_hubs = occupied_hubs or {}
        used_links = used_links or {}

        start_hub = current_hub
        goal_hub = network.end_hub

        if start_hub.is_blocked:
            raise ValueError("[ERROR] Start hub is blocked.")
        if goal_hub.is_blocked:
            raise ValueError("[ERROR] End hub is blocked.")

        open_set: list[tuple[int, int, int, str, Hub]] = []
        came_from: dict[str, str] = {}
        g_score: dict[str, int] = {start_hub.name: 0}

        start_priority = 0 if start_hub.prefered_zone else 1
        heappush(
            open_set,
            (
                self._heuristic(start_hub, goal_hub),
                start_priority,
                0,
                start_hub.name,
                start_hub,
            ),
        )

        while open_set:
            _, _, current_cost, _, current_hub = heappop(open_set)

            if current_hub.name == goal_hub.name:
                path = self._reconstruct_path(
                    came_from,
                    network.hub_map,
                    current_hub.name)
                
                return path[1] if len(path) > 1 else None

            if current_cost > g_score.get(current_hub.name, current_cost):
                continue

            for neighbor in network.neighbors(current_hub.name):
                if neighbor.is_blocked:
                    continue
            
                link_key: tuple[str, str] = (
                    current_hub.name,
                    neighbor.name,
                ) if current_hub.name < neighbor.name else (
                    neighbor.name,
                    current_hub.name,
                )
            
                link_capacity = 0
            
                for left, right, capacity in network.connection:
                    if tuple(sorted((left, right))) == link_key:
                        link_capacity = capacity
                        break
            
                if link_capacity <= 0:
                    continue
            
                # Only the first movement is being scheduled this turn.
                # Do not block links later in the hypothetical A* route.
                if current_hub.name == start_hub.name:
                    if used_links.get(link_key, 0) >= link_capacity:
                        continue
            
                congestion = occupied_hubs.get(neighbor.name, 0)
            
                if (
                    neighbor.max_drones is None
                    or congestion < neighbor.max_drones
                ):
                    congestion_penalty = 0
                else:
                    congestion_penalty = (
                        (congestion - neighbor.max_drones + 1) * 5
                    )
            
                tentative_cost = (
                    current_cost
                    + neighbor.movement_cost
                    + congestion_penalty
                )
            
                if tentative_cost >= g_score.get(
                    neighbor.name,
                    float("inf"),
                ):
                    continue
            
                came_from[neighbor.name] = current_hub.name
                g_score[neighbor.name] = tentative_cost
            
                priority_score = (
                    0 if neighbor.prefered_zone else 1
                )
            
                total_score = (
                    tentative_cost
                    + self._heuristic(neighbor, goal_hub)
                )
            
                heappush(
                    open_set,
                    (
                        total_score,
                        priority_score,
                        tentative_cost,
                        neighbor.name,
                        neighbor,
                    ),
                )

        return None
