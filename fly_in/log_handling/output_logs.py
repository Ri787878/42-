from __future__ import annotations
from typing import cast
from models import Drone, Hub, Zone_Network
from pathfinder.pathfinder import Pathfinder
from variables import DroneStatus


class Logger():
    def _is_restricted(self, hub: Hub) -> bool:
        return hub.zone == "restricted"

    def _movement_label(
        self,
        network: Zone_Network,
        current_hub: Hub,
        next_hub: Hub,
    ) -> str:
        if self._is_restricted(next_hub):
            c_name, n_name = current_hub.name, next_hub.name
            for left, right, _cap in network.connection:
                if {left, right} == {c_name, n_name}:
                    return right if right == n_name else left
            return n_name

        return next_hub.name

    def _hub_has_capacity(self, hub: Hub, occupied_count: int) -> bool:
        limit = hub.max_drones if hub.max_drones is not None else 1
        return occupied_count < limit

    def _get_link_limit(
         self,
         network: Zone_Network,
         hub_a: str,
         hub_b: str
         ) -> float:
        """Return link capacity between two hubs (default: 1, 0 if no link)."""
        for a, b, cap in network.connection:
            if (a == hub_a and b == hub_b) or (a == hub_b and b == hub_a):
                return cap

        for attr_name in (
             "connection_properties",
             "connection_weights",
             "link_capacities"):
            if hasattr(network, attr_name):
                props = getattr(network, attr_name)
                if isinstance(props, dict):
                    val = props.get(
                        (hub_a, hub_b)) or props.get((hub_b, hub_a))
                    if val is not None:
                        if (isinstance(
                                val,
                                dict
                        ) and "max_link_capacity" in val):
                            return float(val["max_link_capacity"])
                        if isinstance(val, (int, float)):
                            return float(val)
        return 0.0

    def simulate_drones(
        self,
        network: Zone_Network,
        drones: list[Drone],
        pathfinder: Pathfinder,
    ) -> list[str]:
        history: list[str] = []
    
        start_hub = network.start_hub
        goal_hub = network.end_hub
    
        occupancy: dict[str, int] = {
            start_hub.name: len(drones)
        }
    
        active = [
            drone for drone in drones
            if not drone.has_reached(goal_hub)
        ]
    
        while active:
            intents: list[dict] = []
            link_usage: dict[tuple[str, str], int] = {}
            inbound_reserved: dict[str, int] = {}
            outbound_reserved: dict[str, int] = {}
            turn_tokens: dict[int, str] = {}
    
            for drone in sorted(active, key=lambda item: item.id):
                current_hub = drone.current_hub
    
                # Complete the second turn of a restricted movement.
                if drone.status == DroneStatus.BLOCKED:
                    if drone.pending_hub is None:
                        raise RuntimeError(
                            f"Drone {drone.id} has no pending destination."
                        )
    
                    intents.append({
                        "drone": drone,
                        "kind": "restricted_arrival",
                        "from_hub": current_hub,
                        "to_hub": drone.pending_hub,
                    })
                    continue
    
                # Let the pathfinder see reservations already made this turn.
                pathfinder_occupancy = {
                    name: (
                        occupancy.get(name, 0)
                        - outbound_reserved.get(name, 0)
                        + inbound_reserved.get(name, 0)
                    )
                    for name in (
                        set(occupancy)
                        | set(outbound_reserved)
                        | set(inbound_reserved)
                    )
                }
    
                next_hub = pathfinder.next_step(
                    network,
                    current_hub,
                    pathfinder_occupancy,
                    link_usage,
                )
    
                if next_hub is None:
                    continue
    
                link_key: tuple[str, str] = (
                    (current_hub.name, next_hub.name)
                    if current_hub.name <= next_hub.name
                    else (next_hub.name, current_hub.name)
                )
    
                link_limit = self._get_link_limit(
                    network,
                    current_hub.name,
                    next_hub.name,
                )
    
                if link_usage.get(link_key, 0) >= link_limit:
                    continue
    
                # Check destination capacity after departures and arrivals.
                if (
                    next_hub.zone != "restricted"
                    and next_hub.name != goal_hub.name
                ):
                    current_count = occupancy.get(next_hub.name, 0)
                    leaving_count = outbound_reserved.get(
                        next_hub.name,
                        0,
                    )
                    entering_count = inbound_reserved.get(
                        next_hub.name,
                        0,
                    )
    
                    effective_count = (
                        current_count
                        - leaving_count
                        + entering_count
                    )
    
                    if not self._hub_has_capacity(
                        next_hub,
                        effective_count,
                    ):
                        continue
    
                    inbound_reserved[next_hub.name] = (
                        entering_count + 1
                    )
    
                link_usage[link_key] = (
                    link_usage.get(link_key, 0) + 1
                )
    
                outbound_reserved[current_hub.name] = (
                    outbound_reserved.get(current_hub.name, 0) + 1
                )
    
                intents.append({
                    "drone": drone,
                    "kind": "move",
                    "from_hub": current_hub,
                    "to_hub": next_hub,
                })
    
            # Commit all accepted movements.
            for intent in intents:
                intent_drone: Drone = intent["drone"]
                intent_current_hub: Hub = intent["from_hub"]
                intent_next_hub: Hub = intent["to_hub"]
    
                if intent["kind"] == "restricted_arrival":
                    intent_drone.pending_hub = None
                    intent_drone.move_to(intent_next_hub)
    
                    occupancy[intent_next_hub.name] = (
                        occupancy.get(intent_next_hub.name, 0) + 1
                    )
    
                    turn_tokens[intent_drone.id] = (
                        f"D{intent_drone.id}-{intent_next_hub.name}"
                    )
                    continue
    
                occupancy[intent_current_hub.name] = max(
                    0,
                    occupancy.get(intent_current_hub.name, 0) - 1,
                )
    
                if intent_next_hub.zone == "restricted":
                    intent_drone.pending_hub = intent_next_hub
                    intent_drone.status = DroneStatus.BLOCKED
    
                    turn_tokens[drone.id] = (
                        f"D{drone.id}-"
                        f"{intent_current_hub.name}-"
                        f"{intent_next_hub.name}"
                    )
                else:
                    intent_drone.move_to(intent_next_hub)
    
                    if intent_next_hub.name != goal_hub.name:
                        occupancy[intent_next_hub.name] = (
                            occupancy.get(intent_next_hub.name, 0) + 1
                        )
    
                    label = self._movement_label(
                        network,
                        intent_current_hub,
                        intent_next_hub,
                    )
    
                    turn_tokens[intent_drone.id] = (
                        f"D{intent_drone.id}-{label}"
                    )
    
            active = [
                drone for drone in active
                if not drone.has_reached(goal_hub)
            ]
    
            if turn_tokens:
                output_line = " ".join(
                    turn_tokens[drone_id]
                    for drone_id in sorted(turn_tokens)
                )
                print(output_line)
                history.append(output_line)
    
            if active and not intents:
                raise RuntimeError(
                    "Simulation stalled: no drone can progress."
                )
    
        with open("output.txt", "w", encoding="utf-8") as file:
            file.write(
                "\n".join(history)
                + ("\n" if history else "")
            )
    
        return history