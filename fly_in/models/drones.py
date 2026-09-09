from .hub import Hub
from .zone_network import Zone_Network
from variables import DroneStatus
from pydantic import BaseModel, Field
from typing import TypedDict


class Drone(BaseModel):
    """Represents one drone moving through the network."""

    current_hub: Hub = Field()
    status: DroneStatus = DroneStatus.IDLE
    previous_hub: Hub | None = None
    pending_hub: Hub | None = None
    id: int

    def move_to(self, hub: Hub) -> None:
        self.previous_hub = self.current_hub
        self.current_hub = hub
        self.status = DroneStatus.MOVING

    def has_reached(self, goal: Hub) -> bool:
        return self.current_hub.name == goal.name

    def has_arrived(self) -> bool:
        return self.status == DroneStatus.ARRIVED


class ActiveDrone(TypedDict):
    id: int
    drone: Drone
    path_index: int
