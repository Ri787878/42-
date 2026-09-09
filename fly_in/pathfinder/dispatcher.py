from models.drones import Drone
from models.zone_network import Zone_Network


class Dispatcher():
    def dispatch_drones(self, network: Zone_Network) -> list[Drone]:
        """
        Build one route per drone.
        Occupied hubs increase routing cost, but do not fully block paths.
        """

        drones: list[Drone] = []
        for drone_id in range(1, network.nb_drones + 1):
            drones.append(
                Drone(
                    current_hub=network.start_hub,
                    id=drone_id,
                )
            )

        return drones
