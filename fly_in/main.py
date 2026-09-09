from pydantic import ValidationError
import sys
import copy

from pathfinder import Dispatcher, Pathfinder
from log_handling import Logger
from ui import Displayer, Map_Selector
from models import Zone_Network, InvalidConfiguration


def main() -> None:
    try:
        test_map: str = Map_Selector.select_map()

        if test_map == "":
            print("[ERROR] No valid map was issued.")
            sys.exit(1)

        displayer = Displayer()
        dispatcher = Dispatcher()
        logger = Logger()

        network = Zone_Network.from_file(test_map)
        drones = dispatcher.dispatch_drones(network)

        display_drones = copy.deepcopy(drones)

        pathfinder = Pathfinder()
        history = logger.simulate_drones(
            network,
            drones,
            pathfinder,
        )

        displayer.start_display(network, display_drones, history)

    except (InvalidConfiguration, ValidationError) as e:
        print(f"Configuration Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
