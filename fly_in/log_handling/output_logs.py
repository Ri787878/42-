from __future__ import annotations
from typing import cast
from models import Drone, Hub, Zone_Network
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
        drones: list[Drone]
    ) -> list[str]:
        history: list[str] = []

        active = [
            {
                "id": i + 1,
                "drone": d,
                "path_index": 0
             } for i, d in enumerate(drones)]
        start_hub, goal_hub = network.start_hub, network.end_hub

        occupancy: dict[str, int] = {start_hub.name: len(active)}

        while active:
            # ---------- Phase 1: collect intents (no mutation) ----------
            intents: list[dict] = []
            link_usage: dict[tuple[str, str], int] = {}
            inbound_reserved: dict[str, int] = {}
            turn_tokens: dict[int, str] = {}
            delivered_ids: set[int] = set()
            any_move = False

            for item in sorted(active, key=lambda x: int(cast(int, x["id"]))):
                d_id = cast(int, item["id"])
                drone: Drone = cast(Drone, item["drone"])
                i = cast(int, item["path_index"])
                path = drone.planned_path

                if i >= len(path) - 1:
                    delivered_ids.add(d_id)
                    continue

                # second turn of restricted move
                if drone.status == DroneStatus.BLOCKED:
                    arrival = path[i + 1]
                    intents.append({
                        "id": d_id, "kind": "restricted_arrival",
                        "from": path[i].name, "to": arrival.name,
                        "item": item
                    })
                    continue

                cur = path[i]
                nxt = path[i + 1]

                n1, n2 = sorted((cur.name, nxt.name))
                link_key = (n1, n2)

                cap = self._get_link_limit(network, cur.name, nxt.name)
                if cap <= 0:
                    continue
                if link_usage.get(link_key, 0) >= cap:
                    continue

                # hub capacity reservation
                if nxt.zone != "restricted" and nxt.name != goal_hub.name:
                    nxt_occ = (
                        occupancy.get(nxt.name, 0)
                        + inbound_reserved.get(nxt.name, 0)
                    )
                    if not self._hub_has_capacity(nxt, nxt_occ):
                        continue
                    inbound_reserved[nxt.name] = inbound_reserved.get(
                        nxt.name,
                        0
                    ) + 1

                link_usage[link_key] = link_usage.get(link_key, 0) + 1
                intents.append({
                    "id": d_id, "kind": "move",
                    "from_hub": cur, "to_hub": nxt,
                    "item": item
                })

            # ---------- Phase 2: commit accepted intents ----------
            for it in intents:
                d_id = it["id"]

                if it["kind"] == "restricted_arrival":
                    item = cast(dict, it["item"])
                    drone = cast(Drone, item["drone"])
                    i = cast(int, item["path_index"])
                    arrival = drone.planned_path[i + 1]

                    drone.status = DroneStatus.MOVING
                    item["path_index"] = i + 1
                    drone.current_hub = arrival
                    turn_tokens[d_id] = f"D{d_id}-{arrival.name}"
                    any_move = True

                    path_len = len(drone.planned_path)
                    if cast(int, item["path_index"]) >= path_len - 1:
                        delivered_ids.add(d_id)

                elif it["kind"] == "move":
                    item = cast(dict, it["item"])
                    drone = cast(Drone, item["drone"])
                    i = cast(int, item["path_index"])
                    cur = it["from_hub"]
                    nxt = it["to_hub"]

                    occupancy[cur.name] = max(
                        0,
                        occupancy.get(cur.name, 0) - 1
                    )

                    if nxt.zone == "restricted":
                        drone.status = DroneStatus.BLOCKED
                        turn_tokens[d_id] = f"D{d_id}-{cur.name}-{nxt.name}"
                    else:
                        if nxt.name != goal_hub.name:
                            occupancy[nxt.name] = occupancy.get(
                                nxt.name,
                                0
                            ) + 1

                        item["path_index"] = i + 1
                        drone.current_hub = nxt
                        lbl = self._movement_label(network, cur, nxt)
                        turn_tokens[d_id] = f"D{d_id}-{lbl}"
                        idx = cast(int, item["path_index"])
                        if idx >= len(drone.planned_path) - 1:
                            delivered_ids.add(d_id)

                    any_move = True

            active = [x for x in active if x["id"] not in delivered_ids]

            if turn_tokens:
                output_line = " ".join(
                    turn_tokens[k] for k in sorted(turn_tokens)
                )
                print(output_line)
                history.append(output_line)

            if active and not any_move:
                raise RuntimeError(
                    "Simulation stalled: no drone can progress."
                )

        with open("output.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(history) + ("\n" if history else ""))
        return history
