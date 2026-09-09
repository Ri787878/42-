
from contextlib import redirect_stdout
from pathlib import Path
from io import BytesIO, StringIO
import base64
import json
import sys
with redirect_stdout(StringIO()):
    import pygame

from models import Drone, Hub, Zone_Network


class Displayer():
    def load_drone_sprite(self) -> pygame.Surface:
        piskel_path = (
            Path(__file__).resolve().parent.parent
            / "sprites"
            / "drone.piskel"
        )
        raw = json.loads(piskel_path.read_text(encoding="utf-8"))

        layer_data = json.loads(raw["piskel"]["layers"][0])
        frame_data = layer_data["chunks"][0]["base64PNG"]
        encoded_png = frame_data.split(",", 1)[1]

        sprite_bytes = base64.b64decode(encoded_png)
        sprite = pygame.image.load(BytesIO(sprite_bytes)).convert_alpha()
        return pygame.transform.smoothscale(sprite, (34, 34))

    def draw_step_counter(
        self,
        screen: pygame.Surface,
        font: pygame.font.Font,
        step_number: int,
        total_steps: int,
    ) -> None:
        label = font.render(
            f"Step {step_number} / {total_steps}",
            True,
            (255, 255, 255),
        )
        label_rect = label.get_rect(topright=(screen.get_width() - 20, 20))

        box_rect = label_rect.inflate(14, 10)
        box = pygame.Surface(box_rect.size, pygame.SRCALPHA)
        box.fill((15, 15, 20, 180))

        screen.blit(box, box_rect.topleft)
        pygame.draw.rect(screen, (255, 215, 0), box_rect, 1, border_radius=6)
        screen.blit(label, label_rect)

    def start_display(
        self,
        network: Zone_Network,
        drones: list[Drone],
        history: list[str]
    ) -> None:
        pygame.display.init()
        pygame.font.init()

        WIDTH, HEIGHT = 1500, 800
        SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Zone Network Visualizer")
        CLOCK = pygame.time.Clock()
        FONT = pygame.font.SysFont("Arial", 14)

        BG_COLOR = (30, 30, 40)
        LINE_COLOR = (180, 180, 180)
        ROUTE_COLOR = (255, 215, 0)
        START_COLOR = (50, 205, 50)
        END_COLOR = (220, 20, 60)
        HUB_COLOR = (70, 130, 180)

        nodes = [network.start_hub, network.end_hub, *network.hubs]
        min_x = min(node.x_coord for node in nodes)
        max_x = max(node.x_coord for node in nodes)
        min_y = min(node.y_coord for node in nodes)
        max_y = max(node.y_coord for node in nodes)

        margin = 70
        span_x = max(1, max_x - min_x)
        span_y = max(1, max_y - min_y)

        cell_size = min(
            (WIDTH - 2 * margin) / span_x,
            (HEIGHT - 2 * margin) / span_y,
        )
        cell_size = max(150, int(cell_size))

        content_width = max(WIDTH, int(span_x * cell_size + (2 * margin)))
        content_height = max(HEIGHT, int(span_y * cell_size + (2 * margin)))

        world_surface = pygame.Surface((content_width, content_height))

        offset_x = margin - min_x * cell_size
        offset_y = margin - min_y * cell_size

        camera_x = 0
        camera_y = 0
        zoom_level = 1.0

        # Build discrete step positions for each drone
        # based on the actual history logs
        total_steps = len(history)
        hub_by_name = {node.name: node for node in nodes}

        # Each drone (using 1-based IDs from "D1", "D2", etc.) gets a path
        # timeline starting at the Start hub
        drone_history_paths: dict[int, list[Hub]] = {
            i + 1: [network.start_hub] for i in range(len(drones))
        }

        # Process the step logs to trace exactly where each drone
        # is at every turn index
        for turn_str in history:
            moved_drones = set()
            if turn_str.strip():
                moves = turn_str.split()
                for move in moves:
                    parts = move.split("-")
                    if len(parts) == 2:
                        d_id = int(parts[0][1:])  # Strip "D" from ID (D3 -> 3)
                        dest_name = parts[1]
                        dest_hub = hub_by_name.get(dest_name)
                        if dest_hub:
                            drone_history_paths[d_id].append(dest_hub)
                            moved_drones.add(d_id)

            # If a drone was delayed or stalled and
            # didn't move this turn, stay in place
            for d_id in drone_history_paths:
                if d_id not in moved_drones:
                    last_pos = drone_history_paths[d_id][-1]
                    drone_history_paths[d_id].append(last_pos)

        def world_to_screen(
             world_x: float,
             world_y: float
             ) -> tuple[int, int]:
            screen_x = int(offset_x + world_x * cell_size)
            screen_y = int(content_height - (offset_y + world_y * cell_size))
            return screen_x, screen_y

        def get_node_style(
             node: Hub
             ) -> tuple[tuple[int, int, int], str]:
            if node.name == network.start_hub.name:
                return START_COLOR, " (Start)"
            if node.name == network.end_hub.name:
                return END_COLOR, " (Goal)"
            return HUB_COLOR, ""

        def draw_route(
            surface: pygame.Surface,
            route: list[Hub],
            color: tuple[int, int, int],
        ) -> None:
            if len(route) < 2:
                return

            for start_hub, end_hub in zip(route, route[1:]):
                pygame.draw.line(
                    surface,
                    color,
                    world_to_screen(
                        start_hub.x_coord,
                        start_hub.y_coord),
                    world_to_screen(
                        end_hub.x_coord,
                        end_hub.y_coord),
                    4,
                )

        def draw_drone(
            surface: pygame.Surface,
            position: tuple[int, int],
            sprite: pygame.Surface,
        ) -> None:
            rect = sprite.get_rect(center=position)
            surface.blit(sprite, rect)

        drone_sprite = self.load_drone_sprite()

        # Smooth step transition speed config
        DRONE_STEP_MS = 1500  # Duration of 1 visual step
        elapsed_time = 0.0

        running = True
        while running:
            dt = CLOCK.tick(60)
            elapsed_time += dt

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 4:
                        zoom_level = min(4.0, zoom_level + 0.1)
                    elif event.button == 5:
                        min_zoom_x = WIDTH / content_width
                        min_zoom_y = HEIGHT / content_height
                        absolute_min_zoom = max(
                            0.1,
                            max(min_zoom_x, min_zoom_y)
                            )
                        zoom_level = max(
                            absolute_min_zoom,
                            zoom_level - 0.1
                            )

            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                camera_x -= 10
            if keys[pygame.K_RIGHT]:
                camera_x += 10
            if keys[pygame.K_UP]:
                camera_y -= 10
            if keys[pygame.K_DOWN]:
                camera_y += 10

            # Progress calculation relative to step index
            step_float = elapsed_time / DRONE_STEP_MS
            current_step_idx = int(step_float)
            progress = step_float - current_step_idx

            # Display-friendly 1-indexed count capped to final step
            display_step = min(current_step_idx + 1, total_steps)

            world_surface.fill(BG_COLOR)

            hub_lookup = {
                node.name: world_to_screen(node.x_coord, node.y_coord)
                for node in nodes
            }

            # Draw grid connections
            for left_name, right_name, _cap in network.connection:
                if left_name in hub_lookup and right_name in hub_lookup:
                    pygame.draw.line(
                        world_surface,
                        LINE_COLOR,
                        hub_lookup[left_name],
                        hub_lookup[right_name],
                        3,
                    )

            # Draw pathways
            for drone_id, route in drone_history_paths.items():
                draw_route(
                    world_surface,
                    route,
                    ROUTE_COLOR,
                )

            # Draw Hub nodes
            for node in nodes:
                pos = world_to_screen(node.x_coord, node.y_coord)
                node_color, _ = get_node_style(node)
                pygame.draw.circle(world_surface, node_color, pos, 16)
                pygame.draw.circle(world_surface, (20, 20, 20), pos, 12)
                pygame.draw.circle(world_surface, node_color, pos, 6)

            # Draw Node text labels
            for node in nodes:
                pos = world_to_screen(node.x_coord, node.y_coord)
                node_color, _ = get_node_style(node)
                text_surface = FONT.render(node.name, True, node_color)
                text_rect = text_surface.get_rect(center=(pos[0], pos[1] - 28))
                bg_rect = text_rect.inflate(6, 4)
                pygame.draw.rect(
                    world_surface,
                    (20, 20, 25),
                    bg_rect,
                    border_radius=4,
                )
                world_surface.blit(text_surface, text_rect)

            # Animate Drones frame-by-frame with smooth interpolation
            for index, drone in enumerate(drones):
                drone_id = index + 1
                hist_path = drone_history_paths.get(
                    drone_id,
                    [network.start_hub])

                if current_step_idx >= len(hist_path) - 1:
                    # Final position arrived
                    final_hub = hist_path[-1]
                    x, y = float(final_hub.x_coord), float(final_hub.y_coord)
                else:
                    curr_hub = hist_path[current_step_idx]
                    next_hub = hist_path[current_step_idx + 1]
                    dx = next_hub.x_coord - curr_hub.x_coord
                    dy = next_hub.y_coord - curr_hub.y_coord
                    x = curr_hub.x_coord + dx * progress
                    y = curr_hub.y_coord + dy * progress

                screen_pos = world_to_screen(x, y)
                # Offset for drone visual representation
                # offset = index * 4
                offset = 0
                screen_pos = (screen_pos[0] + offset, screen_pos[1] + offset)
                draw_drone(world_surface, screen_pos, drone_sprite)

            SCREEN.fill((0, 0, 0))

            visible_width = int(WIDTH / zoom_level)
            visible_height = int(HEIGHT / zoom_level)
            camera_rect = pygame.Rect(
                camera_x,
                camera_y,
                visible_width,
                visible_height,
            )

            camera_rect.clamp_ip(world_surface.get_rect())

            sub_surface = world_surface.subsurface(camera_rect)
            scaled_surface = pygame.transform.smoothscale(
                sub_surface,
                (WIDTH, HEIGHT),
            )

            SCREEN.blit(scaled_surface, (0, 0))

            self.draw_step_counter(
                SCREEN,
                FONT,
                display_step,
                total_steps,
            )
            pygame.display.flip()

        pygame.quit()
        sys.exit()
