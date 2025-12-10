import arcade
import arcade.gui
import random
import numpy as np

class Pong(arcade.View):
    def __init__(self, menu):
        super().__init__()
        self.manager = arcade.gui.UIManager()
        self.menu = menu
        self.sprites = arcade.SpriteList()
        self.paddles = arcade.SpriteList()
        self.current_action = 0

    def apply_agent_action(self, action: int):
        
        if action == 0:
            self.bar_right.change_y = 0

        elif action == 1:
            self.bar_right.change_y = 5

        elif action == 2:
            self.bar_right.change_y = -5

    def setup(self):
        arcade.set_background_color(arcade.color.BLACK)

        self.ball = arcade.Sprite("images/ball.png", .05)
        self.ball.center_y = self.height / 2
        self.ball.center_x = self.width / 2
        vectors = [
            (-3, -3), (-3, -2), (-3, -1), (-3, 1), (-3, 2), (-3, 3),
            (-2, -3), (-2, -2), (-2, -1), (-2, 1), (-2, 2), (-2, 3),
            (-1, -3), (-1, -2), (-1, -1), (-1, 1), (-1, 2), (-1, 3),
            (1, -3), (1, -2), (1, -1), (1, 1), (1, 2), (1, 3),
            (2, -3), (2, -2), (2, -1), (2, 1), (2, 2), (2, 3),
            (3, -3), (3, -2), (3, -1), (3, 1), (3, 2), (3, 3),
        ]
        vector = vectors[random.randint(0, len(vectors) - 1)]
        self.ball.velocity = (vector[0], vector[1])
        self.sprites.append(self.ball)

        self.bar_left = arcade.Sprite("images/bar.png")
        self.bar_left.center_y = self.height / 2
        self.bar_left.left = 10
        self.sprites.append(self.bar_left)
        self.paddles.append(self.bar_left)

        self.bar_right = arcade.Sprite("images/bar.png")
        self.bar_right.center_y = self.height / 2
        self.bar_right.right = self.width - 10
        self.sprites.append(self.bar_right)
        self.paddles.append(self.bar_right)

    def on_show_view(self):
        arcade.set_background_color(arcade.color.BABY_BLUE)
        self.manager.enable()

    def on_hide_view(self):
        self.manager.disable()

    def on_draw(self):
        self.clear()
        self.sprites.draw()

    def on_update(self, delta_time):
        obs = self.build_observation()
        action, _ = self.agent_model.predict(obs, deterministic = True)
        self.apply_agent_action(int(action))

        self.sprites.update()

        if self.ball.collides_with_list(self.paddles):
            self.ball.change_x *= -1

        if self.ball.change_y > 0:
            self.ball.change_y += .001

        if self.ball.change_y < 0:
            self.ball.change_y -= .001

        if self.ball.change_x > 0:
            self.ball.change_x += .001

        if self.ball.change_x < 0:
            self.ball.change_x -= .001

        if self.ball.top >= self.height or self.ball.bottom <= 0:
            self.ball.change_y *= -1

        if self.ball.right <= 0 or self.ball.left >= self.width:
            self.window.show_view(self.menu)

        if self.bar_left.top > self.height:
            self.bar_left.top = self.height

        if self.bar_left.bottom < 0:
            self.bar_left.bottom = 0

        if self.bar_right.top > self.height:
            self.bar_right.top = self.height

        if self.bar_right.bottom < 0:
            self.bar_right.bottom = 0

    def on_key_press(self, symbol, modifiers):
        if symbol == arcade.key.W:
            self.bar_left.change_y = 5

        if symbol == arcade.key.S:
            self.bar_left.change_y = -5

        # if symbol == arcade.key.UP:
        #     self.bar_right.change_y = 5

        # if symbol == arcade.key.DOWN:
        #     self.bar_right.change_y = -5

    def on_key_release(self, symbol, modifiers):
        if symbol == arcade.key.W or symbol == arcade.key.S:
            self.bar_left.change_y = 0

        # if symbol == arcade.key.UP or symbol == arcade.key.DOWN:
        #     self.bar_right.change_y = 0

    def build_observation(self):
        return np.array([
            self.ball.center_x,
            self.ball.center_y,
            self.ball.change_x,
            self.ball.change_y,
            self.bar_left.center_y,
            self.bar_right.center_y,
        ], dtype=np.float32)