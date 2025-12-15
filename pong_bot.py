import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import arcade
import time


class PongEnv(gym.Env):
    """
    Two-paddle Pong.

    - LEFT paddle  = learning agent (this env is built for the NEW model).
    - RIGHT paddle = opponent:
        * If opponent_model is provided → uses its policy.
        * Otherwise → simple heuristic that chases the ball.

    Rewards are from the LEFT paddle's perspective.
    """
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, width=800, height=600, render_mode=None, opponent_model=None):
        super().__init__()

        # Rewards from LEFT (agent) point of view
        self.WIN_REWARD = 0
        self.LOSE_REWARD = 0
        self.HIT_REWARD = 1

        self.width = width
        self.height = height

        self.BALL_SIZE = 20.0
        self.PADDLE_WIDTH = 15.0
        self.PADDLE_HEIGHT = 100.0

        # 0 = stay, 1 = move up, 2 = move down  (FOR LEFT PADDLE / AGENT)
        self.action_space = spaces.Discrete(3)

        low = np.array([0.0, 0.0, -10.0, -10.0, 0.0, 0.0], dtype=np.float32)
        high = np.array(
            [self.width, self.height, 10.0, 10.0, self.height, self.height],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.left_dy = 0.0
        self.right_dy = 0.0

        self.render_mode = render_mode
        self.opponent_model = opponent_model  # OLD model for RIGHT paddle

        # Visuals setup
        self.window = None
        self.sprites = None
        self.ball_sprite = None
        self.left_paddle_sprite = None
        self.right_paddle_sprite = None

        self.reset()

    # --- Geometry helpers ---
    @property
    def ball_top(self):
        return self.ball_y + self.BALL_SIZE / 2

    @property
    def ball_bottom(self):
        return self.ball_y - self.BALL_SIZE / 2

    @property
    def ball_left(self):
        return self.ball_x - self.BALL_SIZE / 2

    @property
    def ball_right(self):
        return self.ball_x + self.BALL_SIZE / 2
    
    @property
    def paddle_half_h(self):
        return self.PADDLE_HEIGHT / 2.0

    @property
    def left_paddle_x(self):
        return 10.0 + self.PADDLE_WIDTH / 2

    @property
    def right_paddle_x(self):
        return self.width - 10.0 - self.PADDLE_WIDTH / 2

    def _rects_overlap(self, x1, y1, w1, h1, x2, y2, w2, h2):
        return abs(x1 - x2) * 2 < (w1 + w2) and abs(y1 - y2) * 2 < (h1 + h2)

    # --- Gym API ---
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.ball_x = self.width / 2.0
        self.ball_y = self.height / 2.0

        vectors = [
            (-3, -3), (-3, -2), (-3, -1), (-3, 1), (-3, 2), (-3, 3),
            (-2, -3), (-2, -2), (-2, -1), (-2, 1), (-2, 2), (-2, 3),
            (-1, -3), (-1, -2), (-1, -1), (-1, 1), (-1, 2), (-1, 3),
            (1, -3), (1, -2), (1, -1), (1, 1), (1, 2), (1, 3),
            (2, -3), (2, -2), (2, -1), (2, 1), (2, 2, ), (2, 3),
            (3, -3), (3, -2), (3, -1), (3, 1), (3, 2), (3, 3),
        ]
        dx, dy = random.choice(vectors)
        self.ball_dx = float(dx)
        self.ball_dy = float(dy)

        # Center paddles
        self.left_y = self.height / 2.0   # agent
        self.right_y = self.height / 2.0  # opponent
        self.left_dy = 0.0
        self.right_dy = 0.0

        return self._get_obs(), {}

    def step(self, action):
        """
        action controls the LEFT (agent) paddle.
        RIGHT paddle is controlled by opponent_model or a built-in heuristic.
        """
        # --- Left paddle: NEW learning agent ---
        if action == 0:
            self.left_dy = 0.0
        elif action == 1:
            self.left_dy = 5.0
        else:
            self.left_dy = -5.0

        # --- Right paddle: OLD model or heuristic ---
        if self.opponent_model is not None:
            # Use the same observation structure the old model was trained on
            opp_action, _ = self.opponent_model.predict(self._get_obs(), deterministic=True)

            if opp_action == 0:
                self.right_dy = 0.0
            elif opp_action == 1:
                self.right_dy = 5.0
            else:
                self.right_dy = -5.0
        else:
            # Fallback: simple AI that chases the ball, slightly slower
            if self.right_y < self.ball_y:
                self.right_dy = 3.0
            elif self.right_y > self.ball_y:
                self.right_dy = -3.0
            else:
                self.right_dy = 0.0

        # --- Move paddles and ball ---
        self.left_y = self._wrap_paddle_y(self.left_y + self.left_dy)
        self.right_y = self._wrap_paddle_y(self.right_y + self.right_dy)


        self.ball_x += self.ball_dx
        self.ball_y += self.ball_dy

        reward = 0.0
        terminated = False

        # --- Collisions with paddles ---
        hit_left = self._rects_overlap(
            self.ball_x, self.ball_y, self.BALL_SIZE, self.BALL_SIZE,
            self.left_paddle_x, self.left_y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT,
        )
        hit_right = self._rects_overlap(
            self.ball_x, self.ball_y, self.BALL_SIZE, self.BALL_SIZE,
            self.right_paddle_x, self.right_y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT,
        )

        if hit_left or hit_right:
            self.ball_dx *= -1.0
            self.ball_dx += 0.1 if self.ball_dx > 0 else -0.1
            self.ball_dy += 0.1 if self.ball_dy > 0 else -0.1

        # Agent (LEFT) reward shaping
        if hit_left:
            reward += self.HIT_REWARD       # good: agent hit ball                     # slight penalty: opponent returned it

        # --- Top/bottom bounce ---
        if self.ball_top >= self.height or self.ball_bottom <= 0.0:
            self.ball_dy *= -1.0

        # --- Terminal conditions from LEFT agent perspective ---
        # LEFT paddle (agent) LOSES if ball exits on the LEFT side
        if self.ball_left <= 0.0:
            reward += self.LOSE_REWARD
            terminated = True

        # LEFT paddle (agent) WINS if ball exits on the RIGHT side
        if self.ball_right >= self.width:
            reward += self.WIN_REWARD
            terminated = True

        return self._get_obs(), reward, terminated, False, {}

    def _get_obs(self):
        # Same order as before so old model can still understand it
        return np.array(
            [self.ball_x, self.ball_y, self.ball_dx, self.ball_dy, self.left_y, self.right_y],
            dtype=np.float32,
        )

    def render(self):
        if self.render_mode == "human":
            # Create the window and sprites once
            if self.window is None:
                self.window = arcade.Window(self.width, self.height, "Pong Self-Play")
                self.sprites = arcade.SpriteList()

                try:
                    self.ball_sprite = arcade.Sprite("images/ball.png", 0.05)
                    self.left_paddle_sprite = arcade.Sprite("images/bar.png")
                    self.right_paddle_sprite = arcade.Sprite("images/bar.png")
                except FileNotFoundError:
                    print("Warning: Images not found, using shapes instead.")
                    return

                self.sprites.append(self.ball_sprite)
                self.sprites.append(self.left_paddle_sprite)
                self.sprites.append(self.right_paddle_sprite)

            # Sync positions
            self.ball_sprite.center_x = self.ball_x
            self.ball_sprite.center_y = self.ball_y

            self.left_paddle_sprite.center_x = self.left_paddle_x
            self.left_paddle_sprite.center_y = self.left_y

            self.right_paddle_sprite.center_x = self.right_paddle_x
            self.right_paddle_sprite.center_y = self.right_y

            # Draw frame
            self.window.clear()
            arcade.set_background_color(arcade.color.BLACK)
            self.sprites.draw()
            self.window.flip()

            time.sleep(0.03)

    def _wrap_paddle_y(self, y):
        top = self.height + self.paddle_half_h
        bottom = -self.paddle_half_h

        if y > top:
            return bottom
        if y < bottom:
            return top
        return y

    def close(self):
        if self.window:
            self.window.close()
            self.window = None
