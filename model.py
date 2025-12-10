from stable_baselines3 import PPO
from pong_bot import PongEnv

old_model = PPO.load("pong_agent")  # whatever you saved it as

# New env: LEFT learns, RIGHT uses old_model
env = PongEnv(render_mode=None, opponent_model=old_model)

new_model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./pong_logs/")
new_model.learn(total_timesteps=500_000)
new_model.save("pong_agent_new")
