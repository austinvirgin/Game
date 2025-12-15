# evaluate.py

from stable_baselines3 import PPO
from pong_bot import PongEnv

while True:
    env = PongEnv(render_mode="human")

    model = PPO.load("pong_agent_fixed")

    obs, _ = env.reset()

    done = False
    final_reward = 0
    total_reward = 0

    while not done:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        total_reward += reward
        final_reward = reward        # keep the last reward
        done = terminated or truncated

    # After the game ends, say who won
    if final_reward == env.WIN_REWARD:
        print("Game over: LEFT paddle (agent) LOST")
    elif final_reward == env.LOSE_REWARD:
        print("Game over: LEFT paddle (agent) WON")
    else:
        print(f"Game over with unexpected final reward {final_reward}")
    print(total_reward)
    env.close()
