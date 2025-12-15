from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from pong_bot import PongEnv

import os, time, requests, traceback

DISCORD_WEBHOOK_URL = "PASTE_NEW_WEBHOOK_URL_HERE"

def notify_discord(msg: str):
    try:
        requests.post(DISCORD_WEBHOOK_URL, json={"content": msg}, timeout=10)
    except Exception as e:
        print("Discord notification failed:", e)

TOTAL_STEPS = 1_000_000
start_time = time.time()

# Train env (no render)
train_env = Monitor(PongEnv(render_mode=None))

# Eval env (also no render; eval should be fast + consistent)
eval_env = Monitor(PongEnv(render_mode=None))

log_dir = "./pong_logs/"
best_dir = "./best_model/"
os.makedirs(best_dir, exist_ok=True)

# This callback evaluates periodically and saves best checkpoint to best_dir
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=best_dir,   # saves best model here as "best_model.zip"
    log_path=log_dir,                # writes eval logs (optional)
    eval_freq=10_000,                # evaluate every N steps (tweak)
    n_eval_episodes=20,              # more episodes = less noisy
    deterministic=True,
    render=False,
)

model = PPO("MlpPolicy", train_env, verbose=1, tensorboard_log=log_dir)

notify_discord(f"🚀 Pong training started • Steps: {TOTAL_STEPS:,} • Eval every 10k • Best→ {best_dir}")

try:
    model.learn(total_timesteps=TOTAL_STEPS, callback=eval_callback)

    # Load the best model (not the last one) and save it under your desired name
    best_path = os.path.join(best_dir, "best_model.zip")
    best_model = PPO.load(best_path, env=train_env)
    best_model.save("pong_agent_best")

    elapsed = int(time.time() - start_time)
    notify_discord(f"✅ Done! Saved BEST as `pong_agent_best` • Time: {elapsed//60}m {elapsed%60}s")

except Exception:
    err = traceback.format_exc()
    notify_discord("💥 Training crashed:\n```" + err[:1800] + "```")
    raise
finally:
    train_env.close()
    eval_env.close()
