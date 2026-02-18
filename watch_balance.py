import matplotlib

try:
    matplotlib.use('TkAgg')
except:
    pass

import matplotlib.pyplot as plt
import numpy as np
import imageio
from stable_baselines3 import TD3

try:
    from new_env import DoublePendulumEnv
except ImportError:
    from double_pendulum_env import DoublePendulumEnv

# --- 配置区 ---
MODEL_PATH = "model_TD3_double_pendulum.zip"
GIF_FILENAME = "double_pendulum.gif"
FPS = 20
DURATION = 10.0
X_LIMIT = (-1.2, 1.2)


def save_gif():
    print(f"🎬 准备录制 GIF: {GIF_FILENAME}")
    print(f"⏱️ 时长: {DURATION}s | 帧率: {FPS}")

    env = DoublePendulumEnv(render_mode="human")
    dt = env.dt

    sim_steps_per_frame = int(1.0 / (FPS * dt))
    if sim_steps_per_frame < 1: sim_steps_per_frame = 1

    max_steps = int(DURATION / dt)

    try:
        model = TD3.load(MODEL_PATH)
        print("✅ 模型加载成功")
    except:
        print("❌ 未找到模型，使用随机动作演示")
        model = None

    obs, _ = env.reset()
    frames = []

    plt.ion()

    print("🚀 开始采集画面...")

    try:
        for step in range(max_steps):
            if model:
                action, _ = model.predict(obs, deterministic=True)
                wind_noise = np.sin(step * dt * 2.0) * 0.5
                action[0] += wind_noise
            else:
                action = env.action_space.sample()

            obs, _, done, truncated, _ = env.step(action)
            env.render()

            if step % sim_steps_per_frame == 0:
                fig = plt.gcf()
                ax = plt.gca()

                ax.set_xlim(X_LIMIT)
                ax.set_ylim(X_LIMIT)
                ax.set_title(f"Simulation: {step * dt:.2f}s")

                # --- 【核心修复部分】 ---
                fig.canvas.draw()

                # 新版 Matplotlib 获取图像数据的方法:
                try:
                    # 尝试使用 buffer_rgba (新版标准)
                    image = np.array(fig.canvas.renderer.buffer_rgba())
                except AttributeError:
                    # 如果是很老的版本，回退旧方法 (以防万一)
                    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                    width, height = fig.canvas.get_width_height()
                    image = image.reshape(height, width, 3)

                # buffer_rgba 返回的是 RGBA (4通道)，如果不想要透明度，可以切片只取前3个通道
                if image.shape[2] == 4:
                    image = image[:, :, :3]

                frames.append(image)

                print(f"\r采集进度: {len(frames)} / {FPS * DURATION:.0f} 帧", end="")

            if done or truncated:
                obs, _ = env.reset()

    except KeyboardInterrupt:
        print("\n🛑 手动停止采集")
    finally:
        env.close()
        plt.close('all')

    print(f"\n\n💾 正在合成 GIF...")
    if len(frames) > 0:
        imageio.mimsave(GIF_FILENAME, frames, fps=FPS, loop=0)
        print(f"✅ 成功！GIF 已保存至: {GIF_FILENAME}")
    else:
        print("❌ 没有采集到任何画面。")


if __name__ == "__main__":
    save_gif()