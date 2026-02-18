import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import TD3, PPO, DDPG, SAC
from new_env import DoublePendulumEnv

# =================配置区域=================
MODELS_TO_EVALUATE = [
    {"name": "TD3", "class": TD3, "path": "model_TD3_double_pendulum"},
    {"name": "PPO",  "class": PPO,  "path": "model_PPO_double_pendulum"},
    {"name": "DDPG", "class": DDPG, "path": "model_DDPG_double_pendulum"},
]

SIMULATION_STEPS = 1000  # 仿真时长 (1000 steps = 10秒)


# =========================================

def evaluate_model(algo_name, model_class, model_path):
    print(f"正在评估算法: {algo_name} ...")

    # 1. 创建环境 (渲染模式设为 human 可以看动画，设为 None 只跑数据)
    # 为了画图，我们通常先用 None 跑数据，最后再根据需要决定是否要看动画
    env = DoublePendulumEnv(render_mode='human')

    # 2. 加载模型
    try:
        model = model_class.load(model_path)
    except FileNotFoundError:
        print(f"错误: 找不到模型文件 {model_path}.zip，跳过此算法。")
        return

    # 3. 运行仿真并记录数据
    obs, _ = env.reset()
    print(obs)
    # 用于存储历史数据
    history = {
        "alpha": [],
        "theta": [],
        "pos": [],
        "voltage": [],
        "steps": []
    }

    for step in range(SIMULATION_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        # 提取状态数据: [xc, xc_dot, alpha, alpha_dot, theta, theta_dot]
        xc, _, alpha, _, theta, _ = obs

        # 记录数据
        history["steps"].append(step)
        history["pos"].append(xc)
        history["alpha"].append(alpha)
        history["theta"].append(theta)

        # 记录电压 (处理 action 格式)
        vm = float(action[0]) if isinstance(action, np.ndarray) else float(action)
        history["voltage"].append(vm)

        if terminated or truncated:
            obs, _ = env.reset()
            # 如果中间断了，我们可以选择继续跑或者重置，这里简单处理为重置后继续记录

    env.close()

    # 4. 绘图
    plot_results(algo_name, history)


def plot_results(algo_name, data):
    steps = data["steps"]

    # 创建画布
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    plt.subplots_adjust(hspace=0.2)

    # === 子图 1: 角度变化 (Pendulum Angles) ===
    ax1.plot(steps, data["alpha"], label="Alpha (Link 1)", color="#008000", linewidth=1.5)  # Green
    ax1.plot(steps, data["theta"], label="Theta (Link 2)", color="#FF0000", linewidth=1.5)  # Red
    ax1.axhline(0, color="gray", linestyle="--", alpha=0.6)  # 0度参考线

    ax1.set_title(f"{algo_name} - Pendulum Angles (Goal = 0)", fontsize=12)
    ax1.set_ylabel("Angle (radians)", fontsize=10)
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # === 子图 2: 位置与电压 (Cart Position & Motor Voltage) ===
    # 左轴：位置
    color_pos = "blue"
    ax2.set_xlabel("Steps (0.01s per step)", fontsize=10)
    ax2.set_ylabel("Position (m)", color=color_pos, fontsize=10)
    line1, = ax2.plot(steps, data["pos"], color=color_pos, label="Cart Position (m)", linewidth=1.5)
    ax2.tick_params(axis='y', labelcolor=color_pos)
    ax2.grid(True, alpha=0.3)

    # 右轴：电压 (Twin axis)
    ax2_twin = ax2.twinx()
    color_vol = "#FFA500"  # Orange/Yellowish
    line2, = ax2_twin.plot(steps, data["voltage"], color=color_vol, label="Voltage (V)", linewidth=0.5, alpha=0.8)
    ax2_twin.set_ylabel("Motor Voltage (V)", color="black", fontsize=10)
    ax2_twin.tick_params(axis='y', labelcolor="black")
    ax2_twin.set_ylim(-12, 12)  # 固定电压范围以便观察

    # 合并图例
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc="upper right")

    ax2.set_title("Cart Position & Motor Voltage", fontsize=12)

    # 保存图片
    filename = f"result_plot_{algo_name}.png"
    plt.savefig(filename, dpi=150)
    print(f"图表已保存为: {filename}")
    plt.show()


if __name__ == "__main__":
    # 循环评估列表中的每一个模型
    for item in MODELS_TO_EVALUATE:
        evaluate_model(item["name"], item["class"], item["path"])