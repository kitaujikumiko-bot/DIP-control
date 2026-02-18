import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import TD3, DDPG, PPO
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from new_env import DoublePendulumEnv


# ==========================================
# 1. 自定义回调函数：用于记录绘图数据
# ==========================================
class DataLoggerCallback(BaseCallback):
    def __init__(self, check_freq: int, verbose=0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.data = {
            "timesteps": [],
            "mean_rewards": [],
            "mean_lengths": []
        }

    def _on_step(self) -> bool:
        # 每隔 check_freq 步记录一次数据
        if self.n_calls % self.check_freq == 0:
            # 获取最近 100 个回合的信息 (由 Monitor 包装器提供)
            ep_infos = self.model.ep_info_buffer

            if ep_infos:
                avg_reward = np.mean([ep_info["r"] for ep_info in ep_infos])
                avg_length = np.mean([ep_info["l"] for ep_info in ep_infos])

                self.data["timesteps"].append(self.num_timesteps)
                self.data["mean_rewards"].append(avg_reward)
                self.data["mean_lengths"].append(avg_length)

                if self.verbose > 0:
                    print(f"Step {self.num_timesteps}: Reward={avg_reward:.2f}, Length={avg_length:.1f}")
        return True


# ==========================================
# 2. 实验设置
# ==========================================
# 训练总步数 (建议至少30万步以观察收敛差异)
TOTAL_TIMESTEPS = 300000
LOG_FREQ = 1000  # 每 1000 步记录一个点

# 存储所有算法的结果
all_results = {}

# 定义要对比的算法
# PPO 是 On-Policy，DDPG/TD3 是 Off-Policy
algorithms_to_test = [
    ("TD3", TD3),
    ("DDPG", DDPG),
    ("PPO", PPO)
]

print(f"开始对比训练: {[name for name, _ in algorithms_to_test]}")
print(f"每种算法训练 {TOTAL_TIMESTEPS} 步...")

# ==========================================
# 3. 循环训练每种算法
# ==========================================
for algo_name, AlgoClass in algorithms_to_test:
    print(f"\n{'=' * 30}")
    print(f"正在训练 {algo_name} ...")
    print(f"{'=' * 30}")

    # 1. 创建环境 (必须用 Monitor 包装以提取数据)
    env = DoublePendulumEnv(render_mode=None)
    env = Monitor(env)

    # 2. 准备动作噪声 (仅用于 DDPG 和 TD3)
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # 3. 初始化模型
    # 根据算法不同，参数略有区别
    if algo_name == "PPO":
        model = AlgoClass(
            "MlpPolicy",
            env,
            verbose=0,
            learning_rate=3e-4,
            n_steps=2048,  # PPO 特有参数
            batch_size=64
        )
    else:  # TD3 和 DDPG
        model = AlgoClass(
            "MlpPolicy",
            env,
            action_noise=action_noise,
            verbose=0,
            learning_rate=1e-3,
            batch_size=256
        )

    # 4. 创建记录器并开始训练
    logger = DataLoggerCallback(check_freq=LOG_FREQ)
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=logger)

    # 5. 保存结果数据
    all_results[algo_name] = logger.data

    # 6. 保存模型文件
    model.save(f"model_{algo_name}_double_pendulum")

    env.close()
    print(f"--> {algo_name} 训练完成")

# ==========================================
# 4. 绘制对比曲线
# ==========================================
print("\n正在生成对比图表...")

# 设置图表风格
plt.style.use('default')  # or 'seaborn'
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

# 颜色映射
colors = {"TD3": "red", "DDPG": "blue", "PPO": "green"}

for name, data in all_results.items():
    steps = data["timesteps"]
    rewards = data["mean_rewards"]
    lengths = data["mean_lengths"]
    c = colors.get(name, "black")

    # 绘制平滑曲线 (计算移动平均以减少抖动)
    # 直接绘制原始记录点
    ax1.plot(steps, rewards, label=name, color=c, linewidth=2, alpha=0.8)
    ax2.plot(steps, lengths, label=name, color=c, linewidth=2, alpha=0.8)

# 图表 1: 平均奖励 (Mean Reward)
ax1.set_ylabel("Mean Reward (Last 100 Episodes)", fontsize=12)
ax1.set_title("Algorithm Comparison: Reward vs Timesteps", fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=12)

# 图表 2: 平均坚持时间 (Mean Episode Length)
# 如果到了 2000，说明完全平衡
ax2.set_ylabel("Mean Upright Time (Steps)", fontsize=12)
ax2.set_xlabel("Total Training Timesteps", fontsize=12)
ax2.set_title("Balancing Performance: Episode Length vs Timesteps", fontsize=14)
ax2.axhline(2000, color='black', linestyle='--', label="Max Steps (2000)", alpha=0.5)  # 标出最大步数
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=12)

plt.tight_layout()

# 保存图表
plt.savefig("algorithm_comparison.png", dpi=300)
print("图表已保存为 algorithm_comparison.png")
plt.show()