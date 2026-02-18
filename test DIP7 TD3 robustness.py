import numpy as np
import matplotlib

try:
    matplotlib.use('TkAgg')  # 修复部分系统下弹窗报错问题
except:
    pass
import matplotlib.pyplot as plt
from stable_baselines3 import TD3

try:
    from new_env import DoublePendulumEnv
except ImportError:
    from double_pendulum_env import DoublePendulumEnv

# --- 核心配置 ---
MODEL_PATH = "model_TD3_double_pendulum.zip"
TEST_DURATION = 12.0  # 测试时长 12秒
RENDER = True  # 是否显示动画窗口
DT_OVERRIDE = None  # 如果需要强制指定dt，否则自动读取环境


def apply_disturbance(env, step_idx, dt):
    """
    定义干扰逻辑：在不同时间点施加不同类型的推力
    返回: external_force (用于绘图，非0表示有干扰)
    """
    current_time = step_idx * dt
    dist_flag = 0.0

    # --- 干扰 1: 时间 2.0s - 2.2s (向右猛推小车) ---
    # 测试对底座位移的抵抗能力
    if 2.0 <= current_time <= 2.2:
        env.state[1] += 1.5 * dt  # 修改 x_dot (给小车加速度)
        dist_flag = 1.0  # 标记为干扰中
        if step_idx % 10 == 0: print(f"[t={current_time:.2f}s] 干扰: 向右推车 (Cart Push)")

    # --- 干扰 2: 时间 5.0s - 5.2s (拨动第一根杆子) ---
    # 测试中间关节的刚度
    elif 5.0 <= current_time <= 5.2:
        env.state[3] += 3.0 * dt  # 修改 alpha_dot (给杆1角速度)
        dist_flag = -1.0
        if step_idx % 10 == 0: print(f"[t={current_time:.2f}s] 干扰: 拨动Link1 (Link 1 Push)")

    # --- 干扰 3: 时间 8.0s - 8.2s (拨动第二根杆子 - 末端) ---
    # 测试最难的末端稳定能力
    elif 8.0 <= current_time <= 8.2:
        env.state[5] -= 3.0 * dt  # 修改 theta_dot (给杆2反向角速度)
        dist_flag = 1.0
        if step_idx % 10 == 0: print(f"[t={current_time:.2f}s] 干扰: 拨动Link2 (Tip Push)")

    return dist_flag


def run_test():
    print(f"\n=== 最终鲁棒性测试 (TD3) : {MODEL_PATH} ===")

    # 1. 初始化环境
    env = DoublePendulumEnv(render_mode="human" if RENDER else None)

    # 获取时间步长
    dt = env.dt if DT_OVERRIDE is None else DT_OVERRIDE
    print(f"环境参数: dt={dt}s, Max Steps={int(TEST_DURATION / dt)}")

    # 2. 加载模型
    try:
        model = TD3.load(MODEL_PATH)
        print(f"模型加载成功: {MODEL_PATH}")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 3. 开始模拟
    obs, _ = env.reset()

    # 数据记录器
    history = {
        'time': [],
        'pos': [],
        'alpha': [],
        'theta': [],
        'action': [],
        'disturbance': []
    }

    max_steps = int(TEST_DURATION / dt)

    print("\n🚀 模拟开始...")

    for step in range(max_steps):
        # 3.1 模型预测 (deterministic=True)
        action, _ = model.predict(obs, deterministic=True)

        # 3.2 施加人为干扰
        dist_val = apply_disturbance(env, step, dt)

        # 3.3 环境步进
        obs, reward, terminated, truncated, _ = env.step(action)

        # 3.4 记录数据
        raw_state = env.state  # [x, x_dot, alpha, alpha_dot, theta, theta_dot]

        history['time'].append(step * dt)
        history['pos'].append(raw_state[0])
        history['alpha'].append(raw_state[2])
        history['theta'].append(raw_state[4])
        history['action'].append(float(action[0]))
        history['disturbance'].append(dist_val)

        if terminated or truncated:
            print(f"失败: 在 t={step * dt:.2f}s 时倒塌或超出范围")
            obs, _ = env.reset()

    env.close()
    print("模拟结束")

    # 4. 绘图
    plot_results(history)


def plot_results(h):
    t = np.array(h['time'])
    pos = np.array(h['pos'])
    alpha = np.array(h['alpha'])
    theta = np.array(h['theta'])
    act = np.array(h['action'])
    dist = np.array(h['disturbance'])

    # --- 打印电压统计信息 (验证是否有控制) ---
    print("\n=== 电压动作统计 ===")
    print(f"最大输出电压: {np.max(act):.4f} V")
    print(f"最小输出电压: {np.min(act):.4f} V")
    print(f"平均绝对电压: {np.mean(np.abs(act)):.4f} V (平时维持平衡用的力)")
    print("======================\n")

    fig, axs = plt.subplots(4, 1, figsize=(10, 14), sharex=True)

    # 辅助函数：画红色干扰带
    def plot_dist_bg(ax):
        # 找到 dist 不为 0 的区域并填充红色
        if np.any(dist != 0):
            # ★★★ 修改点：把 get_ylim 换成 0 和 1 ★★★
            # 配合 transform=ax.get_xaxis_transform()，0代表最底，1代表最顶
            ax.fill_between(t, 0, 1,
                            where=(dist != 0),
                            color='red', alpha=0.2, transform=ax.get_xaxis_transform(),
                            label='External Disturbance')

    # 1. 位移
    axs[0].plot(t, pos, 'b', lw=2, label='Cart Pos')
    plot_dist_bg(axs[0])
    axs[0].set_ylabel('Position (m)')
    axs[0].set_title(f'Robustness Test: 3 Disturbances (Model: {MODEL_PATH})')
    axs[0].legend(loc='upper right')
    axs[0].grid(True)

    # 2. 角度
    axs[1].plot(t, alpha, 'g', label='Link 1 (Alpha)')
    axs[1].plot(t, theta, 'orange', label='Link 2 (Theta)')
    plot_dist_bg(axs[1])
    axs[1].set_ylabel('Angle (rad)')
    # 设置角度范围以便观察微小变化 (可选)
    # axs[1].set_ylim(-0.2, 0.2)
    axs[1].legend(loc='upper right')
    axs[1].grid(True)

    # 3. 电压动作 (修复显示问题)
    axs[2].plot(t, act, 'purple', lw=1.5, label='Voltage (Raw)')
    plot_dist_bg(axs[2])
    axs[2].set_ylabel('Action (V)')

    # 强制限制 Y 轴范围，避免因为瞬间的神经网络极值导致正常波形看不清
    # 物理限制是 +/- 20V
    Y_LIMIT = 20
    axs[2].set_ylim(-Y_LIMIT, Y_LIMIT)
    axs[2].text(t[0], -Y_LIMIT + 2, f"Display Clipped to +/- {Y_LIMIT}V", fontsize=9, color='red')

    axs[2].legend(loc='upper right')
    axs[2].grid(True)

    # 4. 抖动分析
    d_act = np.diff(act, prepend=act[0])
    axs[3].plot(t, d_act, 'k', lw=1, label='Delta Voltage')
    plot_dist_bg(axs[3])
    axs[3].set_ylabel('Stability (Jitter)')
    axs[3].set_xlabel('Time (s)')
    # 同样限制抖动显示范围
    axs[3].set_ylim(-50, 50)
    axs[3].legend(loc='upper right')
    axs[3].grid(True)

    plt.tight_layout()
    plt.show()
    print("绘图完成.")


if __name__ == "__main__":
    run_test()