"""
Double Pendulum on a Cart Environment (Gymnasium style)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Optional rendering (matplotlib)
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle


@dataclass(frozen=True)
class PendulumParams:
    Bc: float = 5.4
    B1: float = 0.0024
    B2: float = 0.0024
    g: float = 9.81
    Jm: float = 3.9e-7
    Kg: float = 3.71
    Km: float = 0.00767
    Kt: float = 0.00767
    L1: float = 0.2096
    L2: float = 0.3365
    l1: float = 0.1143
    l2: float = 0.1778
    Mc: float = 0.57 + 0.37  # Cart mass + Extra weight in lab
    Mh: float = 0.17
    M1: float = 0.072
    M2: float = 0.127
    Rm: float = 2.6
    rmp: float = 6.35e-3


PARAMS = PendulumParams()


def compute_accelerations(state: np.ndarray, Vm: float, p: PendulumParams) -> Tuple[float, float, float]:
    """
    Input: state: [xc, xc_dot, alpha, alpha_dot, theta, theta_dot]
    Returns: [xc_ddot, alpha_ddot, theta_ddot]
    """
    xc, vc, alpha, va, theta, vt = state

    # Parameters
    Mc, M1, M2, Mh, L1, L2, l1, l2, g = p.Mc, p.M1, p.M2, p.Mh, p.L1, p.L2, p.l1, p.l2, p.g
    Jm, Kg, rmp, Kt, Km, Rm, Bc, B1, B2 = p.Jm, p.Kg, p.rmp, p.Kt, p.Km, p.Rm, p.Bc, p.B1, p.B2

    # Precompute trigonometric terms
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    cat, sat = np.cos(alpha + theta), np.sin(alpha + theta)

    DetMSymbol = (
            -(1 / 12) * cat * l2 ** 2 * M2 ** 2 *
            (-12 * ca * (ct * L1 + l2) * (l1 * M1 + L1 * (M2 + Mh))
             +
             cat * (12 * l1 ** 2 * M1 + L1 * (12 * ct * l2 * M2 + L1 * (M1 + 12 * (M2 + Mh))))
             )
            -
            l2 ** 2 * (ct * L1 + l2) * M2 ** 2 *
            (-cat * (ca * l1 * M1 + cat * l2 * M2 + ca * L1 * (M2 + Mh))
             +
             (ct * L1 + l2) * (M1 + M2 + Mc + Mh + (Jm * Kg ** 2) / rmp ** 2))
            +
            (l2 ** 2 * M2 + (L2 ** 2 * M2) / 12)
            *
            (-(ca * l1 * M1 + cat * l2 * M2 + ca * L1 * (M2 + Mh)) ** 2
             +
             (l1 ** 2 * M1 + 2 * ct * L1 * l2 * M2 + l2 ** 2 * M2 + L1 ** 2 * (M1 / 12 + M2 + Mh))
             *
             (M1 + M2 + Mc + Mh + (Jm * Kg ** 2) / rmp ** 2))
    )

    xc_ddot = (
                      l2 * M2 * (12 * ca * (ct * L1 + l2) * (l1 * M1 + L1 * (M2 + Mh))
                                 - cat * (12 * l1 ** 2 * M1 + L1 * (12 * ct * l2 * M2 + L1 * (M1 + 12 * (M2 + Mh))))
                                 )
                      * (-g * l2 * M2 * sat + L1 * l2 * M2 * st * va ** 2 + B2 * vt)
                      +
                      12 * (-l2 ** 2 * (ct * L1 + l2) ** 2 * M2 ** 2
                            + 1 / 144 * (12 * l2 ** 2 + L2 ** 2) * M2
                            * (12 * l1 ** 2 * M1 + 24 * ct * L1 * l2 * M2
                               + 12 * l2 ** 2 * M2 + L1 ** 2 * (M1 + 12 * (M2 + Mh))
                               )
                            )
                      *
                      (-Bc * vc + (Kg * Kt * (-Kg * Km * vc + rmp * Vm)) / (Rm * rmp ** 2)
                       - l2 * M2 * sat * vt * (va + vt)
                       - va * (l1 * M1 * sa * va + L1 * (M2 + Mh) * sa * va
                               + l2 * M2 * sat * (va + vt)))
                      +
                      M2 * (g * (M1 * sa + L1 * (M2 + Mh) * sa + l2 * M2 * sat)
                            - B1 * va + L1 * l2 * M2 * st * vt * (2 * va + vt))
                      *
                      (ca * (l1 * (12 * l2 ** 2 + L2 ** 2) * M1 + L1 * L2 ** 2 * (M2 + Mh)
                             + 6 * L1 * l2 ** 2 * (M2 + 2 * Mh))
                       + l2 * M2 * (cat * L2 ** 2 - 6 * L1 * l2 * np.cos(alpha + 2 * theta)))
              ) / (12 * DetMSymbol)

    alpha_ddot = (
                         l2 * M2 * (cat ** 2 * l2 * M2 + ca * cat * (l1 * M1 + L1 * (M2 + Mh))
                                    - (ct * L1 + l2) * (M1 + M2 + Mc + Mh + (Jm * Kg ** 2) / rmp ** 2))
                         *
                         (g * l2 * M2 * sat - L1 * l2 * M2 * st * va ** 2 - B2 * vt)
                         +
                         (- cat ** 2 * l2 ** 2 * M2 ** 2 + (l2 ** 2 * M2 + (L2 ** 2 * M2) / 12)
                          * (M1 + M2 + Mc + Mh + (Jm * Kg ** 2) / rmp ** 2))
                         *
                         (g * (M1 * sa + L1 * (M2 + Mh) * sa + l2 * M2 * sat)
                          - B1 * va + L1 * l2 * M2 * st * vt * (2 * va + vt))
                         +
                         1 / 12 * M2 * (cat * l2 * (-12 * ct * L1 * l2 + L2 ** 2) * M2
                                        + ca * (12 * l2 ** 2 + L2 ** 2) * (l1 * M1 + L1 * (M2 + Mh))
                                        )
                         *
                         (-Bc * vc + (Kg * Kt * (-Kg * Km * vc + rmp * Vm)) / (Rm * rmp ** 2)
                          - l2 * M2 * sat * vt * (va + vt)
                          - va * (l1 * M1 * sa * va + L1 * (M2 + Mh) * sa * va
                                  + l2 * M2 * sat * (va + vt)))
                 ) / DetMSymbol

    theta_ddot = (
                         (- (ca * l1 * M1 + cat * l2 * M2 + ca * L1 * (M2 + Mh)) ** 2
                          + (l1 ** 2 * M1 + 2 * ct * L1 * l2 * M2 + l2 ** 2 * M2 +
                             L1 ** 2 * (M1 / 12 + M2 + Mh)) * (M1 + M2 + Mc + Mh + (Jm * Kg ** 2) / rmp ** 2)
                          )
                         *
                         (g * l2 * M2 * sat - L1 * l2 * M2 * st * va ** 2 - B2 * vt)
                         +
                         l2 * M2 * (cat ** 2 * l2 * M2 + ca * cat * (l1 * M1 + L1 * (M2 + Mh))
                                    - (ct * L1 + l2) * (M1 + M2 + Mc + Mh + (Jm * Kg ** 2) / rmp ** 2)
                                    )
                         *
                         (g * (M1 * sa + L1 * (M2 + Mh) * sa + l2 * M2 * sat)
                          - B1 * va + L1 * l2 * M2 * st * vt * (2 * va + vt)
                          )
                         +
                         1 / 12 * l2 * M2 * (-12 * ca * (ct * L1 + l2) * (l1 * M1 + L1 * (M2 + Mh))
                                             + cat * (12 * l1 ** 2 * M1 + L1 * (
                                     12 * ct * l2 * M2 + L1 * (M1 + 12 * (M2 + Mh))))
                                             )
                         *
                         (-Bc * vc + (Kg * Kt * (-Kg * Km * vc + rmp * Vm)) / (Rm * rmp ** 2)
                          - l2 * M2 * sat * vt * (va + vt)
                          - va * (l1 * M1 * sa * va + L1 * (M2 + Mh) * sa * va
                                  + l2 * M2 * sat * (va + vt))
                          )
                 ) / DetMSymbol

    return xc_ddot, alpha_ddot, theta_ddot


def rk4_step(state: np.ndarray, Vm: float, p: PendulumParams, dt: float) -> np.ndarray:
    def f(s):
        ax, aa, at = compute_accelerations(s, Vm, p)
        return np.array([s[1], ax, s[3], aa, s[5], at])

    k1 = f(state)
    k2 = f(state + dt / 2 * k1)
    k3 = f(state + dt / 2 * k2)
    k4 = f(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


class DoublePendulumEnv(gym.Env):
    """
    Gymnasium-style environment using analytic dynamics.
    State: [xc, xc_dot, alpha, alpha_dot, theta, theta_dot]
    Action: motor voltage Vm
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
            self,
            render_mode: Optional[str] = None,
            dt: float = 0.01,
            max_episode_steps: int = 2000,
            action_limit: float = 10.0,
            track_limit: float = 0.3,
            angle_limit: float = np.pi / 6,  # radians
            params: PendulumParams = PARAMS,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.dt = dt
        self.max_episode_steps = max_episode_steps
        self.action_limit = action_limit
        self.track_limit = track_limit
        self.angle_limit = angle_limit
        self.params = params

        self.action_space = spaces.Box(
            low=-action_limit, high=action_limit, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )

        self.state: Optional[np.ndarray] = None
        self.elapsed_steps = 0

        # Rendering state
        self._fig = None
        self._ax = None
        self._cart = None
        self._link1 = None
        self._link2 = None
        self._joint1 = None
        self._joint2 = None

    def reset(self, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        self.elapsed_steps = 0

        # Random initial state near upright
        state = self.np_random.uniform(low=-0.08, high=0.08, size=(6,))
        self.state = state.astype(np.float32)

        if self.render_mode == "human":
            self.render()

        return self.state.copy(), {}

    def step(self, action):
        if self.state is None:
            raise RuntimeError("Call reset() before step().")

        if isinstance(action, np.ndarray) and action.ndim > 0:
            Vm = float(np.clip(action[0], -self.action_limit, self.action_limit))
        else:
            Vm = float(np.clip(action, -self.action_limit, self.action_limit))

        self.state = rk4_step(self.state, Vm, self.params, self.dt).astype(np.float32)
        xc, vc, alpha, va, theta, vt = self.state

        terminated = bool(abs(alpha) > self.angle_limit or abs(theta) > self.angle_limit or abs(xc) > self.track_limit)
        self.elapsed_steps += 1
        truncated = bool(self.elapsed_steps >= self.max_episode_steps)

        reward = 0.0
        if terminated:
            # 失败惩罚：给一个较大的负值，告诉它“绝对不要倒下”
            reward = -100.0
        else:
            # 1. 状态误差惩罚 (State Error Penalty)
            # 使用二次型 (squared) 惩罚，距离目标越远，惩罚呈指数级增长
            # 权重解释：
            # alpha (杆1角度): 最重要，权重 10.0
            # theta (杆2角度): 同样重要，权重 10.0
            # xc (小车位置): 次要，只要不跑出轨道即可，权重 1.0
            dist_penalty = 1.0 * xc ** 2 + 10.0 * alpha ** 2 + 10.0 * theta ** 2

            # 2. 速度阻尼惩罚 (Velocity Damping Penalty)
            # 鼓励系统停下来，而不是在平衡点附近剧烈震荡
            vel_penalty = 0.1 * vc ** 2 + 0.5 * va ** 2 + 0.5 * vt ** 2

            # 3. 动作控制惩罚 (Control Effort Penalty)
            # 防止电机疯狂输出 (Bang-Bang control)，鼓励平滑输出
            action_penalty = 0.01 * (Vm ** 2)

            # 4. 存活奖励 (Survival Bonus)
            # 只要没倒下，每一步都给一个正向奖励，鼓励坚持更久
            alive_bonus = 10.0

            # 总奖励 = 存活奖励 - 各项惩罚
            reward = alive_bonus - (dist_penalty + vel_penalty + action_penalty)

        if self.render_mode == "human":
            self.render()

        return self.state.copy(), float(reward), terminated, truncated, {}

    def render(self):
        if self.state is None:
            return None

        if self.render_mode is None:
            return None

        if self._fig is None:
            self._init_render()

        xc, _, alpha, _, theta, _ = self.state
        L1, L2 = self.params.L1, self.params.L2

        # Cart geometry
        cart_w = 0.3
        cart_h = 0.15
        cart_y = 0.0

        # Link positions
        pivot_x = xc
        pivot_y = cart_y + cart_h / 2

        x1 = pivot_x + L1 * np.sin(alpha)
        y1 = pivot_y + L1 * np.cos(alpha)
        x2 = x1 + L2 * np.sin(alpha + theta)
        y2 = y1 + L2 * np.cos(alpha + theta)

        # Update artists
        self._cart.set_xy((pivot_x - cart_w / 2, cart_y - cart_h / 2))
        self._link1.set_data([pivot_x, x1], [pivot_y, y1])
        self._link2.set_data([x1, x2], [y1, y2])
        self._joint1.center = (x1, y1)
        self._joint2.center = (x2, y2)

        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

        if self.render_mode == "rgb_array":
            self._fig.canvas.draw()
            w, h = self._fig.canvas.get_width_height()
            buf = np.frombuffer(self._fig.canvas.tostring_rgb(), dtype=np.uint8)
            return buf.reshape(h, w, 3)

        return None

    def _init_render(self):
        # Use non-interactive backend if running headless
        if self.render_mode == "rgb_array":
            matplotlib.use("Agg")

        self._fig, self._ax = plt.subplots(figsize=(6, 4))
        self._ax.set_xlim(-1.2, 1.2)
        self._ax.set_ylim(-0.8, 1.2)  # Expanded lower limit to see pendulum when down
        self._ax.set_aspect("equal")
        self._ax.grid(True, alpha=0.2)

        cart_w = 0.3
        cart_h = 0.15

        self._cart = Rectangle((-cart_w / 2, -cart_h / 2), cart_w, cart_h, color="#4C72B0")
        self._ax.add_patch(self._cart)

        self._link1, = self._ax.plot([], [], lw=3, color="#55A868")
        self._link2, = self._ax.plot([], [], lw=3, color="#C44E52")
        self._joint1 = Circle((0, 0), 0.02, color="#55A868")
        self._joint2 = Circle((0, 0), 0.02, color="#C44E52")
        self._ax.add_patch(self._joint1)
        self._ax.add_patch(self._joint2)

        self._ax.set_title("Double Pendulum on Cart")
        self._fig.tight_layout()

    def close(self):
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._ax = None
            self._cart = None
            self._link1 = None
            self._link2 = None
            self._joint1 = None
            self._joint2 = None