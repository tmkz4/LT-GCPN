"""
environment_grid.py -- locally coupled Kuramoto lattice with lesions, repaired by a "Traveller".

Dynamics (Yu 2023, arXiv:2304.04636, eq. 1), n x n grid, nearest-neighbour coupling, no periodic boundaries:
    d theta_i / dt = omega_i + K * sum_{j in neighbours(i)} sin(theta_j - theta_i)
Regime: directional wave. Node 0 (corner) is a faster "driver" (omega_low + omega_boost), phases start at 0.
The intact lattice phase-locks when K >= (N-1)/(2N) * omega_boost  (the paper's threshold, called k_star here);
lesions raise that threshold, so with K slightly above k_star a lesioned lattice often loses locking.

Episode: lesion the lattice (random edges and/or a spatially contiguous cut), keep only episodes where the
lesioned system is NOT locked (rejection sampling, so "do nothing" cannot win), then let the Traveller edit
edges within `radius` of its position until the system re-locks (or the structure is restored, by mode).

Actions (fixed size A = 2*M + 1, M = number of lattice offsets within `radius`):
    [0, M)      toggle the edge (u, u + offset_k)       (add if absent, remove if present)
    [M, 2M)     move the Traveller to u + offset_k
    2M          noop (let the dynamics run)
`mask` marks offsets that fall outside the grid. Offsets are sorted, so slot k always means the same direction
(translation-equivariant; no dependence on node ids).

Time: moves are free (dynamics frozen); toggles and noops integrate the dynamics for `t_eval` time units.

Reward modes
    "dynamic"   locking only: counterfactual change in -log10(velocity spread), success = locked
    "structure" potential-based -|E xor E_target| (privileged information), success = edge sets equal
    "mixed"     both (dyn_weight, struct_weight), success = locked
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import torch
    from torch_geometric.data import Data
    HAS_TORCH = True
except ImportError:  # lets the dynamics/logic be tested without torch
    HAS_TORCH = False
    from types import SimpleNamespace


@dataclass
class EnvConfig:
    n: int = 8                       # grid is n x n (N = n*n nodes)
    K: float | None = None           # coupling per edge; None -> 1.2 * k_star
    omega_low: float = 1.0
    omega_boost: float = 1.0         # driver (node 0) has omega_low + omega_boost
    dt: float = 0.05
    t_settle: float = 150.0          # integration time used at reset / to check the intact lattice
    t_eval: float = 40.0             # integration time after each toggle / noop
    lock_tol: float = 0.02           # locked  <=>  max(v) - min(v) < lock_tol
    radius: float = 1.0              # edit radius in lattice units: 1 (4 nbrs), sqrt(2) (8), 2 (12), ...
    lesion_frac: float = 0.1        # fraction of lattice edges removed
    lesion_mode: str = "random"      # "random" | "cluster" | "mixed"
    reward_mode: str = "mixed"       # "dynamic" | "structure" | "mixed"
    dyn_weight: float = 1.0
    struct_weight: float = 0.5
    terminal_bonus: float = 5.0
    step_penalty: float = 0.01
    move_penalty: float = 0.02
    noop_penalty: float = 0.02
    invalid_penalty: float = 0.10
    shaping_gamma: float = 1.0       # 1.0 -> an add/remove cycle sums to exactly 0 (no farming); <1 leaves a small positive residual
    max_steps: int = 60
    max_reject_tries: int = 50
    seed: int | None = None


class GridKuramotoRepairEnv:
    def __init__(self, cfg: EnvConfig | None = None):
        self.cfg = cfg or EnvConfig()
        c = self.cfg
        if c.reward_mode not in ("dynamic", "structure", "mixed"):
            raise ValueError("reward_mode must be 'dynamic', 'structure' or 'mixed'")
        if c.lesion_mode not in ("random", "cluster", "mixed"):
            raise ValueError("lesion_mode must be 'random', 'cluster' or 'mixed'")

        self.n = c.n
        self.N = c.n * c.n
        ids = np.arange(self.N)
        self.pos = np.stack([ids % c.n, ids // c.n], axis=1).astype(np.float64)   # (x, y); node id = y*n + x
        self.omega = np.full(self.N, c.omega_low, dtype=np.float64)
        self.omega[0] = c.omega_low + c.omega_boost
        self.k_star = (self.N - 1) / (2 * self.N) * c.omega_boost
        self.K = float(c.K) if c.K is not None else 1.2 * self.k_star
        self.target_edges = frozenset(self._lattice_edges())

        R = int(np.floor(c.radius + 1e-9))
        offs = [(dx, dy) for dy in range(-R, R + 1) for dx in range(-R, R + 1)
                if (dx, dy) != (0, 0) and dx * dx + dy * dy <= c.radius ** 2 + 1e-9]
        self.offsets = sorted(offs, key=lambda o: (o[0] ** 2 + o[1] ** 2, o[1], o[0]))
        self.M = len(self.offsets)
        self.num_actions = 2 * self.M + 1

        self.rng = np.random.default_rng(c.seed)

        _, v = self._integrate(np.zeros(self.N), self.target_edges, c.t_settle)
        if np.ptp(v) >= c.lock_tol:
            raise ValueError(
                f"The intact lattice does not lock (K={self.K:.3f}, k_star={self.k_star:.3f}). "
                f"Raise K (about 1.1-1.3 x k_star works) or t_settle."
            )

        # episode state (filled by reset)
        self.edges = self.target_edges
        self.theta = np.zeros(self.N)
        self.v = np.zeros(self.N)
        self.spread = 0.0
        self.agent = 0
        self.steps = 0
        self.n_edits = 0
        self.n_lesioned = 0
        self.reject_tries = 0
        self.trivial = False

    # ------------------------------------------------------------------ graph helpers
    def _lattice_edges(self):
        n, out = self.n, []
        for i in range(n):
            for j in range(n):
                a = i * n + j
                if j + 1 < n:
                    out.append((a, a + 1))
                if i + 1 < n:
                    out.append((a, a + n))
        return out

    def _neighbor(self, u: int, k: int) -> int:
        dx, dy = self.offsets[k]
        x, y = int(self.pos[u, 0]) + dx, int(self.pos[u, 1]) + dy
        return y * self.n + x if (0 <= x < self.n and 0 <= y < self.n) else -1

    def candidates(self, u: int | None = None) -> np.ndarray:
        """Node id reached by each offset slot from u (default: the Traveller), -1 if outside the grid."""
        u = self.agent if u is None else u
        return np.array([self._neighbor(u, k) for k in range(self.M)], dtype=np.int64)

    def action_mask(self) -> np.ndarray:
        cand = self.candidates() >= 0
        # Conversion explicite en liste Python puis en array booléen NumPy
        return np.array(cand.tolist() + cand.tolist() + [True], dtype=bool)

    def _decode(self, action: int):
        if action == 2 * self.M:
            return "noop", -1
        return ("toggle", action) if action < self.M else ("move", action - self.M)

    @staticmethod
    def _directed(edges):
        if not edges:
            z = np.zeros(0, dtype=np.int64)
            return z, z
        e = np.array(sorted(edges), dtype=np.int64)
        return np.concatenate([e[:, 0], e[:, 1]]), np.concatenate([e[:, 1], e[:, 0]])

    @property
    def edit_distance(self) -> int:
        return len(self.edges ^ self.target_edges)

    # ------------------------------------------------------------------ dynamics
    def _integrate(self, theta, edges, t):
        """Euler integration of the attractive Kuramoto model; returns (theta, instantaneous velocities)."""
        c = self.cfg
        src, dst = self._directed(edges)
        th = np.array(theta, dtype=np.float64, copy=True)
        for _ in range(int(round(t / c.dt))):
            coup = np.bincount(dst, weights=np.sin(th[src] - th[dst]), minlength=self.N)   # sum_j sin(th_j - th_i)
            th += c.dt * (self.omega + self.K * coup)
        coup = np.bincount(dst, weights=np.sin(th[src] - th[dst]), minlength=self.N)
        return th, self.omega + self.K * coup

    @staticmethod
    def _phi_dyn(spread: float) -> float:
        return float(-np.log10(np.clip(spread, 1e-3, 1.0)))

    def _phi_struct(self, edges) -> float:
        return -float(len(edges ^ self.target_edges))

    # ------------------------------------------------------------------ lesions
    def _lesion(self, mode: str, frac: float):
        elist = sorted(self.target_edges)
        k = int(round(len(elist) * frac))
        if k == 0:
            return self.target_edges
        if mode == "cluster":
            center = self.pos[self.rng.integers(self.N)]
            mids = np.array([(self.pos[a] + self.pos[b]) / 2 for a, b in elist])
            dist = np.linalg.norm(mids - center, axis=1) + 1e-6 * self.rng.random(len(elist))
            removed = [elist[i] for i in np.argsort(dist)[:k]]
        else:
            removed = [elist[i] for i in self.rng.choice(len(elist), k, replace=False)]
        return self.target_edges - frozenset(removed)

    # ------------------------------------------------------------------ gym-like API
    def reset(self, lesion_frac: float | None = None, lesion_mode: str | None = None):
        c = self.cfg
        frac = c.lesion_frac if lesion_frac is None else lesion_frac
        mode = lesion_mode or c.lesion_mode
        tries = 0
        while True:
            tries += 1
            m = mode if mode != "mixed" else ("cluster" if self.rng.random() < 0.5 else "random")
            edges = self._lesion(m, frac)
            theta, v = self._integrate(np.zeros(self.N), edges, c.t_settle)
            if np.ptp(v) >= c.lock_tol or tries >= c.max_reject_tries:
                break
        self.reject_tries = tries
        self.trivial = bool(np.ptp(v) < c.lock_tol)      # True only if rejection sampling gave up
        self.edges, self.theta, self.v = edges, theta, v
        self.spread = float(np.ptp(v))
        self.agent = int(self.rng.integers(self.N))
        self.steps = 0
        self.n_edits = 0
        self.n_lesioned = len(self.target_edges - edges)
        return self._observation()

    def step(self, action: int):
        c = self.cfg
        action = int(action)
        if not 0 <= action < self.num_actions:
            raise ValueError(f"action {action} outside [0, {self.num_actions})")
        self.steps += 1
        u = self.agent
        kind, k = self._decode(action)
        target = self._neighbor(u, k) if kind != "noop" else -1
        if kind != "noop" and target < 0:
            kind = "invalid"                                # masked action: treated as a penalised noop

        w_dyn, w_struct = {"dynamic": (c.dyn_weight, 0.0),
                           "structure": (0.0, 1.0),
                           "mixed": (c.dyn_weight, c.struct_weight)}[c.reward_mode]
        phi_s_before = self._phi_struct(self.edges)
        reward = -c.step_penalty
        dyn = 0.0

        if kind == "toggle":
            e = (min(u, target), max(u, target))
            new_edges = self.edges - {e} if e in self.edges else self.edges | {e}
            # counterfactual: same state, old graph -> the delta isolates the effect of the edit, not elapsed time
            _, v_cf = self._integrate(self.theta, self.edges, c.t_eval)
            theta_new, v_new = self._integrate(self.theta, new_edges, c.t_eval)
            dyn = self._phi_dyn(np.ptp(v_new)) - self._phi_dyn(np.ptp(v_cf))
            self.edges, self.theta, self.v = new_edges, theta_new, v_new
            self.n_edits += 1
        elif kind == "move":
            self.agent = target
            reward -= c.move_penalty
        else:                                               # "noop" or "invalid": time passes, no edit
            reward -= c.noop_penalty if kind == "noop" else c.invalid_penalty
            self.theta, self.v = self._integrate(self.theta, self.edges, c.t_eval)
        self.spread = float(np.ptp(self.v))

        locked = self.spread < c.lock_tol
        structural_success = self.edges == self.target_edges
        success = structural_success if c.reward_mode == "structure" else locked

        reward += w_dyn * dyn + w_struct * (c.shaping_gamma * self._phi_struct(self.edges) - phi_s_before)
        if success:
            reward += c.terminal_bonus
        terminated = bool(success)
        truncated = bool((not success) and self.steps >= c.max_steps)

        info = {
            "is_healed": bool(success),          # success under the current reward_mode (compat with train.py)
            "terminated": terminated,
            "truncated": truncated,              # time limit: bootstrap V(s_T) instead of treating it as terminal
            "locked": bool(locked),
            "structural_success": bool(structural_success),
            "spread": self.spread,
            "edit_distance": self.edit_distance,
            "n_edits": self.n_edits,
            "action_type": kind,
            "invalid": kind == "invalid",
        }
        return self._observation(), float(reward), terminated or truncated, info

    # ------------------------------------------------------------------ observation
    def _observation(self):
        c = self.cfg
        u = self.agent
        dtheta = self.theta - self.theta[u]                       # phases relative to the Traveller (rotation-invariant)
        src, dst = self._directed(self.edges)
        deg = np.bincount(dst, minlength=self.N).astype(np.float64)
        rel = (self.pos - self.pos[u]) / self.n
        flag = np.zeros(self.N)
        flag[u] = 1.0
        x = np.stack([np.sin(dtheta), np.cos(dtheta),
                      self.omega - c.omega_low,                    # intrinsic frequency offset (driver stands out)
                      self.v - self.v.mean(),                      # where locking fails (velocity deviation)
                      flag, rel[:, 0], rel[:, 1], deg / 4.0], axis=1)
        disp = self.pos[dst] - self.pos[src]                       # message src -> dst
        edge_attr = np.concatenate([disp, np.linalg.norm(disp, axis=1, keepdims=True)], axis=1)
        edge_index = np.stack([src, dst], axis=0)
        mask = self.action_mask()
        cand = self.candidates()
        extras = dict(agent_pos=u, steps_left=1.0 - self.steps / c.max_steps, spread=self.spread)

        if HAS_TORCH:
            data = Data(x=torch.as_tensor(x, dtype=torch.float),
                        edge_index=torch.as_tensor(edge_index, dtype=torch.long),
                        edge_attr=torch.as_tensor(edge_attr, dtype=torch.float))
            data.mask = torch.as_tensor(mask.tolist(), dtype=torch.bool)
            data.cand = torch.as_tensor(cand, dtype=torch.long)
            data.offsets = torch.as_tensor(self.offsets, dtype=torch.float)
            for key, val in extras.items():
                setattr(data, key, val)
            return data
        return SimpleNamespace(x=x, edge_index=edge_index, edge_attr=edge_attr, mask=mask, cand=cand,
                               offsets=np.array(self.offsets, dtype=np.float64), **extras)