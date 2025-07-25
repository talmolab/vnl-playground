"""Maze Forage Task for Rodent."""

from mujoco_playground._src import reward

from typing import Any, Dict, Optional, Tuple, Union
import jax
import jax.numpy as jp
import jax.lax as lax
import numpy as np
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env
from vnl_mjx.tasks.rodent import base as rodent_base
from vnl_mjx.tasks.rodent import consts

# -----------------------------------------------------------------------------#
# Maze‑height‑field utilities
# -----------------------------------------------------------------------------#
def _generate_perfect_maze(grid_size: int, rng: jax.Array) -> np.ndarray:
    """Return a (2*grid_size+1, 2*grid_size+1) binary array with 0 = floor, 1 = wall."""
    h = 2 * grid_size + 1
    maze = np.ones((h, h), dtype=np.float32)
    # Cell centers at (2*x+1, 2*y+1)
    stack = [(0, 0)]
    maze[1, 1] = 0
    visited = {(0, 0)}
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    current_rng = rng  # Use a separate variable for the mutable RNG state
    while stack:
        cx, cy = stack[-1]
        # Use JAX random to shuffle directions
        current_rng, subkey = jax.random.split(current_rng)
        dirs = jax.random.permutation(subkey, jp.array(dirs)).tolist()
        for dx, dy in dirs:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < grid_size and 0 <= ny < grid_size and (nx, ny) not in visited:
                # knock down wall between (cx,cy) and (nx,ny)
                maze[2 * cx + 1 + dx, 2 * cy + 1 + dy] = 0
                maze[2 * nx + 1, 2 * ny + 1] = 0
                visited.add((nx, ny))
                stack.append((nx, ny))
                break
        else:
            stack.pop()
    return maze


def default_config() -> config_dict.ConfigDict:
    """Returns the default configuration for the MazeForage environment."""
    return config_dict.create(
        walker_xml_path=consts.RODENT_BOX_FEET_PATH,
        arena_xml_path=consts.ARENA_XML_PATH,
        ctrl_dt=0.01,
        sim_dt=0.002,
        solver="cg",
        iterations=10,
        ls_iterations=5,
        noslip_iterations=0,
        vision=False,
        vision_config=None,
        torque_actuators=True,
        rescale_factor=1.0,
        episode_length=2000,
        action_repeat=1,
        reward_type="dense",  # or "sparse"
        target_pos=(2.0, 2.0),  # example target
        target_radius=0.2,
        reward_scale=1.0,
        target_speed=1.0,     # for speed-based reward
        vision_num_rays=16,      # number of rays for rodent vision
        vision_fov=np.pi,        # field of view in radians
        vision_max_dist=6.0,     # maximum ray distance in metres
        maze_hsize=3,          # horizontal half-size of the maze world
        maze_vsize=0.4,       # wall height
        maze_grid_size=6,      # number of *cells* along one edge (perfect maze will be 2*N+1)
        maze_px_per_cell=8,    # resolution (pixels) per coarse maze cell
        random_goal=True,      # resample goal every episode
    )


class MazeForage(rodent_base.RodentEnv):
    """Maze Forage environment for rodent navigation."""

    def _initialize_maze_hfield(self, rng: jax.Array) -> None:
        """Create a binary maze height‑field and insert into self._spec."""
        self._rng, maze_rng = jax.random.split(rng)
        grid_size = int(self._config.maze_grid_size)
        maze_binary = _generate_perfect_maze(grid_size, maze_rng)
        self._coarse_maze = maze_binary  # keep coarse version
        px_per_cell = int(self._config.maze_px_per_cell)
        if px_per_cell > 1:
            maze_binary = np.kron(maze_binary, np.ones((px_per_cell, px_per_cell), dtype=np.float32))
        size = maze_binary.shape[0]
        # height values: 1.0 for walls, 0.0 for floor
        hfield_data = maze_binary.astype(np.float32)
        hsize = float(self._config.maze_hsize)
        vsize = float(self._config.maze_vsize)
        self._spec.add_hfield(
            name="maze",
            size=[hsize, hsize, vsize, vsize],
            nrow=size,
            ncol=size,
            userdata=hfield_data.flatten(),
        )
        # Apply neutral gray texture so walls visible
        # tex = self._spec.add_texture(
        #     name="maze_tex", type=mujoco.mjtTexture.mjTEXTURE_2D, width=4, height=4
        # )
        # mat = self._spec.add_material(name="maze_mat")
        # mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "maze_tex"
        self._spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname="maze")
        # Save numpy version for fast queries
        self._maze_array = maze_binary

    def _interpolate_maze_height(self, x: float, y: float) -> float:
        """Interpolate the maze height-field at world coordinates (x, y)."""
        hsize = float(self._config.maze_hsize)
        vsize = float(self._config.maze_vsize)
        height_array = self._maze_array
        size = height_array.shape[0]
        # normalized [0,1]
        u = (x + hsize) / (2 * hsize)
        v = (y + hsize) / (2 * hsize)
        # convert to grid indices
        col = int(np.clip(u * (size - 1), 0, size - 1))
        row = int(np.clip(v * (size - 1), 0, size - 1))
        height_norm = height_array[row, col]
        return float(height_norm * vsize)

    def _raycast_lengths(self, agent_xy: jp.ndarray, forward_xy: jp.ndarray) -> jp.ndarray:
        """Return array of ray lengths to first wall for each vision ray."""
        maze = jp.array(self._maze_array)
        size = maze.shape[0]
        hsize = self._config.maze_hsize
        cell = 2 * hsize / (size - 1)
        max_d = self._config.vision_max_dist
        n_rays = self._config.vision_num_rays
        fov = self._config.vision_fov
        base_ang = jp.arctan2(forward_xy[1], forward_xy[0])
        rels = jp.linspace(-fov/2, fov/2, n_rays)
        angles = base_ang + rels

        def ray_fn(angle):
            def cond_fn(state):
                dist, hit = state
                return jp.logical_and(dist < max_d, jp.logical_not(hit))
            def body_fn(state):
                dist, hit = state
                dist = dist + cell * 0.25
                x = agent_xy[0] + jp.cos(angle) * dist
                y = agent_xy[1] + jp.sin(angle) * dist
                gx = jp.clip(jp.round((x + hsize) / cell).astype(jp.int32), 0, size - 1)
                gy = jp.clip(jp.round((y + hsize) / cell).astype(jp.int32), 0, size - 1)
                hit_wall = maze[gy, gx] > 0.5
                return dist, hit_wall
            dist, hit = lax.while_loop(cond_fn, body_fn, (0.0, False))
            return jp.minimum(dist, max_d)
        print(angles)
        return jax.vmap(ray_fn)(angles)

    def __init__(
        self,
        rng: jax.Array = jax.random.PRNGKey(1),
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
        super().__init__(config, config_overrides)
        self._rng = rng
        # Build maze heightfield and store maze array
        self._initialize_maze_hfield(self._rng)
        # Place rodent at first free cell (center of (1,1))
        cell_width = 2 * self._config.maze_hsize / (self._coarse_maze.shape[0] - 1)
        init_x = -self._config.maze_hsize + cell_width * 1.35
        init_y = -self._config.maze_hsize + cell_width * 1.7
        init_z = self._interpolate_maze_height(init_x, init_y) + 0.01
        self.add_rodent(
            self._config.torque_actuators,
            self._config.rescale_factor,
            [init_x, init_y, init_z],
            quat=(0.70710678, 0.0, 0.0, 0.70710678)
        )
        # Visualize target as a sphere
        tx, ty = self._config.target_pos
        tz = self._interpolate_maze_height(tx, ty) + 0.02
        self._spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            pos=[tx, ty, tz],
            size=[self._config.target_radius, self._config.target_radius, self._config.target_radius],
            rgba=[1.0, 0.0, 0.0, 0.8],
            contype=0,
            conaffinity=0,
        )
        self._spec.worldbody.add_light(pos=[0, 0, 10], dir=[0, 0, -1])
        self.compile()

    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Reset the environment state."""
        info = {
            "last_act": jp.zeros(self.mjx_model.nu),
            "last_last_act": jp.zeros(self.mjx_model.nu),
        }
        data = mjx_env.init(self.mjx_model)
        # Sample a random goal in a free cell if requested
        if self._config.random_goal:
            free_cells = np.argwhere(self._maze_array == 0)
            goal_cell = free_cells[np.random.randint(len(free_cells))]
            gx_idx, gy_idx = goal_cell
            cell_w = 2 * self._config.maze_hsize / (self._coarse_maze.shape[0] - 1)
            gx = -self._config.maze_hsize + gx_idx * cell_w
            gy = -self._config.maze_hsize + gy_idx * cell_w
            self._config.target_pos = (float(gx), float(gy))
        obs = self._get_obs(data, info)
        reward = jp.zeros(())
        done = jp.zeros(())
        metrics = {}
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """Step the environment forward by one timestep."""
        data = mjx_env.step(self.mjx_model, state.data, action)
        obs = self._get_obs(data, state.info)
        rewards = self._get_reward(data)
        reward = rewards["distance * upright"] + rewards["speed_reward"]
        done = self._get_termination(data)
        state.info["last_last_act"] = state.info["last_act"]
        state.info["last_act"] = action
        reward = jp.nan_to_num(reward)
        obs = jp.nan_to_num(obs)
        flattened_vals, _ = jax.flatten_util.ravel_pytree(data)
        num_nans = jp.sum(jp.isnan(flattened_vals))
        nan = jp.where(num_nans > 0, 1.0, 0.0)
        done = jp.max(jp.array([nan, done]))
        return state.replace(
            data=data,
            obs=obs,
            reward=reward,
            done=done,
        )

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jp.ndarray:
        """Get the current observation."""
        proprioception = self._get_proprioception(data)
        kinematic_sensors = self._get_kinematic_sensors(data)
        touch_sensors = self._get_touch_sensors(data)
        appendages_pos = self._get_appendages_pos(data)
        origin = self._get_origin(data)
        # Optionally, could include goal position relative to agent
        target_pos = jp.array(self._config.target_pos)
        agent_pos = data.bind(self.mjx_model, self._spec.body(f"torso{self._suffix}")).xpos[:2]
        rel_goal = target_pos - agent_pos
        # Vision: ray lengths to nearest wall
        torso_mat = data.bind(self.mjx_model, self._spec.body(f"torso{self._suffix}")).xmat
        agent_pos = jp.array(agent_pos)
        forward_vec = jp.array(torso_mat[0, :2])
        ray_lengths = self._raycast_lengths(agent_pos, forward_vec)
        obs = jp.concatenate(
            [
                info["last_act"],
                proprioception,
                kinematic_sensors,
                touch_sensors,
                origin,
                rel_goal,
                ray_lengths,
            ]
        )
        return obs

    def _get_reward(self, data: mjx.Data) -> Dict[str, jax.Array]:
        """Compute rewards: distance-to-goal, upright posture, and speed."""
        # Distance-based reward
        target_pos = jp.array(self._config.target_pos)
        agent_pos = data.bind(self.mjx_model, self._spec.body(f"torso{self._suffix}")).xpos[:2]
        dist = jp.linalg.norm(target_pos - agent_pos)
        if self._config.reward_type == "dense":
            distance_reward = self._config.reward_scale * (1.0 - jp.tanh(dist))
        else:
            distance_reward = self._config.reward_scale * jp.where(dist < self._config.target_radius, 1.0, 0.0)

        # Upright reward
        upright_reward = self._upright_reward(data)

        # Speed reward
        speed_reward = self._get_speed_reward(data)

        return {
            "distance_reward": distance_reward,
            "upright_reward": upright_reward,
            "speed_reward": speed_reward,
            "distance * upright": distance_reward * upright_reward,
        }


    def _get_termination(self, data: mjx.Data) -> jp.ndarray:
        """Episode is done if agent reaches the target."""
        target_pos = jp.array(self._config.target_pos)
        agent_pos = data.bind(self.mjx_model, self._spec.body(f"torso{self._suffix}")).xpos[:2]
        dist = jp.linalg.norm(target_pos - agent_pos)
        done = jp.where(dist < self._config.target_radius, 1.0, 0.0)
        return done
    
    def _upright_reward(self, data: mjx.Data) -> jax.Array:
        """Reward torso for maintaining upright orientation."""
        # dot product between torso z-axis and global z-axis
        mat = data.bind(self.mjx_model, self._spec.body(f"torso{self._suffix}")).xmat
        up_dot = mat[2, 2]  # 1 when perfectly upright, -1 when inverted
        # normalize from [-1,1] to [0,1]
        return (up_dot + 1.0) / 2.0

    def _get_speed_reward(self, data: mjx.Data) -> jax.Array:
        """Reward proportional to forward speed up to a target using tolerance."""
        linvel = data.bind(self.mjx_model, self._spec.body(f"torso{self._suffix}")).subtree_linvel
        speed = jp.linalg.norm(linvel)
        ts = float(self._config.target_speed)
        return reward.tolerance(
            speed,
            bounds=(ts, ts),
            margin=ts,
            sigmoid="linear",
            value_at_margin=0.0,
        )