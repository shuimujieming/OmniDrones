import torch
import einops
import numpy as np
from dataclasses import MISSING, fields
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec, DiscreteTensorSpec
from omni_drones.envs.isaac_env import IsaacEnv, AgentSpec
import isaaclab.sim as sim_utils
from omni_drones.robots.drone import MultirotorBase
from isaaclab.assets import AssetBaseCfg
from isaaclab.terrains import TerrainImporterCfg, TerrainImporter, TerrainGeneratorCfg, HfDiscreteObstaclesTerrainCfg
from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils import configclass
from omni_drones.utils.torch import euler_to_quaternion, quat_axis,quaternion_to_euler
from isaaclab.sensors import RayCaster, RayCasterCfg, patterns
from isaacsim.core.utils.viewports import set_camera_view
from utils import vec_to_new_frame, vec_to_world, construct_input
import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
import time
import torch.distributions as D


@height_field_to_mesh
def hf_range_obstacles_terrain(difficulty: float, cfg: "HfRangeObstaclesTerrainCfg") -> np.ndarray:
    """Generate a terrain with randomly generated obstacles as pillars with positive and negative heights.

    The terrain is a flat platform at the center of the terrain with randomly generated obstacles as pillars
    with positive and negative height. The obstacles are randomly generated cuboids with a random width and
    height. They are placed randomly on the terrain with a minimum distance of :obj:`cfg.platform_width`
    from the center of the terrain.

    .. image:: ../../_static/terrains/height_field/discrete_obstacles_terrain.jpg
       :width: 40%
       :align: center

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D numpy array with discretized heights.
        The shape of the array is (width, length), where width and length are the number of points
        along the x and y axis, respectively.
    """
    # resolve terrain configuration
    obs_height = cfg.obstacle_height_range[0] + difficulty * (
        cfg.obstacle_height_range[1] - cfg.obstacle_height_range[0]
    )

    # switch parameters to discrete units
    # -- terrain
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # -- obstacles
    obs_height = int(obs_height / cfg.vertical_scale)
    obs_width_min = int(cfg.obstacle_width_range[0] / cfg.horizontal_scale)
    obs_width_max = int(cfg.obstacle_width_range[1] / cfg.horizontal_scale)
    # -- center of the terrain
    platform_width = int(cfg.platform_width / cfg.horizontal_scale)

    # create discrete ranges for the obstacles
    # -- shape
    obs_width_range = np.arange(obs_width_min, obs_width_max, 4)
    obs_length_range = np.arange(obs_width_min, obs_width_max, 4)
    # -- position
    obs_x_range = np.arange(0, width_pixels, 4)
    obs_y_range = np.arange(0, length_pixels, 4)
    obstacles_history = []

    # create a terrain with a flat platform at the center
    hf_raw = np.zeros((width_pixels, length_pixels))
    # generate the obstacles
    # print("Attention!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # print("height range: ", cfg.obstacle_height_range)
    probability_length = len(cfg.obstacle_height_probability)

    def good_distance(x, y, width, length, obstacles_hist, bad_range = [2, 10]):
        # lower_bound_pixels = bad_range[0] / cfg.horizontal_scale
        # upper_bound_pixels = bad_range[1] / cfg.horizontal_scale


        lower_bound_pixels = bad_range[0]
        upper_bound_pixels = bad_range[1]    
        # previous x, y, w, l. Calculate for closet points
        for (xp, yp, wp, lp) in obstacles_hist:
            dx = abs(xp - x) - width
            dy = abs(yp - y) - length
            if dx<0 and dy<0:
                continue
            distance = np.sqrt(dx**2 + dy**2)
            # print("distance: ", distance)
            # print("lower_bound_pixels: ", lower_bound_pixels)
            # print("upper_bound_pixels: ", upper_bound_pixels)
            # print("x: ", x)
            # print("y: ", y)
            if distance >= lower_bound_pixels and distance <= upper_bound_pixels:
                return False
        return True


    # print("Calculation Start!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    num = 0
    for _ in range(cfg.num_obstacles):
        # print("Number of cylinders generated: ", num)
        # sample size        
        if cfg.obstacle_height_mode == "choice":
            height = np.random.choice([-obs_height, -obs_height // 2, obs_height // 2, obs_height])
        elif cfg.obstacle_height_mode == "fixed":
            height = obs_height
        elif cfg.obstacle_height_mode == "range":
            random_roll = np.random.choice(probability_length, 1, p=cfg.obstacle_height_probability)
            for n in range(probability_length):
                if random_roll == n:
                    height = np.random.uniform(cfg.obstacle_height_range[n]/cfg.vertical_scale, cfg.obstacle_height_range[n+1]/cfg.vertical_scale)
                    break


            # height = np.random.uniform(cfg.obstacle_height_range[0]/cfg.vertical_scale, cfg.obstacle_height_range[1]/cfg.vertical_scale)
        else:
            raise ValueError(f"Unknown obstacle height mode '{cfg.obstacle_height_mode}'. Must be 'choice' or 'fixed' or 'range'.")
        
        attempts = 0
        # print("Start choosing location!!")
        while attempts < 100000:
            width = int(np.random.choice(obs_width_range))
            length = int(np.random.choice(obs_length_range))
            # sample position
            x_start = int(np.random.choice(obs_x_range))
            y_start = int(np.random.choice(obs_y_range))
            if x_start + width > width_pixels:
                x_start = width_pixels - width
            if y_start + length > length_pixels:
                y_start = length_pixels - length
            
            if good_distance(x_start, y_start, width, length, obstacles_history):
                break
            elif obstacles_history == []:
                break
            # print("attempts when generated: ", attempts)
            # print("obstacles_history: ", obstacles_history)
            attempts += 1
        # print("attempts when generated: ", attempts)
        obstacles_history.append((x_start, y_start, width, length))
        num += 1
        # clip start position to the terrain
        # print("x_start: ", x_start)
        # print("y_start: ", y_start)
        # add to terrain
        hf_raw[x_start : x_start + width, y_start : y_start + length] = height
    # print("Calculation End!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")    
    # clip the terrain to the platform
    x1 = (width_pixels - platform_width) // 2
    x2 = (width_pixels + platform_width) // 2
    y1 = (length_pixels - platform_width) // 2
    y2 = (length_pixels + platform_width) // 2
    hf_raw[x1:x2, y1:y2] = 0
    # round off the heights to the nearest vertical step
    return np.rint(hf_raw).astype(np.int16)


@configclass
class HfRangeObstaclesTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a discrete obstacles height field terrain."""

    function = hf_range_obstacles_terrain

    obstacle_height_mode: str = "choice"
    """The mode to use for the obstacle height. Defaults to "choice".

    The following modes are supported: "choice", "fixed".
    """
    obstacle_width_range: tuple[float, float] = MISSING
    """The minimum and maximum width of the obstacles (in m)."""
    # obstacle_height_range: tuple[float, float] = MISSING
    obstacle_height_range: list = MISSING
    """The minimum and maximum height of the obstacles (in m)."""
    obstacle_height_probability: list = MISSING
    num_obstacles: int = MISSING
    """The number of obstacles to generate."""
    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""

class NavigationEnv(IsaacEnv):

    # In one step:
    # 1. _pre_sim_step (apply action) -> step isaac sim
    # 2. _post_sim_step (update lidar)
    # 3. increment progress_buf
    # 4. _compute_state_and_obs (get observation and states, update stats)
    # 5. _compute_reward_and_done (update reward and calculate returns)

    def __init__(self, cfg):
        print("[Navigation Environment]: Initializing Env...")
        # LiDAR params:
        self.lidar_range = cfg.sensor.lidar_range
        self.lidar_vfov = (max(-89., cfg.sensor.lidar_vfov[0]), min(89., cfg.sensor.lidar_vfov[1]))
        self.lidar_vbeams = cfg.sensor.lidar_vbeams
        self.lidar_hres = cfg.sensor.lidar_hres
        self.lidar_hbeams = int(360/self.lidar_hres)
        super().__init__(cfg, cfg.headless)
        
        # Drone Initialization
        self.drone.initialize()
        self.init_vels = torch.zeros_like(self.drone.get_velocities())


        # LiDAR Intialization
        ray_caster_cfg = RayCasterCfg(
            prim_path="/World/envs/env_.*/Hummingbird_0/base_link",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.BpearlPatternCfg(
                horizontal_res=self.lidar_hres, # horizontal default is set to 10
                vertical_ray_angles=torch.linspace(*self.lidar_vfov, self.lidar_vbeams) 
            ),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
            # mesh_prim_paths=["/World"],
        )
        self.lidar = RayCaster(ray_caster_cfg)
        self.lidar._initialize_impl()
        self.lidar_resolution = (self.lidar_hbeams, self.lidar_vbeams) 
        
        self.init_rpy_dist = D.Uniform(
            torch.tensor([0.0, 0.0, 0.], device=self.device) * torch.pi,
            torch.tensor([0.0, 0.0, 2.], device=self.device) * torch.pi
        )

        # start and target 
        with torch.device(self.device):
            # self.start_pos = torch.zeros(self.num_envs, 1, 3)
            self.target_pos = torch.zeros(self.num_envs, 1, 3)
            
            # Coordinate change: add target direction variable
            self.target_dir = torch.zeros(self.num_envs, 1, 3)
            self.height_range = torch.zeros(self.num_envs, 1, 2)
            self.prev_drone_vel_w = torch.zeros(self.num_envs, 1 , 3)
            self.last_distance = torch.zeros(self.num_envs, 1, 1)
            self.last_yaw = torch.zeros(self.num_envs, 1)
            # self.target_pos[:, 0, 0] = torch.linspace(-0.5, 0.5, self.num_envs) * 32.
            # self.target_pos[:, 0, 1] = 24.
            # self.target_pos[:, 0, 2] = 2.     


    def _design_scene(self):
        # Initialize a drone in prim /World/envs/envs_0
        drone_model = MultirotorBase.REGISTRY[self.cfg.drone.model_name] # drone model class
        cfg_kwargs = {}
        if "force_sensor" in {field.name for field in fields(drone_model.cfg_cls)}:
            cfg_kwargs["force_sensor"] = False
        cfg = drone_model.cfg_cls(**cfg_kwargs)
        self.drone = drone_model(cfg=cfg)
        # drone_prim = self.drone.spawn(translations=[(0.0, 0.0, 1.0)])[0]
        drone_prim = self.drone.spawn(translations=[(0.0, 0.0, 2.0)])[0]

        # lighting
        light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
        )
        sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(color=(0.2, 0.2, 0.3), intensity=2000.0),
        )
        light.spawn.func(light.prim_path, light.spawn, light.init_state.pos)
        sky_light.spawn.func(sky_light.prim_path, sky_light.spawn)
        
        # Ground Plane
        cfg_ground = sim_utils.GroundPlaneCfg(color=(0.1, 0.1, 0.1), size=(300., 300.))
        cfg_ground.func("/World/defaultGroundPlane", cfg_ground, translation=(0, 0, 0.01))

        self.map_range = [20.0, 20.0, 4.5]

        terrain_cfg = TerrainImporterCfg(
            num_envs=self.num_envs,
            env_spacing=0.0,
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=TerrainGeneratorCfg(
                seed=0,
                size=(self.map_range[0]*2, self.map_range[1]*2), 
                border_width=5.0,
                num_rows=1, 
                num_cols=1, 
                horizontal_scale=0.1,
                vertical_scale=0.1,
                slope_threshold=0.75,
                use_cache=False,
                color_scheme="height",
                sub_terrains={
                    "obstacles": HfRangeObstaclesTerrainCfg(
                        horizontal_scale=0.1,
                        vertical_scale=0.1,
                        border_width=0.0,
                        num_obstacles=self.cfg.env.num_obstacles,
                        obstacle_height_mode="range",
                        obstacle_width_range=(0.4, 1.1),
                        obstacle_height_range=[1.0, 1.5, 2.0, 4.0, 6.0],
                        obstacle_height_probability=[0.1, 0.15, 0.20, 0.55],
                        platform_width=0.0,
                    ),
                },
            ),
            visual_material = None,
            max_init_terrain_level=None,
            collision_group=-1,
            debug_vis=True,
        )
        terrain_importer = TerrainImporter(terrain_cfg)


    def _set_specs(self):
        observation_dim = 9 # rpos(3), vel_w(3), ang_w(3), current_head_dir_2d(3)
        # Observation Spec
        self.observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": CompositeSpec({
                    "state": UnboundedContinuousTensorSpec((observation_dim,), device=self.device), 
                    "lidar": UnboundedContinuousTensorSpec((1, self.lidar_hbeams, self.lidar_vbeams), device=self.device),
                }),
            }).expand(self.num_envs)
        }, shape=[self.num_envs], device=self.device)
        
        # Action Spec
        self.action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "action": self.drone.action_spec, # number of motor
            })
        }).expand(self.num_envs).to(self.device)
        
        # Reward Spec
        self.reward_spec = CompositeSpec({
            "agents": CompositeSpec({
                "reward": UnboundedContinuousTensorSpec((1,))
            })
        }).expand(self.num_envs).to(self.device)

        # Done Spec
        self.done_spec = CompositeSpec({
            "done": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "terminated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "truncated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
        }).expand(self.num_envs).to(self.device) 


        stats_spec = CompositeSpec({
            "return": UnboundedContinuousTensorSpec(1),
            "episode_len": UnboundedContinuousTensorSpec(1),
            "reach_goal": UnboundedContinuousTensorSpec(1),
            "collision": UnboundedContinuousTensorSpec(1),
            "truncated": UnboundedContinuousTensorSpec(1),
            "reward_safety_static": UnboundedContinuousTensorSpec(1),
            "reward_pos": UnboundedContinuousTensorSpec(1),
            "reward_head": UnboundedContinuousTensorSpec(1),
            "reward_reach": UnboundedContinuousTensorSpec(1),
            "penalty_vel": UnboundedContinuousTensorSpec(1),
            "penalty_acc": UnboundedContinuousTensorSpec(1),
            "penalty_collision": UnboundedContinuousTensorSpec(1),

        }).expand(self.num_envs).to(self.device)

        info_spec = CompositeSpec({
            "drone_state": UnboundedContinuousTensorSpec((self.drone.n, 13), device=self.device),
            "current_head_dir_2d": UnboundedContinuousTensorSpec((1, 3), device=self.device),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.observation_spec["info"] = info_spec
        self.stats = stats_spec.zero()
        self.info = info_spec.zero()

    
    def reset_target(self, env_ids: torch.Tensor):
        if (self.training):
            # decide which side
            masks = torch.tensor([[1., 0., 1.], [1., 0., 1.], [0., 1., 1.], [0., 1., 1.]], dtype=torch.float, device=self.device)
            shifts = torch.tensor([[0., 24., 0.], [0., -24., 0.], [24., 0., 0.], [-24., 0., 0.]], dtype=torch.float, device=self.device)
            mask_indices = np.random.randint(0, masks.size(0), size=env_ids.size(0))
            selected_masks = masks[mask_indices].unsqueeze(1)
            selected_shifts = shifts[mask_indices].unsqueeze(1)


            # generate random positions
            target_pos = 48. * torch.rand(env_ids.size(0), 1, 3, dtype=torch.float, device=self.device) + (-24.)
            heights = 0.5 + torch.rand(env_ids.size(0), dtype=torch.float, device=self.device) * (2.5 - 0.5)
            target_pos[:, 0, 2] = heights# height
            target_pos = target_pos * selected_masks + selected_shifts
            
            # apply target pos
            self.target_pos[env_ids] = target_pos

            # self.target_pos[:, 0, 0] = torch.linspace(-0.5, 0.5, self.num_envs) * 32.
            # self.target_pos[:, 0, 1] = 24.
            # self.target_pos[:, 0, 2] = 2.    
        else:
            self.target_pos[:, 0, 0] = torch.linspace(-0.5, 0.5, self.num_envs) * 32.
            self.target_pos[:, 0, 1] = -24.
            self.target_pos[:, 0, 2] = 2.            


    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids, self.training)
        self.reset_target(env_ids)
        if (self.training):
            masks = torch.tensor([[1., 0., 1.], [1., 0., 1.], [0., 1., 1.], [0., 1., 1.]], dtype=torch.float, device=self.device)
            shifts = torch.tensor([[0., 24., 0.], [0., -24., 0.], [24., 0., 0.], [-24., 0., 0.]], dtype=torch.float, device=self.device)
            mask_indices = np.random.randint(0, masks.size(0), size=env_ids.size(0))
            selected_masks = masks[mask_indices].unsqueeze(1)
            selected_shifts = shifts[mask_indices].unsqueeze(1)

            # generate random positions
            pos = 48. * torch.rand(env_ids.size(0), 1, 3, dtype=torch.float, device=self.device) + (-24.)
            heights = 0.5 + torch.rand(env_ids.size(0), dtype=torch.float, device=self.device) * (2.5 - 0.5)
            pos[:, 0, 2] = heights# height
            pos = pos * selected_masks + selected_shifts
            
            # pos = torch.zeros(len(env_ids), 1, 3, device=self.device)
            # pos[:, 0, 0] = (env_ids / self.num_envs - 0.5) * 32.
            # pos[:, 0, 1] = -24.
            # pos[:, 0, 2] = 2.
        else:
            pos = torch.zeros(len(env_ids), 1, 3, device=self.device)
            pos[:, 0, 0] = (env_ids / self.num_envs - 0.5) * 32.
            pos[:, 0, 1] = 24.
            pos[:, 0, 2] = 2.
        
        # Coordinate change: after reset, the drone's target direction should be changed
        self.target_dir[env_ids] = self.target_pos[env_ids] - pos

        # Coordinate change: after reset, the drone's facing direction should face the current goal
        # rpy = torch.zeros(len(env_ids), 1, 3, device=self.device)
        # diff = self.target_pos[env_ids] - pos
        # facing_yaw = torch.atan2(diff[..., 1], diff[..., 0])
        # rpy[..., 2] = facing_yaw
        # rot = euler_to_quaternion(rpy)

        rpy = self.init_rpy_dist.sample((*env_ids.shape, 1))
        rot = euler_to_quaternion(rpy)
    
        self.drone.set_world_poses(pos, rot, env_ids)
        self.drone.set_velocities(self.init_vels[env_ids], env_ids)
        self.prev_drone_vel_w[env_ids] = 0.
        self.last_distance[env_ids] = self.target_dir[env_ids].norm(dim=-1, keepdim=True)
        self.last_yaw[env_ids] = rpy[..., 2]
        self.height_range[env_ids, 0, 0] = torch.min(pos[:, 0, 2], self.target_pos[env_ids, 0, 2])
        self.height_range[env_ids, 0, 1] = torch.max(pos[:, 0, 2], self.target_pos[env_ids, 0, 2])

        self.stats[env_ids] = 0.  
        
    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")] 
        self.drone.apply_action(actions) 

    def _post_sim_step(self, tensordict: TensorDictBase):
        self.lidar.update(self.dt)
    
    # get current states/observation
    def _compute_state_and_obs(self):
        self.root_state = self.drone.get_state(env_frame=False) # (world_pos, orientation (quat), world_vel_and_angular, heading, up, 4motorsthrust)
        self.info["drone_state"][:] = self.root_state[..., :13] # info is for controller

        # >>>>>>>>>>>>The relevant code starts from here<<<<<<<<<<<<
        # -----------Network Input I: LiDAR range data--------------
        self.lidar_scan = self.lidar_range - (
            (self.lidar.data.ray_hits_w - self.lidar.data.pos_w.unsqueeze(1))
            .norm(dim=-1)
            .clamp_max(self.lidar_range)
            .reshape(self.num_envs, 1, *self.lidar_resolution)
        ) # lidar scan store the data that is range - distance and it is in lidar's local frame

        # Optional render for LiDAR
        if self._should_render(0):
            self.debug_draw.clear()

            heading_dir = quat_axis(self.root_state[..., 3:7], axis=0)
            # print("heading_dir", heading_dir.squeeze(1).shape)
            # Scale the arrow length for better visibility
            arrow_length = 2.0
            self.debug_draw.vector(self.root_state[..., :3], heading_dir.squeeze(1) * arrow_length,
                                 size=3.0, color=(1., 0., 0., 1.))  # Red arrow for heading
        
            diff = self.target_pos - self.root_state[..., :3]
            # print("heading_dir", heading_dir.squeeze(1).shape)
            # Scale the arrow length for better visibility
            arrow_length = 2.0
            self.debug_draw.vector(self.root_state[..., :3], (diff / diff.norm(dim=-1, keepdim=True)) * arrow_length,
                                 size=3.0, color=(0, 1.0, 0., 1.))  # Green arrow for heading

        # ---------Network Input II: Drone's internal states---------
        # a. distance info in horizontal and vertical plane
        rpos = self.target_pos - self.root_state[..., :3]        
        distance = rpos.norm(dim=-1, keepdim=True) # start to goal distance
        distance_2d = rpos[..., :2].norm(dim=-1, keepdim=True)
        distance_z = rpos[..., 2].unsqueeze(-1)
        
        # b. unit direction vector to goal
        target_dir_2d = self.target_dir.clone()
        target_dir_2d[..., 2] = 0

        # current robot heading direction in the horizontal plane
        current_head_dir_2d = quat_axis(self.root_state[..., 3:7], axis=0)
        head_dir_2d = current_head_dir_2d.clone()
        head_dir_2d[..., 2] = 0

        current_head_dir_2d[..., 2] = 0
        current_head_dir_2d = current_head_dir_2d / current_head_dir_2d.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        self.info["current_head_dir_2d"][:] = current_head_dir_2d
        rpos_clipped = rpos / distance.clamp(1e-6) # unit vector: start to goal direction
        # rpos_clipped[...,2] = 0. # only care about the horizontal direction for the input, we will add vertical distance as a separate input
        rpos_clipped_g = vec_to_new_frame(rpos_clipped, target_dir_2d) # express in the goal coodinate
        # print("rpos_clipped_g", rpos_clipped_g)

        # c. velocity in the goal frame
        vel_w = self.root_state[..., 7:10] # world vel
        # vel_w[...,2] = 0. # only care about horizontal velocity for the input, we will add vertical velocity as a separate input
        vel_g = vec_to_new_frame(vel_w, target_dir_2d)   # coordinate change for velocity

        ang_w = self.root_state[..., 10:13] # world angular velocity
        ang_head = vec_to_new_frame(ang_w, head_dir_2d) # coordinate change for angular velocity in the heading frame, where the x axis is the current heading direction
        
        quat = self.root_state[..., 3:7]

        rpos_head = vec_to_new_frame(rpos, head_dir_2d) # express the relative position in the heading coordinate, the x axis is the current heading direction, the y axis is on the horizontal plane and perpendicular to the heading direction, and the z axis is vertical
        vel_head = vec_to_new_frame(vel_w, head_dir_2d)

        # final drone's internal states
        # drone_state = torch.cat([rpos_clipped, distance_2d, distance_z, vel_w , current_head_dir_2d], dim=-1).squeeze(1)
        drone_state = torch.cat([rpos_head, vel_head ,ang_head], dim=-1).squeeze(1)

        # -----------------Network Input Final--------------
        obs = {
            "state": drone_state,
            "lidar": self.lidar_scan,
        }

        # -----------------Reward Calculation-----------------
        # a. safety reward for static obstacles
        reward_safety_static = torch.log((self.lidar_range-self.lidar_scan).clamp(min=1e-6, max=self.lidar_range)).mean(dim=(2, 3))
        # 越靠近障碍物，reward越小
        # 距离惩罚设计为-log(distance)，并且clip在一个合理的范围内，避免过大或者过小的梯度
        # reward_safety_static = -torch.log((self.lidar_scan).clamp(min=1e-6, max=self.lidar_range)).mean(dim=(2, 3))
        # print("reward_safety_static", reward_safety_static)
        # print("lidar_scan", self.lidar_scan)

        # b. safety reward for dynamic obstacles
        reward_pos = (self.last_distance - distance).squeeze(-1)


        # yaw reward for facing the goal direction
        quat = self.root_state[..., 3:7]
        current_yaw = quaternion_to_euler(quat)[..., 2]
        diff = self.target_pos - self.root_state[..., :3]
        target_yaw = torch.atan2(diff[..., 1], diff[..., 0])

        diff_yaw = target_yaw - current_yaw
        diff_yaw = (diff_yaw + np.pi) % (2 * np.pi) - np.pi

        b_t = torch.zeros(self.num_envs, 1, device=self.cfg.device)
        b_t[torch.cos(diff_yaw[..., ]) > 0] = torch.cos(diff_yaw[..., ])[torch.cos(diff_yaw[..., ]) > 0]
        
        d_g = distance.squeeze(-1) / 48.0
        w_t= torch.ones(self.num_envs, 1, device=self.cfg.device)
        w_t[d_g < 1.0] = d_g[d_g < 1.0]


        d_tt = (self.last_distance - distance).squeeze(-1)

        diff_distance = torch.zeros(self.num_envs, 1, device=self.cfg.device)
        diff_distance[d_tt > 0] = d_tt[d_tt > 0]

        g_t = torch.zeros(self.num_envs, 1, device=self.cfg.device)
        g_t[diff_distance > 0.001] = 1.0
 
        reward_head = b_t * w_t * g_t 

        penalty_diffyaw = (self.last_yaw - current_yaw).abs()
        # print("penalty_diffyaw", penalty_diffyaw.shape)

        # 距离目标点距离小于0.5，并且yaw偏角小于30度，就认为达到目标点并且朝向正确，给予较大的reward

        reward_reach = torch.zeros(self.num_envs, 1, device=self.cfg.device)
        reward_reach[(distance.squeeze(-1) < 0.5) & (torch.abs(diff_yaw) < np.pi / 6)] = 1.0

        vel_direction = rpos / distance.clamp_min(1e-6)
        reward_vel = (self.drone.vel_w[..., :3] * vel_direction).sum(-1)#.clip(max=2.0) 

        penalty_vel = torch.zeros(self.num_envs, 1, device=self.cfg.device)
        penalty_vel[self.drone.vel_b[..., 0] > 2.0] = (self.drone.vel_b[..., 0] - 2.0)[self.drone.vel_b[..., 0] > 2.0]
        penalty_vel[self.drone.vel_b[..., 0] < 0.2] = (0.2 - self.drone.vel_b[..., 0])[self.drone.vel_b[..., 0] < 0.2]

        # reward_yaw = (current_head_dir_2d * vel_direction).sum(-1)#.clip(max=2.0)
        # print("reward_yaw", reward_yaw)
        penalty_acc = (self.drone.vel_w[..., :3] - self.prev_drone_vel_w).norm(dim=-1) 

        vel_body = vec_to_new_frame(self.drone.vel_w[..., :3], head_dir_2d)
        penalty_acc_body = (vel_body - vec_to_new_frame(self.prev_drone_vel_w, head_dir_2d)).norm(dim=-1)

        # print("penalty_smooth", penalty_smooth)
        # e. height penalty reward for flying unnessarily high or low
        penalty_height = torch.zeros(self.num_envs, 1, device=self.cfg.device)
        penalty_height[self.drone.pos[..., 2] > (self.height_range[..., 1] + 0.2)] = ( (self.drone.pos[..., 2] - self.height_range[..., 1] - 0.2)**2 )[self.drone.pos[..., 2] > (self.height_range[..., 1] + 0.2)]
        penalty_height[self.drone.pos[..., 2] < (self.height_range[..., 0] - 0.2)] = ( (self.height_range[..., 0] - 0.2 - self.drone.pos[..., 2])**2 )[self.drone.pos[..., 2] < (self.height_range[..., 0] - 0.2)]
        
        # print("penalty_height", penalty_height)

        # f. Collision condition with its penalty
        static_collision = einops.reduce(self.lidar_scan, "n 1 w h -> n 1", "max") >  (self.lidar_range - 0.3) # 0.3 collision radius
        collision = static_collision
        
        penalty_collision = static_collision
        # print("reward_collision", reward_collision)
        
        init_distance = self.target_dir.norm(dim=-1, keepdim=True)# initial distance from start to goal, used for distance reward calculation
        # 
        reward_distance = -((distance.squeeze(-1) / init_distance.squeeze(-1)).clamp_min(1e-6))
        # print("reward_distance", reward_distance)

        self.reward = reward_distance*1.0 + reward_safety_static*(1.0) + reward_pos*(20.0) + reward_head*(5.0) + reward_reach*(100.0) + reward_vel*(1.0) + penalty_collision*(-100.0)+ penalty_height*(-5.0)        
        self.last_distance = distance
        self.last_yaw = current_yaw
        # Terminal reward
        # self.reward[collision] -= 50. # collision

        # Terminate Conditions
        reach_goal = (distance.squeeze(-1) < 0.5)
        below_bound = self.drone.pos[..., 2] < 0.2
        above_bound = self.drone.pos[..., 2] > 4.
        self.terminated = collision | reach_goal | below_bound | above_bound
        self.truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1) # progress buf is to track the step number

        # update previous velocity for smoothness calculation in the next ieteration
        self.prev_drone_vel_w = self.drone.vel_w[..., :3].clone()

        # # -----------------Training Stats-----------------
        self.stats["return"] += self.reward
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)
        self.stats["reach_goal"] = reach_goal.float()
        self.stats["collision"] = collision.float()
        self.stats["truncated"] = self.truncated.float()
        self.stats["reward_safety_static"] = reward_safety_static
        self.stats["reward_pos"] = reward_pos
        self.stats["reward_head"] = reward_head
        self.stats["reward_reach"] = reward_reach
        self.stats["penalty_vel"] = penalty_vel
        self.stats["penalty_acc"] = penalty_acc
        self.stats["penalty_collision"]= penalty_collision.float()

        return TensorDict({
            "agents": TensorDict(
                {
                    "observation": obs,
                }, 
                [self.num_envs]
            ),
            "stats": self.stats.clone(),
            "info": self.info
        }, self.batch_size)

    def _compute_reward_and_done(self):
        reward = self.reward
        terminated = self.terminated
        truncated = self.truncated
        return TensorDict(
            {
                "agents": {
                    "reward": reward
                },
                "done": terminated | truncated,
                "terminated": terminated,
                "truncated": truncated,
            },
            self.batch_size,
        )
