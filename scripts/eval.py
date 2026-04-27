import argparse
import os
import hydra
import datetime
import wandb
import torch
import glob
from omegaconf import DictConfig, OmegaConf
from ppo import PPO
from omni_drones.controllers import LeePositionController
from omni_drones.utils.torchrl.transforms import VelController, ravel_composite
from omni_drones.utils.torchrl import SyncDataCollector, EpisodeStats
from torchrl.envs.transforms import TransformedEnv, Compose
from utils import evaluate
from torchrl.envs.utils import ExplorationType


def get_latest_checkpoint():
    """自动查找最新的checkpoint文件"""
    wandb_dir = "/home/shuimujieming/OmniDrones/wandb"
    
    # 查找所有run目录，按修改时间排序
    run_dirs = glob.glob(os.path.join(wandb_dir, "run-*"))
    if not run_dirs:
        raise FileNotFoundError(f"No wandb run directories found in {wandb_dir}")
    
    # 按修改时间排序，获取最新的
    run_dirs.sort(key=os.path.getmtime, reverse=True)
    latest_run_dir = run_dirs[0]
    
    # 查找该目录下的所有checkpoint文件
    checkpoint_files = glob.glob(os.path.join(latest_run_dir, "files", "checkpoint_*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {latest_run_dir}/files")
    
    # 按时间顺序排序，获取最新的checkpoint
    checkpoint_files.sort(key=os.path.getmtime, reverse=True)
    latest_checkpoint = checkpoint_files[0]
    
    print(f"[NavRL]: Found latest checkpoint: {latest_checkpoint}")
    return latest_checkpoint


FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cfg")
@hydra.main(config_path=FILE_PATH, config_name="train", version_base=None)
def main(cfg):
    # Simulation App
    cfg.headless = False
    cfg.env.num_envs = 1
    # Simulation App
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher({"headless": cfg.headless, "anti_aliasing": 1})
    simulation_app = app_launcher.app


    # Navigation Training Environment
    from env import NavigationEnv
    env = NavigationEnv(cfg)

    # Transformed Environment
    transforms = []
    # transforms.append(ravel_composite(env.observation_spec, ("agents", "intrinsics"), start_dim=-1))
    controller = LeePositionController(9.81, env.drone.params).to(cfg.device)
    vel_transform = VelController(controller, yaw_control=True)
    transforms.append(vel_transform)
    transformed_env = TransformedEnv(env, Compose(*transforms)).train()
    transformed_env.set_seed(cfg.seed)    
    # PPO Policy
    policy = PPO(cfg.algo, transformed_env.observation_spec, transformed_env.action_spec, cfg.device)

    checkpoint = get_latest_checkpoint()
    policy.load_state_dict(torch.load(checkpoint))
    
    # Episode Stats Collector
    episode_stats_keys = [
        k for k in transformed_env.observation_spec.keys(True, True) 
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    episode_stats = EpisodeStats(episode_stats_keys)

    # RL Data Collector
    collector = SyncDataCollector(
        transformed_env,
        policy=policy, 
        frames_per_batch=cfg.env.num_envs * cfg.algo.training_frame_num, 
        total_frames=cfg.max_frame_num,
        device=cfg.device,
        return_same_td=True, # update the return tensordict inplace (should set to false if we need to use replace buffer)
        exploration_type=ExplorationType.RANDOM, # sample from normal distribution
    )

    # Training Loop
    for i, data in enumerate(collector):
        # print("data: ", data)
        # print("============================")
        # Log Info
        info = {"env_frames": collector._frames, "rollout_fps": collector._fps}

        # # Train Policy
        # train_loss_stats = policy.train(data)
        # info.update(train_loss_stats) # log training loss info

        # # Calculate and log training episode stats
        # episode_stats.add(data)
        # if len(episode_stats) >= transformed_env.num_envs: # evaluate once if all agents finished one episode
        #     stats = {
        #         "train/" + (".".join(k) if isinstance(k, tuple) else k): torch.mean(v.float()).item() 
        #         for k, v in episode_stats.pop().items(True, True)
        #     }
        #     info.update(stats)

        # Evaluate policy and log info
        # if i % cfg.eval_interval == 0:
        print("[NavRL]: start evaluating policy at training step: ", i)
        env.eval()
        eval_info = evaluate(
            env=transformed_env, 
            policy=policy,
            seed=cfg.seed, 
            cfg=cfg,
            exploration_type=ExplorationType.MEAN
        )
        env.train()
        env.reset()
        info.update(eval_info)
        print("\n[NavRL]: evaluation done.")
    
    wandb.finish()
    sim_app.close()

if __name__ == "__main__":
    main()
    