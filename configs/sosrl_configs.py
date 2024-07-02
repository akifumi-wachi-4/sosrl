from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

from pyrallis import field


@dataclass
class SOSRLTrainConfig:
    # wandb params
    project: str = "OSRL-baselines"
    group: str = None
    name: Optional[str] = None
    prefix: Optional[str] = "SOSRL"
    suffix: Optional[str] = ""
    logdir: Optional[str] = "logs"
    verbose: bool = True
    # dataset params
    outliers_percent: float = None
    noise_scale: float = None
    inpaint_ranges: Tuple[Tuple[float, float, float, float], ...] = None
    epsilon: float = None
    density: float = 1.0
    # training params
    task: str = "OfflineCarCircle-v0"
    dataset: str = None
    seed: int = 0
    device: str = "cuda"
    threads: int = 4
    reward_scale: float = 0.1
    cost_scale: float = 1
    actor_lr: float = 0.001
    critic_lr: float = 0.001
    vae_lr: float = 0.001
    phi: float = 0.05
    lmbda: float = 0.75
    beta: float = 0.5
    multiplier: float = 100
    cost_limit: int = 40
    episode_len: int = 300
    batch_size: int = 512
    update_steps: int = 100_000
    num_workers: int = 8
    # model params
    a_hidden_sizes: List[float] = field(default=[256, 256], is_mutable=True)
    c_hidden_sizes: List[float] = field(default=[256, 256], is_mutable=True)
    vae_hidden_sizes: int = 400
    sample_action_num: int = 10
    gamma: float = 0.99
    tau: float = 0.005
    num_q: int = 2
    num_qc: int = 2
    # evaluation params
    eval_episodes: int = 10
    eval_every: int = 2500


@dataclass
class SOSRLCarCircleConfig(SOSRLTrainConfig):
    pass


@dataclass
class SOSRLAntRunConfig(SOSRLTrainConfig):
    # training params
    task: str = "OfflineAntRun-v0"
    episode_len: int = 200


@dataclass
class SOSRLDroneRunConfig(SOSRLTrainConfig):
    # training params
    task: str = "OfflineDroneRun-v0"
    episode_len: int = 200


@dataclass
class SOSRLDroneCircleConfig(SOSRLTrainConfig):
    # training params
    task: str = "OfflineDroneCircle-v0"
    episode_len: int = 300


@dataclass
class SOSRLCarRunConfig(SOSRLTrainConfig):
    # training params
    task: str = "OfflineCarRun-v0"
    episode_len: int = 200


@dataclass
class SOSRLAntCircleConfig(SOSRLTrainConfig):
    # training params
    task: str = "OfflineAntCircle-v0"
    episode_len: int = 500


@dataclass
class SOSRLBallRunConfig(SOSRLTrainConfig):
    # training params
    task: str = "OfflineBallRun-v0"
    episode_len: int = 100


@dataclass
class SOSRLBallCircleConfig(SOSRLTrainConfig):
    # training params
    task: str = "OfflineBallCircle-v0"
    episode_len: int = 200


@dataclass
class SOSRLCarButton1Config(SOSRLTrainConfig):
    # training params
    task: str = "OfflineCarButton1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class SOSRLCarButton2Config(SOSRLTrainConfig):
    # training params
    task: str = "OfflineCarButton2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class SOSRLCarCircle1Config(SOSRLTrainConfig):
    # training params
    task: str = "OfflineCarCircle1Gymnasium-v0"
    episode_len: int = 500


@dataclass
class SOSRLCarCircle2Config(SOSRLTrainConfig):
    # training params
    task: str = "OfflineCarCircle2Gymnasium-v0"
    episode_len: int = 500


@dataclass
class SOSRLCarGoal1Config(SOSRLTrainConfig):
    # training params
    task: str = "OfflineCarGoal1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class SOSRLCarGoal2Config(SOSRLTrainConfig):
    # training params
    task: str = "OfflineCarGoal2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class SOSRLCarPush1Config(SOSRLTrainConfig):
    # training params
    task: str = "OfflineCarPush1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class SOSRLCarPush2Config(SOSRLTrainConfig):
    # training params
    task: str = "OfflineCarPush2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class SOSRLPointButton1Config(SOSRLTrainConfig):
    # training params
    task: str = "OfflinePointButton1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class SOSRLPointButton2Config(SOSRLTrainConfig):
    # training params
    task: str = "OfflinePointButton2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class SOSRLPointCircle1Config(SOSRLTrainConfig):
    # training params
    task: str = "OfflinePointCircle1Gymnasium-v0"
    episode_len: int = 500


@dataclass
class SOSRLPointCircle2Config(SOSRLTrainConfig):
    # training params
    task: str = "OfflinePointCircle2Gymnasium-v0"
    episode_len: int = 500


@dataclass
class SOSRLPointGoal1Config(SOSRLTrainConfig):
    # training params
    task: str = "OfflinePointGoal1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class SOSRLPointGoal2Config(SOSRLTrainConfig):
    # training params
    task: str = "OfflinePointGoal2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class SOSRLPointPush1Config(SOSRLTrainConfig):
    # training params
    task: str = "OfflinePointPush1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class SOSRLPointPush2Config(SOSRLTrainConfig):
    # training params
    task: str = "OfflinePointPush2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class SOSRLAntVelocityConfig(SOSRLTrainConfig):
    # training params
    task: str = "OfflineAntVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class SOSRLHalfCheetahVelocityConfig(SOSRLTrainConfig):
    # training params
    task: str = "OfflineHalfCheetahVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class SOSRLHopperVelocityConfig(SOSRLTrainConfig):
    # training params
    task: str = "OfflineHopperVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class SOSRLSwimmerVelocityConfig(SOSRLTrainConfig):
    # training params
    task: str = "OfflineSwimmerVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class SOSRLWalker2dVelocityConfig(SOSRLTrainConfig):
    # training params
    task: str = "OfflineWalker2dVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class SOSRLEasySparseConfig(SOSRLTrainConfig):
    # training params
    task: str = "OfflineMetadrive-easysparse-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class SOSRLEasyMeanConfig(SOSRLTrainConfig):
    # training params
    task: str = "OfflineMetadrive-easymean-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class SOSRLEasyDenseConfig(SOSRLTrainConfig):
    # training params
    task: str = "OfflineMetadrive-easydense-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class SOSRLMediumSparseConfig(SOSRLTrainConfig):
    # training params
    task: str = "OfflineMetadrive-mediumsparse-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class SOSRLMediumMeanConfig(SOSRLTrainConfig):
    # training params
    task: str = "OfflineMetadrive-mediummean-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class SOSRLMediumDenseConfig(SOSRLTrainConfig):
    # training params
    task: str = "OfflineMetadrive-mediumdense-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class SOSRLHardSparseConfig(SOSRLTrainConfig):
    # training params
    task: str = "OfflineMetadrive-hardsparse-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class SOSRLHardMeanConfig(SOSRLTrainConfig):
    # training params
    task: str = "OfflineMetadrive-hardmean-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class SOSRLHardDenseConfig(SOSRLTrainConfig):
    # training params
    task: str = "OfflineMetadrive-harddense-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


SOSRL_DEFAULT_CONFIG = {
    # bullet_safety_gym
    "OfflineCarCircle-v0": SOSRLCarCircleConfig,
    "OfflineAntRun-v0": SOSRLAntRunConfig,
    "OfflineDroneRun-v0": SOSRLDroneRunConfig,
    "OfflineDroneCircle-v0": SOSRLDroneCircleConfig,
    "OfflineCarRun-v0": SOSRLCarRunConfig,
    "OfflineAntCircle-v0": SOSRLAntCircleConfig,
    "OfflineBallCircle-v0": SOSRLBallCircleConfig,
    "OfflineBallRun-v0": SOSRLBallRunConfig,
    # safety_gymnasium: car
    "OfflineCarButton1Gymnasium-v0": SOSRLCarButton1Config,
    "OfflineCarButton2Gymnasium-v0": SOSRLCarButton2Config,
    "OfflineCarCircle1Gymnasium-v0": SOSRLCarCircle1Config,
    "OfflineCarCircle2Gymnasium-v0": SOSRLCarCircle2Config,
    "OfflineCarGoal1Gymnasium-v0": SOSRLCarGoal1Config,
    "OfflineCarGoal2Gymnasium-v0": SOSRLCarGoal2Config,
    "OfflineCarPush1Gymnasium-v0": SOSRLCarPush1Config,
    "OfflineCarPush2Gymnasium-v0": SOSRLCarPush2Config,
    # safety_gymnasium: point
    "OfflinePointButton1Gymnasium-v0": SOSRLPointButton1Config,
    "OfflinePointButton2Gymnasium-v0": SOSRLPointButton2Config,
    "OfflinePointCircle1Gymnasium-v0": SOSRLPointCircle1Config,
    "OfflinePointCircle2Gymnasium-v0": SOSRLPointCircle2Config,
    "OfflinePointGoal1Gymnasium-v0": SOSRLPointGoal1Config,
    "OfflinePointGoal2Gymnasium-v0": SOSRLPointGoal2Config,
    "OfflinePointPush1Gymnasium-v0": SOSRLPointPush1Config,
    "OfflinePointPush2Gymnasium-v0": SOSRLPointPush2Config,
    # safety_gymnasium: velocity
    "OfflineAntVelocityGymnasium-v1": SOSRLAntVelocityConfig,
    "OfflineHalfCheetahVelocityGymnasium-v1": SOSRLHalfCheetahVelocityConfig,
    "OfflineHopperVelocityGymnasium-v1": SOSRLHopperVelocityConfig,
    "OfflineSwimmerVelocityGymnasium-v1": SOSRLSwimmerVelocityConfig,
    "OfflineWalker2dVelocityGymnasium-v1": SOSRLWalker2dVelocityConfig,
    # safe_metadrive
    "OfflineMetadrive-easysparse-v0": SOSRLEasySparseConfig,
    "OfflineMetadrive-easymean-v0": SOSRLEasyMeanConfig,
    "OfflineMetadrive-easydense-v0": SOSRLEasyDenseConfig,
    "OfflineMetadrive-mediumsparse-v0": SOSRLMediumSparseConfig,
    "OfflineMetadrive-mediummean-v0": SOSRLMediumMeanConfig,
    "OfflineMetadrive-mediumdense-v0": SOSRLMediumDenseConfig,
    "OfflineMetadrive-hardsparse-v0": SOSRLHardSparseConfig,
    "OfflineMetadrive-hardmean-v0": SOSRLHardMeanConfig,
    "OfflineMetadrive-harddense-v0": SOSRLHardDenseConfig
}