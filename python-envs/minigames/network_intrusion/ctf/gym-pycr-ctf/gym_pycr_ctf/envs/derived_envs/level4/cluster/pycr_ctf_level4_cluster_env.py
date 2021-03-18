from gym_pycr_ctf.dao.network.env_mode import EnvMode
from gym_pycr_ctf.dao.network.env_config import EnvConfig
from gym_pycr_ctf.dao.network.cluster_config import ClusterConfig
from gym_pycr_ctf.envs.pycr_ctf_env import PyCRCTFEnv
from gym_pycr_ctf.envs.config.level_4.pycr_ctf_level_4_base import PyCrCTFLevel4Base
from gym_pycr_ctf.envs.config.level_4.pycr_ctf_level_4_v1 import PyCrCTFLevel4V1
from gym_pycr_ctf.envs.config.level_4.pycr_ctf_level_4_v2 import PyCrCTFLevel4V2
from gym_pycr_ctf.envs.config.level_4.pycr_ctf_level_4_v3 import PyCrCTFLevel4V3
from gym_pycr_ctf.envs.config.level_4.pycr_ctf_level_4_v4 import PyCrCTFLevel4V4


# -------- Base Version (for testing) ------------
class PyCRCTFLevel4ClusterBaseEnv(PyCRCTFEnv):
    """
    Base version with all set of actions
    """
    def __init__(self, env_config: EnvConfig, cluster_config: ClusterConfig, checkpoint_dir : str):
        if env_config is None:
            render_config = PyCrCTFLevel4Base.render_conf()
            if cluster_config is None:
                cluster_config = PyCrCTFLevel4Base.cluster_conf()
            cluster_config.ids_router = True
            cluster_config.ids_router_ip = PyCrCTFLevel4Base.router_ip()
            network_conf = PyCrCTFLevel4Base.network_conf()
            action_conf = PyCrCTFLevel4Base.all_actions_conf(num_nodes=PyCrCTFLevel4Base.num_nodes(),
                                                                 subnet_mask=PyCrCTFLevel4Base.subnet_mask(),
                                                                 hacker_ip=PyCrCTFLevel4Base.hacker_ip())
            env_config = PyCrCTFLevel4Base.env_config(network_conf=network_conf, action_conf=action_conf,
                                                          cluster_conf=cluster_config, render_conf=render_config)
            env_config.alerts_coefficient = 1
            env_config.cost_coefficient = 0
            env_config.env_mode = EnvMode.CLUSTER
            env_config.save_trajectories = False
            env_config.checkpoint_dir = checkpoint_dir
            env_config.checkpoint_freq = 1000
        super().__init__(env_config=env_config)


# -------- Version 1 ------------

class PyCRCTFLevel4Cluster1Env(PyCRCTFEnv):
    """
    The simplest possible configuration, minimal set of actions. Does not take action costs into account.
    """
    def __init__(self, env_config: EnvConfig, cluster_config: ClusterConfig, checkpoint_dir : str):
        if env_config is None:
            render_config = PyCrCTFLevel4Base.render_conf()
            if cluster_config is None:
                cluster_config = PyCrCTFLevel4Base.cluster_conf()
            cluster_config.ids_router = True
            cluster_config.ids_router_ip = PyCrCTFLevel4Base.router_ip()
            network_conf = PyCrCTFLevel4Base.network_conf()
            action_conf = PyCrCTFLevel4V1.actions_conf(num_nodes=PyCrCTFLevel4Base.num_nodes(),
                                                           subnet_mask=PyCrCTFLevel4Base.subnet_mask(),
                                                           hacker_ip=PyCrCTFLevel4Base.hacker_ip())
            env_config = PyCrCTFLevel4V1.env_config(network_conf=network_conf, action_conf=action_conf,
                                                        cluster_conf=cluster_config, render_conf=render_config)
            env_config.alerts_coefficient = 1
            env_config.cost_coefficient = 0
            env_config.env_mode = EnvMode.CLUSTER
            env_config.save_trajectories = False
            env_config.checkpoint_dir = checkpoint_dir
            env_config.checkpoint_freq = 1000
        super().__init__(env_config=env_config)


# -------- Version 1 with costs------------

class PyCRCTFLevel4ClusterWithCosts1Env(PyCRCTFEnv):
    """
    The simplest possible configuration, minimal set of actions. Does take action costs into account.
    """
    def __init__(self, env_config: EnvConfig, cluster_config: ClusterConfig, checkpoint_dir : str):
        if env_config is None:
            render_config = PyCrCTFLevel4Base.render_conf()
            if cluster_config is None:
                cluster_config = PyCrCTFLevel4Base.cluster_conf()
            cluster_config.ids_router = True
            cluster_config.ids_router_ip = PyCrCTFLevel4Base.router_ip()
            network_conf = PyCrCTFLevel4Base.network_conf()
            action_conf = PyCrCTFLevel4V1.actions_conf(num_nodes=PyCrCTFLevel4Base.num_nodes(),
                                                           subnet_mask=PyCrCTFLevel4Base.subnet_mask(),
                                                           hacker_ip=PyCrCTFLevel4Base.hacker_ip())
            env_config = PyCrCTFLevel4V1.env_config(network_conf=network_conf, action_conf=action_conf,
                                                        cluster_conf=cluster_config, render_conf=render_config)
            env_config.alerts_coefficient = 1
            env_config.cost_coefficient = 1
            env_config.env_mode = EnvMode.CLUSTER
            env_config.save_trajectories = False
            env_config.checkpoint_dir = checkpoint_dir
            env_config.checkpoint_freq = 1000
        super().__init__(env_config=env_config)


# -------- Version 2 ------------

class PyCRCTFLevel4Cluster2Env(PyCRCTFEnv):
    """
    Slightly more set of actions than V3. Does not take action costs into account.
    """
    def __init__(self, env_config: EnvConfig, cluster_config: ClusterConfig, checkpoint_dir : str):
        if env_config is None:
            render_config = PyCrCTFLevel4Base.render_conf()
            if cluster_config is None:
                cluster_config = PyCrCTFLevel4Base.cluster_conf()
            cluster_config.ids_router = True
            cluster_config.ids_router_ip = PyCrCTFLevel4Base.router_ip()
            network_conf = PyCrCTFLevel4Base.network_conf()
            action_conf = PyCrCTFLevel4V2.actions_conf(num_nodes=PyCrCTFLevel4Base.num_nodes(),
                                                           subnet_mask=PyCrCTFLevel4Base.subnet_mask(),
                                                           hacker_ip=PyCrCTFLevel4Base.hacker_ip())
            env_config = PyCrCTFLevel4V2.env_config(network_conf=network_conf, action_conf=action_conf,
                                                        cluster_conf=cluster_config, render_conf=render_config)
            env_config.alerts_coefficient = 1
            env_config.cost_coefficient = 0
            env_config.env_mode = EnvMode.CLUSTER
            env_config.save_trajectories = False
            env_config.checkpoint_dir = checkpoint_dir
            env_config.checkpoint_freq = 1000
        super().__init__(env_config=env_config)


# -------- Version 2 with costs------------

class PyCRCTFLevel4ClusterWithCosts2Env(PyCRCTFEnv):
    """
    Slightly more set of actions than V1. Does take action costs into account.
    """
    def __init__(self, env_config: EnvConfig, cluster_config: ClusterConfig, checkpoint_dir : str):
        if env_config is None:
            render_config = PyCrCTFLevel4Base.render_conf()
            if cluster_config is None:
                cluster_config = PyCrCTFLevel4Base.cluster_conf()
            cluster_config.ids_router = True
            cluster_config.ids_router_ip = PyCrCTFLevel4Base.router_ip()
            network_conf = PyCrCTFLevel4Base.network_conf()
            action_conf = PyCrCTFLevel4V2.actions_conf(num_nodes=PyCrCTFLevel4Base.num_nodes(),
                                                           subnet_mask=PyCrCTFLevel4Base.subnet_mask(),
                                                           hacker_ip=PyCrCTFLevel4Base.hacker_ip())
            env_config = PyCrCTFLevel4V2.env_config(network_conf=network_conf, action_conf=action_conf,
                                                        cluster_conf=cluster_config, render_conf=render_config)
            env_config.alerts_coefficient = 1
            env_config.cost_coefficient = 1
            env_config.env_mode = EnvMode.CLUSTER
            env_config.save_trajectories = False
            env_config.checkpoint_dir = checkpoint_dir
            env_config.checkpoint_freq = 1000
        super().__init__(env_config=env_config)


# -------- Version 3 ------------

class PyCRCTFLevel4Cluster3Env(PyCRCTFEnv):
    """
    Slightly more set of actions than V2. Does not take action costs into account.
    """
    def __init__(self, env_config: EnvConfig, cluster_config: ClusterConfig, checkpoint_dir : str):
        if env_config is None:
            render_config = PyCrCTFLevel4Base.render_conf()
            if cluster_config is None:
                cluster_config = PyCrCTFLevel4Base.cluster_conf()
            cluster_config.ids_router = True
            cluster_config.ids_router_ip = PyCrCTFLevel4Base.router_ip()
            network_conf = PyCrCTFLevel4Base.network_conf()
            action_conf = PyCrCTFLevel4V3.actions_conf(num_nodes=PyCrCTFLevel4Base.num_nodes(),
                                                           subnet_mask=PyCrCTFLevel4Base.subnet_mask(),
                                                           hacker_ip=PyCrCTFLevel4Base.hacker_ip())
            env_config = PyCrCTFLevel4V3.env_config(network_conf=network_conf, action_conf=action_conf,
                                                        cluster_conf=cluster_config, render_conf=render_config)
            env_config.alerts_coefficient = 1
            env_config.cost_coefficient = 0
            env_config.env_mode = EnvMode.CLUSTER
            env_config.save_trajectories = False
            env_config.checkpoint_dir = checkpoint_dir
            env_config.checkpoint_freq = 1000
        super().__init__(env_config=env_config)


# -------- Version 3 with costs------------

class PyCRCTFLevel4ClusterWithCosts3Env(PyCRCTFEnv):
    """
    Slightly more set of actions than V2. Does take action costs into account.
    """
    def __init__(self, env_config: EnvConfig, cluster_config: ClusterConfig, checkpoint_dir : str):
        if env_config is None:
            render_config = PyCrCTFLevel4Base.render_conf()
            if cluster_config is None:
                cluster_config = PyCrCTFLevel4Base.cluster_conf()
            cluster_config.ids_router = True
            cluster_config.ids_router_ip = PyCrCTFLevel4Base.router_ip()
            network_conf = PyCrCTFLevel4Base.network_conf()
            action_conf = PyCrCTFLevel4V3.actions_conf(num_nodes=PyCrCTFLevel4Base.num_nodes(),
                                                           subnet_mask=PyCrCTFLevel4Base.subnet_mask(),
                                                           hacker_ip=PyCrCTFLevel4Base.hacker_ip())
            env_config = PyCrCTFLevel4V3.env_config(network_conf=network_conf, action_conf=action_conf,
                                                        cluster_conf=cluster_config, render_conf=render_config)
            env_config.alerts_coefficient = 1
            env_config.cost_coefficient = 1
            env_config.env_mode = EnvMode.CLUSTER
            env_config.save_trajectories = False
            env_config.checkpoint_dir = checkpoint_dir
            env_config.checkpoint_freq = 1000
        super().__init__(env_config=env_config)


# -------- Version 4 ------------

class PyCRCTFLevel4Cluster4Env(PyCRCTFEnv):
    """
    Slightly more set of actions than V3. Does not take action costs into account.
    """
    def __init__(self, env_config: EnvConfig, cluster_config: ClusterConfig, checkpoint_dir : str):
        if env_config is None:
            render_config = PyCrCTFLevel4Base.render_conf()
            if cluster_config is None:
                cluster_config = PyCrCTFLevel4Base.cluster_conf()
            cluster_config.ids_router = True
            cluster_config.ids_router_ip = PyCrCTFLevel4Base.router_ip()
            network_conf = PyCrCTFLevel4Base.network_conf()
            action_conf = PyCrCTFLevel4V4.actions_conf(num_nodes=PyCrCTFLevel4Base.num_nodes(),
                                                           subnet_mask=PyCrCTFLevel4Base.subnet_mask(),
                                                           hacker_ip=PyCrCTFLevel4Base.hacker_ip())
            env_config = PyCrCTFLevel4V4.env_config(network_conf=network_conf, action_conf=action_conf,
                                                        cluster_conf=cluster_config, render_conf=render_config)
            env_config.alerts_coefficient = 1
            env_config.cost_coefficient = 0
            env_config.env_mode = EnvMode.CLUSTER
            env_config.save_trajectories = False
            env_config.checkpoint_dir = checkpoint_dir
            env_config.checkpoint_freq = 1000
        super().__init__(env_config=env_config)


# -------- Version 4 with costs------------

class PyCRCTFLevel4ClusterWithCosts4Env(PyCRCTFEnv):
    """
    Slightly more set of actions than V3. Does take action costs into account.
    """
    def __init__(self, env_config: EnvConfig, cluster_config: ClusterConfig, checkpoint_dir : str):
        if env_config is None:
            render_config = PyCrCTFLevel4Base.render_conf()
            if cluster_config is None:
                cluster_config = PyCrCTFLevel4Base.cluster_conf()
            cluster_config.ids_router = True
            cluster_config.ids_router_ip = PyCrCTFLevel4Base.router_ip()
            network_conf = PyCrCTFLevel4Base.network_conf()
            action_conf = PyCrCTFLevel4V4.actions_conf(num_nodes=PyCrCTFLevel4Base.num_nodes(),
                                                           subnet_mask=PyCrCTFLevel4Base.subnet_mask(),
                                                           hacker_ip=PyCrCTFLevel4Base.hacker_ip())
            env_config = PyCrCTFLevel4V4.env_config(network_conf=network_conf, action_conf=action_conf,
                                                        cluster_conf=cluster_config, render_conf=render_config)
            env_config.alerts_coefficient = 1
            env_config.cost_coefficient = 1
            env_config.env_mode = EnvMode.CLUSTER
            env_config.save_trajectories = False
            env_config.checkpoint_dir = checkpoint_dir
            env_config.checkpoint_freq = 1000
        super().__init__(env_config=env_config)