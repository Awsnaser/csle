from gym_pycr_ctf.dao.network.env_mode import EnvMode
from gym_pycr_ctf.dao.network.env_config import EnvConfig
from gym_pycr_ctf.dao.network.emulation_config import EmulationConfig
from gym_pycr_ctf.envs.pycr_ctf_env import PyCRCTFEnv
from gym_pycr_ctf.envs_model.config.level_7.pycr_ctf_level_7_base import PyCrCTFLevel7Base
from gym_pycr_ctf.envs_model.config.level_7.pycr_ctf_level_7_v1 import PyCrCTFLevel7V1
from gym_pycr_ctf.envs_model.config.level_7.pycr_ctf_level_7_v2 import PyCrCTFLevel7V2
from gym_pycr_ctf.envs_model.config.level_7.pycr_ctf_level_7_v3 import PyCrCTFLevel7V3
from gym_pycr_ctf.envs_model.config.level_7.pycr_ctf_level_7_v4 import PyCrCTFLevel7V4


# -------- Base Version (for testing) ------------
class PyCRCTFLevel7EmulationBaseEnv(PyCRCTFEnv):
    """
    Base version with all set of actions
    """
    def __init__(self, env_config: EnvConfig, emulation_config: EmulationConfig, checkpoint_dir : str):
        if env_config is None:
            render_config = PyCrCTFLevel7Base.render_conf()
            if emulation_config is None:
                emulation_config = PyCrCTFLevel7Base.emulation_config()
            emulation_config.ids_router = True
            emulation_config.ids_router_ip = PyCrCTFLevel7Base.router_ip()
            network_conf = PyCrCTFLevel7Base.network_conf()
            attacker_action_conf = PyCrCTFLevel7Base.attacker_all_actions_conf(num_nodes=PyCrCTFLevel7Base.num_nodes(),
                                                                      subnet_mask=PyCrCTFLevel7Base.subnet_mask(),
                                                                      hacker_ip=PyCrCTFLevel7Base.hacker_ip())
            defender_action_conf = PyCrCTFLevel7Base.defender_all_actions_conf(
                num_nodes=PyCrCTFLevel7Base.num_nodes(), subnet_mask=PyCrCTFLevel7Base.subnet_mask())
            env_config = PyCrCTFLevel7Base.env_config(network_conf=network_conf,
                                                      attacker_action_conf=attacker_action_conf,
                                                      defender_action_conf=defender_action_conf,
                                                      emulation_config=emulation_config, render_conf=render_config)
            env_config.attacker_alerts_coefficient = 1
            env_config.attacker_cost_coefficient = 0
            env_config.env_mode = EnvMode.emulation
            env_config.save_trajectories = False
            env_config.checkpoint_dir = checkpoint_dir
            env_config.checkpoint_freq = 1000
        super().__init__(env_config=env_config)


# -------- Version 1 ------------

class PyCRCTFLevel7Emulation1Env(PyCRCTFEnv):
    """
    The simplest possible configuration, minimal set of actions. Does not take action costs into account.
    """
    def __init__(self, env_config: EnvConfig, emulation_config: EmulationConfig, checkpoint_dir : str):
        if env_config is None:
            render_config = PyCrCTFLevel7Base.render_conf()
            if emulation_config is None:
                emulation_config = PyCrCTFLevel7Base.emulation_config()
            emulation_config.ids_router = True
            emulation_config.ids_router_ip = PyCrCTFLevel7Base.router_ip()
            network_conf = PyCrCTFLevel7Base.network_conf()
            attacker_action_conf = PyCrCTFLevel7V1.attacker_actions_conf(num_nodes=PyCrCTFLevel7Base.num_nodes(),
                                                                subnet_mask=PyCrCTFLevel7Base.subnet_mask(),
                                                                hacker_ip=PyCrCTFLevel7Base.hacker_ip())
            defender_action_conf = PyCrCTFLevel7V1.defender_actions_conf(
                num_nodes=PyCrCTFLevel7Base.num_nodes(), subnet_mask=PyCrCTFLevel7Base.subnet_mask())
            env_config = PyCrCTFLevel7V1.env_config(network_conf=network_conf,
                                                    attacker_action_conf=attacker_action_conf,
                                                    defender_action_conf=defender_action_conf,
                                                    emulation_config=emulation_config, render_conf=render_config)
            env_config.attacker_alerts_coefficient = 1
            env_config.attacker_cost_coefficient = 0
            env_config.env_mode = EnvMode.emulation
            env_config.save_trajectories = False
            env_config.checkpoint_dir = checkpoint_dir
            env_config.checkpoint_freq = 1000
        super().__init__(env_config=env_config)


# -------- Version 1 with costs------------

class PyCRCTFLevel7EmulationWithCosts1Env(PyCRCTFEnv):
    """
    The simplest possible configuration, minimal set of actions. Does take action costs into account.
    """
    def __init__(self, env_config: EnvConfig, emulation_config: EmulationConfig, checkpoint_dir : str):
        if env_config is None:
            render_config = PyCrCTFLevel7Base.render_conf()
            if emulation_config is None:
                emulation_config = PyCrCTFLevel7Base.emulation_config()
            emulation_config.ids_router = True
            emulation_config.ids_router_ip = PyCrCTFLevel7Base.router_ip()
            network_conf = PyCrCTFLevel7Base.network_conf()
            attacker_action_conf = PyCrCTFLevel7V1.attacker_actions_conf(num_nodes=PyCrCTFLevel7Base.num_nodes(),
                                                                subnet_mask=PyCrCTFLevel7Base.subnet_mask(),
                                                                hacker_ip=PyCrCTFLevel7Base.hacker_ip())
            defender_action_conf = PyCrCTFLevel7V1.defender_actions_conf(
                num_nodes=PyCrCTFLevel7Base.num_nodes(), subnet_mask=PyCrCTFLevel7Base.subnet_mask())
            env_config = PyCrCTFLevel7V1.env_config(network_conf=network_conf,
                                                    attacker_action_conf=attacker_action_conf,
                                                    defender_action_conf=defender_action_conf,
                                                    emulation_config=emulation_config, render_conf=render_config)
            env_config.attacker_alerts_coefficient = 1
            env_config.attacker_cost_coefficient = 1
            env_config.env_mode = EnvMode.emulation
            env_config.save_trajectories = False
            env_config.checkpoint_dir = checkpoint_dir
            env_config.checkpoint_freq = 1000
        super().__init__(env_config=env_config)


# -------- Version 2 ------------

class PyCRCTFLevel7Emulation2Env(PyCRCTFEnv):
    """
    Slightly more set of actions than V3. Does not take action costs into account.
    """
    def __init__(self, env_config: EnvConfig, emulation_config: EmulationConfig, checkpoint_dir : str):
        if env_config is None:
            render_config = PyCrCTFLevel7Base.render_conf()
            if emulation_config is None:
                emulation_config = PyCrCTFLevel7Base.emulation_config()
            emulation_config.ids_router = True
            emulation_config.ids_router_ip = PyCrCTFLevel7Base.router_ip()
            network_conf = PyCrCTFLevel7Base.network_conf()
            attacker_action_conf = PyCrCTFLevel7V2.attacker_actions_conf(num_nodes=PyCrCTFLevel7Base.num_nodes(),
                                                                subnet_mask=PyCrCTFLevel7Base.subnet_mask(),
                                                                hacker_ip=PyCrCTFLevel7Base.hacker_ip())
            defender_action_conf = PyCrCTFLevel7V2.defender_actions_conf(
                num_nodes=PyCrCTFLevel7Base.num_nodes(), subnet_mask=PyCrCTFLevel7Base.subnet_mask())
            env_config = PyCrCTFLevel7V2.env_config(network_conf=network_conf,
                                                    attacker_action_conf=attacker_action_conf,
                                                    defender_action_conf=defender_action_conf,
                                                    emulation_config=emulation_config, render_conf=render_config)
            env_config.attacker_alerts_coefficient = 1
            env_config.attacker_cost_coefficient = 0
            env_config.env_mode = EnvMode.emulation
            env_config.save_trajectories = False
            env_config.checkpoint_dir = checkpoint_dir
            env_config.checkpoint_freq = 1000
        super().__init__(env_config=env_config)


# -------- Version 2 with costs------------

class PyCRCTFLevel7EmulationWithCosts2Env(PyCRCTFEnv):
    """
    Slightly more set of actions than V1. Does take action costs into account.
    """
    def __init__(self, env_config: EnvConfig, emulation_config: EmulationConfig, checkpoint_dir : str):
        if env_config is None:
            render_config = PyCrCTFLevel7Base.render_conf()
            if emulation_config is None:
                emulation_config = PyCrCTFLevel7Base.emulation_config()
            emulation_config.ids_router = True
            emulation_config.ids_router_ip = PyCrCTFLevel7Base.router_ip()
            network_conf = PyCrCTFLevel7Base.network_conf()
            attacker_action_conf = PyCrCTFLevel7V2.attacker_actions_conf(num_nodes=PyCrCTFLevel7Base.num_nodes(),
                                                                subnet_mask=PyCrCTFLevel7Base.subnet_mask(),
                                                                hacker_ip=PyCrCTFLevel7Base.hacker_ip())
            defender_action_conf = PyCrCTFLevel7V2.defender_actions_conf(
                num_nodes=PyCrCTFLevel7Base.num_nodes(), subnet_mask=PyCrCTFLevel7Base.subnet_mask())
            env_config = PyCrCTFLevel7V2.env_config(network_conf=network_conf,
                                                    attacker_action_conf=attacker_action_conf,
                                                    defender_action_conf=defender_action_conf,
                                                    emulation_config=emulation_config, render_conf=render_config)
            env_config.attacker_alerts_coefficient = 1
            env_config.attacker_cost_coefficient = 1
            env_config.env_mode = EnvMode.emulation
            env_config.save_trajectories = False
            env_config.checkpoint_dir = checkpoint_dir
            env_config.checkpoint_freq = 1000
        super().__init__(env_config=env_config)


# -------- Version 3 ------------

class PyCRCTFLevel7Emulation3Env(PyCRCTFEnv):
    """
    Slightly more set of actions than V2. Does not take action costs into account.
    """
    def __init__(self, env_config: EnvConfig, emulation_config: EmulationConfig, checkpoint_dir : str):
        if env_config is None:
            render_config = PyCrCTFLevel7Base.render_conf()
            if emulation_config is None:
                emulation_config = PyCrCTFLevel7Base.emulation_config()
            emulation_config.ids_router = True
            emulation_config.ids_router_ip = PyCrCTFLevel7Base.router_ip()
            network_conf = PyCrCTFLevel7Base.network_conf()
            attacker_action_conf = PyCrCTFLevel7V3.attacker_actions_conf(num_nodes=PyCrCTFLevel7Base.num_nodes(),
                                                                subnet_mask=PyCrCTFLevel7Base.subnet_mask(),
                                                                hacker_ip=PyCrCTFLevel7Base.hacker_ip())
            defender_action_conf = PyCrCTFLevel7V3.defender_actions_conf(
                num_nodes=PyCrCTFLevel7Base.num_nodes(), subnet_mask=PyCrCTFLevel7Base.subnet_mask())
            env_config = PyCrCTFLevel7V3.env_config(network_conf=network_conf,
                                                    attacker_action_conf=attacker_action_conf,
                                                    defender_action_conf=defender_action_conf,
                                                    emulation_config=emulation_config, render_conf=render_config)
            env_config.attacker_alerts_coefficient = 1
            env_config.attacker_cost_coefficient = 0
            env_config.env_mode = EnvMode.emulation
            env_config.save_trajectories = False
            env_config.checkpoint_dir = checkpoint_dir
            env_config.checkpoint_freq = 1000
        super().__init__(env_config=env_config)


# -------- Version 3 with costs------------

class PyCRCTFLevel7EmulationWithCosts3Env(PyCRCTFEnv):
    """
    Slightly more set of actions than V2. Does take action costs into account.
    """
    def __init__(self, env_config: EnvConfig, emulation_config: EmulationConfig, checkpoint_dir : str):
        if env_config is None:
            render_config = PyCrCTFLevel7Base.render_conf()
            if emulation_config is None:
                emulation_config = PyCrCTFLevel7Base.emulation_config()
            emulation_config.ids_router = True
            emulation_config.ids_router_ip = PyCrCTFLevel7Base.router_ip()
            network_conf = PyCrCTFLevel7Base.network_conf()
            attacker_action_conf = PyCrCTFLevel7V3.attacker_actions_conf(num_nodes=PyCrCTFLevel7Base.num_nodes(),
                                                                subnet_mask=PyCrCTFLevel7Base.subnet_mask(),
                                                                hacker_ip=PyCrCTFLevel7Base.hacker_ip())
            defender_action_conf = PyCrCTFLevel7V3.defender_actions_conf(
                num_nodes=PyCrCTFLevel7Base.num_nodes(), subnet_mask=PyCrCTFLevel7Base.subnet_mask())
            env_config = PyCrCTFLevel7V3.env_config(network_conf=network_conf,
                                                    attacker_action_conf=attacker_action_conf,
                                                    defender_action_conf=defender_action_conf,
                                                    emulation_config=emulation_config, render_conf=render_config)
            env_config.attacker_alerts_coefficient = 1
            env_config.attacker_cost_coefficient = 1
            env_config.env_mode = EnvMode.emulation
            env_config.save_trajectories = False
            env_config.checkpoint_dir = checkpoint_dir
            env_config.checkpoint_freq = 1000
        super().__init__(env_config=env_config)


# -------- Version 4 ------------

class PyCRCTFLevel7Emulation4Env(PyCRCTFEnv):
    """
    Slightly more set of actions than V3. Does not take action costs into account.
    """
    def __init__(self, env_config: EnvConfig, emulation_config: EmulationConfig, checkpoint_dir : str):
        if env_config is None:
            render_config = PyCrCTFLevel7Base.render_conf()
            if emulation_config is None:
                emulation_config = PyCrCTFLevel7Base.emulation_config()
            emulation_config.ids_router = True
            emulation_config.ids_router_ip = PyCrCTFLevel7Base.router_ip()
            network_conf = PyCrCTFLevel7Base.network_conf()
            attacker_action_conf = PyCrCTFLevel7V4.attacker_actions_conf(num_nodes=PyCrCTFLevel7Base.num_nodes(),
                                                                subnet_mask=PyCrCTFLevel7Base.subnet_mask(),
                                                                hacker_ip=PyCrCTFLevel7Base.hacker_ip())
            defender_action_conf = PyCrCTFLevel7V4.defender_actions_conf(
                num_nodes=PyCrCTFLevel7Base.num_nodes(), subnet_mask=PyCrCTFLevel7Base.subnet_mask())
            env_config = PyCrCTFLevel7V4.env_config(network_conf=network_conf,
                                                    attacker_action_conf=attacker_action_conf,
                                                    defender_action_conf=defender_action_conf,
                                                    emulation_config=emulation_config, render_conf=render_config)
            env_config.attacker_alerts_coefficient = 1
            env_config.attacker_cost_coefficient = 0
            env_config.env_mode = EnvMode.emulation
            env_config.save_trajectories = False
            env_config.checkpoint_dir = checkpoint_dir
            env_config.checkpoint_freq = 1000
        super().__init__(env_config=env_config)


# -------- Version 4 with costs------------

class PyCRCTFLevel7EmulationWithCosts4Env(PyCRCTFEnv):
    """
    Slightly more set of actions than V3. Does take action costs into account.
    """
    def __init__(self, env_config: EnvConfig, emulation_config: EmulationConfig, checkpoint_dir : str):
        if env_config is None:
            render_config = PyCrCTFLevel7Base.render_conf()
            if emulation_config is None:
                emulation_config = PyCrCTFLevel7Base.emulation_config()
            emulation_config.ids_router = True
            emulation_config.ids_router_ip = PyCrCTFLevel7Base.router_ip()
            network_conf = PyCrCTFLevel7Base.network_conf()
            attacker_action_conf = PyCrCTFLevel7V4.attacker_actions_conf(num_nodes=PyCrCTFLevel7Base.num_nodes(),
                                                                subnet_mask=PyCrCTFLevel7Base.subnet_mask(),
                                                                hacker_ip=PyCrCTFLevel7Base.hacker_ip())
            defender_action_conf = PyCrCTFLevel7V4.defender_actions_conf(
                num_nodes=PyCrCTFLevel7Base.num_nodes(), subnet_mask=PyCrCTFLevel7Base.subnet_mask())
            env_config = PyCrCTFLevel7V4.env_config(network_conf=network_conf,
                                                    attacker_action_conf=attacker_action_conf,
                                                    defender_action_conf=defender_action_conf,
                                                    emulation_config=emulation_config, render_conf=render_config)
            env_config.attacker_alerts_coefficient = 1
            env_config.attacker_cost_coefficient = 1
            env_config.env_mode = EnvMode.emulation
            env_config.save_trajectories = False
            env_config.checkpoint_dir = checkpoint_dir
            env_config.checkpoint_freq = 1000
        super().__init__(env_config=env_config)
