from gym_pycr_ctf.dao.network.env_mode import EnvMode
from gym_pycr_ctf.dao.network.env_config import EnvConfig
from gym_pycr_ctf.dao.network.emulation_config import EmulationConfig
from gym_pycr_ctf.envs.pycr_ctf_env import PyCRCTFEnv
from gym_pycr_ctf.envs_model.config.level_2.pycr_ctf_level_2_base import PyCrCTFLevel2Base
from gym_pycr_ctf.envs_model.config.level_2.pycr_ctf_level_2_v1 import PyCrCTFLevel2V1


# -------- Base Version (for testing) ------------
class PyCRCTFLevel2SimBaseEnv(PyCRCTFEnv):
    """
    Base version with all set of actions
    """
    def __init__(self, env_config: EnvConfig, emulation_config: EmulationConfig, checkpoint_dir : str):
        if env_config is None:
            render_config = PyCrCTFLevel2Base.render_conf()
            network_conf = PyCrCTFLevel2Base.network_conf()
            attacker_action_conf = PyCrCTFLevel2Base.attacker_all_actions_conf(num_nodes=PyCrCTFLevel2Base.num_nodes(),
                                                                      subnet_mask=PyCrCTFLevel2Base.subnet_mask(),
                                                                      hacker_ip=PyCrCTFLevel2Base.hacker_ip())
            defender_action_conf = PyCrCTFLevel2Base.defender_all_actions_conf(
                num_nodes=PyCrCTFLevel2Base.num_nodes(), subnet_mask=PyCrCTFLevel2Base.subnet_mask())
            env_config = PyCrCTFLevel2Base.env_config(network_conf=network_conf,
                                                      attacker_action_conf=attacker_action_conf,
                                                      defender_action_conf=defender_action_conf,
                                                      emulation_config=None, render_conf=render_config)
            env_config.attacker_alerts_coefficient = 1
            env_config.attacker_cost_coefficient = 0
            env_config.simulate_detection = True
            env_config.save_trajectories = False
            # env_config.simulate_detection = False
            env_config.env_mode = EnvMode.SIMULATION
            env_config.checkpoint_dir = checkpoint_dir
            env_config.checkpoint_freq = 1000
        super().__init__(env_config=env_config)

# -------- Simulations ------------

# -------- Version 1 ------------
class PyCRCTFLevel2Sim1Env(PyCRCTFEnv):
    """
    The simplest possible configuration, minimal set of actions. Does not take action costs into account.
    """
    def __init__(self, env_config: EnvConfig, emulation_config: EmulationConfig, checkpoint_dir : str):
        if env_config is None:
            render_config = PyCrCTFLevel2Base.render_conf()
            network_conf = PyCrCTFLevel2Base.network_conf()
            attacker_action_conf = PyCrCTFLevel2V1.attacker_actions_conf(num_nodes=PyCrCTFLevel2Base.num_nodes(),
                                                                subnet_mask=PyCrCTFLevel2Base.subnet_mask(),
                                                                hacker_ip=PyCrCTFLevel2Base.hacker_ip())
            defender_action_conf = PyCrCTFLevel2V1.defender_actions_conf(
                num_nodes=PyCrCTFLevel2Base.num_nodes(), subnet_mask=PyCrCTFLevel2Base.subnet_mask())
            env_config = PyCrCTFLevel2V1.env_config(network_conf=network_conf,
                                                    attacker_action_conf=attacker_action_conf,
                                                    defender_action_conf=defender_action_conf,
                                                    emulation_config=None, render_conf=render_config)
            env_config.attacker_alerts_coefficient = 1
            env_config.attacker_cost_coefficient = 0
            env_config.save_trajectories = False
            env_config.simulate_detection = False
            env_config.env_mode = EnvMode.SIMULATION
            env_config.checkpoint_dir = checkpoint_dir
            env_config.checkpoint_freq = 1000
        super().__init__(env_config=env_config)

# -------- Version 1, Costs ------------
class PyCRCTFLevel2SimWithCosts1Env(PyCRCTFEnv):
    """
    The simplest possible configuration, minimal set of actions. Does take action costs into account.
    """
    def __init__(self, env_config: EnvConfig, emulation_config: EmulationConfig, checkpoint_dir : str):
        if env_config is None:
            render_config = PyCrCTFLevel2Base.render_conf()
            network_conf = PyCrCTFLevel2Base.network_conf()
            attacker_action_conf = PyCrCTFLevel2V1.attacker_actions_conf(num_nodes=PyCrCTFLevel2Base.num_nodes(),
                                                                subnet_mask=PyCrCTFLevel2Base.subnet_mask(),
                                                                hacker_ip=PyCrCTFLevel2Base.hacker_ip())
            defender_action_conf = PyCrCTFLevel2V1.defender_actions_conf(
                num_nodes=PyCrCTFLevel2Base.num_nodes(), subnet_mask=PyCrCTFLevel2Base.subnet_mask())
            env_config = PyCrCTFLevel2V1.env_config(network_conf=network_conf,
                                                    attacker_action_conf=attacker_action_conf,
                                                    defender_action_conf=defender_action_conf,
                                                    emulation_config=None, render_conf=render_config)
            env_config.attacker_alerts_coefficient = 1
            env_config.attacker_cost_coefficient = 1
            env_config.save_trajectories = False
            env_config.simulate_detection = False
            env_config.env_mode = EnvMode.SIMULATION
            env_config.checkpoint_dir = checkpoint_dir
            env_config.checkpoint_freq = 1000
        super().__init__(env_config=env_config)