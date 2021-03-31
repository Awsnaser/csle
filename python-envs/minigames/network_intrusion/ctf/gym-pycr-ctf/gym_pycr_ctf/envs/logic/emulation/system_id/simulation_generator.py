import numpy as np
from gym_pycr_ctf.dao.network.env_config import EnvConfig
from gym_pycr_ctf.dao.network.network_config import NetworkConfig
from gym_pycr_ctf.envs.logic.exploration.exploration_policy import ExplorationPolicy
from gym_pycr_ctf.envs.logic.common.env_dynamics_util import EnvDynamicsUtil

class SimulationGenerator:
    """
    Class for with functions for running an exploration policy in an environment to build a level_1 model
    that later can be used for simulations
    """

    @staticmethod
    def explore(exp_policy: ExplorationPolicy, env_config: EnvConfig, env, render: bool = False) -> np.ndarray:
        """
        Explores the environment to generate trajectories that can be used to learn a dynamics model

        :param exp_policy: the exploration policy to use
        :param env_config: the env config
        :param env: the env to explore
        :param render: whether to render the env or not
        :return: The final observation
        """
        env.reset()
        done = False
        step = 0
        while not done and step < env_config.attacker_max_exploration_steps:
            action = exp_policy.action(env=env, filter_illegal=env_config.attacker_exploration_filter_illegal)
            obs, reward, done, info = env.step(action)
            step +=1
            if render:
                env.render()
        if step >= env_config.attacker_max_exploration_steps:
            print("maximum exploration steps reached")
        return obs

    @staticmethod
    def build_model(exp_policy: ExplorationPolicy, env_config: EnvConfig, env, render: bool = False) -> NetworkConfig:
        print("Starting Exploration Phase to Gather Data to Create Simulation")
        aggregated_observation = env.env_state.attacker_obs_state.copy()
        for i in range(env_config.attacker_max_exploration_trajectories):
            print("Collecting trajectory {}/{}".format(i, env_config.attacker_max_exploration_trajectories))
            SimulationGenerator.explore(exp_policy = exp_policy, env_config=env_config, env=env, render=render)
            observation = env.env_state.attacker_obs_state
            aggregated_observation = EnvDynamicsUtil.merge_complete_obs_state(old_obs_state=aggregated_observation,
                                                                              new_obs_state=observation,
                                                                              env_config=env_config)
        num_machines = len(aggregated_observation.machines)
        num_vulnerabilities = sum(list(map(lambda x: len(x.cve_vulns) + len(x.osvdb_vulns), aggregated_observation.machines)))
        num_credentials = sum(list(map(lambda x: len(x.shell_access_credentials), aggregated_observation.machines)))
        print("Exploration completed, found {} machines, {} vulnerabilities, {} credentials".format(
            num_machines, num_vulnerabilities, num_credentials))
        nodes = list(map(lambda x: x.to_node(), aggregated_observation.machines))
        env_config.network_conf.nodes = nodes
        env_config.network_conf.agent_reachable = aggregated_observation.agent_reachable
        env.cleanup()
        return env_config.network_conf, aggregated_observation