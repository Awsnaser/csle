import numpy as np
import torch
import random
from csle_common.metastore.metastore_facade import MetastoreFacade
from gym_csle_cyborg.envs.cyborg_scenario_two_wrapper import CyborgScenarioTwoWrapper
from gym_csle_cyborg.dao.csle_cyborg_wrapper_config import CSLECyborgWrapperConfig
from gym_csle_cyborg.dao.red_agent_type import RedAgentType
from csle_common.dao.training.ppo_policy import PPOPolicy
from csle_common.dao.training.player_type import PlayerType

if __name__ == '__main__':
    # ppo_policy = MetastoreFacade.get_ppo_policy(id=58)
    ppo_policy = PPOPolicy(model=None, simulation_name="",
                           save_path="/tmp/csle/ppo_test_1707078811.4761195/ppo_model1630_1707115775.1994205.zip",
                           player_type=PlayerType.DEFENDER, actions=[], states=[], experiment_config=None, avg_R=0)
    config = CSLECyborgWrapperConfig(maximum_steps=100, gym_env_name="",
                                     save_trace=False, reward_shaping=False, scenario=2,
                                     red_agent_type=RedAgentType.B_LINE_AGENT)
    env = CyborgScenarioTwoWrapper(config=config)
    num_evaluations = 10000
    max_horizon = 100
    returns = []
    seed = 215125
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print("Starting policy evaluation")
    for i in range(num_evaluations):
        o, _ = env.reset()
        R = 0
        t = 0
        while t < max_horizon:
            a = ppo_policy.action(o=o)
            o, r, done, _, info = env.step(a)
            R += r
            t += 1
        returns.append(R)
        print(f"{i}/{num_evaluations}, avg R: {np.mean(returns)}, R: {R}")
