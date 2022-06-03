import csle_common.constants.constants as constants
from csle_common.dao.training.experiment_config import ExperimentConfig
from csle_common.metastore.metastore_facade import MetastoreFacade
from csle_common.dao.training.agent_type import AgentType
from csle_common.dao.training.hparam import HParam
from csle_common.dao.training.player_type import PlayerType
from csle_agents.agents.random_search.random_search_agent import RandomSearchAgent
import csle_agents.constants.constants as agents_constants
from gym_csle_stopping_game.util.stopping_game_util import StoppingGameUtil


if __name__ == '__main__':
    emulation_env_config = MetastoreFacade.get_emulation_by_name("csle-level9-001")
    simulation_env_config = MetastoreFacade.get_simulation_by_name("csle-stopping-pomdp-defender-001")
    experiment_config = ExperimentConfig(
        output_dir=f"{constants.LOGGING.DEFAULT_LOG_DIR}random_search_test", title="Random search test",
        random_seeds=[399, 98912,999,555],
        agent_type=AgentType.RANDOM_SEARCH,
        log_every=1,
        hparams={
            agents_constants.RANDOM_SEARCH.N: HParam(value=50, name=agents_constants.T_SPSA.N,
                                              descr="the number of training iterations"),
            agents_constants.RANDOM_SEARCH.DELTA: HParam(value=0.5, name=agents_constants.RANDOM_SEARCH.DELTA,
                                                     descr="the step size for random perturbations"),
            agents_constants.RANDOM_SEARCH.L: HParam(value=3, name="L", descr="the number of stop actions"),
            agents_constants.COMMON.EVAL_BATCH_SIZE: HParam(value=100, name=agents_constants.COMMON.EVAL_BATCH_SIZE,
                                                            descr="number of iterations to evaluate theta"),
            agents_constants.RANDOM_SEARCH.THETA1: HParam(value=[-3,-3,-3], name=agents_constants.RANDOM_SEARCH.THETA1,
                                                   descr="initial thresholds"),
            agents_constants.COMMON.SAVE_EVERY: HParam(value=1000, name=agents_constants.COMMON.SAVE_EVERY,
                                                       descr="how frequently to save the model"),
            agents_constants.COMMON.CONFIDENCE_INTERVAL: HParam(
                value=0.95, name=agents_constants.COMMON.CONFIDENCE_INTERVAL,
                descr="confidence interval"),
            agents_constants.COMMON.MAX_ENV_STEPS: HParam(
                value=500, name=agents_constants.COMMON.MAX_ENV_STEPS,
                descr="maximum number of steps in the environment (for envs with infinite horizon generally)"),
            agents_constants.COMMON.RUNNING_AVERAGE: HParam(
                value=100, name=agents_constants.COMMON.RUNNING_AVERAGE,
                descr="the number of samples to include when computing the running avg")
        },
        player_type=PlayerType.DEFENDER, player_idx=0
    )
    agent = RandomSearchAgent(emulation_env_config=emulation_env_config, simulation_env_config=simulation_env_config,
                       experiment_config=experiment_config)
    simulation_env_config.simulation_env_input_config.stopping_game_config.R = list(StoppingGameUtil.reward_tensor(
        R_INT=-1, R_COST=-2, R_SLA=0, R_ST=2, L=3))
    experiment_execution = agent.train()
    MetastoreFacade.save_experiment_execution(experiment_execution)
    for policy in experiment_execution.result.policies.values():
        MetastoreFacade.save_multi_threshold_stopping_policy(multi_threshold_stopping_policy=policy)
