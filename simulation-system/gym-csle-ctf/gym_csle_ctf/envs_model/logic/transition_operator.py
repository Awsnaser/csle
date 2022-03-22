from typing import Tuple
from csle_common.dao.network.env_mode import EnvMode
from csle_common.envs_model.logic.base_transition_operator import BaseTransitionOperator
from gym_csle_ctf.dao.network.env_state import EnvState
from gym_csle_ctf.dao.network.env_config import CSLEEnvConfig
from gym_csle_ctf.envs_model.logic.emulation.emulation_middleware import EmulationMiddleware
from gym_csle_ctf.envs_model.logic.simulation.simulator import Simulator
from gym_csle_ctf.dao.action.attacker.attacker_action import AttackerAction
from gym_csle_ctf.dao.action.defender.defender_action import DefenderAction


class TransitionOperator(BaseTransitionOperator):
    """
    Implements the transition operator of the MDP/Markov Game:

    (s, a) --> (s', r)
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def attacker_transition(s : EnvState, attacker_action : AttackerAction, env_config : CSLEEnvConfig) -> \
            Tuple[EnvState, float, bool]:
        """
        Implements the transition operator of the MDP/Markov Game for attack actions,
        supporting both simulation and emulation mode
        (s, a) --> (s', r)

        :param s: the current state
        :param attacker_action: the attacker's action of the transition
        :param env_config: the environment config
        :return: s', r, done
        """
        if env_config.env_mode == EnvMode.SIMULATION:
            return Simulator.attacker_transition(s=s, attacker_action=attacker_action, env_config=env_config)
        elif env_config.env_mode == EnvMode.EMULATION or env_config.env_mode == EnvMode.GENERATED_SIMULATION:
            return EmulationMiddleware.attacker_transition(s=s, attacker_action=attacker_action, env_config=env_config)
            # try:
            #     return emulationMiddleware.transition(s=s,a=a,env_config=env_config)
            # except Exception as e:
            #     print("Could not execute action on the emulation, using simulation instead. \n Error:{}".format(str(e)))
            #     return Simulator.transition(s=s, a=a, env_config=env_config)

        else:
            raise ValueError("Invalid environment mode")

    @staticmethod
    def defender_transition(s: EnvState, defender_action: DefenderAction, env_config: CSLEEnvConfig,
                            attacker_action : AttackerAction = None) \
            -> Tuple[EnvState, float, bool]:
        """
        Implements the transition operator of the MDP/Markov Game for defense actions,
        supporting both simulation and emulation mode
        (s, a) --> (s', r)

        :param s: the current state
        :param defender_action: the defender's action of the transition
        :param env_config: the environment config
        :param attacker_action: previous attacker action
        :return: s', r, done
        """
        if env_config.env_mode == EnvMode.SIMULATION:
            return Simulator.defender_transition(s=s, defender_action=defender_action, env_config=env_config,
                                          attacker_action=attacker_action)
        elif env_config.env_mode == EnvMode.EMULATION or env_config.env_mode == EnvMode.GENERATED_SIMULATION:
            return EmulationMiddleware.defender_transition(s=s, defender_action=defender_action,
                                                           attacker_action=attacker_action,
                                                           env_config=env_config)
        else:
            raise ValueError("Invalid environment mode")