from gym_csle_cyborg.envs.cyborg_scenario_two_wrapper import CyborgScenarioTwoWrapper
from gym_csle_cyborg.dao.csle_cyborg_wrapper_config import CSLECyborgWrapperConfig
from gym_csle_cyborg.dao.cyborg_wrapper_state import CyborgWrapperState
from gym_csle_cyborg.dao.red_agent_type import RedAgentType


class TestCSLECyborgVersionTwoWrapperSuite:
    """
    Test suite for csle_cyborg_config.py
    """

    def test_set_state_1(self) -> None:
        """
        Tests the set_state method

        :return: None
        """
        config = CSLECyborgWrapperConfig(maximum_steps=100, gym_env_name="", save_trace=False, reward_shaping=True,
                                         scenario=2, red_agent_type=RedAgentType.B_LINE_AGENT)
        env = CyborgScenarioTwoWrapper(config=config)
        state = CyborgWrapperState(
            s=[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
               [0, 0, 0, 0], [1, 0, 2, 0], [1, 1, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]],
            scan_state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
            op_server_restored=True,
            obs=[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                 [0, 0, 0, 0], [0, 0, 0, 0], [1, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            red_agent_state=2,
            privilege_escalation_detected=None,
            red_agent_target=9, red_action_targets={0: 0, 1: 9},
            attacker_observed_decoy=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            detected=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            malware_state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ssh_access=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            escalated=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            exploited=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            bline_base_jump=False
        )
        action = 1
        max_tries = 1000
        match = False
        i = 0
        while i < max_tries and not match:
            env.set_state(state)
            o, r, done, _, info = env.step(action)
            if env.last_obs == [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                                [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                                [0, 0, 0, 0]] and env.s == [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                                                            [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                                                            [1, 0, 2, 0], [1, 1, 2, 0], [1, 0, 0, 0], [1, 0, 0, 0],
                                                            [1, 0, 0, 0]]:
                match = True
            i += 1
        assert match

    def test_set_state_2(self) -> None:
        """
        Tests the set_state method

        :return: None
        """
        config = CSLECyborgWrapperConfig(maximum_steps=100, gym_env_name="", save_trace=False, reward_shaping=True,
                                         scenario=2, red_agent_type=RedAgentType.B_LINE_AGENT)
        env = CyborgScenarioTwoWrapper(config=config)
        state = CyborgWrapperState(
            s=[[0, 0, 0, 0], [1, 1, 0, 4], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
               [0, 0, 0, 0], [1, 0, 2, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 1, 2, 0]],
            scan_state=[0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            op_server_restored=True,
            obs=[[0, 0, 0, 0], [1, 2, 0, 4], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                 [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 1, 0]],
            red_agent_state=2,
            privilege_escalation_detected=None,
            red_agent_target=12, red_action_targets={0: 0, 1: 12, 2: 12, 3: 12, 4: 1, 5: 1},
            attacker_observed_decoy=[0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            detected=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            malware_state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ssh_access=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            escalated=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            exploited=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            bline_base_jump=False
        )
        action = 33
        max_tries = 1000
        match = False
        i = 0
        while i < max_tries and not match:
            env.set_state(state)
            o, r, done, _, info = env.step(action)
            if env.last_obs == [[0, 0, 0, 0], [0, 2, 0, 4], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                                [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                                [0, 1, 1, 1]] \
                    and env.s == [[0, 0, 0, 0], [1, 1, 0, 4], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                                  [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 2, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0],
                                  [1, 1, 2, 1]]:
                match = True
            i += 1
        assert match

    def test_set_state_3(self) -> None:
        """
        Tests the set_state method

        :return: None
        """
        config = CSLECyborgWrapperConfig(maximum_steps=100, gym_env_name="", save_trace=False, reward_shaping=True,
                                         scenario=2, red_agent_type=RedAgentType.B_LINE_AGENT)
        env = CyborgScenarioTwoWrapper(config=config)
        state = CyborgWrapperState(
            s=[[0, 0, 0, 0], [1, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
               [0, 0, 0, 0], [1, 0, 2, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 1, 2, 0], [1, 0, 0, 1]],
            scan_state=[0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            op_server_restored=True,
            obs=[[0, 0, 0, 0], [0, 2, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                 [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 1]],
            red_agent_state=5,
            privilege_escalation_detected=None,
            red_agent_target=1, red_action_targets={0: 0, 1: 11, 2: 11, 3: 11, 4: 1, 5: 1, 6: 1},
            attacker_observed_decoy=[0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            detected=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            malware_state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ssh_access=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            escalated=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            exploited=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            bline_base_jump=False
        )
        action = 27
        max_tries = 1000
        match = False
        i = 0
        while i < max_tries and not match:
            env.set_state(state)
            o, r, done, _, info = env.step(action)
            if env.last_obs == [[0, 0, 0, 0], [2, 2, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                                [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 1, 0],
                                [0, 0, 0, 1]] \
                    and env.s == [[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                                  [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 2, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 1, 2, 0],
                                  [1, 0, 0, 1]]:
                match = True
            i += 1
        assert match

    def test_set_state_4(self) -> None:
        """
        Tests the set_state method

        :return: None
        """
        config = CSLECyborgWrapperConfig(maximum_steps=100, gym_env_name="", save_trace=False, reward_shaping=True,
                                         scenario=2, red_agent_type=RedAgentType.B_LINE_AGENT)
        env = CyborgScenarioTwoWrapper(config=config)
        state = CyborgWrapperState(
            s=[[0, 0, 0, 0], [1, 1, 2, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
               [0, 0, 0, 0], [1, 0, 2, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 1, 2, 0], [1, 0, 0, 1]],
            scan_state=[0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            op_server_restored=True,
            obs=[[0, 0, 0, 0], [0, 2, 3, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                 [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 1]],
            red_agent_state=7,
            privilege_escalation_detected=None,
            red_agent_target=1, red_action_targets={0: 0, 1: 11, 2: 11, 3: 11, 4: 1, 5: 1, 6: 1},
            attacker_observed_decoy=[0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            detected=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            malware_state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ssh_access=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            escalated=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            exploited=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            bline_base_jump=False
        )
        action = 9
        max_tries = 1000
        match = False
        i = 0
        env.set_state(state)
        o, r, done, _, info = env.step(action)
        while i < max_tries and not match:
            env.set_state(state)
            o, r, done, _, info = env.step(action)
            if env.last_obs == [[0, 0, 0, 0], [0, 2, 3, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                                [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 1, 0],
                                [0, 0, 0, 1]] \
                    and env.s == [[1, 0, 0, 0], [1, 1, 2, 1], [1, 0, 0, 1], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                                  [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 2, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 1, 2, 0],
                                  [1, 0, 0, 1]]:
                match = True
            i += 1
        assert match

    def test_set_state_5(self) -> None:
        """
        Tests the set_state method

        :return: None
        """
        config = CSLECyborgWrapperConfig(maximum_steps=100, gym_env_name="", save_trace=False, reward_shaping=True,
                                         scenario=2, red_agent_type=RedAgentType.B_LINE_AGENT)
        env = CyborgScenarioTwoWrapper(config=config)
        state = CyborgWrapperState(
            s=[[1, 0, 0, 4], [1, 0, 0, 4], [1, 1, 0, 1], [1, 1, 2, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
               [1, 0, 0, 4], [1, 0, 2, 0], [1, 1, 2, 4], [1, 0, 0, 4], [1, 0, 0, 2], [1, 0, 0, 1]],
            scan_state=[0, 0, 1, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            op_server_restored=False,
            obs=[[0, 0, 0, 4], [0, 0, 0, 4], [0, 1, 0, 1], [0, 2, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                 [0, 0, 0, 4], [0, 0, 0, 0], [0, 1, 3, 4], [0, 0, 0, 4], [0, 0, 0, 2], [0, 0, 0, 1]],
            red_agent_state=11,
            privilege_escalation_detected=3,
            red_agent_target=7, red_action_targets={0: 0, 1: 9, 2: 9, 3: 9, 4: 2, 5: 2, 6: 2, 7: 1, 8: 3, 9: 3, 10: 3},
            attacker_observed_decoy=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            detected=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            malware_state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ssh_access=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            escalated=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            exploited=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            bline_base_jump=False
        )
        action = 2
        max_tries = 1000
        match = False
        i = 0
        env.set_state(state)
        o, r, done, _, info = env.step(action)
        while i < max_tries and not match:
            env.set_state(state)
            o, r, done, _, info = env.step(action)
            if env.last_obs == [[0, 0, 0, 4], [0, 0, 0, 4], [0, 1, 0, 1], [0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                                [0, 0, 0, 0], [0, 0, 0, 4], [0, 0, 0, 0], [0, 1, 3, 4], [0, 0, 0, 4], [0, 0, 0, 2],
                                [0, 0, 0, 1]] \
                    and env.s == [[1, 0, 0, 4], [1, 0, 0, 4], [1, 1, 0, 1], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                                  [0, 0, 0, 0], [1, 0, 0, 4], [1, 0, 2, 0], [1, 1, 2, 4], [1, 0, 0, 4], [1, 0, 0, 2],
                                  [1, 0, 0, 1]] and env.red_agent_state == 12:
                match = True
            i += 1
        assert match

    def test_set_state_6(self) -> None:
        """
        Tests the set_state method

        :return: None
        """
        config = CSLECyborgWrapperConfig(maximum_steps=100, gym_env_name="", save_trace=False, reward_shaping=True,
                                         scenario=2, red_agent_type=RedAgentType.B_LINE_AGENT)
        env = CyborgScenarioTwoWrapper(config=config)
        state = CyborgWrapperState(
            s=[[1, 0, 0, 0], [1, 1, 0, 0], [1, 0, 0, 1], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
               [0, 0, 0, 0], [1, 0, 2, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 1, 2, 0], [1, 0, 0, 1]],
            scan_state=[0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            op_server_restored=True,
            obs=[[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                 [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 1]],
            red_agent_state=5,
            privilege_escalation_detected=None,
            red_agent_target=1, red_action_targets={0: 0, 1: 11, 2: 11, 3: 11, 4: 1, 5: 1, 6: 1, 7: 1, 8: 3, 9: 3},
            attacker_observed_decoy=[0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            detected=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            malware_state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ssh_access=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            escalated=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            exploited=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            bline_base_jump=False
        )
        action = 27
        max_tries = 1000
        match = False
        i = 0
        while i < max_tries and not match:
            env.set_state(state)
            o, r, done, _, info = env.step(action)
            if env.last_obs == [[0, 0, 0, 0], [2, 1, 1, 1], [0, 0, 0, 1], [0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                                [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 1, 0],
                                [0, 0, 0, 1]] \
                    and env.s == [[1, 0, 0, 0], [1, 1, 1, 1], [1, 0, 0, 1], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                                  [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 2, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 1, 2, 0],
                                  [1, 0, 0, 1]]:
                match = True
            i += 1
        assert match

    def test_set_state_7(self) -> None:
        """
        Tests the set_state method

        :return: None
        """
        config = CSLECyborgWrapperConfig(maximum_steps=100, gym_env_name="", save_trace=False, reward_shaping=True,
                                         scenario=2, red_agent_type=RedAgentType.B_LINE_AGENT)
        env = CyborgScenarioTwoWrapper(config=config)
        state = CyborgWrapperState(
            s=[[1, 0, 0, 0], [1, 1, 2, 1], [1, 0, 0, 1], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
               [0, 0, 0, 0], [1, 0, 2, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 1, 2, 0], [1, 0, 0, 1]],
            scan_state=[0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            op_server_restored=True,
            obs=[[0, 0, 0, 0], [0, 1, 1, 1], [0, 0, 0, 1], [0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                 [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 1]],
            red_agent_state=7,
            privilege_escalation_detected=None,
            red_agent_target=1, red_action_targets={0: 0, 1: 11, 2: 11, 3: 11, 4: 1, 5: 1, 6: 1, 7: 1, 8: 3, 9: 3},
            attacker_observed_decoy=[0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            detected=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            malware_state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ssh_access=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            escalated=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            exploited=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            bline_base_jump=False
        )
        action = 27
        max_tries = 1000
        match = False
        i = 0
        while i < max_tries and not match:
            env.set_state(state)
            o, r, done, _, info = env.step(action)
            if env.last_obs == [[0, 0, 0, 0], [0, 1, 1, 2], [0, 0, 0, 1], [0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                                [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 1, 0],
                                [0, 0, 0, 1]] \
                    and env.s == [[1, 0, 0, 0], [1, 1, 2, 2], [1, 0, 0, 1], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                                  [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 2, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 1, 2, 0],
                                  [1, 0, 0, 1]]:
                match = True
            i += 1
        assert match

    def test_set_state_8(self) -> None:
        """
        Tests the set_state method

        :return: None
        """
        config = CSLECyborgWrapperConfig(maximum_steps=100, gym_env_name="", save_trace=False, reward_shaping=True,
                                         scenario=2, red_agent_type=RedAgentType.B_LINE_AGENT)
        env = CyborgScenarioTwoWrapper(config=config)
        state = CyborgWrapperState(
            s=[[0, 0, 0, 1], [0, 0, 0, 4], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
               [0, 0, 0, 1], [1, 0, 2, 0], [1, 0, 0, 0], [1, 1, 2, 0], [1, 0, 0, 0], [1, 0, 0, 0]],
            scan_state=[0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            op_server_restored=True,
            obs=[[0, 0, 0, 1], [0, 0, 0, 4], [0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                 [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            red_agent_state=5,
            privilege_escalation_detected=None,
            red_agent_target=2, red_action_targets={0: 0, 1: 10, 2: 10, 3: 10, 4: 2, 5: 2, 6: 2},
            attacker_observed_decoy=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            detected=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            malware_state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ssh_access=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            escalated=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            exploited=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            bline_base_jump=False
        )
        action = 28
        max_tries = 1000
        match = False
        i = 0
        while i < max_tries and not match:
            env.set_state(state)
            o, r, done, _, info = env.step(action)
            if env.last_obs == [[0, 0, 0, 1], [0, 0, 0, 4], [2, 2, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                                [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 0],
                                [0, 0, 0, 0]] \
                    and env.s == [[0, 0, 0, 1], [0, 0, 0, 4], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                                  [0, 0, 0, 0], [0, 0, 0, 1], [1, 0, 2, 0], [1, 0, 0, 0], [1, 1, 2, 0], [1, 0, 0, 0],
                                  [1, 0, 0, 0]]:
                match = True
            i += 1
        assert match

    def test_set_state_9(self) -> None:
        """
        Tests the set_state method

        :return: None
        """
        config = CSLECyborgWrapperConfig(maximum_steps=100, gym_env_name="", save_trace=False, reward_shaping=True,
                                         scenario=2, red_agent_type=RedAgentType.B_LINE_AGENT)
        env = CyborgScenarioTwoWrapper(config=config)
        state = CyborgWrapperState(
            s=[[0, 0, 0, 1], [0, 0, 0, 4], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
               [0, 0, 0, 1], [1, 0, 2, 0], [1, 0, 0, 0], [1, 1, 2, 0], [1, 0, 0, 0], [1, 0, 0, 0]],
            scan_state=[0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            op_server_restored=True,
            obs=[[0, 0, 0, 1], [0, 0, 0, 4], [2, 2, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                 [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            red_agent_state=6,
            privilege_escalation_detected=None,
            red_agent_target=2, red_action_targets={0: 0, 1: 10, 2: 10, 3: 10, 4: 2, 5: 2, 6: 2},
            attacker_observed_decoy=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            detected=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            malware_state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ssh_access=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            escalated=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            exploited=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            bline_base_jump=False
        )
        action = 28
        max_tries = 1000
        match = False
        i = 0
        env.set_state(state)
        o, r, done, _, info = env.step(action)
        while i < max_tries and not match:
            env.set_state(state)
            o, r, done, _, info = env.step(action)
            if env.last_obs == [[0, 0, 0, 1], [0, 0, 0, 4], [0, 2, 3, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                                [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 0],
                                [0, 0, 0, 0]] \
                    and env.s == [[0, 0, 0, 1], [0, 0, 0, 4], [1, 1, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                                  [0, 0, 0, 0], [0, 0, 0, 1], [1, 0, 2, 0], [1, 0, 0, 0], [1, 1, 2, 0], [1, 0, 0, 0],
                                  [1, 0, 0, 0]]:
                match = True
            i += 1
        assert match

    def test_set_state_10(self) -> None:
        """
        Tests the set_state method

        :return: None
        """
        config = CSLECyborgWrapperConfig(maximum_steps=100, gym_env_name="", save_trace=False, reward_shaping=True,
                                         scenario=2, red_agent_type=RedAgentType.B_LINE_AGENT)
        env = CyborgScenarioTwoWrapper(config=config)
        state = CyborgWrapperState(
            s=[[0, 0, 0, 1], [0, 0, 0, 4], [1, 1, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
               [0, 0, 0, 1], [1, 0, 2, 0], [1, 0, 0, 0], [1, 1, 2, 0], [1, 0, 0, 0], [1, 0, 0, 0]],
            scan_state=[0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            op_server_restored=True,
            obs=[[0, 0, 0, 1], [0, 0, 0, 4], [0, 2, 3, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                 [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            red_agent_state=5,
            privilege_escalation_detected=None,
            red_agent_target=2, red_action_targets={0: 0, 1: 10, 2: 10, 3: 10, 4: 2, 5: 2, 6: 2},
            attacker_observed_decoy=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            detected=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            malware_state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ssh_access=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            escalated=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            exploited=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            bline_base_jump=False
        )
        action = 1
        max_tries = 1000
        match = False
        i = 0
        while i < max_tries and not match:
            env.set_state(state)
            o, r, done, _, info = env.step(action)
            if env.last_obs == [[0, 0, 0, 1], [0, 0, 0, 4], [2, 2, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                                [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 0],
                                [0, 0, 0, 0]] \
                    and env.s == [[0, 0, 0, 1], [0, 0, 0, 4], [1, 1, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                                  [0, 0, 0, 0], [0, 0, 0, 1], [1, 0, 2, 0], [1, 0, 0, 0], [1, 1, 2, 0], [1, 0, 0, 0],
                                  [1, 0, 0, 0]]:
                match = True
            i += 1
        assert match

    def test_set_state_11(self) -> None:
        """
        Tests the set_state method

        :return: None
        """
        config = CSLECyborgWrapperConfig(maximum_steps=100, gym_env_name="", save_trace=False, reward_shaping=True,
                                         scenario=2, red_agent_type=RedAgentType.B_LINE_AGENT)
        env = CyborgScenarioTwoWrapper(config=config)
        state = CyborgWrapperState(
            s=[[0, 0, 0, 3], [1, 1, 2, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
               [0, 0, 0, 3], [1, 0, 2, 0], [1, 0, 0, 4], [1, 0, 0, 3], [1, 0, 0, 2], [1, 1, 2, 0]],
            scan_state=[0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            op_server_restored=False,
            obs=[[0, 0, 0, 3], [0, 2, 3, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                 [0, 0, 0, 3], [0, 0, 0, 0], [0, 0, 0, 4], [0, 0, 0, 3], [0, 0, 0, 2], [0, 1, 1, 0]],
            red_agent_state=7,
            privilege_escalation_detected=None,
            red_agent_target=1, red_action_targets={0: 0, 1: 12, 2: 12, 3: 12, 4: 1, 5: 1, 6: 1},
            attacker_observed_decoy=[0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            detected=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            malware_state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ssh_access=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            escalated=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            exploited=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            bline_base_jump=False
        )
        action = 27
        max_tries = 1000
        match = False
        i = 0
        while i < max_tries and not match:
            env.set_state(state)
            o, r, done, _, info = env.step(action)
            if env.last_obs == [[0, 0, 0, 3], [0, 2, 3, 2], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0],
                                [0, 0, 0, 0], [0, 0, 0, 3], [0, 0, 0, 0], [0, 0, 0, 4], [0, 0, 0, 3], [0, 0, 0, 2],
                                [0, 1, 1, 0]] \
                    and env.s == [[1, 0, 0, 3], [1, 1, 2, 2], [1, 0, 0, 1], [1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0],
                                  [0, 0, 0, 0], [0, 0, 0, 3], [1, 0, 2, 0], [1, 0, 0, 4], [1, 0, 0, 3], [1, 0, 0, 2],
                                  [1, 1, 2, 0]]:
                match = True
            i += 1
        assert match

    def test_set_state_12(self) -> None:
        """
        Tests the set_state method

        :return: None
        """
        config = CSLECyborgWrapperConfig(maximum_steps=100, gym_env_name="", save_trace=False, reward_shaping=True,
                                         scenario=2, red_agent_type=RedAgentType.B_LINE_AGENT)
        env = CyborgScenarioTwoWrapper(config=config)
        state = CyborgWrapperState(
            s=[[0, 0, 0, 0], [1, 1, 1, 4], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
               [0, 0, 0, 0], [1, 0, 2, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 1, 2, 0], [1, 0, 0, 1]],
            scan_state=[0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            op_server_restored=True,
            obs=[[0, 0, 0, 0], [2, 2, 1, 4], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                 [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 1]],
            red_agent_state=6,
            privilege_escalation_detected=None,
            red_agent_target=1, red_action_targets={0: 0, 1: 11, 2: 11, 3: 11, 4: 1, 5: 1},
            attacker_observed_decoy=[0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            detected=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            malware_state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ssh_access=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            escalated=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            exploited=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            bline_base_jump=False
        )
        action = 8
        max_tries = 1000
        match = False
        i = 0
        while i < max_tries and not match:
            env.set_state(state)
            o, r, done, _, info = env.step(action)
            if env.last_obs == [[0, 0, 0, 0], [0, 2, 3, 4], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                                [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 1, 0],
                                [0, 0, 0, 1]] \
                    and env.s == [[0, 0, 0, 0], [1, 1, 2, 4], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                                  [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 2, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 1, 2, 0],
                                  [1, 0, 0, 1]]:
                match = True
            i += 1
        assert match

    def test_set_state_13(self) -> None:
        """
        Tests the set_state method

        :return: None
        """
        config = CSLECyborgWrapperConfig(maximum_steps=100, gym_env_name="", save_trace=False, reward_shaping=True,
                                         scenario=2, red_agent_type=RedAgentType.B_LINE_AGENT)
        env = CyborgScenarioTwoWrapper(config=config)
        state = CyborgWrapperState(
            s=[[0, 0, 0, 0], [1, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
               [0, 0, 0, 0], [1, 0, 2, 0], [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 0, 0], [1, 1, 2, 1]],
            scan_state=[0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            op_server_restored=True,
            obs=[[0, 0, 0, 0], [0, 2, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                 [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 1, 1, 1]],
            red_agent_state=5,
            privilege_escalation_detected=None,
            red_agent_target=1, red_action_targets={0: 0, 1: 12, 2: 12, 3: 12, 4: 1, 5: 1, 6: 1},
            attacker_observed_decoy=[0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            detected=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            malware_state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ssh_access=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            escalated=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            exploited=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            bline_base_jump=False
        )
        action = 27
        max_tries = 1000
        match = False
        i = 0
        while i < max_tries and not match:
            env.set_state(state)
            o, r, done, _, info = env.step(action)
            if env.last_obs == [[0, 0, 0, 0], [2, 2, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                                [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0],
                                [0, 1, 1, 1]] \
                    and env.s == [[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                                  [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 2, 0], [1, 0, 0, 0], [1, 0, 0, 1],
                                  [1, 0, 0, 0], [1, 1, 2, 1]]:
                match = True
            i += 1
        assert match

    def test_set_state_14(self) -> None:
        """
        Tests the set_state method

        :return: None
        """
        config = CSLECyborgWrapperConfig(maximum_steps=100, gym_env_name="", save_trace=False, reward_shaping=True,
                                         scenario=2, red_agent_type=RedAgentType.B_LINE_AGENT)
        env = CyborgScenarioTwoWrapper(config=config)
        state = CyborgWrapperState(
            s=[[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
               [0, 0, 0, 0], [1, 0, 2, 0], [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 0, 0], [1, 1, 2, 1]],
            scan_state=[0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            op_server_restored=True,
            obs=[[0, 0, 0, 0], [2, 2, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                 [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 1, 1, 1]],
            red_agent_state=6,
            privilege_escalation_detected=None,
            red_agent_target=1, red_action_targets={0: 0, 1: 12, 2: 12, 3: 12, 4: 1, 5: 1, 6: 1},
            attacker_observed_decoy=[0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            detected=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            malware_state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ssh_access=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            escalated=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            exploited=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            bline_base_jump=False
        )
        action = 8
        max_tries = 1000
        match = False
        i = 0
        while i < max_tries and not match:
            env.set_state(state)
            o, r, done, _, info = env.step(action)
            if env.last_obs == [[0, 0, 0, 0], [0, 2, 3, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                                [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0],
                                [0, 1, 1, 1]] \
                    and env.s == [[0, 0, 0, 0], [1, 1, 2, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                                  [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 2, 0], [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 0, 0],
                                  [1, 1, 2, 1]]:
                match = True
            i += 1
        assert match

    def test_set_state_15(self) -> None:
        """
        Tests the set_state method

        :return: None
        """
        config = CSLECyborgWrapperConfig(maximum_steps=100, gym_env_name="", save_trace=False, reward_shaping=True,
                                         scenario=2, red_agent_type=RedAgentType.B_LINE_AGENT)
        env = CyborgScenarioTwoWrapper(config=config)
        state = CyborgWrapperState(
            s=[[0, 0, 0, 0], [1, 1, 0, 3], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
               [0, 0, 0, 3], [1, 0, 2, 0], [1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 0], [1, 1, 2, 0]],
            scan_state=[0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            op_server_restored=False,
            obs=[[0, 0, 0, 0], [1, 2, 0, 3], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                 [0, 0, 0, 3], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 1, 2, 0]],
            red_agent_state=2,
            privilege_escalation_detected=None,
            red_agent_target=12, red_action_targets={0: 0, 1: 12, 2: 12, 3: 12, 4: 1, 5: 1},
            attacker_observed_decoy=[0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            detected=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            malware_state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ssh_access=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            escalated=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            exploited=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            bline_base_jump=False
        )
        action = 21
        max_tries = 1000
        match = False
        i = 0
        while i < max_tries and not match:
            env.set_state(state)
            o, r, done, _, info = env.step(action)
            if env.last_obs == [[0, 0, 0, 0], [0, 2, 0, 3], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0],
                                [0, 0, 0, 0], [0, 0, 0, 3], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0],
                                [2, 1, 1, 0]] \
                    and env.s == [[0, 0, 0, 0], [1, 1, 0, 3], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0],
                                  [0, 0, 0, 0], [0, 0, 0, 3], [1, 0, 2, 0], [1, 0, 0, 1], [1, 0, 0, 1],
                                  [1, 0, 0, 0], [1, 1, 2, 0]]:
                match = True
            i += 1
        assert match

    def test_set_state_16(self) -> None:
        """
        Tests the set_state method

        :return: None
        """
        config = CSLECyborgWrapperConfig(maximum_steps=100, gym_env_name="", save_trace=False, reward_shaping=True,
                                         scenario=2, red_agent_type=RedAgentType.B_LINE_AGENT)
        env = CyborgScenarioTwoWrapper(config=config)
        state = CyborgWrapperState(
            s=[[0, 0, 0, 0], [1, 1, 0, 4], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
               [0, 0, 0, 4], [1, 0, 2, 0], [1, 0, 0, 2], [1, 0, 0, 4], [1, 0, 0, 2], [1, 1, 2, 1]],
            scan_state=[0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            op_server_restored=False,
            obs=[[0, 0, 0, 0], [1, 2, 0, 4], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                 [0, 0, 0, 4], [0, 0, 0, 0], [0, 0, 0, 2], [0, 0, 0, 4], [0, 0, 0, 2], [0, 1, 3, 1]],
            red_agent_state=2,
            privilege_escalation_detected=None,
            red_agent_target=12, red_action_targets={0: 0, 1: 12, 2: 12, 3: 12, 4: 1, 5: 1},
            attacker_observed_decoy=[0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            detected=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            malware_state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            ssh_access=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            escalated=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            exploited=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            bline_base_jump=False
        )
        action = 15
        max_tries = 1000
        match = False
        i = 0
        env.set_state(state)
        o, r, done, _, info = env.step(action)
        while i < max_tries and not match:
            env.set_state(state)
            o, r, done, _, info = env.step(action)
            if env.last_obs == [[0, 0, 0, 0], [0, 2, 0, 4], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0],
                                [0, 0, 0, 0], [0, 0, 0, 4], [0, 0, 0, 0], [0, 0, 0, 2], [0, 0, 0, 4], [0, 0, 0, 2],
                                [2, 1, 2, 1]] \
                    and env.s == [[0, 0, 0, 0], [1, 1, 0, 4], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0],
                                  [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 4], [1, 0, 2, 0], [1, 0, 0, 2], [1, 0, 0, 4],
                                  [1, 0, 0, 2], [1, 1, 2, 1]]:
                match = True
            i += 1
        assert match

    def test_set_state_17(self) -> None:
        """
        Tests the set_state method

        :return: None
        """
        config = CSLECyborgWrapperConfig(maximum_steps=100, gym_env_name="", save_trace=False, reward_shaping=True,
                                         scenario=2, red_agent_type=RedAgentType.B_LINE_AGENT)
        env = CyborgScenarioTwoWrapper(config=config)
        state = CyborgWrapperState(
            s=[[0, 0, 0, 1], [1, 1, 2, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
               [0, 0, 0, 4], [1, 0, 2, 0], [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 2, 1]],
            scan_state=[0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            op_server_restored=False,
            obs=[[0, 0, 0, 1], [0, 2, 1, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                 [0, 0, 0, 4], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 1, 3, 1]],
            red_agent_state=7,
            privilege_escalation_detected=1,
            red_agent_target=1, red_action_targets={0: 0, 1: 12, 2: 12, 3: 12, 4: 1, 5: 1, 6: 1},
            attacker_observed_decoy=[0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            detected=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            malware_state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ssh_access=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            escalated=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            exploited=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            bline_base_jump=False
        )
        action = 0
        max_tries = 1000
        match = False
        i = 0
        while i < max_tries and not match:
            env.set_state(state)
            o, r, done, _, info = env.step(action)
            if env.last_obs == [[0, 0, 0, 1], [0, 2, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0],
                                [0, 0, 0, 0], [0, 0, 0, 4], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1],
                                [0, 1, 3, 1]] \
                    and env.s == [[1, 0, 0, 1], [1, 1, 0, 0], [1, 0, 0, 1], [1, 0, 0, 1], [0, 0, 0, 0],
                                  [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 4], [1, 0, 2, 0], [1, 0, 0, 0], [1, 0, 0, 1],
                                  [1, 0, 0, 1], [1, 1, 2, 1]]:
                match = True
            i += 1
        assert match

    def test_set_state_18(self) -> None:
        """
        Tests the set_state method

        :return: None
        """
        config = CSLECyborgWrapperConfig(maximum_steps=100, gym_env_name="", save_trace=False, reward_shaping=True,
                                         scenario=2, red_agent_type=RedAgentType.B_LINE_AGENT)
        env = CyborgScenarioTwoWrapper(config=config)
        state = CyborgWrapperState(
            s=[[0, 0, 0, 4], [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
               [0, 0, 0, 2], [1, 0, 2, 0], [1, 0, 0, 1], [1, 1, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]],
            scan_state=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
            op_server_restored=False,
            obs=[[0, 0, 0, 4], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                 [0, 0, 0, 2], [0, 0, 0, 0], [0, 0, 0, 1], [0, 2, 3, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            red_agent_state=2,
            privilege_escalation_detected=None,
            red_agent_target=10, red_action_targets={0: 0, 1: 10, 2: 10, 3: 10, 4: 2, 5: 2, 6: 2},
            attacker_observed_decoy=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            detected=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            malware_state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            ssh_access=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            escalated=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            exploited=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            bline_base_jump=False
        )
        action = 13
        max_tries = 1000
        match = False
        i = 0
        while i < max_tries and not match:
            env.set_state(state)
            o, r, done, _, info = env.step(action)
            if env.last_obs == [[0, 0, 0, 4], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0],
                                [0, 0, 0, 0], [0, 0, 0, 2], [0, 0, 0, 0], [0, 0, 0, 1], [0, 2, 2, 0], [0, 0, 0, 0],
                                [0, 0, 0, 0]] \
                    and env.s == [[0, 0, 0, 4], [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0],
                                  [0, 0, 0, 0], [0, 0, 0, 2], [1, 0, 2, 0], [1, 0, 0, 1], [1, 1, 0, 0], [1, 0, 0, 0],
                                  [1, 0, 0, 0]]:
                match = True
            i += 1
        assert match

    def test_set_state_19(self) -> None:
        """
        Tests the set_state method

        :return: None
        """
        config = CSLECyborgWrapperConfig(maximum_steps=100, gym_env_name="", save_trace=False, reward_shaping=True,
                                         scenario=2, red_agent_type=RedAgentType.B_LINE_AGENT)
        env = CyborgScenarioTwoWrapper(config=config)
        state = CyborgWrapperState(
            s=[[0, 0, 0, 0], [1, 1, 0, 3], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
               [0, 0, 0, 0], [1, 0, 2, 0], [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 0, 0], [1, 1, 2, 0]],
            scan_state=[0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            op_server_restored=False,
            obs=[[0, 0, 0, 0], [1, 2, 0, 3], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                 [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 1, 1, 0]],
            red_agent_state=5,
            privilege_escalation_detected=None,
            red_agent_target=1, red_action_targets={0: 0, 1: 12, 2: 12, 3: 12, 4: 1, 5: 1},
            attacker_observed_decoy=[0, 3, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            detected=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            malware_state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            ssh_access=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            escalated=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            exploited=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            bline_base_jump=False
        )
        action = 4
        max_tries = 1000
        match = False
        i = 0
        while i < max_tries and not match:
            env.set_state(state)
            o, r, done, _, info = env.step(action)
            if env.last_obs == [[0, 0, 0, 0], [1, 2, 0, 3], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                                [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0],
                                [0, 1, 1, 0]] \
                    and env.s == [[0, 0, 0, 0], [1, 1, 0, 3], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                                  [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 2, 0], [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 0, 0],
                                  [1, 1, 2, 0]]:
                match = True
            i += 1
        assert match

    def test_set_state_20(self) -> None:
        """
        Tests the set_state method

        :return: None
        """
        config = CSLECyborgWrapperConfig(maximum_steps=100, gym_env_name="", save_trace=False, reward_shaping=True,
                                         scenario=2, red_agent_type=RedAgentType.B_LINE_AGENT)
        env = CyborgScenarioTwoWrapper(config=config)
        state = CyborgWrapperState(
            s=[[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
               [0, 0, 0, 0], [1, 0, 2, 0], [1, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0]],
            scan_state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
            op_server_restored=False,
            obs=[[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                 [0, 0, 0, 0], [0, 0, 0, 0], [2, 2, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
            red_agent_state=3,
            privilege_escalation_detected=None,
            red_agent_target=9, red_action_targets={0: 0, 1: 9, 2: 9},
            attacker_observed_decoy=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            detected=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            malware_state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ssh_access=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            escalated=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            exploited=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            bline_base_jump=False
        )
        action = 12
        max_tries = 1000
        match = False
        i = 0
        while i < max_tries and not match:
            env.set_state(state)
            o, r, done, _, info = env.step(action)
            if env.last_obs == [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                                [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 2, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0],
                                [0, 0, 0, 0]] \
                    and env.s == [[0, 0, 0, 0], [0, 0, 0, 1], [1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                                  [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 2, 0], [1, 1, 2, 0], [1, 0, 0, 1], [1, 0, 0, 0],
                                  [1, 0, 0, 0]]:
                match = True
            i += 1
        assert match

    def test_set_state_21(self) -> None:
        """
        Tests the set_state method

        :return: None
        """
        config = CSLECyborgWrapperConfig(maximum_steps=100, gym_env_name="", save_trace=False, reward_shaping=True,
                                         scenario=2, red_agent_type=RedAgentType.B_LINE_AGENT)
        env = CyborgScenarioTwoWrapper(config=config)
        state = CyborgWrapperState(
            s=[[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
               [0, 0, 0, 0], [1, 0, 2, 0], [1, 0, 0, 0], [1, 1, 0, 0], [1, 0, 0, 0], [1, 0, 0, 1]],
            scan_state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
            op_server_restored=False,
            obs=[[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                 [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 2, 3, 0], [0, 0, 0, 0], [0, 0, 0, 1]],
            red_agent_state=2,
            privilege_escalation_detected=None,
            red_agent_target=10, red_action_targets={0: 0, 1: 10, 2: 10, 3: 10},
            attacker_observed_decoy=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            detected=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            malware_state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            ssh_access=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            escalated=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            exploited=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            bline_base_jump=False
        )
        action = 13
        max_tries = 1000
        match = False
        i = 0
        while i < max_tries and not match:
            env.set_state(state)
            o, r, done, _, info = env.step(action)
            if env.last_obs == [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                                [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [2, 2, 2, 0], [0, 0, 0, 0],
                                [0, 0, 0, 1]] \
                    and env.s == [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                                  [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 2, 0], [1, 0, 0, 0], [1, 1, 2, 0], [1, 0, 0, 0],
                                  [1, 0, 0, 1]]:
                match = True
            i += 1
        assert match

    def test_set_state_22(self) -> None:
        """
        Tests the set_state method

        :return: None
        """
        config = CSLECyborgWrapperConfig(maximum_steps=100, gym_env_name="", save_trace=False, reward_shaping=True,
                                         scenario=2, red_agent_type=RedAgentType.B_LINE_AGENT)
        env = CyborgScenarioTwoWrapper(config=config)
        state = CyborgWrapperState(
            s=[[0, 0, 0, 2], [0, 0, 0, 2], [1, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
               [0, 0, 0, 3], [1, 0, 2, 0], [1, 1, 2, 1], [1, 0, 0, 4], [1, 0, 0, 0], [1, 0, 0, 1]],
            scan_state=[0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            op_server_restored=False,
            obs=[[0, 0, 0, 2], [0, 0, 0, 2], [0, 2, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0],
                 [0, 0, 0, 0], [0, 0, 0, 3], [0, 0, 0, 0], [0, 1, 2, 1], [0, 0, 0, 4], [0, 0, 0, 0], [0, 0, 0, 1]],
            red_agent_state=5,
            privilege_escalation_detected=None,
            red_agent_target=2, red_action_targets={0: 0, 1: 9, 2: 9, 3: 9, 4: 2, 5: 2, 6: 2},
            attacker_observed_decoy=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            detected=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            malware_state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            ssh_access=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            escalated=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            exploited=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            bline_base_jump=False
        )
        action = 5
        max_tries = 1000
        match = False
        i = 0
        while i < max_tries and not match:
            env.set_state(state)
            o, r, done, _, info = env.step(action)
            if env.last_obs == [[0, 0, 0, 2], [0, 0, 0, 2], [2, 2, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0],
                                [0, 0, 0, 0], [0, 0, 0, 3], [0, 0, 0, 0], [0, 1, 2, 1], [0, 0, 0, 4], [0, 0, 0, 0],
                                [0, 0, 0, 1]] \
                    and env.s == [[0, 0, 0, 2], [0, 0, 0, 2], [1, 1, 2, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0],
                                  [0, 0, 0, 0], [0, 0, 0, 3], [1, 0, 2, 0], [1, 1, 2, 1], [1, 0, 0, 4], [1, 0, 0, 0],
                                  [1, 0, 0, 1]]:
                match = True
            i += 1
        assert match

    def test_set_state_23(self) -> None:
        """
        Tests the set_state method

        :return: None
        """
        config = CSLECyborgWrapperConfig(maximum_steps=100, gym_env_name="", save_trace=False, reward_shaping=True,
                                         scenario=2, red_agent_type=RedAgentType.B_LINE_AGENT)
        env = CyborgScenarioTwoWrapper(config=config)
        state = CyborgWrapperState(
            s=[[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
               [0, 0, 0, 0], [1, 0, 2, 0], [1, 0, 0, 1], [1, 1, 0, 2], [1, 0, 0, 0], [1, 0, 0, 0]],
            scan_state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
            op_server_restored=False,
            obs=[[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                 [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [1, 2, 1, 2], [0, 0, 0, 0], [0, 0, 0, 0]],
            red_agent_state=2,
            privilege_escalation_detected=None,
            red_agent_target=10, red_action_targets={0: 0, 1: 10, 2: 10},
            attacker_observed_decoy=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            detected=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            malware_state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ssh_access=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            escalated=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            exploited=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            bline_base_jump=False
        )
        action = 13
        max_tries = 1000
        match = False
        i = 0
        while i < max_tries and not match:
            env.set_state(state)
            o, r, done, _, info = env.step(action)
            if env.last_obs == [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0],
                                [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [1, 2, 1, 2], [0, 0, 0, 0],
                                [0, 0, 0, 0]] \
                    and env.s == [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0],
                                  [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 2, 0], [1, 0, 0, 1], [1, 1, 0, 2], [1, 0, 0, 0],
                                  [1, 0, 0, 0]]:
                match = True
            i += 1
        assert match

    def test_set_state_24(self) -> None:
        """
        Tests the set_state method

        :return: None
        """
        config = CSLECyborgWrapperConfig(maximum_steps=100, gym_env_name="", save_trace=False, reward_shaping=True,
                                         scenario=2, red_agent_type=RedAgentType.B_LINE_AGENT)
        env = CyborgScenarioTwoWrapper(config=config)
        state = CyborgWrapperState(
            s=[[0, 0, 0, 2], [0, 0, 0, 1], [1, 1, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
               [0, 0, 0, 0], [1, 0, 2, 0], [1, 1, 2, 1], [1, 0, 0, 3], [1, 0, 0, 2], [1, 0, 0, 1]],
            scan_state=[0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            op_server_restored=False,
            obs=[[0, 0, 0, 2], [0, 0, 0, 1], [1, 2, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                 [0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 1, 1], [0, 0, 0, 3], [0, 0, 0, 2], [0, 0, 0, 1]],
            red_agent_state=5,
            privilege_escalation_detected=None,
            red_agent_target=2, red_action_targets={0: 0, 1: 9, 2: 9, 3: 9, 4: 2, 5: 2},
            attacker_observed_decoy=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            detected=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            malware_state=[0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            ssh_access=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            escalated=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            exploited=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            bline_base_jump=False
        )
        action = 5
        max_tries = 1000
        match = False
        i = 0
        while i < max_tries and not match:
            env.set_state(state)
            o, r, done, _, info = env.step(action)
            if env.last_obs == [[0, 0, 0, 2], [0, 0, 0, 1], [2, 2, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0],
                                [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 1, 1], [0, 0, 0, 3], [0, 0, 0, 2],
                                [0, 0, 0, 1]] \
                    and env.s == [[0, 0, 0, 2], [0, 0, 0, 1], [1, 1, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0],
                                  [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 2, 0], [1, 1, 2, 1], [1, 0, 0, 3], [1, 0, 0, 2],
                                  [1, 0, 0, 1]]:
                match = True
            i += 1
        assert match

    def test_set_state_25(self) -> None:
        """
        Tests the set_state method

        :return: None
        """
        config = CSLECyborgWrapperConfig(maximum_steps=100, gym_env_name="", save_trace=False, reward_shaping=True,
                                         scenario=2, red_agent_type=RedAgentType.B_LINE_AGENT)
        env = CyborgScenarioTwoWrapper(config=config)
        state = CyborgWrapperState(
            s=[[0, 0, 0, 0], [0, 0, 0, 2], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
               [0, 0, 0, 2], [1, 0, 2, 0], [1, 0, 0, 0], [1, 1, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]],
            scan_state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
            op_server_restored=False,
            obs=[[0, 0, 0, 0], [0, 0, 0, 2], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                 [0, 0, 0, 2], [0, 0, 0, 0], [0, 0, 0, 0], [1, 2, 3, 1], [0, 0, 0, 1], [0, 0, 0, 1]],
            red_agent_state=2,
            privilege_escalation_detected=None,
            red_agent_target=10, red_action_targets={0: 0, 1: 10, 2: 10, 3: 10},
            attacker_observed_decoy=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            detected=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            malware_state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            ssh_access=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            escalated=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            exploited=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            bline_base_jump=False
        )
        action = 13
        max_tries = 1000
        match = False
        i = 0
        while i < max_tries and not match:
            env.set_state(state)
            o, r, done, _, info = env.step(action)
            if env.last_obs == [[0, 0, 0, 0], [0, 0, 0, 2], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                                [0, 0, 0, 0], [0, 0, 0, 2], [0, 0, 0, 0], [0, 0, 0, 0], [1, 2, 2, 1], [0, 0, 0, 1],
                                [0, 0, 0, 1]] \
                    and env.s == [[0, 0, 0, 0], [0, 0, 0, 2], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                                  [0, 0, 0, 0], [0, 0, 0, 2], [1, 0, 2, 0], [1, 0, 0, 0], [1, 1, 0, 1], [1, 0, 0, 1],
                                  [1, 0, 0, 1]]:
                match = True
            i += 1
        assert match

    def test_set_state_26(self) -> None:
        """
        Tests the set_state method

        :return: None
        """
        config = CSLECyborgWrapperConfig(maximum_steps=100, gym_env_name="", save_trace=False, reward_shaping=True,
                                         scenario=2, red_agent_type=RedAgentType.B_LINE_AGENT)
        env = CyborgScenarioTwoWrapper(config=config)
        state = CyborgWrapperState(
            s=[[0, 0, 0, 4], [0, 0, 0, 4], [1, 1, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
               [0, 0, 0, 1], [1, 0, 2, 0], [1, 0, 0, 2], [1, 1, 0, 1], [1, 0, 0, 2], [1, 0, 0, 1]],
            scan_state=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
            op_server_restored=False,
            obs=[[0, 0, 0, 4], [0, 0, 0, 4], [0, 1, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                 [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 2], [1, 2, 1, 1], [0, 0, 0, 2], [0, 0, 0, 1]],
            red_agent_state=2,
            privilege_escalation_detected=None,
            red_agent_target=10, red_action_targets={0: 0, 1: 10, 2: 10, 3: 10, 4: 2, 5: 2, 6: 2},
            attacker_observed_decoy=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            detected=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            malware_state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ssh_access=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            escalated=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            exploited=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            bline_base_jump=False
        )
        action = 13
        max_tries = 1000
        match = False
        i = 0
        while i < max_tries and not match:
            env.set_state(state)
            o, r, done, _, info = env.step(action)
            if env.last_obs == [[0, 0, 0, 4], [0, 0, 0, 4], [0, 1, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0],
                                [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 2], [1, 2, 1, 1], [0, 0, 0, 2],
                                [0, 0, 0, 1]] \
                    and env.s == [[0, 0, 0, 4], [0, 0, 0, 4], [1, 1, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0],
                                  [0, 0, 0, 0], [0, 0, 0, 1], [1, 0, 2, 0], [1, 0, 0, 2], [1, 1, 1, 1], [1, 0, 0, 2],
                                  [1, 0, 0, 1]]:
                match = True
            i += 1
        assert match

    def test_set_state_27(self) -> None:
        """
        Tests the set_state method

        :return: None
        """
        config = CSLECyborgWrapperConfig(maximum_steps=100, gym_env_name="", save_trace=False, reward_shaping=True,
                                         scenario=2, red_agent_type=RedAgentType.B_LINE_AGENT)
        env = CyborgScenarioTwoWrapper(config=config)
        state = CyborgWrapperState(
            s=[[0, 0, 0, 4], [1, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
               [0, 0, 0, 4], [1, 0, 2, 0], [1, 0, 0, 4], [1, 0, 0, 4], [1, 1, 2, 2], [1, 0, 0, 1]],
            scan_state=[0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            op_server_restored=False,
            obs=[[0, 0, 0, 4], [0, 2, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                 [0, 0, 0, 4], [0, 0, 0, 0], [0, 0, 0, 4], [0, 0, 0, 4], [0, 1, 3, 2], [0, 0, 0, 1]],
            red_agent_state=5,
            privilege_escalation_detected=None,
            red_agent_target=1, red_action_targets={0: 0, 1: 11, 2: 11, 3: 11, 4: 1, 5: 1, 6: 1},
            attacker_observed_decoy=[0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            detected=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            malware_state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            ssh_access=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            escalated=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            exploited=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            bline_base_jump=False
        )
        action = 27
        max_tries = 1000
        match = False
        i = 0
        while i < max_tries and not match:
            env.set_state(state)
            o, r, done, _, info = env.step(action)
            if env.last_obs == [[0, 0, 0, 4], [2, 2, 1, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0],
                                [0, 0, 0, 0], [0, 0, 0, 4], [0, 0, 0, 0], [0, 0, 0, 4], [0, 0, 0, 4], [0, 1, 3, 2],
                                [0, 0, 0, 1]] \
                    and env.s == [[0, 0, 0, 4], [1, 1, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0],
                                  [0, 0, 0, 0], [0, 0, 0, 4], [1, 0, 2, 0], [1, 0, 0, 4], [1, 0, 0, 4], [1, 1, 2, 2],
                                  [1, 0, 0, 1]] and env.detected == [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]:
                match = True
            i += 1
        assert match

    def test_set_state_28(self) -> None:
        """
        Tests the set_state method

        :return: None
        """
        config = CSLECyborgWrapperConfig(maximum_steps=100, gym_env_name="", save_trace=False, reward_shaping=True,
                                         scenario=2, red_agent_type=RedAgentType.B_LINE_AGENT)
        env = CyborgScenarioTwoWrapper(config=config)
        state = CyborgWrapperState(
            s=[[0, 0, 0, 4], [1, 1, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
               [0, 0, 0, 4], [1, 0, 2, 0], [1, 0, 0, 4], [1, 0, 0, 4], [1, 1, 2, 2], [1, 0, 0, 1]],
            scan_state=[0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            op_server_restored=False,
            obs=[[0, 0, 0, 4], [2, 2, 1, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                 [0, 0, 0, 4], [0, 0, 0, 0], [0, 0, 0, 4], [0, 0, 0, 4], [0, 1, 3, 2], [0, 0, 0, 1]],
            red_agent_state=2,
            privilege_escalation_detected=None,
            red_agent_target=11, red_action_targets={0: 0, 1: 11, 2: 11, 3: 11, 4: 1, 5: 1, 6: 1},
            attacker_observed_decoy=[0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            detected=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            malware_state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            ssh_access=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            escalated=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            exploited=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            bline_base_jump=False
        )
        action = 4
        max_tries = 1000
        match = False
        i = 0
        while i < max_tries and not match:
            env.set_state(state)
            o, r, done, _, info = env.step(action)
            if env.last_obs == [[0, 0, 0, 4], [0, 2, 1, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0],
                                [0, 0, 0, 0], [0, 0, 0, 4], [0, 0, 0, 0], [0, 0, 0, 4], [0, 0, 0, 4], [2, 1, 1, 2],
                                [0, 0, 0, 1]] \
                    and env.s == [[0, 0, 0, 4], [1, 1, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0],
                                  [0, 0, 0, 0], [0, 0, 0, 4], [1, 0, 2, 0], [1, 0, 0, 4], [1, 0, 0, 4], [1, 1, 2, 2],
                                  [1, 0, 0, 1]]:
                match = True
            i += 1
        assert match

    def test_set_state_29(self) -> None:
        """
        Tests the set_state method

        :return: None
        """
        config = CSLECyborgWrapperConfig(maximum_steps=100, gym_env_name="", save_trace=False, reward_shaping=True,
                                         scenario=2, red_agent_type=RedAgentType.B_LINE_AGENT)
        env = CyborgScenarioTwoWrapper(config=config)
        state = CyborgWrapperState(
            s=[[0, 0, 0, 0], [0, 0, 0, 1], [1, 1, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
               [0, 0, 0, 0], [1, 0, 2, 0], [1, 1, 2, 1], [1, 0, 0, 2], [1, 0, 0, 1], [1, 0, 0, 1]],
            scan_state=[0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            op_server_restored=False,
            obs=[[0, 0, 0, 0], [0, 0, 0, 1], [0, 2, 3, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                 [0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 1, 1], [0, 0, 0, 2], [0, 0, 0, 1], [0, 0, 0, 1]],
            red_agent_state=4,
            privilege_escalation_detected=None,
            red_agent_target=2, red_action_targets={0: 0, 1: 9, 2: 9, 3: 9, 4: 2, 5: 2, 6: 2},
            attacker_observed_decoy=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            detected=[0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            malware_state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            ssh_access=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            escalated=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            exploited=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            bline_base_jump=False
        )
        action = 5
        max_tries = 1000
        match = False
        i = 0
        while i < max_tries and not match:
            env.set_state(state)
            o, r, done, _, info = env.step(action)
            if env.last_obs == [[0, 0, 0, 0], [0, 0, 0, 1], [1, 2, 3, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                                [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 1, 1], [0, 0, 0, 2], [0, 0, 0, 1],
                                [0, 0, 0, 1]] \
                    and env.s == [[0, 0, 0, 0], [0, 0, 0, 1], [1, 1, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                                  [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 2, 0], [1, 1, 2, 1], [1, 0, 0, 2], [1, 0, 0, 1],
                                  [1, 0, 0, 1]]:
                match = True
            i += 1
        assert match

    def test_set_state_30(self) -> None:
        """
        Tests the set_state method

        :return: None
        """
        config = CSLECyborgWrapperConfig(maximum_steps=100, gym_env_name="", save_trace=False, reward_shaping=True,
                                         scenario=2, red_agent_type=RedAgentType.B_LINE_AGENT)
        env = CyborgScenarioTwoWrapper(config=config)
        state = CyborgWrapperState(
            s=[[0, 0, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
               [0, 0, 0, 0], [1, 0, 2, 0], [1, 0, 0, 0], [1, 0, 0, 1], [1, 1, 2, 0], [1, 0, 0, 1]],
            scan_state=[0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            op_server_restored=False,
            obs=[[0, 0, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                 [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 1, 2, 0], [0, 0, 0, 1]],
            red_agent_state=5,
            privilege_escalation_detected=None,
            red_agent_target=1, red_action_targets={0: 0, 1: 11, 2: 11, 3: 11, 4: 1, 5: 1, 6: 1},
            attacker_observed_decoy=[0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            detected=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            malware_state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            ssh_access=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            escalated=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            exploited=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            bline_base_jump=False
        )
        action = 27
        max_tries = 1000
        match = False
        i = 0
        while i < max_tries and not match:
            env.set_state(state)
            o, r, done, _, info = env.step(action)
            if env.last_obs == [[0, 0, 0, 0], [0, 2, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                                [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 1, 2, 0],
                                [0, 0, 0, 1]] \
                    and env.s == [[0, 0, 0, 0], [1, 1, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                                  [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 2, 0], [1, 0, 0, 0], [1, 0, 0, 1], [1, 1, 2, 0],
                                  [1, 0, 0, 1]]:
                match = True
            i += 1
        assert match

    def test_set_state_31(self) -> None:
        """
        Tests the set_state method

        :return: None
        """
        config = CSLECyborgWrapperConfig(maximum_steps=100, gym_env_name="", save_trace=False, reward_shaping=True,
                                         scenario=2, red_agent_type=RedAgentType.B_LINE_AGENT)
        env = CyborgScenarioTwoWrapper(config=config)
        state = CyborgWrapperState(
            s=[[0, 0, 0, 3], [0, 0, 0, 3], [1, 1, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
               [0, 0, 0, 0], [1, 0, 2, 0], [1, 0, 0, 2], [1, 1, 2, 4], [1, 0, 0, 0], [1, 0, 0, 0]],
            scan_state=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
            op_server_restored=False,
            obs=[[0, 0, 0, 3], [0, 0, 0, 3], [0, 1, 3, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0],
                 [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 2], [1, 2, 2, 4], [0, 0, 0, 0], [0, 0, 0, 0]],
            red_agent_state=2,
            privilege_escalation_detected=None,
            red_agent_target=10, red_action_targets={0: 0, 1: 10, 2: 10, 3: 10, 4: 2, 5: 2, 6: 2},
            attacker_observed_decoy=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            detected=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            malware_state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            ssh_access=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            escalated=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            exploited=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            bline_base_jump=False
        )
        action = 5
        max_tries = 1000
        match = False
        i = 0
        while i < max_tries and not match:
            env.set_state(state)
            o, r, done, _, info = env.step(action)
            if env.last_obs == [[0, 0, 0, 3], [0, 0, 0, 3], [0, 1, 3, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0],
                                [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 2], [2, 2, 1, 4], [0, 0, 0, 0],
                                [0, 0, 0, 0]] \
                    and env.s == [[0, 0, 0, 3], [0, 0, 0, 3], [1, 1, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0],
                                  [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 2, 0], [1, 0, 0, 2], [1, 1, 2, 4], [1, 0, 0, 0],
                                  [1, 0, 0, 0]]:
                match = True
            i += 1
        assert match

    def test_set_state_32(self) -> None:
        """
        Tests the set_state method

        :return: None
        """
        config = CSLECyborgWrapperConfig(maximum_steps=100, gym_env_name="", save_trace=False, reward_shaping=True,
                                         scenario=2, red_agent_type=RedAgentType.B_LINE_AGENT)
        env = CyborgScenarioTwoWrapper(config=config)
        state = CyborgWrapperState(
            s=[[0, 0, 0, 2], [1, 1, 1, 2], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
               [0, 0, 0, 2], [1, 0, 2, 0], [1, 0, 0, 1], [1, 0, 0, 4], [1, 1, 2, 0], [1, 0, 0, 1]],
            scan_state=[0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            op_server_restored=False,
            obs=[[0, 0, 0, 2], [2, 2, 1, 2], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                 [0, 0, 0, 2], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 4], [0, 1, 2, 0], [0, 0, 0, 1]],
            red_agent_state=6,
            privilege_escalation_detected=None,
            red_agent_target=1, red_action_targets={0: 0, 1: 11, 2: 11, 3: 11, 4: 1, 5: 1, 6: 1},
            attacker_observed_decoy=[0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            detected=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            malware_state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            ssh_access=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            escalated=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            exploited=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            bline_base_jump=False
        )
        action = 8
        max_tries = 1000
        match = False
        i = 0
        while i < max_tries and not match:
            env.set_state(state)
            o, r, done, _, info = env.step(action)
            if env.last_obs == [[0, 0, 0, 2], [0, 2, 3, 2], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0],
                                [0, 0, 0, 0], [0, 0, 0, 2], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 4], [0, 1, 2, 0],
                                [0, 0, 0, 1]] \
                    and env.s == [[0, 0, 0, 2], [1, 1, 2, 2], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0],
                                  [0, 0, 0, 0], [0, 0, 0, 2], [1, 0, 2, 0], [1, 0, 0, 1], [1, 0, 0, 4], [1, 1, 2, 0],
                                  [1, 0, 0, 1]]:
                match = True
            i += 1
        assert match

    def test_set_state_33(self) -> None:
        """
        Tests the set_state method

        :return: None
        """
        config = CSLECyborgWrapperConfig(maximum_steps=100, gym_env_name="", save_trace=False, reward_shaping=True,
                                         scenario=2, red_agent_type=RedAgentType.B_LINE_AGENT)
        env = CyborgScenarioTwoWrapper(config=config)
        state = CyborgWrapperState(
            s=[[1, 0, 0, 0], [1, 1, 0, 1], [1, 0, 0, 0], [1, 1, 2, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
               [0, 0, 0, 0], [1, 0, 2, 0], [1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 2, 1], [1, 0, 0, 1]],
            scan_state=[0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            op_server_restored=False,
            obs=[[0, 0, 0, 0], [0, 1, 0, 1], [0, 0, 0, 0], [2, 2, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                 [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 1, 1, 1], [0, 0, 0, 1]],
            red_agent_state=10,
            privilege_escalation_detected=None,
            red_agent_target=3,
            red_action_targets={0: 0, 1: 11, 2: 11, 3: 11, 4: 1, 5: 1, 6: 1, 7: 1, 8: 3, 9: 3, 10: 3},
            attacker_observed_decoy=[0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            detected=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            malware_state=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            ssh_access=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            escalated=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            exploited=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            bline_base_jump=False
        )
        action = 29
        max_tries = 1000
        match = False
        i = 0
        while i < max_tries and not match:
            env.set_state(state)
            o, r, done, _, info = env.step(action)
            if env.last_obs == [[0, 0, 0, 0], [0, 1, 0, 1], [0, 0, 0, 0], [0, 2, 3, 1], [0, 0, 0, 0], [0, 0, 0, 0],
                                [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 1, 1, 1],
                                [0, 0, 0, 1]] \
                    and env.s == [[1, 0, 0, 0], [1, 1, 0, 1], [1, 0, 0, 0], [1, 1, 2, 1], [0, 0, 0, 0], [0, 0, 0, 0],
                                  [0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 2, 0], [1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 2, 1],
                                  [1, 0, 0, 1]]:
                match = True
            i += 1
        assert match

    def test_set_state_34(self) -> None:
        """
        Tests the set_state method

        :return: None
        """
        config = CSLECyborgWrapperConfig(maximum_steps=100, gym_env_name="", save_trace=False, reward_shaping=True,
                                         scenario=2, red_agent_type=RedAgentType.B_LINE_AGENT)
        env = CyborgScenarioTwoWrapper(config=config)
        state = CyborgWrapperState(
            s=[[1, 0, 0, 0], [1, 1, 0, 1], [1, 0, 0, 0], [1, 1, 2, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
               [1, 0, 0, 0], [1, 0, 2, 0], [1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 2, 1], [1, 0, 0, 1]],
            scan_state=[0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            op_server_restored=False,
            obs=[[0, 0, 0, 0], [0, 1, 0, 1], [0, 0, 0, 0], [0, 2, 3, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                 [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 1, 1, 1], [0, 0, 0, 1]],
            red_agent_state=11,
            privilege_escalation_detected=None,
            red_agent_target=7, red_action_targets={0: 0, 1: 11, 2: 11, 3: 11, 4: 1, 5: 1, 6: 1, 7: 1, 8: 3, 9: 3,
                                                    10: 3, 11: 7},
            attacker_observed_decoy=[0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            detected=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            malware_state=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            ssh_access=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            escalated=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            exploited=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            bline_base_jump=False
        )
        action = 35
        max_tries = 1000
        match = False
        i = 0
        while i < max_tries and not match:
            env.set_state(state)
            o, r, done, _, info = env.step(action)
            if env.last_obs == [[0, 0, 0, 0], [0, 1, 0, 1], [0, 0, 0, 0], [0, 1, 3, 1], [0, 0, 0, 0], [0, 0, 0, 0],
                                [0, 0, 0, 0], [1, 2, 0, 1], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 1, 1, 1],
                                [0, 0, 0, 1]] \
                    and env.s == [[1, 0, 0, 0], [1, 1, 0, 1], [1, 0, 0, 0], [1, 1, 2, 1], [0, 0, 0, 0], [0, 0, 0, 0],
                                  [0, 0, 0, 0], [1, 1, 0, 1], [1, 0, 2, 0], [1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 2, 1],
                                  [1, 0, 0, 1]]:
                match = True
            i += 1
        assert match

    def test_set_state_35(self) -> None:
        """
        Tests the set_state method

        :return: None
        """
        config = CSLECyborgWrapperConfig(maximum_steps=100, gym_env_name="", save_trace=False, reward_shaping=True,
                                         scenario=2, red_agent_type=RedAgentType.B_LINE_AGENT)
        env = CyborgScenarioTwoWrapper(config=config)
        state = CyborgWrapperState(
            s=[[0, 0, 0, 0], [1, 0, 0, 2], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
               [0, 0, 0, 0], [1, 0, 2, 0], [1, 0, 0, 0], [1, 0, 0, 1], [1, 1, 2, 0], [1, 0, 0, 0]],
            scan_state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
            op_server_restored=False,
            obs=[[0, 0, 0, 0], [0, 0, 0, 2], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                 [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 2, 1, 0], [0, 0, 0, 0]],
            red_agent_state=4,
            privilege_escalation_detected=None,
            red_agent_target=1, red_action_targets={0: 0, 1: 11, 2: 11, 3: 11},
            attacker_observed_decoy=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            detected=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            malware_state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            ssh_access=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            escalated=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            exploited=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            bline_base_jump=False
        )
        action = 27
        max_tries = 1000
        match = False
        i = 0
        while i < max_tries and not match:
            env.set_state(state)
            o, r, done, _, info = env.step(action)
            if env.last_obs == [[0, 0, 0, 0], [1, 2, 0, 3], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                                [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 1, 1, 0],
                                [0, 0, 0, 0]] \
                    and env.s == [[0, 0, 0, 0], [1, 1, 0, 3], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                                  [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 2, 0], [1, 0, 0, 0], [1, 0, 0, 1], [1, 1, 2, 0],
                                  [1, 0, 0, 0]]:
                match = True
            i += 1
        assert match

    def test_set_state_36(self) -> None:
        """
        Tests the set_state method

        :return: None
        """
        config = CSLECyborgWrapperConfig(maximum_steps=100, gym_env_name="", save_trace=False, reward_shaping=True,
                                         scenario=2, red_agent_type=RedAgentType.B_LINE_AGENT)
        env = CyborgScenarioTwoWrapper(config=config)
        state = CyborgWrapperState(
            s=[[0, 0, 0, 1], [1, 1, 0, 2], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
               [0, 0, 0, 0], [1, 0, 2, 0], [1, 0, 0, 0], [1, 0, 0, 2], [1, 1, 2, 0], [1, 0, 0, 0]],
            scan_state=[0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            op_server_restored=False,
            obs=[[0, 0, 0, 1], [1, 2, 0, 2], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                 [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 2], [0, 1, 1, 0], [0, 0, 0, 0]],
            red_agent_state=4,
            privilege_escalation_detected=None,
            red_agent_target=1, red_action_targets={0: 0, 1: 11, 2: 11, 3: 11, 4: 1, 5: 1},
            attacker_observed_decoy=[0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            detected=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            malware_state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            ssh_access=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            escalated=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            exploited=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            bline_base_jump=False
        )
        action = 4
        max_tries = 1000
        match = False
        i = 0
        while i < max_tries and not match:
            env.set_state(state)
            o, r, done, _, info = env.step(action)
            if env.last_obs == [[0, 0, 0, 1], [1, 2, 0, 2], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0],
                                [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 2], [0, 1, 1, 0],
                                [0, 0, 0, 0]] \
                    and env.s == [[0, 0, 0, 1], [1, 1, 0, 2], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0],
                                  [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 2, 0], [1, 0, 0, 0], [1, 0, 0, 2], [1, 1, 2, 0],
                                  [1, 0, 0, 0]]:
                match = True
            i += 1
        assert match

    def test_set_state_37(self) -> None:
        """
        Tests the set_state method

        :return: None
        """
        config = CSLECyborgWrapperConfig(maximum_steps=100, gym_env_name="", save_trace=False, reward_shaping=True,
                                         scenario=2, red_agent_type=RedAgentType.B_LINE_AGENT)
        env = CyborgScenarioTwoWrapper(config=config)
        state = CyborgWrapperState(
            s=[[0, 0, 0, 1], [1, 1, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
               [0, 0, 0, 2], [1, 0, 2, 0], [1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 2, 1]],
            scan_state=[0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            op_server_restored=False,
            obs=[[0, 0, 0, 1], [0, 2, 1, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                 [0, 0, 0, 2], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 1, 1, 1]],
            red_agent_state=4,
            privilege_escalation_detected=None,
            red_agent_target=1, red_action_targets={0: 0, 1: 12, 2: 12, 3: 12, 4: 1, 5: 1, 6: 1},
            attacker_observed_decoy=[0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            detected=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            malware_state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            ssh_access=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            escalated=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            exploited=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            bline_base_jump=False
        )
        action = 4
        max_tries = 1000
        match = False
        i = 0
        while i < max_tries and not match:
            env.set_state(state)
            o, r, done, _, info = env.step(action)
            if env.last_obs == [[0, 0, 0, 1], [1, 2, 1, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0],
                                [0, 0, 0, 0], [0, 0, 0, 2], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1],
                                [0, 1, 1, 1]] \
                    and env.s == [[0, 0, 0, 1], [1, 1, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0],
                                  [0, 0, 0, 0], [0, 0, 0, 2], [1, 0, 2, 0], [1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1],
                                  [1, 1, 2, 1]]:
                match = True
            i += 1
        assert match

    def test_set_state_38(self) -> None:
        """
        Tests the set_state method

        :return: None
        """
        config = CSLECyborgWrapperConfig(maximum_steps=100, gym_env_name="", save_trace=False, reward_shaping=True,
                                         scenario=2, red_agent_type=RedAgentType.B_LINE_AGENT)
        env = CyborgScenarioTwoWrapper(config=config)
        state = CyborgWrapperState(
            s=[[0, 0, 0, 0], [0, 0, 0, 2], [1, 1, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
               [0, 0, 0, 0], [1, 0, 2, 0], [1, 1, 2, 4], [1, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0]],
            scan_state=[0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            op_server_restored=False,
            obs=[[0, 0, 0, 0], [0, 0, 0, 2], [0, 2, 3, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                 [0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 3, 4], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
            red_agent_state=4,
            privilege_escalation_detected=None,
            red_agent_target=2, red_action_targets={0: 0, 1: 9, 2: 9, 3: 9, 4: 2, 5: 2, 6: 2},
            attacker_observed_decoy=[0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            detected=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            malware_state=[0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            ssh_access=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            escalated=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            exploited=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            bline_base_jump=False
        )
        action = 5
        max_tries = 1000
        match = False
        i = 0
        while i < max_tries and not match:
            env.set_state(state)
            o, r, done, _, info = env.step(action)
            if env.last_obs == [[0, 0, 0, 0], [0, 0, 0, 2], [1, 2, 2, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0],
                                [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 3, 4], [0, 0, 0, 1], [0, 0, 0, 0],
                                [0, 0, 0, 0]] \
                    and env.s == [[0, 0, 0, 0], [0, 0, 0, 2], [1, 1, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0],
                                  [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 2, 0], [1, 1, 2, 4], [1, 0, 0, 1], [1, 0, 0, 0],
                                  [1, 0, 0, 0]]:
                match = True
            i += 1
        assert match

    def test_set_state_39(self) -> None:
        """
        Tests the set_state method

        :return: None
        """
        config = CSLECyborgWrapperConfig(maximum_steps=100, gym_env_name="", save_trace=False, reward_shaping=True,
                                         scenario=2, red_agent_type=RedAgentType.B_LINE_AGENT)
        env = CyborgScenarioTwoWrapper(config=config)
        state = CyborgWrapperState(
            s=[[0, 0, 0, 0], [1, 1, 0, 4], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
               [0, 0, 0, 0], [1, 0, 2, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 1, 2, 2], [1, 0, 0, 0]],
            scan_state=[0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            op_server_restored=False,
            obs=[[0, 0, 0, 0], [0, 2, 0, 4], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                 [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [2, 1, 1, 2], [0, 0, 0, 0]],
            red_agent_state=2,
            privilege_escalation_detected=None,
            red_agent_target=11, red_action_targets={0: 0, 1: 11, 2: 11, 3: 11, 4: 1, 5: 1, 6: 1},
            attacker_observed_decoy=[0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
            detected=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            malware_state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            ssh_access=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            escalated=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            exploited=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            bline_base_jump=False
        )
        action = 32
        max_tries = 1000
        match = False
        i = 0
        while i < max_tries and not match:
            env.set_state(state)
            o, r, done, _, info = env.step(action)
            if env.last_obs == [[0, 0, 0, 0], [0, 2, 0, 4], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                                [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [2, 1, 1, 2],
                                [0, 0, 0, 0]] \
                    and env.s == [[0, 0, 0, 0], [1, 1, 0, 4], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                                  [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 2, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 1, 2, 2],
                                  [1, 0, 0, 0]]:
                match = True
            i += 1
        assert match

    def test_set_state_40(self) -> None:
        """
        Tests the set_state method

        :return: None
        """
        config = CSLECyborgWrapperConfig(maximum_steps=100, gym_env_name="", save_trace=False, reward_shaping=True,
                                         scenario=2, red_agent_type=RedAgentType.B_LINE_AGENT)
        env = CyborgScenarioTwoWrapper(config=config)
        state = CyborgWrapperState(
            s=[[0, 0, 0, 0], [1, 1, 0, 4], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
               [0, 0, 0, 0], [1, 0, 2, 0], [1, 0, 0, 0], [1, 0, 0, 1], [1, 1, 2, 2], [1, 0, 0, 0]],
            scan_state=[0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], op_server_restored=False,
            obs=[[0, 0, 0, 0], [1, 2, 0, 4], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                 [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 1, 3, 2], [0, 0, 0, 0]],
            red_action_targets={0: 0, 1: 11, 2: 11, 3: 11, 4: 1, 5: 1, 6: 1},
            privilege_escalation_detected=None, red_agent_state=2, red_agent_target=11,
            attacker_observed_decoy=[0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            detected=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            malware_state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            ssh_access=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            escalated=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            exploited=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            bline_base_jump=False
        )
        action = 27
        max_tries = 1000
        match = False
        i = 0
        while i < max_tries and not match:
            env.set_state(state)
            o, r, done, _, info = env.step(action)
            if env.last_obs == [[0, 0, 0, 0], [0, 2, 0, 4], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                                [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 1, 3, 2],
                                [0, 0, 0, 0]] \
                    and env.s == [[0, 0, 0, 0], [1, 1, 0, 4], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                                  [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 2, 0], [1, 0, 0, 0], [1, 0, 0, 1], [1, 1, 2, 2],
                                  [1, 0, 0, 0]]:
                match = True
            i += 1
        assert match

    def test_set_state_41(self) -> None:
        """
        Tests the set_state method

        :return: None
        """
        config = CSLECyborgWrapperConfig(maximum_steps=100, gym_env_name="", save_trace=False, reward_shaping=True,
                                         scenario=2, red_agent_type=RedAgentType.B_LINE_AGENT)
        env = CyborgScenarioTwoWrapper(config=config)
        state = CyborgWrapperState(
            s=[[0, 0, 0, 2], [0, 0, 0, 2], [1, 1, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
               [0, 0, 0, 1], [1, 0, 2, 0], [1, 0, 0, 4], [1, 1, 2, 3], [1, 0, 0, 2], [1, 0, 0, 1]],
            scan_state=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0], op_server_restored=False,
            obs=[[0, 0, 0, 2], [0, 0, 0, 2], [0, 1, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                 [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 4], [1, 2, 2, 3], [0, 0, 0, 2], [0, 0, 0, 1]],
            red_action_targets={0: 0, 1: 10, 2: 10, 3: 10, 4: 2, 5: 2, 6: 2}, privilege_escalation_detected=None,
            red_agent_state=2, red_agent_target=10, attacker_observed_decoy=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            detected=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], malware_state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            ssh_access=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], escalated=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            exploited=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            bline_base_jump=False
        )
        action = 31
        max_tries = 1000
        match = False
        i = 0
        while i < max_tries and not match:
            env.set_state(state)
            o, r, done, _, info = env.step(action)
            if env.last_obs == [[0, 0, 0, 2], [0, 0, 0, 2], [0, 1, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0],
                                [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 4], [2, 2, 1, 4], [0, 0, 0, 2],
                                [0, 0, 0, 1]] \
                    and env.s == [[0, 0, 0, 2], [0, 0, 0, 2], [1, 1, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0],
                                  [0, 0, 0, 0], [0, 0, 0, 1], [1, 0, 2, 0], [1, 0, 0, 4], [1, 1, 2, 4], [1, 0, 0, 2],
                                  [1, 0, 0, 1]]:
                match = True
            i += 1
        assert match

    def test_set_state_42(self) -> None:
        """
        Tests the set_state method

        :return: None
        """
        config = CSLECyborgWrapperConfig(maximum_steps=100, gym_env_name="", save_trace=False, reward_shaping=True,
                                         scenario=2, red_agent_type=RedAgentType.B_LINE_AGENT)
        env = CyborgScenarioTwoWrapper(config=config)
        state = CyborgWrapperState(
            s=[[0, 0, 0, 1], [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
               [0, 0, 0, 1], [1, 0, 2, 0], [1, 0, 0, 0], [1, 1, 2, 1], [1, 0, 0, 0], [1, 0, 0, 1]],
            scan_state=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0], op_server_restored=False,
            obs=[[0, 0, 0, 1], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                 [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 2, 2, 1], [0, 0, 0, 0], [0, 0, 0, 1]],
            red_action_targets={0: 0, 1: 10, 2: 10, 3: 10, 4: 2, 5: 2, 6: 2},
            privilege_escalation_detected=None, red_agent_state=2, red_agent_target=10,
            attacker_observed_decoy=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            detected=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            malware_state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            ssh_access=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            escalated=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            exploited=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            bline_base_jump=False
        )
        action = 31
        max_tries = 1000
        match = False
        i = 0
        while i < max_tries and not match:
            env.set_state(state)
            o, r, done, _, info = env.step(action)
            if env.last_obs == [[0, 0, 0, 1], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0],
                                [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [2, 2, 1, 2], [0, 0, 0, 0],
                                [0, 0, 0, 1]] \
                    and env.s == [[0, 0, 0, 1], [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0],
                                  [0, 0, 0, 0], [0, 0, 0, 1], [1, 0, 2, 0], [1, 0, 0, 0], [1, 1, 2, 2], [1, 0, 0, 0],
                                  [1, 0, 0, 1]] and env.red_agent_state == 2:
                match = True
            i += 1
        assert match

    def test_set_state_43(self) -> None:
        """
        Tests the set_state method

        :return: None
        """
        config = CSLECyborgWrapperConfig(maximum_steps=100, gym_env_name="", save_trace=False, reward_shaping=True,
                                         scenario=2, red_agent_type=RedAgentType.B_LINE_AGENT)
        env = CyborgScenarioTwoWrapper(config=config)
        state = CyborgWrapperState(
            s=[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
               [0, 0, 0, 0], [1, 0, 2, 0], [1, 0, 0, 0], [1, 1, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]],
            scan_state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0], op_server_restored=False,
            obs=[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                 [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            red_action_targets={0: 0, 1: 10}, privilege_escalation_detected=None, red_agent_state=2,
            red_agent_target=10, attacker_observed_decoy=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            detected=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], malware_state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ssh_access=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], escalated=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            exploited=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            bline_base_jump=False
        )
        action = 6
        max_tries = 1000
        match = False
        i = 0
        while i < max_tries and not match:
            env.set_state(state)
            o, r, done, _, info = env.step(action)
            if env.last_obs == [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                                [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [2, 2, 1, 0], [0, 0, 0, 0],
                                [0, 0, 0, 0]] \
                    and env.s == [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                                  [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 2, 0], [1, 0, 0, 0], [1, 1, 1, 0], [1, 0, 0, 0],
                                  [1, 0, 0, 0]] and env.red_agent_state == 3:
                match = True
            i += 1
        assert match


    def test_set_state_44(self) -> None:
        """
        Tests the set_state method

        :return: None
        """
        config = CSLECyborgWrapperConfig(maximum_steps=100, gym_env_name="", save_trace=False, reward_shaping=True,
                                         scenario=2, red_agent_type=RedAgentType.B_LINE_AGENT)
        env = CyborgScenarioTwoWrapper(config=config)
        state = CyborgWrapperState(
            s=[[1, 0, 0, 4], [1, 0, 0, 3], [1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
               [1, 1, 0, 4], [1, 0, 2, 0], [1, 0, 0, 3], [1, 1, 2, 0], [1, 0, 0, 2], [1, 0, 0, 1]],
            scan_state=[0, 0, 1, 2, 0, 0, 0, 1, 0, 0, 1, 0, 0], op_server_restored=False,
            obs=[[0, 0, 0, 4], [0, 0, 0, 3], [0, 1, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                 [0, 1, 0, 4], [0, 0, 0, 0], [0, 0, 0, 3], [0, 1, 2, 0], [0, 0, 0, 2], [0, 0, 0, 1]],
            red_action_targets={0: 0, 1: 10, 2: 10, 3: 10, 4: 2, 5: 2, 6: 2, 7: 1, 8: 3, 9: 3, 10: 3, 11: 7, 12: 7},
            privilege_escalation_detected=None, red_agent_state=12, red_agent_target=7,
            attacker_observed_decoy=[0, 0, 1, 1, 0, 0, 0, 4, 0, 0, 0, 0, 0],
            detected=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], malware_state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            ssh_access=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], escalated=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            exploited=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            bline_base_jump=False
        )
        action = 34
        max_tries = 1000
        match = False
        i = 0
        while i < max_tries and not match:
            env.set_state(state)
            o, r, done, _, info = env.step(action)
            if env.last_obs == [[0, 0, 0, 4], [0, 0, 0, 3], [0, 1, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                                [0, 0, 0, 0], [0, 1, 0, 4], [0, 0, 0, 0], [0, 0, 0, 3], [0, 1, 2, 0], [0, 0, 0, 2],
                                [0, 0, 0, 1]] \
                    and env.s == [[1, 0, 0, 4], [1, 0, 0, 3], [1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                                  [0, 0, 0, 0], [1, 1, 0, 4], [1, 0, 2, 0], [1, 0, 0, 3], [1, 1, 2, 0], [1, 0, 0, 2],
                                  [1, 0, 0, 1]]:
                match = True
            i += 1
        assert match

    def test_set_state_45(self) -> None:
        """
        Tests the set_state method

        :return: None
        """
        config = CSLECyborgWrapperConfig(maximum_steps=100, gym_env_name="", save_trace=False, reward_shaping=True,
                                         scenario=2, red_agent_type=RedAgentType.B_LINE_AGENT)
        env = CyborgScenarioTwoWrapper(config=config)
        state = CyborgWrapperState(
            s=[[0, 0, 0, 0], [0, 0, 0, 1], [1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
               [0, 0, 0, 0], [1, 0, 2, 0], [1, 1, 2, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]],
            scan_state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0], op_server_restored=False,
            obs=[[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                 [0, 0, 0, 0], [0, 0, 0, 0], [0, 2, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            red_action_targets={0: 0, 1: 9, 2: 9, 3: 9}, privilege_escalation_detected=None, red_agent_state=4,
            red_agent_target=2, attacker_observed_decoy=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            detected=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], malware_state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            ssh_access=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], escalated=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            exploited=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], bline_base_jump=False
        )
        action = 26
        max_tries = 1000
        match = False
        i = 0
        while i < max_tries and not match:
            env.set_state(state)
            o, r, done, _, info = env.step(action)
            if env.last_obs == [[0, 0, 0, 0], [0, 0, 0, 1], [1, 2, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                                [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                                [0, 0, 0, 0]] \
                    and env.s == [[0, 0, 0, 0], [0, 0, 0, 1], [1, 1, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                                  [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 2, 0], [1, 1, 2, 0], [1, 0, 0, 0], [1, 0, 0, 0],
                                  [1, 0, 0, 0]]:
                match = True
            i += 1
        assert match

    def test_set_state_46(self) -> None:
        """
        Tests the set_state method

        :return: None
        """
        config = CSLECyborgWrapperConfig(maximum_steps=100, gym_env_name="", save_trace=False, reward_shaping=True,
                                         scenario=2, red_agent_type=RedAgentType.B_LINE_AGENT)
        env = CyborgScenarioTwoWrapper(config=config)
        state = CyborgWrapperState(
            s=[[1, 0, 0, 4], [1, 1, 0, 0], [1, 0, 0, 1], [1, 1, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
               [0, 0, 0, 2], [1, 0, 2, 0], [1, 0, 0, 4], [1, 0, 0, 4], [1, 1, 2, 2], [1, 0, 0, 1]],
            scan_state=[0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0], op_server_restored=False,
            obs=[[0, 0, 0, 4], [0, 1, 0, 0], [0, 0, 0, 1], [1, 2, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                 [0, 0, 0, 2], [0, 0, 0, 0], [0, 0, 0, 4], [0, 0, 0, 4], [0, 1, 2, 2], [0, 0, 0, 1]],
            red_action_targets={0: 0, 1: 11, 2: 11, 3: 11, 4: 1, 5: 1, 6: 1, 7: 1, 8: 3},
            privilege_escalation_detected=None, red_agent_state=9, red_agent_target=3,
            attacker_observed_decoy=[0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            detected=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], malware_state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            ssh_access=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], escalated=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            exploited=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], bline_base_jump=False
        )
        action = 35
        max_tries = 1000
        match = False
        i = 0
        while i < max_tries and not match:
            env.set_state(state)
            o, r, done, _, info = env.step(action)
            if env.last_obs == [[0, 0, 0, 4], [0, 1, 0, 0], [0, 0, 0, 1], [1, 2, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0],
                                [0, 0, 0, 0], [0, 0, 0, 3], [0, 0, 0, 0], [0, 0, 0, 4], [0, 0, 0, 4], [0, 1, 2, 2],
                                [0, 0, 0, 1]] \
                    and env.s == [[1, 0, 0, 4], [1, 1, 0, 0], [1, 0, 0, 1], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0],
                                  [0, 0, 0, 0], [0, 0, 0, 3], [1, 0, 2, 0], [1, 0, 0, 4], [1, 0, 0, 4], [1, 1, 2, 2],
                                  [1, 0, 0, 1]] and env.red_agent_state == 10:
                match = True
            i += 1
        assert match

    def test_set_state_47(self) -> None:
        """
        Tests the set_state method

        :return: None
        """
        config = CSLECyborgWrapperConfig(maximum_steps=100, gym_env_name="", save_trace=False, reward_shaping=True,
                                         scenario=2, red_agent_type=RedAgentType.B_LINE_AGENT)
        env = CyborgScenarioTwoWrapper(config=config)
        state = CyborgWrapperState(
            s=[[1, 0, 0, 4], [1, 1, 0, 0], [1, 0, 0, 1], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0],
               [0, 0, 0, 0], [0, 0, 0, 3], [1, 0, 2, 0], [1, 0, 0, 4], [1, 0, 0, 4], [1, 1, 2, 2],
               [1, 0, 0, 1]],
            scan_state=[0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0], op_server_restored=False,
            obs=[[0, 0, 0, 4], [0, 1, 0, 0], [0, 0, 0, 1], [1, 2, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0],
                 [0, 0, 0, 0], [0, 0, 0, 3], [0, 0, 0, 0], [0, 0, 0, 4], [0, 0, 0, 4], [0, 1, 2, 2],
                 [0, 0, 0, 1]],
            red_action_targets={0: 0, 1: 11, 2: 11, 3: 11, 4: 1, 5: 1, 6: 1, 7: 1, 8: 3, 9: 3},
            privilege_escalation_detected=None, red_agent_state=10, red_agent_target=3,
            attacker_observed_decoy=[0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            detected=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], malware_state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            ssh_access=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], escalated=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            exploited=[0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0], bline_base_jump=False
        )
        action = 25
        max_tries = 1000
        match = False
        i = 0
        while i < max_tries and not match:
            env.set_state(state)
            o, r, done, _, info = env.step(action)
            if env.last_obs == [[0, 0, 0, 4], [0, 1, 0, 0], [0, 0, 0, 1], [0, 2, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0],
                                [0, 0, 0, 0], [0, 0, 0, 3], [0, 0, 0, 0], [0, 0, 0, 4], [0, 0, 0, 4], [0, 1, 2, 2],
                                [0, 0, 0, 1]] \
                    and env.s == [[1, 0, 0, 4], [1, 1, 0, 0], [1, 0, 0, 1], [1, 1, 2, 1], [0, 0, 0, 0], [0, 0, 0, 0],
                                  [0, 0, 0, 0], [1, 0, 0, 3], [1, 0, 2, 0], [1, 0, 0, 4], [1, 0, 0, 4], [1, 1, 2, 2],
                                  [1, 0, 0, 1]] and env.red_agent_state == 11:
                match = True
            i += 1
        assert match


    def test_set_state_48(self) -> None:
        """
        Tests the set_state method

        :return: None
        """
        config = CSLECyborgWrapperConfig(maximum_steps=100, gym_env_name="", save_trace=False, reward_shaping=True,
                                         scenario=2, red_agent_type=RedAgentType.B_LINE_AGENT)
        env = CyborgScenarioTwoWrapper(config=config)
        state = CyborgWrapperState(
            s=[[1, 0, 0, 4], [1, 1, 0, 0], [1, 0, 0, 1], [1, 1, 2, 1], [0, 0, 0, 0], [0, 0, 0, 0],
               [0, 0, 0, 0], [1, 0, 0, 3], [1, 0, 2, 0], [1, 0, 0, 4], [1, 0, 0, 4], [1, 1, 2, 2],
               [1, 0, 0, 1]],
            scan_state=[0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0], op_server_restored=False,
            obs=[[0, 0, 0, 4], [0, 1, 0, 0], [0, 0, 0, 1], [0, 2, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0],
                 [0, 0, 0, 0], [0, 0, 0, 3], [0, 0, 0, 0], [0, 0, 0, 4], [0, 0, 0, 4], [0, 1, 2, 2],
                 [0, 0, 0, 1]],
            red_action_targets={0: 0, 1: 11, 2: 11, 3: 11, 4: 1, 5: 1, 6: 1, 7: 1, 8: 3, 9: 3},
            privilege_escalation_detected=None, red_agent_state=11, red_agent_target=7,
            attacker_observed_decoy=[0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            detected=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], malware_state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            ssh_access=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], escalated=[0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            exploited=[0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0], bline_base_jump=False
        )
        action = 35
        max_tries = 1000
        match = False
        i = 0
        while i < max_tries and not match:
            env.set_state(state)
            o, r, done, _, info = env.step(action)
            # if env.s == [[1, 0, 0, 4], [1, 1, 0, 0], [1, 0, 0, 1], [1, 1, 2, 1], [0, 0, 0, 0], [0, 0, 0, 0],
            #              [0, 0, 0, 0], [1, 1, 0, 4], [1, 0, 2, 0], [1, 0, 0, 4], [1, 0, 0, 4], [1, 1, 2, 2],
            #              [1, 0, 0, 1]]:
            #     import logging
            #     logging.info("State match")
            #     logging.info(env.last_obs)
            if env.last_obs == [[0, 0, 0, 4], [0, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0],
                                [0, 0, 0, 0], [1, 2, 0, 4], [0, 0, 0, 0], [0, 0, 0, 4], [0, 0, 0, 4], [0, 1, 2, 2],
                                [0, 0, 0, 1]] \
                    and env.s == [[1, 0, 0, 4], [1, 1, 0, 0], [1, 0, 0, 1], [1, 1, 2, 1], [0, 0, 0, 0], [0, 0, 0, 0],
                                  [0, 0, 0, 0], [1, 1, 0, 4], [1, 0, 2, 0], [1, 0, 0, 4], [1, 0, 0, 4], [1, 1, 2, 2],
                                  [1, 0, 0, 1]]:
                match = True
            i += 1
        assert match
