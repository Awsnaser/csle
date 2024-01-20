from gym_csle_stopping_game.util.stopping_game_util import StoppingGameUtil
from csle_common.dao.training.multi_threshold_stopping_policy import MultiThresholdStoppingPolicy
from gym_csle_stopping_game.dao.stopping_game_config import StoppingGameConfig


class TestStoppingGameUtilSuite(object):
    """
    Test suite for stopping_game_util.py
    """

    def test_b1(self) -> None:
        """
        Tests the b1 function

        :return: None
        """
        assert sum(StoppingGameUtil.b1()) == 1

    def test_state_space(self) -> None:
        """
        Tests the state space function

        :return: None
        """
        assert sum(StoppingGameUtil.state_space()) == 3
        assert len(StoppingGameUtil.state_space()) == 3

    def test_defender_actions(self) -> None:
        """
                Tests the defender actions function

                :return: None
                """
        assert sum(StoppingGameUtil.defender_actions()) == 1
        assert len(StoppingGameUtil.defender_actions()) == 2

    def test_attacker_actions(self) -> None:
        """
        Tests the attacker actions function

        :return: None
        """
        assert sum(StoppingGameUtil.attacker_actions()) == 1
        assert len(StoppingGameUtil.attacker_actions()) == 2

    def test_observation_space(self) -> None:
        """
        Tests the observation space function

        :return: None
        """
        n = 6
        assert len(StoppingGameUtil.observation_space(n)) == n + 1

    def test_reward_tensor(self) -> None:
        """
        Tests the observation space function

        :return: None
        """
        l = 6
        example_reward_vector = StoppingGameUtil.reward_tensor(R_SLA=1, R_INT=3, R_COST=2, L=l, R_ST=5)
        assert example_reward_vector.shape == (l, 2, 2, 3)

    def test_transition_tensor(self) -> None:
        """
        Tests the transition tensor function

        :return: None
        """
        l = 6
        example_transition_tensor = StoppingGameUtil.transition_tensor(L=l, p=0.1)
        for i in range(l):
            for j in range(2):
                for k in range(2):
                    for m in range(3):
                        assert sum(example_transition_tensor[i, j, k, m])
        assert example_transition_tensor.shape == (l, 2, 2, 3, 3)

    def test_observation_tensor(self) -> None:
        """
        Tests the observation tensor function

        :return: None
        """
        n = 6
        example_observation_tensor = StoppingGameUtil.observation_tensor(n=n)
        assert example_observation_tensor.shape == (2, 2, 3, n + 1)

    def test_sample_next_state(self, example_stopping_game_util: StoppingGameUtil) -> None:
        """
        Tests the sample next state function

        :param: example_stopping_game_util an example object of StoppingGameUtil
        :return: None
        """
        example_sample_next_state = StoppingGameUtil.sample_next_state(
            T=example_stopping_game_util.transition_tensor(L=3, p=0.1), l=3, s=2, a1=1, a2=1,
            S=example_stopping_game_util.state_space())
        assert example_sample_next_state in [0, 1, 2]

    def test_sample_initial_state(self, example_stopping_game_util: StoppingGameUtil) -> None:
        """
        Tests the sample initial state function

        :param: example_stopping_game_util an example object of StoppingGameUtil
        :return: None
        """
        example_sample_initial_state = StoppingGameUtil.sample_initial_state(example_stopping_game_util.b1())
        assert example_sample_initial_state in example_stopping_game_util.b1()
        assert sum(example_stopping_game_util.b1()) == 1

    def test_sample_next_observation(self, example_stopping_game_util: StoppingGameUtil) -> None:
        """
        Tests the sample next observation function

        :param: example_stopping_game_util an example object of StoppingGameUtil
        :return: None
        """
        n = 6
        z = example_stopping_game_util.observation_tensor(n)
        O = StoppingGameUtil.observation_space(n=n)
        example_next_obs = StoppingGameUtil.sample_next_observation(
            Z=z, s_prime=2, O=O)
        assert example_next_obs in O

    def test_bayes_filter(self, example_stopping_game_util: StoppingGameUtil,
                          example_attacker_strategy: MultiThresholdStoppingPolicy,
                          example_stopping_game_config: StoppingGameConfig) -> None:
        """
        Tests the byte filter function

        :param: example_stopping_game_util an example object of StoppingGameUtil
        :return: None
        """
        b_prime_s_prime = StoppingGameUtil.bayes_filter(s_prime=0, o=0, a1=0, b=example_stopping_game_util.b1(),
                                                        pi2=example_attacker_strategy.stage_policy([0, 0]), l=1,
                                                        config=example_stopping_game_config)
        assert b_prime_s_prime <= 1 and b_prime_s_prime >= 0
