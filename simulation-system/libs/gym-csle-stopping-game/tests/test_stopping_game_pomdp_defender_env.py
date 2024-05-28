from gym_csle_stopping_game.envs.stopping_game_pomdp_defender_env import (
    StoppingGamePomdpDefenderEnv,
)
from gym_csle_stopping_game.dao.stopping_game_config import StoppingGameConfig
from gym_csle_stopping_game.dao.stopping_game_defender_pomdp_config import (
    StoppingGameDefenderPomdpConfig,
)
from gym_csle_stopping_game.envs.stopping_game_env import StoppingGameEnv
from csle_common.dao.training.policy import Policy
import pytest
from unittest.mock import MagicMock
import numpy as np


class TestStoppingGamePomdpDefenderEnvSuite:
    """
    Test suite for stopping_game_pomdp_defender_env.py
    """

    @pytest.fixture(autouse=True)
    def setup_env(self) -> None:
        """
        Sets up the configuration of the stopping game

        :return: None
        """
        env_name = "test_env"
        T = np.array([[[0.1, 0.9], [0.4, 0.6]], [[0.7, 0.3], [0.2, 0.8]]])
        O = np.array([0, 1])
        Z = np.array([[[0.8, 0.2], [0.5, 0.5]], [[0.4, 0.6], [0.9, 0.1]]])
        R = np.zeros((2, 3, 3, 3))
        S = np.array([0, 1, 2])
        A1 = np.array([0, 1, 2])
        A2 = np.array([0, 1, 2])
        L = 2
        R_INT = 1
        R_COST = 2
        R_SLA = 3
        R_ST = 4
        b1 = np.array([0.6, 0.4])
        save_dir = "save_directory"
        checkpoint_traces_freq = 100
        gamma = 0.9
        compute_beliefs = True
        save_trace = True
        self.config = StoppingGameConfig(
            env_name,
            T,
            O,
            Z,
            R,
            S,
            A1,
            A2,
            L,
            R_INT,
            R_COST,
            R_SLA,
            R_ST,
            b1,
            save_dir,
            checkpoint_traces_freq,
            gamma,
            compute_beliefs,
            save_trace,
        )

    def test_init_(self) -> None:
        """
        Tests the initializing function

        :return: None
        """
        # Mock the attacker strategy
        attacker_strategy = MagicMock(spec=Policy)
        # Create the defender POMDP configuration
        defender_pomdp_config = StoppingGameDefenderPomdpConfig(
            env_name="test_env",
            stopping_game_config=self.config,
            attacker_strategy=attacker_strategy,
            stopping_game_name="csle-stopping-game-v1",
        )
        # Initialize the StoppingGamePomdpDefenderEnv
        env = StoppingGamePomdpDefenderEnv(config=defender_pomdp_config)
        assert env.config == defender_pomdp_config
        assert env.observation_space == self.config.defender_observation_space()
        assert env.action_space == self.config.defender_action_space()
        assert env.static_attacker_strategy == attacker_strategy
        assert not env.viewer

    def test_reset(self) -> None:
        """
        Tests the function for reseting the environment state

        :return: None
        """
        attacker_strategy = MagicMock(spec=Policy)
        defender_pomdp_config = StoppingGameDefenderPomdpConfig(
            env_name="test_env",
            stopping_game_config=self.config,
            attacker_strategy=attacker_strategy,
            stopping_game_name="csle-stopping-game-v1",
        )
        env = StoppingGamePomdpDefenderEnv(config=defender_pomdp_config)
        _, info = env.reset()
        assert info

    def test_render(self) -> None:
        """
        Tests the function for rendering the environment

        :return: None
        """
        attacker_strategy = MagicMock(spec=Policy)
        defender_pomdp_config = StoppingGameDefenderPomdpConfig(
            env_name="test_env",
            stopping_game_config=self.config,
            attacker_strategy=attacker_strategy,
            stopping_game_name="csle-stopping-game-v1",
        )
        env = StoppingGamePomdpDefenderEnv(config=defender_pomdp_config)
        with pytest.raises(NotImplementedError):
            env.render("human")

    def test_is_defense_action_legal(self) -> None:
        """
        Tests the function of checking whether a defender action in the environment is legal or not

        :return: None
        """
        attacker_strategy = MagicMock(spec=Policy)
        defender_pomdp_config = StoppingGameDefenderPomdpConfig(
            env_name="test_env",
            stopping_game_config=self.config,
            attacker_strategy=attacker_strategy,
            stopping_game_name="csle-stopping-game-v1",
        )
        env = StoppingGamePomdpDefenderEnv(config=defender_pomdp_config)
        assert env.is_defense_action_legal(1)

    def test_is_attack_action_legal(self) -> None:
        """
        Tests the function of checking whether an attacker action in the environment is legal or not

        :return: None
        """
        attacker_strategy = MagicMock(spec=Policy)
        defender_pomdp_config = StoppingGameDefenderPomdpConfig(
            env_name="test_env",
            stopping_game_config=self.config,
            attacker_strategy=attacker_strategy,
            stopping_game_name="csle-stopping-game-v1",
        )
        env = StoppingGamePomdpDefenderEnv(config=defender_pomdp_config)
        assert env.is_attack_action_legal(1)

    def test_get_traces(self) -> None:
        """
        Tests the function of getting the list of simulation traces

        :return: None
        """
        attacker_strategy = MagicMock(spec=Policy)
        defender_pomdp_config = StoppingGameDefenderPomdpConfig(
            env_name="test_env",
            stopping_game_config=self.config,
            attacker_strategy=attacker_strategy,
            stopping_game_name="csle-stopping-game-v1",
        )
        env = StoppingGamePomdpDefenderEnv(config=defender_pomdp_config)
        assert env.get_traces() == StoppingGameEnv(self.config).traces

    def test_reset_traces(self) -> None:
        """
        Tests the function of resetting the list  of traces

        :return: None
        """
        attacker_strategy = MagicMock(spec=Policy)
        defender_pomdp_config = StoppingGameDefenderPomdpConfig(
            env_name="test_env",
            stopping_game_config=self.config,
            attacker_strategy=attacker_strategy,
            stopping_game_name="csle-stopping-game-v1",
        )
        env = StoppingGamePomdpDefenderEnv(config=defender_pomdp_config)
        env.traces = ["trace1", "trace2"]
        env.reset_traces()
        assert StoppingGameEnv(self.config).traces == []

    def test_set_model(self) -> None:
        """
        Tests the function for setting the model

        :return: None
        """
        attacker_strategy = MagicMock(spec=Policy)
        defender_pomdp_config = StoppingGameDefenderPomdpConfig(
            env_name="test_env",
            stopping_game_config=self.config,
            attacker_strategy=attacker_strategy,
            stopping_game_name="csle-stopping-game-v1",
        )
        env = StoppingGamePomdpDefenderEnv(config=defender_pomdp_config)
        mock_model = MagicMock()
        env.set_model(mock_model)
        assert env.model == mock_model

    def test_set_state(self) -> None:
        """
        Tests the function for setting the state

        :return: None
        """
        attacker_strategy = MagicMock(spec=Policy)
        defender_pomdp_config = StoppingGameDefenderPomdpConfig(
            env_name="test_env",
            stopping_game_config=self.config,
            attacker_strategy=attacker_strategy,
            stopping_game_name="csle-stopping-game-v1",
        )
        env = StoppingGamePomdpDefenderEnv(config=defender_pomdp_config)
        assert not env.set_state(1)

    def test_get_observation_from_history(self) -> None:
        """
        Tests the function for getting a defender observatin (belief) from a history
        """
        attacker_strategy = MagicMock(spec=Policy)
        defender_pomdp_config = StoppingGameDefenderPomdpConfig(
            env_name="test_env",
            stopping_game_config=self.config,
            attacker_strategy=attacker_strategy,
            stopping_game_name="csle-stopping-game-v1",
        )
        env = StoppingGamePomdpDefenderEnv(config=defender_pomdp_config)
        history = [1, 2, 3]
        l = self.config.L
        pi2 = env.static_attacker_strategy.stage_policy(o=0)
        assert env.get_observation_from_history(history) == StoppingGameEnv(
            self.config
        ).get_observation_from_history(history, pi2, l)

    def test_is_state_terminal(self) -> None:
        """
        Tests the funciton for checking whether a state is terminal or not

        :return: None
        """
        attacker_strategy = MagicMock(spec=Policy)
        defender_pomdp_config = StoppingGameDefenderPomdpConfig(
            env_name="test_env",
            stopping_game_config=self.config,
            attacker_strategy=attacker_strategy,
            stopping_game_name="csle-stopping-game-v1",
        )
        env = StoppingGamePomdpDefenderEnv(config=defender_pomdp_config)
        assert env.is_state_terminal(1) == StoppingGameEnv(
            self.config
        ).is_state_terminal(1)

    def test_generate_random_particles(self) -> None:
        """
        Tests the funtion of generating a random list of state particles from a given observation

        :return: None
        """
        attacker_strategy = MagicMock(spec=Policy)
        defender_pomdp_config = StoppingGameDefenderPomdpConfig(
            env_name="test_env",
            stopping_game_config=self.config,
            attacker_strategy=attacker_strategy,
            stopping_game_name="csle-stopping-game-v1",
        )
        env = StoppingGamePomdpDefenderEnv(config=defender_pomdp_config)
        num_particles = 10
        particles = env.generate_random_particles(o=1, num_particles=num_particles)
        assert len(particles) == num_particles
        assert all(p in [0, 1] for p in particles)

        num_particles = 0
        particles = env.generate_random_particles(o=1, num_particles=num_particles)
        assert len(particles) == num_particles

    def test_get_actions_from_particles(self) -> None:
        """
        Tests the function for pruning the set of actions based on the current particle set

        :return: None
        """
        attacker_strategy = MagicMock(spec=Policy)
        defender_pomdp_config = StoppingGameDefenderPomdpConfig(
            env_name="test_env",
            stopping_game_config=self.config,
            attacker_strategy=attacker_strategy,
            stopping_game_name="csle-stopping-game-v1",
        )
        env = StoppingGamePomdpDefenderEnv(config=defender_pomdp_config)
        particles = [1, 2, 3]
        t = 0
        observation = 0
        expected_actions = [0, 1, 2]
        assert (
            env.get_actions_from_particles(particles, t, observation)
            == expected_actions
        )

    def test_step(self) -> None:
        """
        Tests the function for taking a step in the environment by executing the given action

        :return: None
        """
        attacker_strategy = MagicMock(spec=Policy)
        defender_pomdp_config = StoppingGameDefenderPomdpConfig(
            env_name="test_env",
            stopping_game_config=self.config,
            attacker_strategy=attacker_strategy,
            stopping_game_name="csle-stopping-game-v1",
        )
        env = StoppingGamePomdpDefenderEnv(config=defender_pomdp_config)
        a1 = 2
        defender_obs, reward, terminated, truncated, info = env.step(a1)
        assert isinstance(defender_obs, int)
        assert isinstance(reward, int)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
