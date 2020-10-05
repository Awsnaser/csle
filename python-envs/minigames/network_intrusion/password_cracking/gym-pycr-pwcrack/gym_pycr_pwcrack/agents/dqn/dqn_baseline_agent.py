"""
An agent for the cgc-bta env that uses the DQN algorithm from OpenAI stable baselines
"""
import time
import torch
import math

from gym_pycr_pwcrack.envs.rendering.video.pycr_pwcrack_monitor import PycrPwCrackMonitor
from gym_pycr_pwcrack.envs.pycr_pwcrack_env import PyCRPwCrackEnv
from gym_pycr_pwcrack.dao.experiment.experiment_result import ExperimentResult
from gym_pycr_pwcrack.agents.train_agent import TrainAgent
from gym_pycr_pwcrack.agents.config.agent_config import AgentConfig
from gym_pycr_pwcrack.agents.dqn.impl.dqn import DQN

class DQNBaselineAgent(TrainAgent):
    """
    An agent for the cgc-bta env that uses the PPO Policy Gradient algorithm from OpenAI stable baselines
    """

    def __init__(self, env: PyCRPwCrackEnv, config: AgentConfig):
        """
        Initialize environment and hyperparameters

        :param config: the configuration
        """
        super(DQNBaselineAgent, self).__init__(env, config)

    def train(self) -> ExperimentResult:
        """
        Starts the training loop and returns the result when complete

        :return: the training result
        """

        # Custom MLP policy
        net_arch = []
        pi_arch = []
        vf_arch = []
        for l in range(self.config.shared_layers):
            net_arch.append(self.config.shared_hidden_dim)
        for l in range(self.config.pi_hidden_layers):
            pi_arch.append(self.config.pi_hidden_dim)
        for l in range(self.config.vf_hidden_layers):
            vf_arch.append(self.config.vf_hidden_dim)


        net_dict = {"pi":pi_arch, "vf":vf_arch}
        net_arch.append(net_dict)

        policy_kwargs = dict(activation_fn=self.get_hidden_activation(), net_arch=net_arch)
        device = "cpu" if not self.config.gpu else "cuda:" + str(self.config.gpu_id)
        policy = "MlpPolicy"

        if self.config.lr_progress_decay:
            temp = self.config.alpha
            lr_decay_func = lambda x: temp*math.pow(x, self.config.lr_progress_power_decay)
            self.config.alpha = lr_decay_func
        model = DQN(
            policy, self.env, learning_rate=self.config.alpha, buffer_size=self.config.buffer_size,
            learning_starts=self.config.learning_starts,batch_size=self.config.batch_size,
            tau=self.config.tau,gamma=self.config.gamma,train_freq=self.config.train_freq,
            gradient_steps=self.config.gradient_steps,target_update_interval=self.config.target_update_interval,
            exploration_fraction=self.config.exploration_fraction,
            exploration_initial_eps=self.config.exploration_initial_eps,
            exploration_final_eps=self.config.exploration_final_eps,
            agent_config=self.config
        )

        if self.config.load_path is not None:
            DQN.load(self.config.load_path, policy, agent_config=self.config)


        # Video config
        if self.config.video or self.config.gifs:
            time_str = str(time.time())
            if self.config.video_dir is None:
                raise AssertionError("Video is set to True but no video_dir is provided, please specify "
                                     "the video_dir argument")
            eval_env = PycrPwCrackMonitor(self.env, self.config.video_dir + "/" + time_str, force=True,
                                      video_frequency=self.config.video_frequency, openai_baseline=True)
            eval_env.metadata["video.frames_per_second"] = self.config.video_fps

        model.learn(total_timesteps=self.config.num_episodes,
                    log_interval=self.config.train_log_frequency,
                    eval_freq=self.config.eval_frequency,
                    n_eval_episodes=self.config.eval_episodes,
                    eval_env=eval_env)

        self.config.logger.info("Training Complete")

        # Save networks
        model.save_model()

        # Save other game data
        if self.config.save_dir is not None:
            time_str = str(time.time())
            model.train_result.to_csv(self.config.save_dir + "/" + time_str + "_train_results_checkpoint.csv")
            model.eval_result.to_csv(self.config.save_dir + "/" + time_str + "_eval_results_checkpoint.csv")

        self.train_result = model.train_result
        self.eval_result = model.eval_result
        return model.train_result

    def get_hidden_activation(self):
        """
        Interprets the hidden activation

        :return: the hidden activation function
        """
        return torch.nn.Tanh
        if self.config.hidden_activation == "ReLU":
            return torch.nn.ReLU
        elif self.config.hidden_activation == "LeakyReLU":
            return torch.nn.LeakyReLU
        elif self.config.hidden_activation == "LogSigmoid":
            return torch.nn.LogSigmoid
        elif self.config.hidden_activation == "PReLU":
            return torch.nn.PReLU
        elif self.config.hidden_activation == "Sigmoid":
            return torch.nn.Sigmoid
        elif self.config.hidden_activation == "Softplus":
            return torch.nn.Softplus
        elif self.config.hidden_activation == "Tanh":
            return torch.nn.Tanh
        else:
            raise ValueError("Activation type: {} not recognized".format(self.config.hidden_activation))


    def get_action(self, s, eval=False, attacker=True) -> int:
        raise NotImplemented("not implemented")

    def eval(self, log=True) -> ExperimentResult:
        raise NotImplemented("not implemented")