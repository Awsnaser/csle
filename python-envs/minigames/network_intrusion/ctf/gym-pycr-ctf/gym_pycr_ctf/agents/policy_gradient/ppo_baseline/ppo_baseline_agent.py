"""
An agent for the cgc-bta env that uses the PPO Policy Gradient algorithm from OpenAI stable baselines
"""
import time
import torch
import math

from gym_pycr_ctf.rendering import PyCrCTFMonitor
#from gym_pycr_ctf.envs.pycr_ctf_env import PyCRCTFEnv
from gym_pycr_ctf.dao.experiment.experiment_result import ExperimentResult
from gym_pycr_ctf.agents.train_agent import TrainAgent
from gym_pycr_ctf.agents.config.agent_config import AgentConfig
from gym_pycr_ctf.agents.policy_gradient.ppo_baseline.impl.ppo.ppo import PPO
from gym_pycr_ctf.agents.openai_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from gym_pycr_ctf.agents.openai_baselines.common.vec_env.subproc_vec_env import SubprocVecEnv


class PPOBaselineAgent(TrainAgent):
    """
    An agent for the pycr-ctf env that uses the PPO Policy Gradient algorithm from OpenAI stable baselines
    """

    def __init__(self, env, config: AgentConfig, eval_env):
        """
        Initialize environment and hyperparameters

        :param config: the configuration
        """
        super(PPOBaselineAgent, self).__init__(env, config, eval_env)

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

        model = PPO(policy, self.env,
                    batch_size=self.config.mini_batch_size,
                    learning_rate=self.config.alpha,
                    n_steps=self.config.batch_size,
                    n_epochs=self.config.optimization_iterations,
                    gamma=self.config.gamma,
                    gae_lambda=self.config.gae_lambda,
                    clip_range=self.config.eps_clip,
                    max_grad_norm=self.config.max_gradient_norm,
                    verbose=1,
                    seed=self.config.random_seed,
                    policy_kwargs=policy_kwargs,
                    device=device,
                    agent_config=self.config,
                    vf_coef=self.config.vf_coef,
                    ent_coef=self.config.ent_coef,
                    use_sde=self.config.use_sde,
                    sde_sample_freq=self.config.sde_sample_freq,
                    env_2=self.eval_env
                    )

        if self.config.load_path is not None:
            PPO.load(self.config.load_path, policy, agent_config=self.config)

        # Eval config
        time_str = str(time.time())
        if self.config.video_dir is None:
            raise AssertionError("Video is set to True but no video_dir is provided, please specify "
                                 "the video_dir argument")
        if isinstance(self.env, DummyVecEnv):
            train_eval_env_i = self.env
            train_eval_env = train_eval_env_i
        elif isinstance(self.env, SubprocVecEnv):
            train_eval_env_i = self.env
            train_eval_env = train_eval_env_i
        else:
            train_eval_env_i = self.env
            if train_eval_env_i is not None:
                train_eval_env = PyCrCTFMonitor(train_eval_env_i, self.config.video_dir + "/" + time_str, force=True,
                                          video_frequency=self.config.video_frequency, openai_baseline=True)
                train_eval_env.metadata["video.frames_per_second"] = self.config.video_fps
            else:
                train_eval_env = None

        eval_env = None
        if isinstance(self.eval_env, DummyVecEnv) or isinstance(self.eval_env, SubprocVecEnv):
            eval_env = self.eval_env
        else:
            if self.eval_env is not None:
                eval_env = PyCrCTFMonitor(self.eval_env, self.config.video_dir + "/" + time_str, force=True,
                                                    video_frequency=self.config.video_frequency, openai_baseline=True)
                eval_env.metadata["video.frames_per_second"] = self.config.video_fps

        model.learn(total_timesteps=self.config.num_episodes,
                    log_interval=self.config.train_log_frequency,
                    eval_freq=self.config.eval_frequency,
                    n_eval_episodes=self.config.eval_episodes,
                    eval_env=train_eval_env,
                    eval_env_2=eval_env
                    )

        self.config.logger.info("Training Complete")

        # Save networks
        try:
            model.save_model()
        except Exception as e:
            print("There was en error saving the model:{}".format(str(e)))

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