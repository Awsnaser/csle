from gym_pycr_ctf.envs.derived_envs.level1.simulation.pycr_ctf_level1_sim_env import PyCRCTFLevel1Sim1Env
from gym_pycr_ctf.envs.derived_envs.level1.emulation.pycr_ctf_level1_emulation_env import PyCRCTFLevel1Emulation1Env
from gym_pycr_ctf.dao.network.emulation_config import EmulationConfig
import gym
import time
import numpy as np
from gym_pycr_ctf.util.experiments_util import util
from gym_pycr_ctf.envs.config.generator.env_config_generator import EnvConfigGenerator

def test_env(env_name : str, num_steps : int):
    # emulation_config = emulationConfig(server_ip="172.31.212.91", agent_ip="172.18.1.191",
    #                                agent_username="agent", agent_pw="agent", server_connection=True,
    #                                server_private_key_file="/Users/kimham/.ssh/pycr_id_rsa",
    #                                server_username="kim")
    # emulation_config = emulationConfig(server_ip="172.31.212.91", agent_ip="172.18.1.191",
    #                                agent_username="agent", agent_pw="agent", server_connection=True,
    #                                server_private_key_file="/home/kim/.ssh/id_rsa",
    #                                server_username="kim")
    # containers_configs = EnvConfigGenerator.get_all_envs_containers_config(
    #     "/home/kim/pycr/emulation-envs/minigames/network_intrusion/ctf/001/random_many/")
    # flags_configs = EnvConfigGenerator.get_all_envs_flags_config(
    #     "/home/kim/pycr/emulation-envs/minigames/network_intrusion/ctf/001/random_many/")
    # eval_containers_configs = EnvConfigGenerator.get_all_envs_containers_config(
    #     "/home/kim/pycr/emulation-envs/minigames/network_intrusion/ctf/001/random_many_2/")
    # eval_flags_configs = EnvConfigGenerator.get_all_envs_flags_config(
    #     "/home/kim/pycr/emulation-envs/minigames/network_intrusion/ctf/001/random_many_2/")

    containers_configs = EnvConfigGenerator.get_all_envs_containers_config(
        "/home/kim/storage/workspace/pycr/emulation-envs/minigames/network_intrusion/ctf/001/random_many_2/backup/random_many/")
    flags_configs = EnvConfigGenerator.get_all_envs_flags_config(
        "/home/kim/storage/workspace/pycr/emulation-envs/minigames/network_intrusion/ctf/001/random_many_2/backup/random_many/")
    eval_containers_configs = EnvConfigGenerator.get_all_envs_containers_config(
        "/home/kim/storage/workspace/pycr/emulation-envs/minigames/network_intrusion/ctf/001/random_many/backup/random_many_2/")
    eval_env_flags_configs = EnvConfigGenerator.get_all_envs_flags_config(
        "/home/kim/storage/workspace/pycr/emulation-envs/minigames/network_intrusion/ctf/001/random_many/backup/random_many_2/")


    containers_config = util.read_containers_config(
        "/home/kim/storage/workspace/pycr/emulation-envs/minigames/network_intrusion/ctf/001/random/containers.json")
    flags_config = util.read_flags_config(
        "/home/kim/storage/workspace/pycr/emulation-envs/minigames/network_intrusion/ctf/001/random/flags.json")
    max_num_nodes_train = max(list(map(lambda x: len(x.containers), containers_configs)))
    max_num_nodes_eval = max(list(map(lambda x: len(x.containers), eval_containers_configs)))
    max_num_nodes = max(max_num_nodes_train, max_num_nodes_eval)

    emulation_config = EmulationConfig(agent_ip=containers_config.agent_ip, agent_username="agent", agent_pw="agent",
                                     port_forward_next_port=9600,
                                     server_connection=False, warmup=True, warmup_iterations=500
                                     )

    # emulation_config = emulationConfig(agent_ip=containers_config.agent_ip, agent_username="agent", agent_pw="agent",
    #                                port_forward_next_port=9600,
    #                                server_connection=True, server_private_key_file="/home/kim/.ssh/id_rsa",
    #                                server_username="kim", server_ip="172.31.212.92",
    #                                warmup=True, warmup_iterations=500
    #                                )

    env = gym.make(env_name, env_config=None, emulation_config=emulation_config,
                   containers_config=containers_config, flags_config=flags_config, num_nodes=max_num_nodes)
    env.env_config.max_episode_length = 1000000000
    env.env_config.manual_play = True

    env.reset()

    num_actions = env.env_config.attacker_action_conf.num_actions
    actions = np.array(list(range(num_actions)))
    print("num actions:{}".format(num_actions))
    tot_rew = 0
    for i in range(num_steps):
        print(i)
        legal_actions = list(filter(lambda x: env.is_attack_action_legal(x, env.env_config, env.env_state), actions))
        action = np.random.choice(legal_actions)
        obs, reward, done, info = env.step(action)
        tot_rew += reward
        #env.render()
        if done:
            print("done")
            tot_rew = 0
            env.reset()
        #time.sleep(0.001)
        #time.sleep(0.5)
    env.reset()
    env.close()


def test_all():
    #test_env("pycr-ctf-random-emulation-v1", num_steps=1000000000)
    test_env("pycr-ctf-random-generated-sim-v1", num_steps=1000000000)
    #pycr-ctf-random-generated-sim-v1

if __name__ == '__main__':
    test_all()