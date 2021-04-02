from typing import Tuple
from gym_pycr_ctf.dao.network.env_state import EnvState
from gym_pycr_ctf.dao.network.env_config import EnvConfig
from gym_pycr_ctf.dao.action.attacker.attacker_action import AttackerAction
from gym_pycr_ctf.dao.defender_dynamics.defender_dynamics_model import DefenderDynamicsModel
from gym_pycr_ctf.envs.logic.common.env_dynamics_util import EnvDynamicsUtil


class DefenderBeliefStateSimulator:
    """
    Class that simulates belief state transitions of the defender
    """

    @staticmethod
    def transition(s: EnvState, a: AttackerAction, env_config: EnvConfig,
                   defender_dynamics_model: DefenderDynamicsModel) \
            -> Tuple[EnvState, int, bool]:
        """
        Simulates a belief state transition of the defender

        :param s: the current state
        :param a: the action to take
        :param env_config: the environment configuration
        :param defender_dynamics_model: dynamics model of the defender
        :return: s_prime, reward, done
        """
        logged_in_ips_str = EnvDynamicsUtil.logged_in_ips_str(env_config=env_config, a=a, s=a)

        num_new_alerts = 0
        num_new_priority = 0
        num_new_severe_alerts = 0
        num_new_warning_alerts = 0

        # Sample transitions
        if (a.id.value, logged_in_ips_str) in defender_dynamics_model.norm_num_new_alerts:
            num_new_alerts = defender_dynamics_model.norm_num_new_alerts[(a.id.value, logged_in_ips_str)].rvs()

        if (a.id.value, logged_in_ips_str) in defender_dynamics_model.norm_num_new_priority:
            num_new_priority = defender_dynamics_model.norm_num_new_priority[(a.id.value, logged_in_ips_str)].rvs()

        if (a.id.value, logged_in_ips_str) in defender_dynamics_model.norm_num_new_severe_alerts:
            num_new_severe_alerts = defender_dynamics_model.norm_num_new_severe_alerts[(a.id.value, logged_in_ips_str)].rvs()

        if (a.id.value, logged_in_ips_str) in defender_dynamics_model.norm_num_new_warning_alerts:
            num_new_warning_alerts = defender_dynamics_model.norm_num_new_warning_alerts[(a.id.value, logged_in_ips_str)].rvs()

        # Update network state
        s.defender_obs_state.num_alerts_total = s.defender_obs_state.num_alerts_total + num_new_alerts
        s.defender_obs_state.num_alerts_recent = num_new_alerts
        s.defender_obs_state.sum_priority_alerts_total = s.defender_obs_state.sum_priority_alerts_total + num_new_priority
        s.defender_obs_state.sum_priority_alerts_recent = num_new_priority
        s.defender_obs_state.num_warning_alerts_total = s.defender_obs_state.num_warning_alerts_total + num_new_warning_alerts
        s.defender_obs_state.num_warning_alerts_recent = num_new_warning_alerts
        s.defender_obs_state.num_severe_alerts_total = s.defender_obs_state.num_severe_alerts_total + num_new_severe_alerts
        s.defender_obs_state.num_severe_alerts_recent = num_new_severe_alerts

        # Update machines state
        for m in s.defender_obs_state.machines:

            if m.ip in defender_dynamics_model.machines_dynamics_model:
                m_dynamics = defender_dynamics_model.machines_dynamics_model[m.ip]

                num_new_open_connections = 0
                num_new_failed_login_attempts = 0
                num_new_users = 0
                num_new_logged_in_users = 0
                num_new_login_events = 0
                num_new_processes = 0

                # Sample transitions
                if (a.id.value, logged_in_ips_str) in m_dynamics.norm_num_new_open_connections:
                    num_new_open_connections = m_dynamics.norm_num_new_open_connections[(a.id.value, logged_in_ips_str)].rvs()

                if (a.id.value, logged_in_ips_str) in m_dynamics.norm_num_new_failed_login_attempts:
                    num_new_failed_login_attempts = m_dynamics.norm_num_new_failed_login_attempts[
                        (a.id.value, logged_in_ips_str)].rvs()

                if (a.id.value, logged_in_ips_str) in m_dynamics.norm_num_new_users:
                    num_new_users = m_dynamics.norm_num_new_users[
                        (a.id.value, logged_in_ips_str)].rvs()

                if (a.id.value, logged_in_ips_str) in m_dynamics.norm_num_new_logged_in_users:
                    num_new_logged_in_users = m_dynamics.norm_num_new_logged_in_users[
                        (a.id.value, logged_in_ips_str)].rvs()

                if (a.id.value, logged_in_ips_str) in m_dynamics.norm_num_new_login_events:
                    num_new_login_events = m_dynamics.norm_num_new_login_events[
                        (a.id.value, logged_in_ips_str)].rvs()

                if (a.id.value, logged_in_ips_str) in m_dynamics.norm_num_new_processes:
                    num_new_processes = m_dynamics.norm_num_new_processes[
                        (a.id.value, logged_in_ips_str)].rvs()

                # Update network state
                m.num_open_connections = m.num_open_connections + num_new_open_connections
                m.num_open_connections_recent = num_new_open_connections
                m.num_failed_login_attempts = m.num_failed_login_attempts + num_new_failed_login_attempts
                m.num_failed_login_attempts_recent = num_new_failed_login_attempts
                m.num_users = m.num_users + num_new_users
                m.num_users_recent = num_new_users
                m.num_logged_in_users = m.num_logged_in_users + num_new_logged_in_users
                m.num_logged_in_users_recent = num_new_logged_in_users
                m.num_login_events = m.num_login_events + num_new_login_events
                m.num_login_events_recent= num_new_login_events
                m.num_processes = m.num_processes + num_new_processes
                m.num_processes_recent = num_new_processes

        return s, 0, True
