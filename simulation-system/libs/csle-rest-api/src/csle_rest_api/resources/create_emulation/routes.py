"""
Routes and sub-resources for the /create-emulation resource
"""
from typing import Tuple
import csle_common.constants.constants as constants
from flask import Blueprint, jsonify, request, Response
import csle_rest_api.constants.constants as api_constants
import csle_rest_api.util.rest_api_util as rest_api_util

import json

from typing import Dict, List, Union
import argparse
import os
import multiprocessing
import csle_common.constants.constants as constants
import csle_collector.constants.constants as collector_constants
from csle_collector.client_manager.dao.constant_arrival_config import ConstantArrivalConfig
from csle_collector.client_manager.dao.workflows_config import WorkflowsConfig
from csle_collector.client_manager.dao.workflow_service import WorkflowService
from csle_collector.client_manager.dao.workflow_markov_chain import WorkflowMarkovChain
from csle_collector.client_manager.dao.client import Client
from csle_common.dao.emulation_config.topology_config import TopologyConfig
from csle_common.dao.emulation_config.node_firewall_config import NodeFirewallConfig
from csle_common.dao.emulation_config.default_network_firewall_config import DefaultNetworkFirewallConfig
from csle_common.dao.emulation_config.containers_config import ContainersConfig
from csle_common.dao.emulation_config.node_container_config import NodeContainerConfig
from csle_common.dao.emulation_config.container_network import ContainerNetwork
from csle_common.dao.emulation_config.flags_config import FlagsConfig
from csle_common.dao.emulation_config.node_flags_config import NodeFlagsConfig
from csle_common.dao.emulation_config.resources_config import ResourcesConfig
from csle_common.dao.emulation_config.node_resources_config import NodeResourcesConfig
from csle_common.dao.emulation_config.node_network_config import NodeNetworkConfig
from csle_common.dao.emulation_config.packet_loss_type import PacketLossType
from csle_common.dao.emulation_config.packet_delay_distribution_type import PacketDelayDistributionType
from csle_common.dao.emulation_config.traffic_config import TrafficConfig
from csle_common.dao.emulation_config.node_traffic_config import NodeTrafficConfig
from csle_common.dao.emulation_config.users_config import UsersConfig
from csle_common.dao.emulation_config.node_users_config import NodeUsersConfig
from csle_common.dao.emulation_config.vulnerabilities_config import VulnerabilitiesConfig
from csle_common.dao.emulation_config.emulation_env_config import EmulationEnvConfig
from csle_common.controllers.emulation_env_controller import EmulationEnvController
from csle_common.dao.emulation_config.client_population_config import ClientPopulationConfig
from csle_common.dao.emulation_config.kafka_config import KafkaConfig
from csle_common.dao.emulation_config.kafka_topic import KafkaTopic
from csle_common.util.experiment_util import ExperimentUtil
from csle_common.dao.emulation_config.flag import Flag
from csle_common.dao.emulation_config.node_vulnerability_config import NodeVulnerabilityConfig
from csle_common.dao.emulation_config.credential import Credential
from csle_common.dao.emulation_config.vulnerability_type import VulnType
from csle_common.dao.emulation_config.transport_protocol import TransportProtocol
from csle_common.dao.emulation_config.node_services_config import NodeServicesConfig
from csle_common.dao.emulation_config.services_config import ServicesConfig
from csle_common.dao.emulation_config.network_service import NetworkService
from csle_common.dao.emulation_config.ovs_config import OVSConfig
from csle_common.dao.emulation_config.sdn_controller_config import SDNControllerConfig
from csle_common.dao.emulation_config.user import User
from csle_common.dao.emulation_action.attacker.emulation_attacker_action import EmulationAttackerAction
from csle_common.dao.emulation_config.host_manager_config import HostManagerConfig
from csle_common.dao.emulation_config.snort_ids_manager_config import SnortIDSManagerConfig
from csle_common.dao.emulation_config.ossec_ids_manager_config import OSSECIDSManagerConfig
from csle_common.dao.emulation_config.docker_stats_manager_config import DockerStatsManagerConfig
from csle_common.dao.emulation_config.elk_config import ElkConfig
from csle_common.dao.emulation_config.beats_config import BeatsConfig
from csle_common.dao.emulation_config.node_beats_config import NodeBeatsConfig

def vulnerabilities(emulation_data:json) -> VulnerabilitiesConfig:

    # This function has problems. In this function in the front end the credentials are not defined and prepared on
    # the web page.

    print(emulation_data)

    vulns = []
    # *** We need to add credentials in the front end
    # *** Moreover here we consider one interface ip however each container can have more than one interface,
    # how should we condsier the ip? Can we consider the service ip defined for that vuln?
    credentials = []

    emulation_containers = emulation_data["emulationContainer"]
    for containers in emulation_containers:
        container_interfaces = containers["interfaces"]
        interface_ip = ""
        for interfaces in container_interfaces:
            interface_ip = interfaces["ip"]
        container_vulns = containers["vulns"]
        for vuln in container_vulns:
            vuln_name = vuln["vulnName"]
            vuln_type = vuln["vulnType"]
            vuln_service_name = vuln["vulnService"]["name"]
            vuln_service_protocol = vuln["vulnService"]["protocol"]
            vuln_service_port = vuln["vulnService"]["port"]
            vuln_service_ip = vuln["vulnService"]["serviceIp"]
            vuln_root_access = vuln["vulnRoot"]

            node_vuln_config = NodeVulnerabilityConfig(
                name=vuln_name,
                ip=interface_ip,
                vuln_type=vuln_type,
                credentials=credentials,
                cvss=constants.EXPLOIT_VULNERABILITES.WEAK_PASSWORD_CVSS,
                cve=None,
                root=vuln_root_access, port=vuln_service_port,
                protocol=vuln_service_protocol, service=vuln_service_name)
            vulns.append(node_vuln_config)
    vulns_config = VulnerabilitiesConfig(node_vulnerability_configs=vulns)
    return vulns_config

def default_config(emulation_data:json) -> EmulationEnvConfig:
    name = emulation_data["emulationName"]
    network_id = emulation_data["emulationNetworkId"]
    level = emulation_data["emulationLevel"]
    version = emulation_data["emulationVersion"]
    time_step_len_seconds = emulation_data["emulationTimeStepLengh"]
    descr = emulation_data["emulationDescription"]

    containers_cfg = default_containers_config(emulation_data=emulation_data)
    flags_cfg = default_flags_config(emulation_data=emulation_data)
    resources_cfg = default_resource_constraints_config(network_id=network_id, level=level)
    topology_cfg = default_topology_config(network_id=network_id)
    traffic_cfg = default_traffic_config(network_id=network_id, time_step_len_seconds=time_step_len_seconds)
    users_cfg = default_users_config(network_id=network_id)
    vuln_cfg = default_vulns_config(network_id=network_id)
    services_cfg = default_services_config(network_id=network_id)
    kafka_cfg = default_kafka_config(emulation_data=emulation_data)
    static_attackers_cfg = default_static_attacker_sequences(topology_cfg.subnetwork_masks)
    ovs_cfg = default_ovs_config(emulation_data=emulation_data)
    sdn_controller_cfg = default_sdn_controller_config(emulation_data=emulation_data)
    host_manager_cfg = default_host_manager_config(emulation_data=emulation_data)
    snort_ids_manager_cfg = default_snort_ids_manager_config(emulation_data=emulation_data)
    ossec_ids_manager_cfg = default_ossec_ids_manager_config(emulation_data=emulation_data)
    docker_stats_manager_cfg = default_docker_stats_manager_config(emulation_data=emulation_data)
    elk_cfg = default_elk_config(emulation_data=emulation_data)
    beats_cfg = default_beats_config(emulation_data=emulation_data)
    emulation_env_cfg = EmulationEnvConfig(
        name=name, containers_config=containers_cfg, users_config=users_cfg, flags_config=flags_cfg,
        vuln_config=vuln_cfg, topology_config=topology_cfg, traffic_config=traffic_cfg, resources_config=resources_cfg,
        kafka_config=kafka_cfg, services_config=services_cfg,
        descr=descr, static_attacker_sequences=static_attackers_cfg, ovs_config=ovs_cfg,
        sdn_controller_config=sdn_controller_cfg, host_manager_config=host_manager_cfg,
        snort_ids_manager_config=snort_ids_manager_cfg, ossec_ids_manager_config=ossec_ids_manager_cfg,
        docker_stats_manager_config=docker_stats_manager_cfg, elk_config=elk_cfg,
        level=level, execution_id=-1, version=version, beats_config=beats_cfg
    )
    return emulation_env_cfg

def default_containers_config(emulation_data:json) -> ContainersConfig:
    containers = []
    vulnerable_nodes = []
    # *** We need to define the agent reachable nodes
    agent_reachable_nodes = []
    # *** We need to check with Kim what is networks? Which interfaces are considered here?
    networks=[]
    agent_ip = ""
    router_ip = ""
    emulation_ids_enabled = emulation_data["emulatioIdsEnabled"]
    emulation_containers = emulation_data["emulationContainer"]
    for container in emulation_containers:
        container_name = container["name"]
        container_os = container["os"]
        container_version = container["version"]
        containers_level = container["level"]
        container_restart_policy = container["restartPolicy"]
        container_interfaces = container["interfaces"]
        ips_and_networks = []
        for interfaces in container_interfaces:
            interface_name = interfaces["name"]
            interface_ip = interfaces["ip"]
            interface_subnet_mask = interfaces["subnetMask"]
            interface_subnet_prefix = interfaces["subnetPrefix"]
            interface_physical_interface = interfaces["physicalInterface"]
            interface_bit_mask = interfaces["bitmask"]
            ips_and_networks.append((interface_ip,
             ContainerNetwork(
                 name=interface_name,
                 subnet_mask=interface_subnet_mask,
                 subnet_prefix=interface_subnet_prefix,
                 interface=interface_physical_interface,
                 bitmask=interface_bit_mask
             )))
            if ("hacker" in container_name):
                agent_ip = interface_ip
            if ("router" in container_name):
                router_ip = interface_ip
        container_vulns = containers["vulns"]
        for vuln in container_vulns:
            vuln_service_ip = vuln["vulnService"]["serviceIp"]
            vulnerable_nodes.append(vuln_service_ip)

        node = NodeContainerConfig(
            name=container_name,
            os=container_os,
            ips_and_networks=ips_and_networks,
            version=container_version, level=containers_level, restart_policy=container_restart_policy, suffix="_1")
        containers.append(node)
    containers_cfg = ContainersConfig(containers=containers,agent_ip=agent_ip, router_ip=router_ip,
                                      ids_enabled=emulation_ids_enabled, vulnerable_nodes=vulnerable_nodes,
                                      agent_reachable_nodes=agent_reachable_nodes, networks=networks)

    return containers_cfg

def default_flags_config(emulation_data:json) -> FlagsConfig:
    """
    Generates default flags config

    :param emulation_data: the emulation data in JSON format received from front-end
    :return: The flags confguration
    """
    flags = []
    emulation_containers = emulation_data["emulationContainer"]
    for containers in emulation_containers:
        container_falg_id = containers["flagId"]
        container_flag_score = containers["flagScore"]
        container_flag_permission = containers["flagPermission"]
        container_interfaces = containers["interfaces"]
        for interfaces in container_interfaces:
            interface_ip = interfaces["ip"]
        NodeFlagsConfig(ip=interface_ip,
                        flags=[Flag(
                            name=f"{constants.COMMON.FLAG_FILENAME_PREFIX}{container_falg_id}",
                            path=f"/{constants.COMMANDS.TMP_DIR}/{constants.COMMON.FLAG_FILENAME_PREFIX}{container_falg_id}"
                                 f"{constants.FILE_PATTERNS.TXT_FILE_SUFFIX}",
                            dir=f"/{constants.COMMANDS.TMP_DIR}/",
                            id=container_falg_id, requires_root=container_flag_permission, score=container_flag_score
                        )])
    flags_config = FlagsConfig(node_flag_configs=flags)
    return flags_config

def default_static_attacker_sequences(subnet_masks: List[str]) -> Dict[str, List[EmulationAttackerAction]]:
    """
    Generates default attacker sequences config

    :param subnetmasks: list of subnet masks for the emulation
    :return: the default static attacker sequences configuration
    """
    return {}


def default_ovs_config(emulation_data: json) -> OVSConfig:
    """
    Generates default OVS config

    :param emulation_data: the emulation data in JSON format received from front-end
    """
    name = emulation_data["emulationName"]
    network_id = emulation_data["emulationNetworkId"]
    level = emulation_data["emulationLevel"]
    version = emulation_data["emulationVersion"]
    time_step_len_seconds = emulation_data["emulationTimeStepLengh"]

    ovs_config = OVSConfig(switch_configs=[])
    return ovs_config


def default_sdn_controller_config(emulation_data: json) \
        -> Union[None, SDNControllerConfig]:
    """
    Generates the default SDN controller config

    :param emulation_data: the emulation data in JSON format received from front-end
    """
    name = emulation_data["emulationName"]
    network_id = emulation_data["emulationNetworkId"]
    level = emulation_data["emulationLevel"]
    version = emulation_data["emulationVersion"]
    time_step_len_seconds = emulation_data["emulationTimeStepLengh"]
    return None


def default_host_manager_config(emulation_data: json) \
        -> HostManagerConfig:
    """
    Generates the default host manager configuration

    :param emulation_data: the emulation data in JSON format received from front-end
    :return: the host manager configuration
    """
    name = emulation_data["emulationName"]
    network_id = emulation_data["emulationNetworkId"]
    level = emulation_data["emulationLevel"]
    version = emulation_data["emulationVersion"]
    time_step_len_seconds = emulation_data["emulationTimeStepLengh"]
    config = HostManagerConfig(version=version, time_step_len_seconds=time_step_len_seconds,
                               host_manager_port=collector_constants.MANAGER_PORTS.HOST_MANAGER_DEFAULT_PORT,
                               host_manager_log_file=collector_constants.LOG_FILES.HOST_MANAGER_LOG_FILE,
                               host_manager_log_dir=collector_constants.LOG_FILES.HOST_MANAGER_LOG_DIR,
                               host_manager_max_workers=collector_constants.GRPC_WORKERS.DEFAULT_MAX_NUM_WORKERS)
    return config


def default_snort_ids_manager_config(emulation_data: json) \
        -> SnortIDSManagerConfig:
    """
    Generates the default Snort IDS manager configuration

    :param emulation_data: the emulation data in JSON format received from front-end
    :return: the Snort IDS manager configuration
    """
    name = emulation_data["emulationName"]
    network_id = emulation_data["emulationNetworkId"]
    level = emulation_data["emulationLevel"]
    version = emulation_data["emulationVersion"]
    time_step_len_seconds = emulation_data["emulationTimeStepLengh"]

    config = SnortIDSManagerConfig(
        version=version, time_step_len_seconds=time_step_len_seconds,
        snort_ids_manager_port=collector_constants.MANAGER_PORTS.SNORT_IDS_MANAGER_DEFAULT_PORT,
        snort_ids_manager_log_dir=collector_constants.LOG_FILES.SNORT_IDS_MANAGER_LOG_DIR,
        snort_ids_manager_log_file=collector_constants.LOG_FILES.SNORT_IDS_MANAGER_LOG_FILE,
        snort_ids_manager_max_workers=collector_constants.GRPC_WORKERS.DEFAULT_MAX_NUM_WORKERS)
    return config


def default_ossec_ids_manager_config(emulation_data: json) \
        -> OSSECIDSManagerConfig:
    """
    Generates the default OSSEC IDS manager configuration

    :param emulation_data: the emulation data in JSON format received from front-end
    :return: the OSSEC IDS manager configuration
    """
    name = emulation_data["emulationName"]
    network_id = emulation_data["emulationNetworkId"]
    level = emulation_data["emulationLevel"]
    version = emulation_data["emulationVersion"]
    time_step_len_seconds = emulation_data["emulationTimeStepLengh"]

    config = OSSECIDSManagerConfig(
        version=version, time_step_len_seconds=time_step_len_seconds,
        ossec_ids_manager_port=collector_constants.MANAGER_PORTS.OSSEC_IDS_MANAGER_DEFAULT_PORT,
        ossec_ids_manager_log_file=collector_constants.LOG_FILES.OSSEC_IDS_MANAGER_LOG_FILE,
        ossec_ids_manager_log_dir=collector_constants.LOG_FILES.OSSEC_IDS_MANAGER_LOG_DIR,
        ossec_ids_manager_max_workers=collector_constants.GRPC_WORKERS.DEFAULT_MAX_NUM_WORKERS)
    return config


def default_docker_stats_manager_config(emulation_data: json) \
        -> DockerStatsManagerConfig:
    """
    Generates the default docker stats manager configuration

    :param emulation_data: the emulation data in JSON format received from front-end
    :return: the docker stats manager configuration
    """
    name = emulation_data["emulationName"]
    network_id = emulation_data["emulationNetworkId"]
    level = emulation_data["emulationLevel"]
    version = emulation_data["emulationVersion"]
    time_step_len_seconds = emulation_data["emulationTimeStepLengh"]

    config = DockerStatsManagerConfig(
        version=version, time_step_len_seconds=time_step_len_seconds,
        docker_stats_manager_port=collector_constants.MANAGER_PORTS.DOCKER_STATS_MANAGER_DEFAULT_PORT,
        docker_stats_manager_log_file=collector_constants.LOG_FILES.DOCKER_STATS_MANAGER_LOG_FILE,
        docker_stats_manager_log_dir=collector_constants.LOG_FILES.DOCKER_STATS_MANAGER_LOG_DIR,
        docker_stats_manager_max_workers=collector_constants.GRPC_WORKERS.DEFAULT_MAX_NUM_WORKERS)
    return config


def default_elk_config(emulation_data: json) -> ElkConfig:
    """
    Generates the default ELK configuration

    :param emulation_data: the emulation data in JSON format received from front-end
    :return: the ELK configuration
    """

    # *** This function I am not sure if we have already collected all the paramters needed.

    name = emulation_data["emulationName"]
    network_id = emulation_data["emulationNetworkId"]
    level = emulation_data["emulationLevel"]
    version = emulation_data["emulationVersion"]
    time_step_len_seconds = emulation_data["emulationTimeStepLengh"]

    container = NodeContainerConfig(
        name=f"{constants.CONTAINER_IMAGES.ELK_1}",
        os=constants.CONTAINER_OS.ELK_1_OS,
        ips_and_networks=[
            (f"{constants.CSLE.CSLE_SUBNETMASK_PREFIX}{network_id}."
             f"{collector_constants.ELK_CONFIG.NETWORK_ID_THIRD_OCTET}."
             f"{collector_constants.ELK_CONFIG.NETWORK_ID_FOURTH_OCTET}",
             ContainerNetwork(
                 name=f"{constants.CSLE.CSLE_NETWORK_PREFIX}{network_id}_"
                      f"{collector_constants.ELK_CONFIG.NETWORK_ID_THIRD_OCTET}",
                 subnet_mask=f"{constants.CSLE.CSLE_SUBNETMASK_PREFIX}"
                             f"{network_id}.{collector_constants.ELK_CONFIG.NETWORK_ID_THIRD_OCTET}"
                             f"{constants.CSLE.CSLE_EDGE_SUBNETMASK_SUFFIX}",
                 subnet_prefix=f"{constants.CSLE.CSLE_SUBNETMASK_PREFIX}{network_id}",
                 bitmask=constants.CSLE.CSLE_EDGE_BITMASK
             )),
        ],
        version=version, level=str(level),
        restart_policy=constants.DOCKER.ON_FAILURE_3, suffix=collector_constants.ELK_CONFIG.SUFFIX)

    resources = NodeResourcesConfig(
        container_name=f"{constants.CSLE.NAME}-"
                       f"{constants.CONTAINER_IMAGES.ELK_1}_1-{constants.CSLE.LEVEL}{level}",
        num_cpus=2, available_memory_gb=16,
        ips_and_network_configs=[
            (f"{constants.CSLE.CSLE_SUBNETMASK_PREFIX}{network_id}."
             f"{collector_constants.ELK_CONFIG.NETWORK_ID_THIRD_OCTET}."
             f"{collector_constants.ELK_CONFIG.NETWORK_ID_FOURTH_OCTET}",
             None)])

    firewall_config = NodeFirewallConfig(
        hostname=f"{constants.CONTAINER_IMAGES.ELK_1}_1",
        ips_gw_default_policy_networks=[
            DefaultNetworkFirewallConfig(
                ip=f"{constants.CSLE.CSLE_SUBNETMASK_PREFIX}{network_id}."
                   f"{collector_constants.ELK_CONFIG.NETWORK_ID_THIRD_OCTET}."
                   f"{collector_constants.ELK_CONFIG.NETWORK_ID_FOURTH_OCTET}",
                default_gw=None,
                default_input=constants.FIREWALL.ACCEPT,
                default_output=constants.FIREWALL.ACCEPT,
                default_forward=constants.FIREWALL.ACCEPT,
                network=ContainerNetwork(
                    name=f"{constants.CSLE.CSLE_NETWORK_PREFIX}{network_id}_"
                         f"{collector_constants.ELK_CONFIG.NETWORK_ID_THIRD_OCTET}",
                    subnet_mask=f"{constants.CSLE.CSLE_SUBNETMASK_PREFIX}"
                                f"{network_id}.{collector_constants.ELK_CONFIG.NETWORK_ID_THIRD_OCTET}"
                                f"{constants.CSLE.CSLE_EDGE_SUBNETMASK_SUFFIX}",
                    subnet_prefix=f"{constants.CSLE.CSLE_SUBNETMASK_PREFIX}{network_id}",
                    bitmask=constants.CSLE.CSLE_EDGE_BITMASK
                )
            )
        ],
        output_accept=set([]),
        input_accept=set([]),
        forward_accept=set([]),
        output_drop=set(), input_drop=set(), forward_drop=set(), routes=set())

    config = ElkConfig(version=version, time_step_len_seconds=time_step_len_seconds,
                       elastic_port=collector_constants.ELK.ELASTIC_PORT,
                       kibana_port=collector_constants.ELK.KIBANA_PORT,
                       logstash_port=collector_constants.ELK.LOGSTASH_PORT,
                       elk_manager_port=collector_constants.MANAGER_PORTS.ELK_MANAGER_DEFAULT_PORT,
                       container=container,
                       resources=resources, firewall_config=firewall_config,
                       elk_manager_log_file=collector_constants.LOG_FILES.ELK_MANAGER_LOG_FILE,
                       elk_manager_log_dir=collector_constants.LOG_FILES.ELK_MANAGER_LOG_DIR,
                       elk_manager_max_workers=collector_constants.GRPC_WORKERS.DEFAULT_MAX_NUM_WORKERS)
    return config

def default_beats_config(emulation_data: json) -> BeatsConfig:
    """
    Generates default beats config

    :param emulation_data: the emulation data in JSON format received from front-end
    :return: the beats configuration
    """
    # *** This file I am not very sure if all the parameters are set correctly
    # *** Compared to other files this one seems incomplete.

    network_id = emulation_data["emulationNetworkId"]
    level = emulation_data["emulationLevel"]
    version = emulation_data["emulationVersion"]
    time_step_len_seconds = emulation_data["emulationTimeStepLengh"]

    node_beats_configs = []
    emulation_containers = emulation_data["emulationContainer"]
    for containers in emulation_containers:
        container_interfaces = containers["interfaces"]
        for interfaces in container_interfaces:
            interface_ip = interfaces["ip"]
            interface_subnet_mask = interfaces["subnetMask"]
            interface_subnet_prefix = interfaces["subnetPrefix"]
        node_beats = NodeBeatsConfig(ip=interface_ip,
                        log_files_paths=collector_constants.LOG_FILES.DEFAULT_LOG_FILE_PATHS,
                        filebeat_modules=[collector_constants.FILEBEAT.SYSTEM_MODULE],
                        kafka_input=False, start_filebeat_automatically=False,
                        start_packetbeat_automatically=False,
                        metricbeat_modules=[collector_constants.METRICBEAT.SYSTEM_MODULE,
                                            collector_constants.METRICBEAT.LINUX_MODULE],
                        start_metricbeat_automatically=False,
                        start_heartbeat_automatically=False,
                        heartbeat_hosts_to_monitor=[
                            f"{constants.CSLE.CSLE_SUBNETMASK_PREFIX}{network_id}."
                            f"{collector_constants.KAFKA_CONFIG.NETWORK_ID_THIRD_OCTET}."
                            f"{collector_constants.KAFKA_CONFIG.NETWORK_ID_FOURTH_OCTET}",
                            f"{constants.CSLE.CSLE_SUBNETMASK_PREFIX}{network_id}."
                            f"{collector_constants.ELK_CONFIG.NETWORK_ID_THIRD_OCTET}."
                            f"{collector_constants.ELK_CONFIG.NETWORK_ID_FOURTH_OCTET}"
                        ])
        node_beats_configs.append(node_beats)
    beats_conf = BeatsConfig(node_beats_configs=node_beats_configs, num_elastic_shards=1, reload_enabled=False)
    return beats_conf

def default_kafka_config(emulation_data: json) -> KafkaConfig:
    """
    Generates the default kafka configuration

    :param emulation_data: the emulation data in JSON format received from front-end
    :return: the kafka configuration
    """

    network_id = emulation_data["emulationNetworkId"]
    level = emulation_data["emulationLevel"]
    version = emulation_data["emulationVersion"]
    time_step_len_seconds = emulation_data["emulationTimeStepLengh"]

    container = NodeContainerConfig(
        name=f"{constants.CONTAINER_IMAGES.KAFKA_1}",
        os=constants.CONTAINER_OS.KAFKA_1_OS,
        ips_and_networks=[
            (f"{constants.CSLE.CSLE_SUBNETMASK_PREFIX}{network_id}."
             f"{collector_constants.KAFKA_CONFIG.NETWORK_ID_THIRD_OCTET}."
             f"{collector_constants.KAFKA_CONFIG.NETWORK_ID_FOURTH_OCTET}",
             ContainerNetwork(
                 name=f"{constants.CSLE.CSLE_NETWORK_PREFIX}{network_id}_"
                      f"{collector_constants.KAFKA_CONFIG.NETWORK_ID_THIRD_OCTET}",
                 subnet_mask=f"{constants.CSLE.CSLE_SUBNETMASK_PREFIX}"
                             f"{network_id}.{collector_constants.KAFKA_CONFIG.NETWORK_ID_THIRD_OCTET}"
                             f"{constants.CSLE.CSLE_EDGE_SUBNETMASK_SUFFIX}",
                 subnet_prefix=f"{constants.CSLE.CSLE_SUBNETMASK_PREFIX}{network_id}",
                 bitmask=constants.CSLE.CSLE_EDGE_BITMASK
             )),
        ],
        version=version, level=str(level),
        restart_policy=constants.DOCKER.ON_FAILURE_3, suffix=collector_constants.KAFKA_CONFIG.SUFFIX)

    resources = NodeResourcesConfig(
        container_name=f"{constants.CSLE.NAME}-"
                       f"{constants.CONTAINER_IMAGES.KAFKA_1}_1-{constants.CSLE.LEVEL}{level}",
        num_cpus=1, available_memory_gb=4,
        ips_and_network_configs=[
            (f"{constants.CSLE.CSLE_SUBNETMASK_PREFIX}{network_id}."
             f"{collector_constants.KAFKA_CONFIG.NETWORK_ID_THIRD_OCTET}."
             f"{collector_constants.KAFKA_CONFIG.NETWORK_ID_FOURTH_OCTET}",
             None)])

    firewall_config = NodeFirewallConfig(
        hostname=f"{constants.CONTAINER_IMAGES.KAFKA_1}_1",
        ips_gw_default_policy_networks=[
            DefaultNetworkFirewallConfig(
                ip=f"{constants.CSLE.CSLE_SUBNETMASK_PREFIX}{network_id}."
                   f"{collector_constants.KAFKA_CONFIG.NETWORK_ID_THIRD_OCTET}."
                   f"{collector_constants.KAFKA_CONFIG.NETWORK_ID_FOURTH_OCTET}",
                default_gw=None,
                default_input=constants.FIREWALL.ACCEPT,
                default_output=constants.FIREWALL.ACCEPT,
                default_forward=constants.FIREWALL.ACCEPT,
                network=ContainerNetwork(
                    name=f"{constants.CSLE.CSLE_NETWORK_PREFIX}{network_id}_"
                         f"{collector_constants.KAFKA_CONFIG.NETWORK_ID_THIRD_OCTET}",
                    subnet_mask=f"{constants.CSLE.CSLE_SUBNETMASK_PREFIX}"
                                f"{network_id}.{collector_constants.KAFKA_CONFIG.NETWORK_ID_THIRD_OCTET}"
                                f"{constants.CSLE.CSLE_EDGE_SUBNETMASK_SUFFIX}",
                    subnet_prefix=f"{constants.CSLE.CSLE_SUBNETMASK_PREFIX}{network_id}",
                    bitmask=constants.CSLE.CSLE_EDGE_BITMASK
                )
            )
        ],
        output_accept=set([]),
        input_accept=set([]),
        forward_accept=set([]),
        output_drop=set(), input_drop=set(), forward_drop=set(), routes=set())

    topics = [
        KafkaTopic(
            name=collector_constants.KAFKA_CONFIG.CLIENT_POPULATION_TOPIC_NAME,
            num_replicas=collector_constants.KAFKA_CONFIG.DEFAULT_NUM_REPLICAS,
            num_partitions=collector_constants.KAFKA_CONFIG.DEFAULT_NUM_PARTITIONS,
            retention_time_hours=collector_constants.KAFKA_CONFIG.DEFAULT_RETENTION_TIME_HOURS,
            attributes=collector_constants.KAFKA_CONFIG.CLIENT_POPULATION_TOPIC_ATTRIBUTES
        ),
        KafkaTopic(
            name=collector_constants.KAFKA_CONFIG.SNORT_IDS_LOG_TOPIC_NAME,
            num_replicas=collector_constants.KAFKA_CONFIG.DEFAULT_NUM_REPLICAS,
            num_partitions=collector_constants.KAFKA_CONFIG.DEFAULT_NUM_PARTITIONS,
            retention_time_hours=collector_constants.KAFKA_CONFIG.DEFAULT_RETENTION_TIME_HOURS,
            attributes=collector_constants.KAFKA_CONFIG.SNORT_IDS_LOG_TOPIC_ATTRIBUTES
        ),
        KafkaTopic(
            name=collector_constants.KAFKA_CONFIG.OSSEC_IDS_LOG_TOPIC_NAME,
            num_replicas=collector_constants.KAFKA_CONFIG.DEFAULT_NUM_REPLICAS,
            num_partitions=collector_constants.KAFKA_CONFIG.DEFAULT_NUM_PARTITIONS,
            retention_time_hours=collector_constants.KAFKA_CONFIG.DEFAULT_RETENTION_TIME_HOURS,
            attributes=collector_constants.KAFKA_CONFIG.OSSEC_IDS_LOG_TOPIC_ATTRIBUTES
        ),
        KafkaTopic(
            name=collector_constants.KAFKA_CONFIG.HOST_METRICS_TOPIC_NAME,
            num_replicas=collector_constants.KAFKA_CONFIG.DEFAULT_NUM_REPLICAS,
            num_partitions=collector_constants.KAFKA_CONFIG.DEFAULT_NUM_PARTITIONS,
            retention_time_hours=collector_constants.KAFKA_CONFIG.DEFAULT_RETENTION_TIME_HOURS,
            attributes=collector_constants.KAFKA_CONFIG.HOST_METRICS_TOPIC_ATTRIBUTES
        ),
        KafkaTopic(
            name=collector_constants.KAFKA_CONFIG.DOCKER_STATS_TOPIC_NAME,
            num_replicas=collector_constants.KAFKA_CONFIG.DEFAULT_NUM_REPLICAS,
            num_partitions=collector_constants.KAFKA_CONFIG.DEFAULT_NUM_PARTITIONS,
            retention_time_hours=collector_constants.KAFKA_CONFIG.DEFAULT_RETENTION_TIME_HOURS,
            attributes=collector_constants.KAFKA_CONFIG.DOCKER_STATS_TOPIC_ATTRIBUTES
        ),
        KafkaTopic(
            name=collector_constants.KAFKA_CONFIG.ATTACKER_ACTIONS_TOPIC_NAME,
            num_replicas=collector_constants.KAFKA_CONFIG.DEFAULT_NUM_REPLICAS,
            num_partitions=collector_constants.KAFKA_CONFIG.DEFAULT_NUM_PARTITIONS,
            retention_time_hours=collector_constants.KAFKA_CONFIG.DEFAULT_RETENTION_TIME_HOURS,
            attributes=collector_constants.KAFKA_CONFIG.ATTACKER_ACTIONS_ATTRIBUTES
        ),
        KafkaTopic(
            name=collector_constants.KAFKA_CONFIG.DEFENDER_ACTIONS_TOPIC_NAME,
            num_replicas=collector_constants.KAFKA_CONFIG.DEFAULT_NUM_REPLICAS,
            num_partitions=collector_constants.KAFKA_CONFIG.DEFAULT_NUM_PARTITIONS,
            retention_time_hours=collector_constants.KAFKA_CONFIG.DEFAULT_RETENTION_TIME_HOURS,
            attributes=collector_constants.KAFKA_CONFIG.DEFENDER_ACTIONS_ATTRIBUTES
        ),
        KafkaTopic(
            name=collector_constants.KAFKA_CONFIG.DOCKER_HOST_STATS_TOPIC_NAME,
            num_replicas=collector_constants.KAFKA_CONFIG.DEFAULT_NUM_REPLICAS,
            num_partitions=collector_constants.KAFKA_CONFIG.DEFAULT_NUM_PARTITIONS,
            retention_time_hours=collector_constants.KAFKA_CONFIG.DEFAULT_RETENTION_TIME_HOURS,
            attributes=collector_constants.KAFKA_CONFIG.DOCKER_STATS_TOPIC_ATTRIBUTES
        ),
        KafkaTopic(
            name=collector_constants.KAFKA_CONFIG.OPENFLOW_FLOW_STATS_TOPIC_NAME,
            num_replicas=collector_constants.KAFKA_CONFIG.DEFAULT_NUM_REPLICAS,
            num_partitions=collector_constants.KAFKA_CONFIG.DEFAULT_NUM_PARTITIONS,
            retention_time_hours=collector_constants.KAFKA_CONFIG.DEFAULT_RETENTION_TIME_HOURS,
            attributes=collector_constants.KAFKA_CONFIG.OPENFLOW_FLOW_STATS_TOPIC_ATTRIBUTES
        ),
        KafkaTopic(
            name=collector_constants.KAFKA_CONFIG.OPENFLOW_PORT_STATS_TOPIC_NAME,
            num_replicas=collector_constants.KAFKA_CONFIG.DEFAULT_NUM_REPLICAS,
            num_partitions=collector_constants.KAFKA_CONFIG.DEFAULT_NUM_PARTITIONS,
            retention_time_hours=collector_constants.KAFKA_CONFIG.DEFAULT_RETENTION_TIME_HOURS,
            attributes=collector_constants.KAFKA_CONFIG.OPENFLOW_PORT_STATS_TOPIC_ATTRIBUTES
        ),
        KafkaTopic(
            name=collector_constants.KAFKA_CONFIG.AVERAGE_OPENFLOW_FLOW_STATS_PER_SWITCH_TOPIC_NAME,
            num_replicas=collector_constants.KAFKA_CONFIG.DEFAULT_NUM_REPLICAS,
            num_partitions=collector_constants.KAFKA_CONFIG.DEFAULT_NUM_PARTITIONS,
            retention_time_hours=collector_constants.KAFKA_CONFIG.DEFAULT_RETENTION_TIME_HOURS,
            attributes=collector_constants.KAFKA_CONFIG.AVERAGE_OPENFLOW_FLOW_STATS_PER_SWITCH_TOPIC_ATTRIBUTES
        ),
        KafkaTopic(
            name=collector_constants.KAFKA_CONFIG.AVERAGE_OPENFLOW_PORT_STATS_PER_SWITCH_TOPIC_NAME,
            num_replicas=collector_constants.KAFKA_CONFIG.DEFAULT_NUM_REPLICAS,
            num_partitions=collector_constants.KAFKA_CONFIG.DEFAULT_NUM_PARTITIONS,
            retention_time_hours=collector_constants.KAFKA_CONFIG.DEFAULT_RETENTION_TIME_HOURS,
            attributes=collector_constants.KAFKA_CONFIG.AVERAGE_OPENFLOW_PORT_STATS_PER_SWITCH_TOPIC_ATTRIBUTES
        ),
        KafkaTopic(
            name=collector_constants.KAFKA_CONFIG.OPENFLOW_AGG_FLOW_STATS_TOPIC_NAME,
            num_replicas=collector_constants.KAFKA_CONFIG.DEFAULT_NUM_REPLICAS,
            num_partitions=collector_constants.KAFKA_CONFIG.DEFAULT_NUM_PARTITIONS,
            retention_time_hours=collector_constants.KAFKA_CONFIG.DEFAULT_RETENTION_TIME_HOURS,
            attributes=collector_constants.KAFKA_CONFIG.OPENFLOW_AGG_FLOW_STATS_TOPIC_ATTRIBUTES
        ),
        KafkaTopic(
            name=collector_constants.KAFKA_CONFIG.SNORT_IDS_RULE_LOG_TOPIC_NAME,
            num_replicas=collector_constants.KAFKA_CONFIG.DEFAULT_NUM_REPLICAS,
            num_partitions=collector_constants.KAFKA_CONFIG.DEFAULT_NUM_PARTITIONS,
            retention_time_hours=collector_constants.KAFKA_CONFIG.DEFAULT_RETENTION_TIME_HOURS,
            attributes=collector_constants.KAFKA_CONFIG.SNORT_IDS_RULE_LOG_ATTRIBUTES
        ),
        KafkaTopic(
            name=collector_constants.KAFKA_CONFIG.SNORT_IDS_IP_LOG_TOPIC_NAME,
            num_replicas=collector_constants.KAFKA_CONFIG.DEFAULT_NUM_REPLICAS,
            num_partitions=collector_constants.KAFKA_CONFIG.DEFAULT_NUM_PARTITIONS,
            retention_time_hours=collector_constants.KAFKA_CONFIG.DEFAULT_RETENTION_TIME_HOURS,
            attributes=collector_constants.KAFKA_CONFIG.SNORT_IDS_IP_LOG_ATTRIBUTES
        )
    ]

    config = KafkaConfig(container=container, resources=resources, topics=topics, firewall_config=firewall_config,
                         version=version,
                         kafka_port=collector_constants.KAFKA.PORT,
                         kafka_port_external=collector_constants.KAFKA.EXTERNAL_PORT,
                         kafka_manager_port=collector_constants.MANAGER_PORTS.KAFKA_MANAGER_DEFAULT_PORT,
                         time_step_len_seconds=time_step_len_seconds,
                         kafka_manager_log_file=collector_constants.LOG_FILES.KAFKA_MANAGER_LOG_FILE,
                         kafka_manager_log_dir=collector_constants.LOG_FILES.KAFKA_MANAGER_LOG_DIR,
                         kafka_manager_max_workers=collector_constants.GRPC_WORKERS.DEFAULT_MAX_NUM_WORKERS)
    return config

def default_resource_constraints_config(emulation_data: json) -> ResourcesConfig:
    """
    Generates default resource constraints config

    :param emulation_data: the emulation data in JSON format received from front-end
    :return: generates the ResourcesConfig
    """
    node_resources_configurations = []
    emulation_containers = emulation_data["emulationContainer"]
    for containers in emulation_containers:
        container_name = containers["name"]
        container_cpu = containers["cpu"]
        container_memory = containers["mem"]
        container_interfaces = containers["interfaces"]
        ips_and_network_configs = []
        for interfaces in container_interfaces:
            interface_ip = interfaces["ip"]
            interface_physical_interface = interfaces["physicalInterface"]
            interface_limit_packet_queue = interfaces["limitPacketsQueue"]
            interface_packet_delay_ms = interfaces["packetDelayMs"]
            interface_packet_delay_jitter_ms = interfaces["packetDelayJitterMs"]
            interface_packet_delay_correlation_percentage = interfaces["packetDelayCorrelationPercentage"]
            interfaces_packet_delay_distribution = interfaces["packetDelayDistribution"]
            interface_packet_loss_type = interfaces["packetLossType"]
            interface_loss_gmodel_p = interfaces["lossGemodelp"]
            interface_loss_gmodel_r = interfaces["lossGemodelr"]
            interface_loss_gmodel_k = interfaces["lossGemodelk"]
            interface_loss_gmodel_h = interfaces["lossGemodelh"]
            interface_packet_corruption_percentage = interfaces["packetCorruptPercentage"]
            interface_packet_corruption_correlation_percentage = interfaces["packetCorruptCorrelationPercentage"]
            interface_packet_duplication_percentage = interfaces["packetDuplicatePercentage"]
            interface_packet_duplicate_correlation_percentage = interfaces["packetDuplicateCorrelationPercentage"]
            interface_packet_reorder_percentage = interfaces["packetReorderPercentage"]
            interface_packet_reorder_correlation_percentage = interfaces["packetReorderCorrelationPercentage"]
            interface_packet_reorder_gap = interfaces["packetReorderGap"]
            interface_rate_limit_m_bit = interfaces["rateLimitMbit"]
            interface_packet_overhead_bytes = interfaces["packetOverheadBytes"]
            interface_cell_overhead_bytes = interfaces["cellOverheadBytes"]
            ips_and_network_configs.append(
                (interface_ip, NodeNetworkConfig(
                    interface=interface_physical_interface,
                    limit_packets_queue=interface_limit_packet_queue, packet_delay_ms=interface_packet_delay_ms,
                    packet_delay_jitter_ms=interface_packet_delay_jitter_ms,
                    packet_delay_correlation_percentage=interface_packet_delay_correlation_percentage,
                    packet_delay_distribution=interfaces_packet_delay_distribution,
                    packet_loss_type=interface_packet_loss_type,
                    loss_gemodel_p=interface_loss_gmodel_p,
                    loss_gemodel_r=interface_loss_gmodel_r,
                    loss_gemodel_k=interface_loss_gmodel_k,
                    loss_gemodel_h=interface_loss_gmodel_h,
                    packet_corrupt_percentage=interface_packet_corruption_percentage,
                    packet_corrupt_correlation_percentage=interface_packet_corruption_correlation_percentage,
                    packet_duplicate_percentage=interface_packet_duplication_percentage,
                    packet_duplicate_correlation_percentage=interface_packet_duplicate_correlation_percentage,
                    packet_reorder_percentage=interface_packet_reorder_percentage,
                    packet_reorder_correlation_percentage=interface_packet_reorder_correlation_percentage,
                    packet_reorder_gap=interface_packet_reorder_gap,
                    rate_limit_mbit=interface_rate_limit_m_bit,
                    packet_overhead_bytes=interface_packet_overhead_bytes,
                    cell_overhead_bytes=interface_cell_overhead_bytes
                 ))
            )

        node_resource_config = NodeResourcesConfig(
            container_name=container_name,
            num_cpus=container_cpu, available_memory_gb=container_memory,
            ips_and_network_configs=ips_and_network_configs)
        node_resources_configurations.append(node_resource_config)
    resources_config = ResourcesConfig(node_resources_configurations=node_resources_configurations)
    return resources_config

def default_topology_config(emulation_data: json) -> TopologyConfig:
    """
    Generates default topology config

    :param emulation_data: the emulation data in JSON format received from front-end
    :return: the Topology configuration
    """
    # *** The function is incomplete


# Creates a blueprint "sub application" of the main REST app
create_emulation_bp = Blueprint(
    api_constants.MGMT_WEBAPP.CREATE_EMULATION_RESOURCE, __name__,
    url_prefix=f"{constants.COMMANDS.SLASH_DELIM}{api_constants.MGMT_WEBAPP.CREATE_EMULATION_RESOURCE}")


@create_emulation_bp.route("", methods=[api_constants.MGMT_WEBAPP.HTTP_REST_POST])
def create_emulation() -> Tuple[Response, int]:
    """
    The /create-emulation resource.

    :return: The given policy or deletes the policy
    """
    print("Create emulation")
    requires_admin = True
    authorized = rest_api_util.check_if_user_is_authorized(request=request, requires_admin=requires_admin)
    if authorized is not None:
        return authorized

    # print(request.data)
    emulation_data = json.loads(request.data)
    emulation_name = emulation_data["emulationName"]
    emulation_network_id = emulation_data["emulationNetworkId"]
    emulation_level = emulation_data["emulationLevel"]
    emulation_version = emulation_data["emulationVersion"]
    emulation_time_step_length = emulation_data["emulationTimeStepLengh"]
    emulation_ids_enabled = emulation_data["emulatioIdsEnabled"]
    emulation_description = emulation_data["emulationDescription"]
    emulation_containers = emulation_data["emulationContainer"]
    for containers in emulation_containers:
        container_name = containers["name"]
        container_os = containers["os"]
        container_version = containers["version"]
        containers_level = containers["level"]
        container_restart_policy = containers["restartPolicy"]
        container_network_id = containers["networkId"]
        container_subnet_mask = containers["subnetMask"]
        container_subnet_prefix = containers["subnetPrefix"]
        container_cpu = containers["cpu"]
        container_memory = containers["mem"]
        container_falg_id = containers["flagId"]
        container_flag_score = containers["flagScore"]
        container_flag_permission = containers["flagPermission"]
        container_interfaces = containers["interfaces"]
        for interfaces in container_interfaces:
            interface_name = interfaces["name"]
            interface_ip = interfaces["ip"]
            interface_subnet_mask = interfaces["subnetMask"]
            interface_subnet_prefix = interfaces["subnetPrefix"]
            interface_physical_interface = interfaces["physicalInterface"]
            interface_bit_mask = interfaces["bitmask"]
            interface_limit_packet_queue = interfaces["limitPacketsQueue"]
            interface_packet_delay_ms = interfaces["packetDelayMs"]
            interface_packet_delay_jitter_ms = interfaces["packetDelayJitterMs"]
            interface_packet_delay_correlation_percentage = interfaces["packetDelayCorrelationPercentage"]
            interfaces_packet_delay_distribution = interfaces["packetDelayDistribution"]
            interface_packet_loss_type = interfaces["packetLossType"]
            interface_loss_gmodel_p = interfaces["lossGemodelp"]
            interface_loss_gmodel_p = interfaces["lossGemodelr"]
            interface_loss_gmodel_p = interfaces["lossGemodelk"]
            interface_loss_gmodel_p = interfaces["lossGemodelh"]
            interface_packet_corruption_percentage = interfaces["packetCorruptPercentage"]
            interface_packet_corruption_correlation_percentage = interfaces["packetCorruptCorrelationPercentage"]
            interface_packet_duplication_percentage = interfaces["packetDuplicatePercentage"]
            interface_packet_duplicate_correlation_percentage = interfaces["packetDuplicateCorrelationPercentage"]
            interface_packet_reorder_percentage = interfaces["packetReorderPercentage"]
            interface_packet_reorder_correlation_percentage = interfaces["packetReorderCorrelationPercentage"]
            interface_packet_reorder_gap = interfaces["packetReorderGap"]
            interface_rate_limit_m_bit = interfaces["rateLimitMbit"]
            interface_packet_overhead_bytes = interfaces["packetOverheadBytes"]
            interface_cell_overhead_bytes = interfaces["cellOverheadBytes"]
            interface_default_gateway = interfaces["defaultGateway"]
            interface_default_input = interfaces["defaultInput"]
            interface_default_output = interfaces["defaultOutput"]
            interface_default_forward = interfaces["defaultForward"]
            interfaces_traffic_manager_port = interfaces["trafficManagerPort"]
            interface_traffic_manager_log_file = interfaces["trafficManagerLogFile"]
            interface_traffic_manager_log_dir = interfaces["trafficManagerLogDir"]
            interface_traffic_manager_max_workers = interfaces["trafficManagerMaxWorkers"]
            print("Container name: ", container_name, " interface is:", interface_name)
        container_reachable_by_agent = containers["reachableByAgent"]
        container_users = containers["users"]
        for user in container_users:
            user_name = user["userName"]
            user_pw = user["pw"]
            user_access = user["root"]
        container_services = containers["services"]
        for service in container_services:
            service_name = service["name"]
            service_protocol = service["protocol"]
            service_port = service["port"]
            service_ip = service["serviceIp"]
        container_vulns = containers["vulns"]
        for vuln in container_vulns:
            vuln_name = vuln["vulnName"]
            vuln_type = vuln["vulnType"]
            vuln_service_name = vuln["vulnService"]["name"]
            vuln_service_protocol = vuln["vulnService"]["protocol"]
            vuln_service_port = vuln["vulnService"]["port"]
            vuln_service_ip = vuln["vulnService"]["serviceIp"]
            vuln_root_access = vuln["vulnRoot"]
            print("Container vuln service name: ", vuln_service_name)

    vulnerabilities(emulation_data)

    response = jsonify({"TEST": "TEST"})
    response.headers.add(api_constants.MGMT_WEBAPP.ACCESS_CONTROL_ALLOW_ORIGIN_HEADER, "*")
    return response, constants.HTTPS.OK_STATUS_CODE
