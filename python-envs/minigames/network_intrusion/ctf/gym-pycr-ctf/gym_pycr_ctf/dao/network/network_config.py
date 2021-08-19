from typing import List, Tuple
import pickle
from gym_pycr_ctf.dao.network.node import Node
from gym_pycr_ctf.dao.network.node_type import NodeType
from gym_pycr_ctf.dao.defender_dynamics.defender_dynamics_model import DefenderDynamicsModel
import gym_pycr_ctf.constants.constants as constants
from gym_pycr_ctf.util.experiments_util import util
import numpy as np
import os

class NetworkConfig:
    """
    DTO Representing a network configuration
    """

    def __init__(self, subnet_mask: str, nodes: List[Node], adj_matrix: np.ndarray, flags_lookup: dict,
                 agent_reachable: set, vulnerable_nodes = None):
        self.subnet_mask = subnet_mask
        self.nodes = nodes
        self.adj_matrix = adj_matrix
        node_d, hacker, router, levels_d = self.create_lookup_dicts()
        self.node_d = node_d
        self.hacker = hacker
        self.router = router
        self.levels_d = levels_d
        self.flags_lookup = flags_lookup
        self.agent_reachable = agent_reachable
        self.vulnerable_nodes = vulnerable_nodes
        self.defender_dynamics_model = DefenderDynamicsModel()

    def __str__(self) -> str:
        """
        :return: a string representation of the DTO
        """
        return "subnet_mask:{}, nodes:{}, adj_matrix:{}, hacker:{}, router: {}, flags_lookup: {}, agent_reachable: {}, " \
               "vulnerable_nodes: {}, defender_dynamics_model:{}".format(
            self.subnet_mask, list(map(lambda x: str(x), self.nodes)), self.adj_matrix, self.hacker, self.router, self.flags_lookup,
            self.agent_reachable, self.vulnerable_nodes, self.defender_dynamics_model)

    def create_lookup_dicts(self) -> Tuple[dict, Node, Node, dict]:
        """
        Utility function for creating lookup dictionaries, useful when rending the network

        :return: A lookup dictionary for nodes, the hacker node, the router node, and a lookup dictionary
                 for levels in the network
        """
        levels_d = {}
        node_d = {}
        hacker = None
        router = None
        for node in self.nodes:
            node_d[node.id] = node
            if node.type == NodeType.HACKER:
                if hacker is not None:
                    raise ValueError("Invalid Network Config: 2 Hackers")
                hacker = node
            elif node.type == NodeType.ROUTER:
                if router is not None:
                    raise ValueError("Invalid Network Config: 2 Routers")
                router = node
            if node.level in levels_d:
                levels_d[node.level].append(node)
                #levels_d[node.level] = n_level
            else:
                levels_d[node.level] = [node]

        return node_d, hacker, router, levels_d

    def copy(self):
        """
        :return: a copy of the network configuration
        """
        return NetworkConfig(
            subnet_mask=self.subnet_mask, nodes=self.nodes, adj_matrix=self.adj_matrix, flags_lookup=self.flags_lookup,
            agent_reachable=self.agent_reachable)

    def shortest_paths(self) -> List[Tuple[List[str], List[int]]]:
        """
        Utility function for finding the shortest paths to find all flags using brute-force search

        :return: a list of the shortest paths (list of ips and flags)
        """
        shortest_paths = self._find_nodes(reachable=self.agent_reachable, path=[], flags=[])
        return shortest_paths

    def _find_nodes(self, reachable, path, flags) -> List[Tuple[List[str], List[int]]]:
        """
        Utility function for finding the next node in a brute-force-search procedure for finding all flags
        in the network

        :param reachable: the set of reachable nodes from the current attacker state
        :param path: the current path
        :param flags: the set of flags
        :return: a list of the shortest paths (list of ips and flags)
        """
        paths = []
        min_path_len = len(self.nodes)
        for n in self.nodes:
            l_reachable = reachable.copy()
            l_path = path.copy()
            l_flags = flags.copy()
            if n.ip in l_reachable and n.ip in self.vulnerable_nodes and n.ip not in l_path:
                l_reachable.update(n.reachable_nodes)
                l_path.append(n.ip)
                l_flags = l_flags + n.flags
                if len(l_flags)== len(self.flags_lookup):
                    paths.append((l_path, l_flags.copy()))
                    if len(l_path) < min_path_len:
                        min_path_len = len(l_path)
                elif len(l_path) < min_path_len:
                    paths = paths + self._find_nodes(l_reachable, l_path, l_flags)
        return paths

    def save(self, dir_path: str, file_name: str) -> None:
        """
        Utility function for saving the network config to disk

        :param dir_path: the path to save it to
        :param file_name: the name o the file to save to
        :return: None
        """
        if file_name is None:
            file_name = constants.SYSTEM_IDENTIFICATION.NETWORK_CONF_FILE
        if dir_path is not None:
            load_dir = dir_path + "/" + file_name
        else:
            load_dir = util.get_script_path() + "/" + file_name
        with open(load_dir, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(dir_path: str, file_name: str) -> "NetworkConfig":
        """
        Utility function for loading the network config from a pickled file on disk

        :param dir_path: the path to load it from
        :param file_name: the filename
        :return: the loaded network config
        """
        if file_name is None:
            file_name = constants.SYSTEM_IDENTIFICATION.NETWORK_CONF_FILE
        load_dir = None
        if dir_path is not None:
            load_dir = dir_path + "/" + file_name
        else:
            load_dir = util.get_script_path() + "/" + file_name
        if os.path.exists(load_dir):
            with open(load_dir, 'rb') as file:
                obj = pickle.load(file)
                return obj

    def merge(self, network_conf: "NetworkConfig") -> None:
        """
        Merges the network config with another one

        :param network_conf: the network config to merge with
        :return: None
        """
        for node in network_conf.nodes:
            new_node = True
            for n in self.nodes:
                if node.ip == n.ip:
                    new_node = False
                    for vuln in node.vulnerabilities:
                        new_vuln = True
                        for vuln2 in n.vulnerabilities:
                            if vuln.name == vuln2.name:
                                new_vuln = False
                        if new_vuln:
                            n.vulnerabilities.append(vuln)
            if new_node:
                self.nodes.append(node)
        self.vulnerable_nodes = self.vulnerable_nodes.union(network_conf.vulnerable_nodes)
        for node in self.nodes:
            for vuln in node.vulnerabilities:
                if vuln.name == constants.SAMBA.VULNERABILITY_NAME:
                    vuln.credentials[0].username = constants.SAMBA.BACKDOOR_USER
                    vuln.credentials[0].pw = constants.SAMBA.BACKDOOR_PW


