from typing import List, Tuple
from csle_common.dao.container_config.node_network_config import NodeNetworkConfig


class NodeResourcesConfig:
    """
    A DTO object representing the resources of a specific container in an emulation environment
    """

    def __init__(self, container_name: str,
                 num_cpus: int, available_memory_gb :int,
                 ips_and_network_configs = List[Tuple[str, NodeNetworkConfig]]):
        """
        Initializes the DTO

        :param container_name: the name of the container
        :param num_cpus: the number of CPUs available to the node
        :param available_memory_gb: the number of RAM GB available to the node
        :param ips_and_network_configs: list of ip adresses and network configurations
        """
        self.container_name = container_name
        self.num_cpus = num_cpus
        self.available_memory_gb = available_memory_gb
        self.ips_and_network_configs = ips_and_network_configs


    def get_ips(self) -> List[str]:
        """
        :return: a list of ips
        """
        return list(map(lambda x: x[0], self.ips_and_network_configs))


    def __str__(self) -> str:
        """
        :return: a string representation of the node's resources
        """
        return f"num_cpus: {self.num_cpus}, available_memory_gb:{self.available_memory_gb}, " \
               f"container_name:{self.container_name}, ips_and_network_configs: {self.ips_and_network_configs}"
