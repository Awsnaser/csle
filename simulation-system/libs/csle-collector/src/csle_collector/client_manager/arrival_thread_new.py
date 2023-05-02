import logging
import threading
import time
from typing import List
from scipy.stats import poisson
from csle_collector.client_manager.client_thread_new import ClientThreadNew
from csle_collector.client_manager.client_type import ClientType
from csle_collector.client_manager.service import Service


class ArrivalThreadNew(threading.Thread):
    """
    Thread that generates client threads accoriding to time-varying Poisson processes with custom arrival rate functions.
    """
    def __init__(self, time_step_len_seconds: float, client_types: List[ClientType], services: List[Service]) -> None:
        """
        Initializes a new arrival thread.

        :param time_step_len_seconds: The time step length in seconds.
        :param commands: A list of commands to be executed by the client.
        """
        threading.Thread.__init__(self)
        self.t = 0
        self.client_threads = []
        self.stopped = False
        self.time_step_len_seconds = time_step_len_seconds
        self.client_types = client_types
        self.services = services
        logging.info("ArrivalThreadNew initialized, time_step_len_seconds: " + str(time_step_len_seconds) + ", client_types: " + str(client_types) + ", services: " + str(services))

    def run(self) -> None:
        """
        Runs the arrival generator.
        The arrival generator generates client threads according to a time-varying Poisson process with a custom arrival rate function.
        A sequence of commands is generated for each client thread according to its workflow distribution.
        """
        while not self.stopped:
            self.t += 1
            new_client_threads = []

            # Add alive client threads to the list of new client threads
            for client_thread in self.client_threads:
                if client_thread.is_alive():
                    new_client_threads.append(client_thread)
            self.client_threads = new_client_threads

            for client_type in self.client_types:
                # Generate new client threads
                arrival_rate = client_type.rate_function(self.t)
                logging.info("t: " + str(self.t) + ", arrival_rate: " + str(arrival_rate))
                if arrival_rate > 0:
                    num_new_clients = poisson.rvs(arrival_rate, size=1)[0]
                    for _ in range(num_new_clients):
                        new_client_thread = ClientThreadNew(client_type.generate_commands(self.services), self.time_step_len_seconds)
                        new_client_thread.start()
                        self.client_threads.append(new_client_thread)
                    
            time.sleep(self.time_step_len_seconds)

