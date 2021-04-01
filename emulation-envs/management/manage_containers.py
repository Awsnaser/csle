import argparse
from gym_pycr_ctf.envs_model.config import ContainerManager

def parse_args():
    parser = argparse.ArgumentParser(description='Parse flags for command to apply to the containers')
    parser.add_argument("-c", "--cmd", help="command to apply", type=str)
    args = parser.parse_args()
    return args


def execute_cmd(cmd: str):
    ContainerManager.run_command(cmd=cmd)


if __name__ == '__main__':
    args = parse_args()
    cmd = args.cmd
    execute_cmd(cmd=cmd)