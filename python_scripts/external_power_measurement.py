import paramiko
import subprocess
import sys
import os
import re
from pathlib import Path

ENERGY_UJ_FILE = "/sys/class/powercap/intel-rapl:0/energy_uj"
NUMBER_OF_RUNS = 10
PROBLEM_SIZE = 960_000_000
NUMBER_OF_STREAMS = 2
POWERCAP = 0
NUMBER_OF_NODES = 2

SSH_USER_NAME = 'student'
SSH_PASSWORD = 'student'
SSH_HOST = '172.20.83.214'

def ssh_read_energy(username: str, password: str, host: str = SSH_HOST, port: int = 22):
    """Read energy from remote slave node"""
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname=host, port=port, username=username, password=password)
        
        stdin, stdout, stderr = ssh.exec_command(f"cat {ENERGY_UJ_FILE}")
        output = stdout.read().decode().strip()
        error = stderr.read().decode().strip()
        ssh.close()
        
        if error:
            return f"Error: {error}"
        return float(output)
    except Exception as e:
        return f"An exception occurred: {e}"

# def read_energy(file_path: str = ENERGY_UJ_FILE):
#     """Reads the energy from master local node."""
#     try:
#         with open(file_path, 'r') as file:
#             return float(file.read())
#     except FileNotFoundError:
#         print(f"Error: File '{file_path}' not found.")
#     except PermissionError:
#         print(f"Error: Insufficient permissions to read file '{file_path}'.")
#     except Exception as e:
#         print(f"Unexpected error reading energy file: {e}")
#     return None

def read_main_elapsed_time(stderr_output: str):
    match = re.search(r'Main elapsed time=([\d.]+)', stderr_output)
    if match:
        return float(match.group(1))
    print("Error: Could not find 'Main elapsed time' in stderr output.")
    return None

def run_script(command: str):
    try:
        result = subprocess.run(
            command.split(" "),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        return result.stderr
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        print(f"Command: {e.cmd}")
        print(f"Return Code: {e.returncode}")
        print(f"Error Output:\n{e.stderr}")
        sys.exit(1)

if __name__ == "__main__":
    """
    Master runs on local node, slave runs on remote node.
    Energy is measured on remote node. To do this there is a need to log in remote node using ssh and read energy_uj file.
    
    To run script fill SSH_USER_NAME, SSH_PASSWORD AND SSH_HOST (remote node ip).
    """

    os.chdir(Path.home() / "parallel-processing-cpu-and-gpu-env-and-lib-with-powercap/cudampilib")
    commands = [
        f"./run_scripts/run-app twinprime B {NUMBER_OF_NODES} --cpu-enabled=0 --number-of-streams={NUMBER_OF_STREAMS} --batch-size=480000 --powercap={POWERCAP} --initial-cpu-batch-size-scaling=100",
        f"./run_scripts/run-app twinprime B {NUMBER_OF_NODES} --cpu-enabled=1 --number-of-streams={NUMBER_OF_STREAMS} --batch-size=480000 --powercap={POWERCAP} --initial-cpu-batch-size-scaling=100"
    ]
    
    for command in commands:
        print(f"[COMMAND] {command}")
        energy_before = ssh_read_energy(username=SSH_USER_NAME, password=SSH_PASSWORD)
        print(f"ENERGY BEFORE: {energy_before}")
        total_main_elapsed_time = 0
        for i in range(NUMBER_OF_RUNS):
            print(f"RUN: {i}")
            stderr_output = run_script(command=command)
            elapsed_time = read_main_elapsed_time(stderr_output)
            total_main_elapsed_time += elapsed_time
            print(f"After run {i}. Elapsed time {elapsed_time}, energy used: {ssh_read_energy(username=SSH_USER_NAME, password=SSH_PASSWORD)}")
        print(f"Total main elapsed time: {total_main_elapsed_time}")
