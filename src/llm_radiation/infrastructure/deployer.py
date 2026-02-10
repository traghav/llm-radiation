"""Deployment utilities for managing experiments on Lambda Labs instances."""

import subprocess
from pathlib import Path
from typing import Any

from .lambda_client import LambdaClient


class LambdaDeployer:
    """Handles deployment and execution of experiments on Lambda Labs instances."""

    def __init__(self, client: LambdaClient, ssh_key_path: str):
        """Initialize the deployer.

        Args:
            client: LambdaClient instance for API operations
            ssh_key_path: Path to SSH private key for instance access
        """
        self.client = client
        self.ssh_key_path = Path(ssh_key_path).expanduser()
        if not self.ssh_key_path.exists():
            raise FileNotFoundError(f"SSH key not found: {self.ssh_key_path}")

    def provision(
        self,
        instance_type: str,
        region: str,
        ssh_key_names: list[str],
        name: str = "llm-radiation"
    ) -> dict[str, Any]:
        """Provision a new instance and wait for it to be ready.

        Args:
            instance_type: Type of instance to launch (e.g., "gpu_1x_a100")
            region: Region to launch in (e.g., "us-west-1")
            ssh_key_names: List of SSH key names registered with Lambda Labs
            name: Name for the instance (default: "llm-radiation")

        Returns:
            Dictionary containing instance details including instance_id and IP

        Raises:
            RuntimeError: If launch fails or instance doesn't become active
        """
        print(f"Launching {instance_type} instance in {region}...")
        launch_result = self.client.launch_instance(
            instance_type=instance_type,
            region=region,
            ssh_key_names=ssh_key_names,
            name=name
        )

        instance_ids = launch_result.get("instance_ids", [])
        if not instance_ids:
            raise RuntimeError("No instance ID returned from launch")

        instance_id = instance_ids[0]
        print(f"Instance {instance_id} launching, waiting for active state...")

        instance_data = self.client.wait_for_active(instance_id)
        print(f"Instance active at IP: {instance_data.get('ip')}")

        return instance_data

    def deploy_code(
        self,
        instance_ip: str,
        local_project_dir: str,
        remote_dir: str = "~/llm-radiation"
    ) -> None:
        """Deploy code to the remote instance using rsync.

        Args:
            instance_ip: IP address of the instance
            local_project_dir: Local directory containing the project
            remote_dir: Remote directory to deploy to (default: ~/llm-radiation)

        Raises:
            subprocess.CalledProcessError: If rsync fails
        """
        print(f"Deploying code from {local_project_dir} to {instance_ip}:{remote_dir}")

        # Ensure remote directory exists
        self._ssh_cmd(instance_ip, f"mkdir -p {remote_dir}")

        # Rsync code to remote
        self._rsync(
            src=str(Path(local_project_dir).resolve()) + "/",
            dst=remote_dir,
            to_remote=True,
            ip=instance_ip
        )

        print("Code deployment complete")

    def run_experiment(
        self,
        instance_ip: str,
        config_path: str,
        remote_dir: str = "~/llm-radiation"
    ) -> subprocess.CompletedProcess:
        """Run an experiment on the remote instance.

        Args:
            instance_ip: IP address of the instance
            config_path: Path to experiment config (relative to remote_dir)
            remote_dir: Remote directory containing the project (default: ~/llm-radiation)

        Returns:
            CompletedProcess object with stdout, stderr, and return code

        Raises:
            subprocess.CalledProcessError: If the experiment fails
        """
        print(f"Running experiment with config: {config_path}")

        command = (
            f"cd {remote_dir} && "
            f"python -m llm_radiation.experiments.runner --config {config_path}"
        )

        result = self._ssh_cmd(instance_ip, command)
        print("Experiment complete")

        return result

    def collect_results(
        self,
        instance_ip: str,
        remote_dir: str = "~/llm-radiation",
        local_dir: str = "results/"
    ) -> None:
        """Collect experiment results from the remote instance.

        Args:
            instance_ip: IP address of the instance
            remote_dir: Remote directory containing results (default: ~/llm-radiation)
            local_dir: Local directory to save results to (default: results/)

        Raises:
            subprocess.CalledProcessError: If rsync fails
        """
        print(f"Collecting results from {instance_ip}:{remote_dir}/results")

        # Create local results directory
        Path(local_dir).mkdir(parents=True, exist_ok=True)

        # Rsync results back
        self._rsync(
            src=f"{remote_dir}/results/",
            dst=str(Path(local_dir).resolve()),
            to_remote=False,
            ip=instance_ip
        )

        print(f"Results collected to {local_dir}")

    def teardown(self, instance_id: str) -> None:
        """Terminate an instance.

        Args:
            instance_id: ID of the instance to terminate

        Raises:
            requests.HTTPError: If termination fails
        """
        print(f"Terminating instance {instance_id}...")
        self.client.terminate_instance(instance_id)
        print("Instance terminated")

    def _ssh_cmd(self, ip: str, command: str) -> subprocess.CompletedProcess:
        """Execute a command on the remote instance via SSH.

        Args:
            ip: IP address of the instance
            command: Command to execute

        Returns:
            CompletedProcess object with stdout, stderr, and return code

        Raises:
            subprocess.CalledProcessError: If the command fails
        """
        ssh_command = [
            "ssh",
            "-i", str(self.ssh_key_path),
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            f"ubuntu@{ip}",
            command
        ]

        result = subprocess.run(
            ssh_command,
            capture_output=True,
            text=True,
            check=True
        )

        return result

    def _rsync(
        self,
        src: str,
        dst: str,
        to_remote: bool = True,
        ip: str = ""
    ) -> None:
        """Sync files using rsync over SSH.

        Args:
            src: Source path
            dst: Destination path
            to_remote: If True, sync to remote; if False, sync from remote
            ip: IP address (required for remote operations)

        Raises:
            subprocess.CalledProcessError: If rsync fails
        """
        if to_remote:
            remote_path = f"ubuntu@{ip}:{dst}"
            rsync_src = src
            rsync_dst = remote_path
        else:
            remote_path = f"ubuntu@{ip}:{src}"
            rsync_src = remote_path
            rsync_dst = dst

        rsync_command = [
            "rsync",
            "-avz",
            "-e", f"ssh -i {self.ssh_key_path} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null",
            "--exclude=.git",
            "--exclude=__pycache__",
            "--exclude=*.pyc",
            "--exclude=.venv",
            "--exclude=venv",
            rsync_src,
            rsync_dst
        ]

        subprocess.run(rsync_command, check=True)
