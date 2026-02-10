#!/usr/bin/env python3
"""Provision Lambda Labs GPU instance, deploy code, run experiment, collect results."""

import argparse
import logging
import os
import sys

from llm_radiation.infrastructure.lambda_client import LambdaClient
from llm_radiation.infrastructure.deployer import LambdaDeployer


def main():
    parser = argparse.ArgumentParser(description="Launch experiment on Lambda Labs")
    parser.add_argument("config", type=str, help="Path to experiment YAML config")
    parser.add_argument("--instance-type", default="gpu_1x_a100", help="Lambda instance type")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--ssh-key-name", required=True, help="SSH key name registered in Lambda")
    parser.add_argument("--ssh-key-path", required=True, help="Local path to SSH private key")
    parser.add_argument("--no-terminate", action="store_true", help="Don't terminate instance after completion")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    api_key = os.environ.get("LAMBDA_API_KEY", "")
    if not api_key:
        logger.error("LAMBDA_API_KEY environment variable not set")
        return 1

    client = LambdaClient(api_key=api_key)
    deployer = LambdaDeployer(client=client, ssh_key_path=args.ssh_key_path)

    # Provision
    logger.info(f"Provisioning {args.instance_type} in {args.region}...")
    instance = deployer.provision(
        instance_type=args.instance_type,
        region=args.region,
        ssh_key_names=[args.ssh_key_name],
    )
    instance_id = instance["id"]
    instance_ip = instance["ip"]
    logger.info(f"Instance {instance_id} active at {instance_ip}")

    try:
        # Deploy
        project_dir = str(__import__("pathlib").Path(__file__).parent.parent)
        deployer.deploy_code(instance_ip, project_dir)

        # Run
        logger.info("Running experiment...")
        result = deployer.run_experiment(instance_ip, args.config)
        logger.info(f"Experiment exit code: {result.returncode}")

        # Collect results
        deployer.collect_results(instance_ip)
        logger.info("Results collected to results/")
    finally:
        if not args.no_terminate:
            logger.info(f"Terminating instance {instance_id}...")
            deployer.teardown(instance_id)

    return 0


if __name__ == "__main__":
    sys.exit(main())
