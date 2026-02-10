"""Lambda Labs Cloud API client for GPU instance management."""

import time
from typing import Any

import requests


class LambdaClient:
    """Client for interacting with Lambda Labs Cloud API."""

    def __init__(self, api_key: str):
        """Initialize the Lambda Labs API client.

        Args:
            api_key: Lambda Labs API key for authentication
        """
        self.api_key = api_key
        self.base_url = "https://cloud.lambdalabs.com/api/v1"
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        })

    def _request(self, method: str, endpoint: str, **kwargs: Any) -> dict[str, Any]:
        """Make an authenticated request to the Lambda Labs API.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path (will be appended to base_url)
            **kwargs: Additional arguments to pass to requests

        Returns:
            JSON response as dictionary

        Raises:
            requests.HTTPError: If the request fails
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        response = self.session.request(method, url, **kwargs)
        response.raise_for_status()
        return response.json()

    def list_instance_types(self) -> list[dict[str, Any]]:
        """List all available instance types.

        Returns:
            List of instance type dictionaries with details about GPU types,
            pricing, and availability
        """
        result = self._request("GET", "/instance-types")
        return result.get("data", {})

    def list_instances(self) -> list[dict[str, Any]]:
        """List all instances in the account.

        Returns:
            List of instance dictionaries with details about running instances
        """
        result = self._request("GET", "/instances")
        return result.get("data", [])

    def launch_instance(
        self,
        instance_type: str,
        region: str,
        ssh_key_names: list[str],
        name: str = ""
    ) -> dict[str, Any]:
        """Launch a new instance.

        Args:
            instance_type: Type of instance to launch (e.g., "gpu_1x_a100")
            region: Region to launch in (e.g., "us-west-1")
            ssh_key_names: List of SSH key names to add to the instance
            name: Optional name for the instance

        Returns:
            Dictionary containing instance_ids of launched instances

        Raises:
            requests.HTTPError: If the launch fails
        """
        payload = {
            "region_name": region,
            "instance_type_name": instance_type,
            "ssh_key_names": ssh_key_names,
        }
        if name:
            payload["name"] = name

        result = self._request("POST", "/instance-operations/launch", json=payload)
        return result.get("data", {})

    def get_instance(self, instance_id: str) -> dict[str, Any]:
        """Get details about a specific instance.

        Args:
            instance_id: ID of the instance to query

        Returns:
            Dictionary containing instance details including status, IP, etc.

        Raises:
            requests.HTTPError: If the request fails
        """
        result = self._request("GET", f"/instances/{instance_id}")
        return result.get("data", {})

    def terminate_instance(self, instance_id: str) -> dict[str, Any]:
        """Terminate an instance.

        Args:
            instance_id: ID of the instance to terminate

        Returns:
            Dictionary containing termination confirmation

        Raises:
            requests.HTTPError: If the termination fails
        """
        payload = {"instance_ids": [instance_id]}
        result = self._request("POST", "/instance-operations/terminate", json=payload)
        return result.get("data", {})

    def wait_for_active(
        self,
        instance_id: str,
        timeout: int = 600,
        poll_interval: int = 10
    ) -> dict[str, Any]:
        """Wait for an instance to reach active state.

        Args:
            instance_id: ID of the instance to wait for
            timeout: Maximum time to wait in seconds (default: 600)
            poll_interval: Time between status checks in seconds (default: 10)

        Returns:
            Dictionary containing final instance details

        Raises:
            TimeoutError: If instance doesn't become active within timeout
            requests.HTTPError: If status checks fail
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            instance_data = self.get_instance(instance_id)
            status = instance_data.get("status")

            if status == "active":
                return instance_data
            elif status in ["terminated", "terminating"]:
                raise RuntimeError(
                    f"Instance {instance_id} entered terminal state: {status}"
                )

            time.sleep(poll_interval)

        raise TimeoutError(
            f"Instance {instance_id} did not become active within {timeout} seconds"
        )
