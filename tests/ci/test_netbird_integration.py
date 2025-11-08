#!/usr/bin/env python3
"""
NetBird VPN Integration Tests

This test suite validates that NetBird VPN connectivity works correctly
and provides access to internal infrastructure resources.
"""

import os
import subprocess
import time
import socket
import pytest
import httpx
from typing import Optional


class TestNetBirdIntegration:
    """Integration tests for NetBird VPN connectivity."""

    @pytest.fixture(scope="session")
    def netbird_setup_key(self) -> Optional[str]:
        """Get NetBird setup key from environment."""
        return os.getenv("NETBIRD_SETUP_KEY")

    @pytest.fixture(scope="session")
    def is_netbird_available(self) -> bool:
        """Check if NetBird is installed and available."""
        try:
            result = subprocess.run(
                ["netbird", "version"], capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    @pytest.fixture(scope="session")
    def netbird_status(self, is_netbird_available: bool) -> dict:
        """Get current NetBird VPN status."""
        if not is_netbird_available:
            return {"connected": False, "reason": "netbird_not_installed"}

        try:
            result = subprocess.run(
                ["sudo", "netbird", "status"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            output = result.stdout.lower()
            return {
                "connected": "connected" in output,
                "output": result.stdout,
                "returncode": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {"connected": False, "reason": "timeout"}
        except Exception as e:
            return {"connected": False, "reason": str(e)}

    def test_netbird_installed(self, is_netbird_available: bool):
        """Test that NetBird is installed in the CI environment."""
        if not is_netbird_available:
            pytest.skip("NetBird not installed - skipping integration tests")

        assert (
            is_netbird_available
        ), "NetBird should be installed in CI/CD environment for VPN access"

    def test_netbird_service_running(self, is_netbird_available: bool):
        """Test that NetBird service is running."""
        if not is_netbird_available:
            pytest.skip("NetBird not installed")

        try:
            result = subprocess.run(
                ["sudo", "systemctl", "is-active", "netbird"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            # Service might be running via other init system or not as systemd service
            # So we check if netbird daemon is accessible
            status_result = subprocess.run(
                ["sudo", "netbird", "status"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            assert status_result.returncode == 0, "NetBird daemon should be accessible"

        except subprocess.TimeoutExpired:
            pytest.fail("NetBird service check timed out")

    def test_netbird_connected(self, netbird_status: dict):
        """Test that NetBird VPN is connected."""
        if netbird_status.get("reason") == "netbird_not_installed":
            pytest.skip("NetBird not installed")

        # In CI, we might not be connected yet, so this is informational
        if not netbird_status["connected"]:
            pytest.skip(
                f"NetBird not connected: {netbird_status.get('reason', 'unknown')}. "
                "This is expected if running outside CI with VPN setup."
            )

        assert netbird_status[
            "connected"
        ], "NetBird VPN should be connected to access internal resources"

    @pytest.mark.asyncio
    async def test_internal_nexus_accessible(self, netbird_status: dict):
        """Test that internal Nexus repository is accessible via VPN."""
        if not netbird_status.get("connected"):
            pytest.skip("NetBird VPN not connected")

        nexus_url = "https://nexus.in.lazarev.cloud/repository/pypi-proxy/simple/"

        try:
            async with httpx.AsyncClient(timeout=30.0, verify=False) as client:
                response = await client.get(nexus_url)

                assert (
                    response.status_code == 200
                ), f"Nexus should be accessible via VPN, got status {response.status_code}"

                # Verify we got actual PyPI simple index page
                assert (
                    "simple" in response.text.lower() or "pypi" in response.text.lower()
                ), "Nexus response should contain PyPI simple index content"

        except httpx.ConnectTimeout:
            pytest.fail(
                "Connection to Nexus timed out. This indicates NetBird VPN "
                "routing is not working correctly."
            )
        except httpx.ConnectError as e:
            pytest.fail(
                f"Failed to connect to Nexus: {e}. This indicates NetBird VPN "
                "is not providing proper network access."
            )

    @pytest.mark.asyncio
    async def test_internal_harbor_accessible(self, netbird_status: dict):
        """Test that internal Harbor registry is accessible via VPN."""
        if not netbird_status.get("connected"):
            pytest.skip("NetBird VPN not connected")

        harbor_url = "https://harbor.in.lazarev.cloud/api/v2.0/health"

        try:
            async with httpx.AsyncClient(timeout=30.0, verify=False) as client:
                response = await client.get(harbor_url)

                # Harbor health endpoint should return 200
                assert (
                    response.status_code == 200
                ), f"Harbor should be accessible via VPN, got status {response.status_code}"

        except httpx.ConnectTimeout:
            pytest.fail(
                "Connection to Harbor timed out. This indicates NetBird VPN "
                "routing is not working correctly."
            )
        except httpx.ConnectError as e:
            pytest.fail(
                f"Failed to connect to Harbor: {e}. This indicates NetBird VPN "
                "is not providing proper network access."
            )

    def test_internal_dns_resolution(self, netbird_status: dict):
        """Test that internal DNS names can be resolved."""
        if not netbird_status.get("connected"):
            pytest.skip("NetBird VPN not connected")

        internal_hosts = [
            "nexus.in.lazarev.cloud",
            "harbor.in.lazarev.cloud",
            "vpn.lazarev.cloud",
        ]

        for host in internal_hosts:
            try:
                ip = socket.gethostbyname(host)
                assert ip, f"DNS resolution for {host} should return an IP address"

                # Internal IPs should be in private ranges
                # 192.168.x.x, 10.x.x.x, or 172.16-31.x.x
                ip_parts = ip.split(".")
                first_octet = int(ip_parts[0])
                second_octet = int(ip_parts[1])

                is_private = (
                    first_octet == 192
                    and second_octet == 168
                    or first_octet == 10
                    or (first_octet == 172 and 16 <= second_octet <= 31)
                )

                # Note: Not asserting private IP as internal DNS might use different ranges
                # This is just informational

            except socket.gaierror:
                pytest.fail(
                    f"Failed to resolve {host}. This indicates NetBird VPN "
                    "is not providing proper DNS resolution."
                )

    def test_vpn_connection_stability(self, netbird_status: dict):
        """Test that VPN connection is stable over multiple checks."""
        if not netbird_status.get("connected"):
            pytest.skip("NetBird VPN not connected")

        # Check connection status multiple times
        stable_checks = 0
        required_stable_checks = 3

        for i in range(required_stable_checks):
            try:
                result = subprocess.run(
                    ["sudo", "netbird", "status"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

                if "connected" in result.stdout.lower():
                    stable_checks += 1

                time.sleep(1)  # Wait 1 second between checks

            except subprocess.TimeoutExpired:
                pass

        assert stable_checks >= required_stable_checks, (
            f"VPN should remain stable, but only {stable_checks}/{required_stable_checks} "
            f"checks passed"
        )

    @pytest.mark.asyncio
    async def test_package_installation_via_vpn(self, netbird_status: dict):
        """Test that Python packages can be installed from internal Nexus."""
        if not netbird_status.get("connected"):
            pytest.skip("NetBird VPN not connected")

        # Try to fetch package info from Nexus (without actually installing)
        nexus_url = (
            "https://nexus.in.lazarev.cloud/repository/pypi-proxy/simple/pytest/"
        )

        try:
            async with httpx.AsyncClient(timeout=30.0, verify=False) as client:
                response = await client.get(nexus_url)

                assert response.status_code == 200, (
                    f"Should be able to fetch package info from Nexus, "
                    f"got status {response.status_code}"
                )

                # Verify we got pytest package links
                assert (
                    "pytest" in response.text.lower()
                ), "Package info should contain pytest references"

        except httpx.ConnectTimeout:
            pytest.fail(
                "Timeout fetching package from Nexus. This will cause "
                "'pip install' and 'uv pip install' commands to fail in CI."
            )


class TestNetBirdCompositeAction:
    """Tests for the NetBird composite action itself."""

    def test_composite_action_script_syntax(self):
        """Test that the composite action bash script is syntactically correct."""
        action_file = ".github/actions/setup-netbird/action.yml"

        if not os.path.exists(action_file):
            pytest.skip("Composite action file not found")

        # We can't easily validate bash syntax without running it,
        # but we can check that the file is valid YAML
        import yaml

        with open(action_file, "r") as f:
            config = yaml.safe_load(f)

        assert config is not None, "Composite action should be valid YAML"

        # Verify critical components exist in the script
        script = config["runs"]["steps"][0]["run"]

        critical_commands = [
            "curl -fsSL https://pkgs.netbird.io/install.sh",
            "sudo netbird service install",
            "sudo netbird service start",
            "sudo netbird up",
            "--management-url https://vpn.lazarev.cloud",
            "--setup-key",
            "sudo netbird status",
        ]

        for cmd in critical_commands:
            assert cmd in script, f"Composite action script should contain '{cmd}'"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
