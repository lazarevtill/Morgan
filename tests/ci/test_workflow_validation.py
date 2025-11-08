#!/usr/bin/env python3
"""
Workflow validation tests for CI/CD configuration.

This test suite validates that all GitHub Actions workflows are properly
configured with necessary components like NetBird VPN setup, environment
variables, and proper action sequencing.
"""

import os
from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml


class TestWorkflowValidation:
    """Test suite for validating GitHub Actions workflows."""

    @pytest.fixture
    def workflows_dir(self) -> Path:
        """Get the workflows directory path."""
        repo_root = Path(__file__).parent.parent.parent
        return repo_root / ".github" / "workflows"

    @pytest.fixture
    def workflow_files(self, workflows_dir: Path) -> Dict[str, Path]:
        """Get all workflow YAML files."""
        return {f.stem: f for f in workflows_dir.glob("*.yml") if f.is_file()}

    @pytest.fixture
    def composite_actions_dir(self) -> Path:
        """Get the composite actions directory path."""
        repo_root = Path(__file__).parent.parent.parent
        return repo_root / ".github" / "actions"

    def load_workflow(self, workflow_path: Path) -> Dict[str, Any]:
        """Load and parse a workflow YAML file."""

        # Use custom resolver to handle 'on' keyword correctly
        class CustomLoader(yaml.SafeLoader):
            pass

        # Prevent 'on' from being interpreted as boolean
        CustomLoader.yaml_implicit_resolvers = {
            k: [r for r in v if r[0] != "tag:yaml.org,2002:bool"]
            for k, v in CustomLoader.yaml_implicit_resolvers.items()
        }

        with open(workflow_path, "r") as f:
            content = f.read()
            # Alternative: just read as text and check structure
            workflow = yaml.load(content, Loader=CustomLoader)
            # If 'on' was parsed as True, rename it
            if True in workflow and "on" not in workflow:
                workflow["on"] = workflow.pop(True)
            return workflow

    def get_jobs_with_pip_or_uv(self, workflow: Dict[str, Any]) -> List[str]:
        """Find all jobs that install Python packages with pip or uv."""
        jobs_with_installs = []

        if "jobs" not in workflow:
            return jobs_with_installs

        for job_name, job_config in workflow["jobs"].items():
            if "steps" not in job_config:
                continue

            for step in job_config["steps"]:
                if "run" not in step:
                    continue

                run_cmd = step["run"]
                # Check for pip or uv install commands
                if any(cmd in run_cmd for cmd in ["pip install", "uv pip install"]):
                    jobs_with_installs.append(job_name)
                    break

        return jobs_with_installs

    def job_has_netbird_setup(self, job_config: Dict[str, Any]) -> bool:
        """Check if a job has NetBird VPN setup."""
        if "steps" not in job_config:
            return False

        for step in job_config["steps"]:
            if "uses" in step:
                if "setup-netbird" in step["uses"]:
                    return True
            if "name" in step:
                if "NetBird" in step["name"]:
                    return True

        return False

    def job_has_checkout(self, job_config: Dict[str, Any]) -> bool:
        """Check if a job has checkout action before other actions."""
        if "steps" not in job_config:
            return False

        for step in job_config["steps"]:
            if "uses" in step and "checkout" in step["uses"]:
                return True

        return False

    def test_composite_action_exists(self, composite_actions_dir: Path):
        """Test that the NetBird composite action exists and is valid."""
        setup_netbird_action = composite_actions_dir / "setup-netbird" / "action.yml"

        assert (
            setup_netbird_action.exists()
        ), "NetBird composite action should exist at .github/actions/setup-netbird/action.yml"

        with open(setup_netbird_action, "r") as f:
            action_config = yaml.safe_load(f)

        # Validate composite action structure
        assert "name" in action_config, "Composite action should have a name"
        assert (
            "description" in action_config
        ), "Composite action should have a description"
        assert "inputs" in action_config, "Composite action should define inputs"
        assert (
            "setup-key" in action_config["inputs"]
        ), "Composite action should have 'setup-key' input"
        assert (
            action_config["inputs"]["setup-key"]["required"] is True
        ), "setup-key input should be required"
        assert (
            "runs" in action_config
        ), "Composite action should have runs configuration"
        assert (
            action_config["runs"]["using"] == "composite"
        ), "Action should use composite type"

    def test_workflows_have_proper_structure(self, workflow_files: Dict[str, Path]):
        """Test that all workflows have proper basic structure."""
        for workflow_name, workflow_path in workflow_files.items():
            workflow = self.load_workflow(workflow_path)

            assert "name" in workflow, f"Workflow {workflow_name} should have a name"
            assert (
                "on" in workflow
            ), f"Workflow {workflow_name} should have triggers defined"
            assert (
                "jobs" in workflow
            ), f"Workflow {workflow_name} should have jobs defined"

    def test_jobs_with_package_installs_have_netbird(
        self, workflow_files: Dict[str, Path]
    ):
        """Test that all jobs installing Python packages have NetBird VPN setup."""
        workflows_needing_vpn = ["lint", "test", "security"]

        for workflow_name in workflows_needing_vpn:
            if workflow_name not in workflow_files:
                pytest.skip(f"Workflow {workflow_name}.yml not found")
                continue

            workflow = self.load_workflow(workflow_files[workflow_name])
            jobs_with_installs = self.get_jobs_with_pip_or_uv(workflow)

            for job_name in jobs_with_installs:
                job_config = workflow["jobs"][job_name]
                has_netbird = self.job_has_netbird_setup(job_config)

                assert has_netbird, (
                    f"Job '{job_name}' in {workflow_name}.yml installs packages "
                    f"but doesn't have NetBird VPN setup. This will cause timeouts "
                    f"when accessing nexus.in.lazarev.cloud"
                )

    def test_netbird_setup_has_checkout_first(self, workflow_files: Dict[str, Path]):
        """Test that jobs using composite actions have checkout step first."""
        for workflow_name, workflow_path in workflow_files.items():
            workflow = self.load_workflow(workflow_path)

            if "jobs" not in workflow:
                continue

            for job_name, job_config in workflow["jobs"].items():
                if not self.job_has_netbird_setup(job_config):
                    continue

                has_checkout = self.job_has_checkout(job_config)
                assert has_checkout, (
                    f"Job '{job_name}' in {workflow_name}.yml uses composite action "
                    f"but doesn't have checkout step. Composite actions require "
                    f"repository to be checked out first."
                )

    def test_workflows_use_consistent_env_vars(self, workflow_files: Dict[str, Path]):
        """Test that workflows use consistent environment variable configuration."""
        env_var_requirements = {
            "lint": ["UV_INDEX_URL", "PIP_INDEX_URL"],
            "test": ["UV_INDEX_URL"],
            "security": ["PIP_INDEX_URL"],
        }

        for workflow_name, required_vars in env_var_requirements.items():
            if workflow_name not in workflow_files:
                continue

            workflow = self.load_workflow(workflow_files[workflow_name])

            # Check for workflow-level env vars
            workflow_env = workflow.get("env", {})

            for var in required_vars:
                assert var in workflow_env, (
                    f"Workflow {workflow_name}.yml should have {var} at workflow level "
                    f"to avoid duplication across jobs"
                )

                # Verify it points to the internal Nexus
                assert (
                    "nexus.in.lazarev.cloud" in workflow_env[var]
                ), f"{var} in {workflow_name}.yml should point to internal Nexus repository"

    def test_no_redundant_env_vars_in_steps(self, workflow_files: Dict[str, Path]):
        """Test that steps don't redundantly define env vars that exist at workflow level."""
        for workflow_name, workflow_path in workflow_files.items():
            workflow = self.load_workflow(workflow_path)
            workflow_env = workflow.get("env", {})

            if not workflow_env:
                continue

            if "jobs" not in workflow:
                continue

            for job_name, job_config in workflow["jobs"].items():
                if "steps" not in job_config:
                    continue

                for step_idx, step in enumerate(job_config["steps"]):
                    step_env = step.get("env", {})

                    for env_var in workflow_env.keys():
                        assert env_var not in step_env, (
                            f"Step {step_idx} in job '{job_name}' of {workflow_name}.yml "
                            f"redundantly defines {env_var} which already exists at "
                            f"workflow level"
                        )

    def test_docker_build_jobs_have_host_network(self, workflow_files: Dict[str, Path]):
        """Test that Docker build jobs use host network mode for VPN access."""
        workflows_with_docker = ["test", "deploy"]

        for workflow_name in workflows_with_docker:
            if workflow_name not in workflow_files:
                continue

            workflow = self.load_workflow(workflow_files[workflow_name])

            if "jobs" not in workflow:
                continue

            for job_name, job_config in workflow["jobs"].items():
                # Skip jobs that don't build Docker images
                if "docker" not in job_name.lower() and workflow_name != "deploy":
                    continue

                if "steps" not in job_config:
                    continue

                # Find Docker Buildx setup
                has_buildx_with_host_network = False
                for step in job_config["steps"]:
                    if "uses" in step and "docker/setup-buildx-action" in step["uses"]:
                        if "with" in step:
                            driver_opts = step["with"].get("driver-opts", "")
                            if "network=host" in driver_opts:
                                has_buildx_with_host_network = True

                # If job uses Docker Buildx and has NetBird, it should have host network
                if self.job_has_netbird_setup(job_config):
                    assert has_buildx_with_host_network, (
                        f"Job '{job_name}' in {workflow_name}.yml uses Docker build "
                        f"with NetBird VPN but doesn't configure host network mode. "
                        f"This will prevent Docker from accessing internal resources."
                    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
