#!/usr/bin/env python3
"""
Morgan Distributed Deployment CLI

Easy-to-use command-line tool for managing Morgan across 6 hosts.

Usage:
    # Deploy to all hosts
    python -m morgan.cli.distributed_cli deploy

    # Update all hosts (rolling, zero-downtime)
    python -m morgan.cli.distributed_cli update

    # Health check
    python -m morgan.cli.distributed_cli health

    # Restart service
    python -m morgan.cli.distributed_cli restart ollama

    # Sync configuration
    python -m morgan.cli.distributed_cli sync-config

    # Show status
    python -m morgan.cli.distributed_cli status
"""

import asyncio
import json
import sys

try:
    import click

    CLICK_AVAILABLE = True
except ImportError:
    CLICK_AVAILABLE = False
    print("click package required. Install with: pip install click")
    sys.exit(1)

from morgan.infrastructure.distributed_manager import (
    ServiceType,
    get_distributed_manager,
)
from morgan.utils.logger import get_logger

logger = get_logger(__name__)


@click.group()
@click.option("--ssh-key", default=None, help="Path to SSH private key")
@click.pass_context
def cli(ctx, ssh_key):
    """Morgan Distributed Deployment Manager"""
    ctx.ensure_object(dict)
    ctx.obj["manager"] = get_distributed_manager(
        ssh_key_path=ssh_key, auto_load_config=True
    )


@cli.command()
@click.option("--branch", default="v2-0.0.1", help="Git branch to deploy")
@click.option("--force", is_flag=True, help="Force update (discard local changes)")
@click.option("--parallel", is_flag=True, help="Deploy in parallel (faster)")
@click.pass_context
def deploy(ctx, branch, force, parallel):
    """Deploy Morgan to all hosts"""
    manager = ctx.obj["manager"]

    click.echo(f"Deploying Morgan (branch: {branch}) to all hosts...")
    if force:
        click.echo("⚠ WARNING: Force mode will discard local changes!")

    results = asyncio.run(
        manager.deploy_all(git_branch=branch, force=force, parallel=parallel)
    )

    # Display results
    click.echo("\n" + "=" * 60)
    click.echo("Deployment Results")
    click.echo("=" * 60)

    for result in results:
        status = "✓" if result.success else "✗"
        click.echo(f"{status} {result.host}: {result.message} ({result.duration:.2f}s)")

        if result.error:
            click.echo(f"   Error: {result.error}")

    successful = sum(1 for r in results if r.success)
    click.echo(f"\nTotal: {successful}/{len(results)} hosts deployed successfully")


@cli.command()
@click.option("--branch", default="v2-0.0.1", help="Git branch to update to")
@click.option(
    "--rolling/--parallel", default=True, help="Rolling update (zero-downtime)"
)
@click.pass_context
def update(ctx, branch, rolling):
    """Update all hosts with latest code"""
    manager = ctx.obj["manager"]

    if rolling:
        click.echo("Starting rolling update (zero-downtime)...")
        click.echo(
            "Update order: Background → Reranking → Embeddings → LLM#2 → LLM#1 → Manager"
        )
    else:
        click.echo("Starting parallel update (faster, but downtime expected)...")

    results = asyncio.run(manager.update_all(git_branch=branch, rolling=rolling))

    # Display results
    click.echo("\n" + "=" * 60)
    click.echo("Update Results")
    click.echo("=" * 60)

    for result in results:
        status = "✓" if result.success else "✗"
        click.echo(f"{status} {result.host}: {result.message} ({result.duration:.2f}s)")

    successful = sum(1 for r in results if r.success)
    click.echo(f"\nTotal: {successful}/{len(results)} hosts updated successfully")


@cli.command()
@click.pass_context
def health(ctx):
    """Check health of all hosts"""
    manager = ctx.obj["manager"]

    click.echo("Checking health of all hosts...")

    status = asyncio.run(manager.health_check_all())

    # Display results
    click.echo("\n" + "=" * 60)
    click.echo("Health Status")
    click.echo("=" * 60)

    click.echo(f"Total Hosts: {status['total_hosts']}")
    click.echo(f"Healthy: {status['healthy_hosts']}")
    click.echo(f"Unhealthy: {status['unhealthy_hosts']}")

    click.echo("\nDetailed Status:")
    for host in status["hosts"]:
        status_icon = "✓" if host.get("healthy") else "✗"
        click.echo(f"\n{status_icon} {host['hostname']} ({host['role']})")

        if host.get("error"):
            click.echo(f"   Error: {host['error']}")

        if host.get("services"):
            click.echo("   Services:")
            for service, active in host["services"].items():
                service_icon = "✓" if active else "✗"
                click.echo(f"     {service_icon} {service}")

        if host.get("gpu"):
            gpu = host["gpu"]
            click.echo(f"   GPU: {gpu['model']}")
            click.echo(f"     Utilization: {gpu['utilization']}")
            click.echo(f"     Memory: {gpu['memory_used']}")


@cli.command()
@click.argument("service")
@click.option(
    "--hosts", multiple=True, help="Specific hosts (default: all with service)"
)
@click.pass_context
def restart(ctx, service, hosts):
    """Restart a service on hosts"""
    manager = ctx.obj["manager"]

    # Map service name to enum
    service_map = {
        "ollama": ServiceType.OLLAMA,
        "morgan": ServiceType.MORGAN_CORE,
        "qdrant": ServiceType.QDRANT,
        "redis": ServiceType.REDIS,
        "reranking": ServiceType.RERANKING_API,
        "prometheus": ServiceType.PROMETHEUS,
        "grafana": ServiceType.GRAFANA,
    }

    if service not in service_map:
        click.echo(f"Unknown service: {service}")
        click.echo(f"Available: {', '.join(service_map.keys())}")
        sys.exit(1)

    service_type = service_map[service]
    hosts_list = list(hosts) if hosts else None

    click.echo(f"Restarting {service} on hosts...")

    results = asyncio.run(
        manager.restart_service(service=service_type, hosts=hosts_list)
    )

    # Display results
    click.echo("\n" + "=" * 60)
    click.echo("Restart Results")
    click.echo("=" * 60)

    for hostname, success in results.items():
        status = "✓" if success else "✗"
        click.echo(f"{status} {hostname}")

    successful = sum(1 for s in results.values() if s)
    click.echo(f"\nTotal: {successful}/{len(results)} hosts restarted successfully")


@cli.command()
@click.option("--config-file", default=".env", help="Config file to sync")
@click.option("--source", default=None, help="Source host (default: local file)")
@click.pass_context
def sync_config(ctx, config_file, source):
    """Synchronize configuration across all hosts"""
    manager = ctx.obj["manager"]

    if source:
        click.echo(f"Syncing {config_file} from {source} to all hosts...")
    else:
        click.echo(f"Syncing {config_file} from local to all hosts...")

    results = asyncio.run(
        manager.sync_config(config_file=config_file, source_host=source)
    )

    # Display results
    click.echo("\n" + "=" * 60)
    click.echo("Sync Results")
    click.echo("=" * 60)

    for hostname, success in results.items():
        status = "✓" if success else "✗"
        click.echo(f"{status} {hostname}")

    successful = sum(1 for s in results.values() if s)
    click.echo(f"\nTotal: {successful}/{len(results)} hosts synced successfully")


@cli.command()
@click.pass_context
def status(ctx):
    """Show current configuration and status"""
    manager = ctx.obj["manager"]

    config = manager.get_config()

    click.echo("=" * 60)
    click.echo("Morgan Distributed Deployment Configuration")
    click.echo("=" * 60)

    click.echo(f"\nSSH Key: {config['ssh_key']}")
    click.echo(f"Total Hosts: {config['total_hosts']}")

    click.echo("\nHosts:")
    for host in config["hosts"]:
        click.echo(f"\n  {host['hostname']} ({host['role']})")
        click.echo(f"    GPU: {host['gpu'] or 'None'}")
        click.echo(f"    Services: {', '.join(host['services'])}")


@cli.command()
@click.pass_context
def config(ctx):
    """Show configuration as JSON"""
    manager = ctx.obj["manager"]
    config = manager.get_config()
    click.echo(json.dumps(config, indent=2))


if __name__ == "__main__":
    cli(obj={})
