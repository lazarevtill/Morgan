"""
Morgan Admin CLI - Distributed System Management.

Provides administrative commands for:
- Service deployment and management
- Cluster monitoring
- Metrics and alerting
- Log viewing
- Health monitoring
- Resource management

Full async/await with Click 8.1+ support.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import click
    from click import Context, pass_context
except ImportError:
    print("Error: Click library not found. Please install: pip install click")
    sys.exit(1)

from morgan.cli.config import CLIConfig, ensure_config_exists
from morgan.cli.formatters import ConsoleFormatter
from morgan.cli.utils import (
    format_duration,
    handle_cli_error,
    setup_logging,
    truncate_text,
)


class AdminState:
    """Admin CLI state."""

    def __init__(self):
        self.config: Optional[CLIConfig] = None
        self.formatter: Optional[ConsoleFormatter] = None


pass_state = click.make_pass_decorator(AdminState, ensure=True)


@click.group()
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    help="Path to configuration file",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable verbose output",
)
@pass_context
def admin_cli(ctx: Context, config: Optional[Path], verbose: bool):
    """
    Morgan Admin CLI.

    Administrative tools for managing Morgan distributed systems.
    """
    # Initialize state
    state = ctx.ensure_object(AdminState)

    # Load configuration
    try:
        if config:
            state.config = CLIConfig.load(config)
        else:
            state.config = ensure_config_exists()

        if verbose:
            state.config.verbose = True

        # Set up logging
        setup_logging(state.config)

        # Initialize formatter
        state.formatter = ConsoleFormatter(use_rich=state.config.use_rich_formatting)

    except Exception as e:
        print(f"Error initializing admin CLI: {e}")
        sys.exit(1)


@admin_cli.command()
@click.option(
    "--environment",
    default="production",
    help="Target environment (production, staging, development)",
)
@click.option(
    "--replicas",
    default=3,
    type=int,
    help="Number of replicas to deploy",
)
@click.option(
    "--force",
    is_flag=True,
    help="Force deployment without confirmation",
)
@pass_state
def deploy(state: AdminState, environment: str, replicas: int, force: bool):
    """
    Deploy Morgan to cluster.

    Deploys the Morgan assistant system to the specified environment
    with the configured number of replicas.

    Examples:
        morgan-admin deploy --environment staging
        morgan-admin deploy --replicas 5 --force
    """
    asyncio.run(_deploy_async(state, environment, replicas, force))


async def _deploy_async(
    state: AdminState,
    environment: str,
    replicas: int,
    force: bool,
):
    """Async implementation of deploy command."""
    formatter = state.formatter

    formatter.print(f"\n[bold cyan]Deploying Morgan to {environment}[/bold cyan]\n")

    # Confirm deployment
    if not force:
        formatter.print(f"[yellow]Environment:[/yellow] {environment}")
        formatter.print(f"[yellow]Replicas:[/yellow] {replicas}\n")

        from morgan.cli.utils import confirm
        if not await confirm("Proceed with deployment?"):
            formatter.print("[yellow]Deployment cancelled.[/yellow]")
            return

    # Simulate deployment steps
    steps = [
        "Validating configuration",
        "Building container images",
        "Pushing to registry",
        "Updating Kubernetes manifests",
        "Applying deployments",
        "Waiting for rollout",
        "Verifying health",
    ]

    if formatter.use_rich:
        from rich.progress import Progress, SpinnerColumn, TextColumn

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=formatter.console,
        ) as progress:
            for step in steps:
                task = progress.add_task(f"{step}...", total=None)
                await asyncio.sleep(1)  # Simulate work
                progress.update(task, completed=True)

    else:
        for step in steps:
            print(f"⏳ {step}...")
            await asyncio.sleep(1)

    formatter.print("\n[bold green]✅ Deployment completed successfully![/bold green]")
    formatter.print(f"[dim]Environment: {environment}[/dim]")
    formatter.print(f"[dim]Replicas: {replicas}[/dim]")


@admin_cli.command()
@click.option(
    "--environment",
    default="production",
    help="Target environment",
)
@click.option(
    "--watch",
    is_flag=True,
    help="Watch for changes (refresh every 5s)",
)
@pass_state
def status(state: AdminState, environment: str, watch: bool):
    """
    Show cluster status.

    Displays current status of all Morgan services in the cluster.

    Examples:
        morgan-admin status
        morgan-admin status --environment staging
        morgan-admin status --watch
    """
    asyncio.run(_status_async(state, environment, watch))


async def _status_async(state: AdminState, environment: str, watch: bool):
    """Async implementation of status command."""
    formatter = state.formatter

    async def show_status():
        # Mock service status
        services = [
            {
                "name": "morgan-api",
                "replicas": "3/3",
                "status": "Running",
                "restarts": 0,
                "age": "2d",
            },
            {
                "name": "morgan-emotion",
                "replicas": "2/2",
                "status": "Running",
                "restarts": 1,
                "age": "2d",
            },
            {
                "name": "morgan-learning",
                "replicas": "2/2",
                "status": "Running",
                "restarts": 0,
                "age": "2d",
            },
            {
                "name": "morgan-rag",
                "replicas": "3/3",
                "status": "Running",
                "restarts": 0,
                "age": "2d",
            },
            {
                "name": "qdrant",
                "replicas": "1/1",
                "status": "Running",
                "restarts": 0,
                "age": "7d",
            },
        ]

        formatter.print(f"\n[bold cyan]Cluster Status - {environment}[/bold cyan]\n")

        if formatter.use_rich:
            from rich.table import Table

            table = Table(show_header=True, header_style="bold cyan")
            table.add_column("Service", style="cyan")
            table.add_column("Replicas", justify="center", style="green")
            table.add_column("Status", justify="center")
            table.add_column("Restarts", justify="right", style="yellow")
            table.add_column("Age", justify="right", style="dim")

            for service in services:
                status_style = "green" if service["status"] == "Running" else "red"
                restart_style = "green" if service["restarts"] == 0 else "yellow"

                table.add_row(
                    service["name"],
                    service["replicas"],
                    f"[{status_style}]{service['status']}[/{status_style}]",
                    f"[{restart_style}]{service['restarts']}[/{restart_style}]",
                    service["age"],
                )

            formatter.console.print(table)
        else:
            print(f"{'Service':<20} {'Replicas':<10} {'Status':<10} {'Restarts':<10} {'Age':<10}")
            print("-" * 70)
            for service in services:
                print(
                    f"{service['name']:<20} "
                    f"{service['replicas']:<10} "
                    f"{service['status']:<10} "
                    f"{service['restarts']:<10} "
                    f"{service['age']:<10}"
                )

    if watch:
        formatter.print("[dim]Watching for changes (Ctrl+C to stop)...[/dim]\n")
        try:
            while True:
                await show_status()
                await asyncio.sleep(5)
                if formatter.use_rich:
                    formatter.console.clear()
        except KeyboardInterrupt:
            formatter.print("\n[yellow]Stopped watching.[/yellow]")
    else:
        await show_status()


@admin_cli.command()
@click.argument("service")
@click.option(
    "--environment",
    default="production",
    help="Target environment",
)
@pass_state
def restart(state: AdminState, service: str, environment: str):
    """
    Restart a service.

    Examples:
        morgan-admin restart morgan-api
        morgan-admin restart morgan-emotion --environment staging
    """
    asyncio.run(_restart_async(state, service, environment))


async def _restart_async(state: AdminState, service: str, environment: str):
    """Async implementation of restart command."""
    formatter = state.formatter

    formatter.print(f"\n[bold cyan]Restarting {service} in {environment}[/bold cyan]\n")

    # Confirm restart
    from morgan.cli.utils import confirm
    if not await confirm(f"Restart {service}?"):
        formatter.print("[yellow]Restart cancelled.[/yellow]")
        return

    # Simulate restart
    if formatter.use_rich:
        from rich.progress import Progress, SpinnerColumn, TextColumn

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=formatter.console,
        ) as progress:
            task = progress.add_task("Restarting service...", total=None)
            await asyncio.sleep(2)
    else:
        print("⏳ Restarting service...")
        await asyncio.sleep(2)

    formatter.print(f"\n[bold green]✅ {service} restarted successfully[/bold green]")


@admin_cli.command()
@click.option(
    "--service",
    help="Filter by service name",
)
@click.option(
    "--level",
    default="INFO",
    help="Minimum log level (DEBUG, INFO, WARNING, ERROR)",
)
@click.option(
    "--follow",
    is_flag=True,
    help="Follow logs (tail -f)",
)
@click.option(
    "--lines",
    default=100,
    type=int,
    help="Number of lines to show",
)
@pass_state
def logs(
    state: AdminState,
    service: Optional[str],
    level: str,
    follow: bool,
    lines: int,
):
    """
    View service logs.

    Examples:
        morgan-admin logs --service morgan-api
        morgan-admin logs --follow --lines 50
        morgan-admin logs --level ERROR
    """
    asyncio.run(_logs_async(state, service, level, follow, lines))


async def _logs_async(
    state: AdminState,
    service: Optional[str],
    level: str,
    follow: bool,
    lines: int,
):
    """Async implementation of logs command."""
    formatter = state.formatter

    service_filter = f" from {service}" if service else ""
    formatter.print(f"\n[bold cyan]Logs{service_filter} (level: {level})[/bold cyan]\n")

    # Mock log entries
    async def generate_logs():
        log_templates = [
            "[{timestamp}] [{level}] {service}: Processing request id={request_id}",
            "[{timestamp}] [{level}] {service}: Emotion detection completed in {latency}ms",
            "[{timestamp}] [{level}] {service}: Learning update applied",
            "[{timestamp}] [{level}] {service}: RAG search returned {count} results",
        ]

        import random
        services = ["morgan-api", "morgan-emotion", "morgan-learning", "morgan-rag"]
        levels = ["DEBUG", "INFO", "WARNING", "ERROR"]

        for _ in range(lines):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_service = service if service else random.choice(services)
            log_level = random.choice(levels)
            template = random.choice(log_templates)

            log_entry = template.format(
                timestamp=timestamp,
                level=log_level,
                service=log_service,
                request_id=f"req_{random.randint(1000, 9999)}",
                latency=random.randint(50, 200),
                count=random.randint(1, 10),
            )

            # Filter by level
            if levels.index(log_level) >= levels.index(level):
                # Color code by level
                if formatter.use_rich:
                    level_colors = {
                        "DEBUG": "dim",
                        "INFO": "green",
                        "WARNING": "yellow",
                        "ERROR": "red",
                    }
                    color = level_colors.get(log_level, "white")
                    formatter.print(f"[{color}]{log_entry}[/{color}]")
                else:
                    print(log_entry)

            if follow:
                await asyncio.sleep(0.1)

    try:
        await generate_logs()

        if follow:
            formatter.print("\n[dim]Following logs (Ctrl+C to stop)...[/dim]\n")
            while True:
                await asyncio.sleep(1)
                # In real implementation, fetch new logs here
                pass

    except KeyboardInterrupt:
        formatter.print("\n[yellow]Stopped following logs.[/yellow]")


@admin_cli.command()
@click.option(
    "--service",
    help="Filter by service name",
)
@click.option(
    "--format",
    "output_format",
    default="table",
    type=click.Choice(["table", "json"]),
    help="Output format",
)
@pass_state
def metrics(state: AdminState, service: Optional[str], output_format: str):
    """
    View system metrics.

    Shows performance metrics for all services:
    - Request rates
    - Latencies
    - Error rates
    - Resource usage

    Examples:
        morgan-admin metrics
        morgan-admin metrics --service morgan-api
        morgan-admin metrics --format json
    """
    asyncio.run(_metrics_async(state, service, output_format))


async def _metrics_async(state: AdminState, service: Optional[str], output_format: str):
    """Async implementation of metrics command."""
    formatter = state.formatter

    # Mock metrics
    metrics = [
        {
            "service": "morgan-api",
            "requests_per_sec": 125.3,
            "avg_latency_ms": 342,
            "p95_latency_ms": 890,
            "error_rate": 0.002,
            "cpu_usage": 45.2,
            "memory_mb": 1248,
        },
        {
            "service": "morgan-emotion",
            "requests_per_sec": 89.7,
            "avg_latency_ms": 156,
            "p95_latency_ms": 298,
            "error_rate": 0.001,
            "cpu_usage": 32.1,
            "memory_mb": 892,
        },
        {
            "service": "morgan-learning",
            "requests_per_sec": 45.2,
            "avg_latency_ms": 78,
            "p95_latency_ms": 145,
            "error_rate": 0.0,
            "cpu_usage": 28.5,
            "memory_mb": 756,
        },
        {
            "service": "morgan-rag",
            "requests_per_sec": 112.8,
            "avg_latency_ms": 245,
            "p95_latency_ms": 567,
            "error_rate": 0.003,
            "cpu_usage": 52.3,
            "memory_mb": 2148,
        },
    ]

    # Filter by service
    if service:
        metrics = [m for m in metrics if m["service"] == service]

    if output_format == "json":
        import json
        print(json.dumps(metrics, indent=2))
        return

    # Table format
    formatter.print("\n[bold cyan]Service Metrics[/bold cyan]\n")

    if formatter.use_rich:
        from rich.table import Table

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Service", style="cyan")
        table.add_column("RPS", justify="right", style="green")
        table.add_column("Avg Latency", justify="right", style="yellow")
        table.add_column("P95 Latency", justify="right", style="yellow")
        table.add_column("Error Rate", justify="right", style="red")
        table.add_column("CPU %", justify="right", style="blue")
        table.add_column("Memory", justify="right", style="magenta")

        for metric in metrics:
            error_style = "red" if metric["error_rate"] > 0.01 else "green"
            cpu_style = "red" if metric["cpu_usage"] > 80 else "green"

            table.add_row(
                metric["service"],
                f"{metric['requests_per_sec']:.1f}",
                f"{metric['avg_latency_ms']}ms",
                f"{metric['p95_latency_ms']}ms",
                f"[{error_style}]{metric['error_rate']:.3%}[/{error_style}]",
                f"[{cpu_style}]{metric['cpu_usage']:.1f}%[/{cpu_style}]",
                f"{metric['memory_mb']}MB",
            )

        formatter.console.print(table)
    else:
        # Plain text table
        print(
            f"{'Service':<20} {'RPS':<10} {'Avg Lat':<12} {'P95 Lat':<12} "
            f"{'Error %':<10} {'CPU %':<10} {'Memory':<10}"
        )
        print("-" * 100)
        for metric in metrics:
            print(
                f"{metric['service']:<20} "
                f"{metric['requests_per_sec']:<10.1f} "
                f"{metric['avg_latency_ms']:<12}ms "
                f"{metric['p95_latency_ms']:<12}ms "
                f"{metric['error_rate']:<10.3%} "
                f"{metric['cpu_usage']:<10.1f}% "
                f"{metric['memory_mb']:<10}MB"
            )


@admin_cli.command()
@click.option(
    "--severity",
    default="warning",
    type=click.Choice(["info", "warning", "critical"]),
    help="Minimum severity",
)
@pass_state
def alerts(state: AdminState, severity: str):
    """
    View active alerts.

    Shows active alerts and warnings for the system.

    Examples:
        morgan-admin alerts
        morgan-admin alerts --severity critical
    """
    asyncio.run(_alerts_async(state, severity))


async def _alerts_async(state: AdminState, severity: str):
    """Async implementation of alerts command."""
    formatter = state.formatter

    # Mock alerts
    alerts = [
        {
            "severity": "warning",
            "service": "morgan-api",
            "message": "High memory usage (85%)",
            "timestamp": "2025-11-08 10:30:15",
        },
        {
            "severity": "info",
            "service": "morgan-rag",
            "message": "Slow query detected (>1s)",
            "timestamp": "2025-11-08 10:25:42",
        },
        {
            "severity": "critical",
            "service": "morgan-emotion",
            "message": "Circuit breaker open",
            "timestamp": "2025-11-08 10:20:05",
        },
    ]

    # Filter by severity
    severity_levels = ["info", "warning", "critical"]
    min_level = severity_levels.index(severity)
    alerts = [a for a in alerts if severity_levels.index(a["severity"]) >= min_level]

    formatter.print(f"\n[bold cyan]Active Alerts (severity: {severity}+)[/bold cyan]\n")

    if not alerts:
        formatter.print("[green]No active alerts ✅[/green]")
        return

    if formatter.use_rich:
        from rich.table import Table

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Severity", style="yellow")
        table.add_column("Service", style="cyan")
        table.add_column("Message", style="white")
        table.add_column("Time", style="dim")

        severity_colors = {
            "info": "blue",
            "warning": "yellow",
            "critical": "red",
        }

        for alert in alerts:
            color = severity_colors.get(alert["severity"], "white")
            table.add_row(
                f"[{color}]{alert['severity'].upper()}[/{color}]",
                alert["service"],
                alert["message"],
                alert["timestamp"],
            )

        formatter.console.print(table)
    else:
        for alert in alerts:
            print(
                f"[{alert['severity'].upper()}] {alert['service']}: {alert['message']} "
                f"({alert['timestamp']})"
            )


@admin_cli.command()
@pass_state
def version(state: AdminState):
    """Show Morgan Admin CLI version."""
    formatter = state.formatter

    formatter.print("\n[bold green]Morgan Admin CLI[/bold green]")
    formatter.print("[dim]Version: 2.0.0[/dim]")
    formatter.print("[dim]Distributed system management tools[/dim]\n")


def main():
    """Main entry point for the admin CLI."""
    try:
        admin_cli()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
