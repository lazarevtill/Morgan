"""
Demonstration of Morgan RAG monitoring and observability system.

This example shows how to set up and use the comprehensive monitoring system
including metrics collection, performance monitoring, companion metrics,
health monitoring, alerting, and dashboard functionality.
"""

import asyncio
import time
import random
import sys
import os
from datetime import datetime, timedelta

# Add the parent directory to the path so we can import morgan
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from morgan.monitoring import (
    MetricsCollector,
    PerformanceMonitor,
    CompanionMetrics,
    HealthMonitor,
    AlertManager,
    MonitoringDashboard,
)
from morgan.monitoring.companion_metrics import EmotionType, MilestoneType
from morgan.monitoring.alerting import AlertingConfig, AlertChannel, AlertSeverity


async def main():
    """Main demonstration function."""
    print("🚀 Morgan RAG Monitoring System Demo")
    print("=" * 50)

    # Initialize monitoring components
    print("\n📊 Initializing monitoring components...")

    # Core metrics collector
    metrics_collector = MetricsCollector()

    # Performance monitor
    performance_monitor = PerformanceMonitor(metrics_collector)

    # Companion metrics
    companion_metrics = CompanionMetrics(metrics_collector)

    # Health monitor
    health_monitor = HealthMonitor(metrics_collector)

    # Alert manager with configuration
    alerting_config = AlertingConfig(
        email_to=["admin@example.com"], webhook_urls=["http://localhost:8080/webhook"]
    )
    alert_manager = AlertManager(alerting_config)

    # Monitoring dashboard
    dashboard = MonitoringDashboard(
        metrics_collector=metrics_collector,
        performance_monitor=performance_monitor,
        companion_metrics=companion_metrics,
        health_monitor=health_monitor,
        alert_manager=alert_manager,
    )

    print("✅ All monitoring components initialized")

    # Start monitoring services
    print("\n🔄 Starting monitoring services...")

    performance_monitor.start_system_monitoring(interval_seconds=10)
    await health_monitor.start_monitoring()
    await alert_manager.start_processing()
    await dashboard.start_dashboard()

    print("✅ All monitoring services started")

    # Demonstrate metrics collection
    print("\n📈 Demonstrating metrics collection...")
    await demonstrate_metrics_collection(metrics_collector, performance_monitor)

    # Demonstrate companion metrics
    print("\n💝 Demonstrating companion metrics...")
    await demonstrate_companion_metrics(companion_metrics)

    # Demonstrate health monitoring
    print("\n🏥 Demonstrating health monitoring...")
    await demonstrate_health_monitoring(health_monitor)

    # Demonstrate alerting
    print("\n🚨 Demonstrating alerting system...")
    await demonstrate_alerting(alert_manager, performance_monitor)

    # Demonstrate dashboard
    print("\n📊 Demonstrating monitoring dashboard...")
    await demonstrate_dashboard(dashboard)

    # Generate comprehensive report
    print("\n📋 Generating comprehensive health report...")
    health_report = await dashboard.generate_health_report()
    print(f"Health report generated with {len(health_report)} sections")

    # Export metrics
    print("\n📤 Exporting metrics...")
    prometheus_metrics = dashboard.export_prometheus_metrics()
    json_dashboard = dashboard.export_dashboard_json()

    print(f"Prometheus metrics: {len(prometheus_metrics)} characters")
    print(f"JSON dashboard: {len(json_dashboard)} characters")

    # Clean up
    print("\n🧹 Cleaning up...")

    performance_monitor.stop_system_monitoring()
    await health_monitor.stop_monitoring()
    await alert_manager.stop_processing()
    await dashboard.stop_dashboard()

    print("✅ Monitoring system demo completed successfully!")


async def demonstrate_metrics_collection(
    metrics_collector: MetricsCollector, performance_monitor: PerformanceMonitor
):
    """Demonstrate metrics collection capabilities."""

    # Simulate document processing
    print("  📄 Simulating document processing...")

    for i in range(5):
        # Simulate processing with performance tracking
        with performance_monitor.track_operation(
            "document_ingestion", "processing", {"document_type": "pdf"}
        ):
            # Simulate processing time
            processing_time = random.uniform(0.5, 3.0)
            await asyncio.sleep(processing_time * 0.1)  # Speed up for demo

            # Record processing metrics
            metrics_collector.record_processing_time(
                operation_type="pdf_processing",
                duration=processing_time,
                document_type="pdf",
                success=True,
            )

    # Simulate search operations
    print("  🔍 Simulating search operations...")

    for i in range(10):
        search_time = random.uniform(0.1, 0.8)
        relevance_score = random.uniform(0.6, 0.95)

        metrics_collector.record_search_time(
            search_type="semantic_search",
            duration=search_time,
            collection="documents",
            success=True,
        )

        metrics_collector.record_search_relevance(
            search_type="semantic_search", relevance_score=relevance_score
        )

    # Simulate embedding generation
    print("  🧠 Simulating embedding generation...")

    for scale in ["coarse", "medium", "fine"]:
        embedding_time = random.uniform(0.2, 1.5)
        metrics_collector.record_embedding_time(
            model_type="sentence_transformer", scale=scale, duration=embedding_time
        )

    # Simulate cache performance
    cache_hit_rate = random.uniform(75, 95)
    metrics_collector.record_cache_hit_rate("semantic_cache", cache_hit_rate)

    # Get metrics summary
    processing_summary = metrics_collector.get_metrics_summary("processing_time")
    search_summary = metrics_collector.get_metrics_summary("search_time")

    print(
        f"  ✅ Processed {processing_summary.total_count if processing_summary else 0} documents"
    )
    print(
        f"  ✅ Executed {search_summary.total_count if search_summary else 0} searches"
    )
    print(f"  ✅ Cache hit rate: {cache_hit_rate:.1f}%")


async def demonstrate_companion_metrics(companion_metrics: CompanionMetrics):
    """Demonstrate companion experience metrics."""

    # Simulate emotional interactions
    print("  💭 Simulating emotional interactions...")

    users = ["user_1", "user_2", "user_3"]
    emotions = list(EmotionType)

    for i in range(15):
        user_id = random.choice(users)
        emotion = random.choice(emotions)
        intensity = random.uniform(0.3, 0.9)
        confidence = random.uniform(0.7, 0.95)
        empathy_score = random.uniform(0.6, 0.9)
        satisfaction = random.uniform(0.5, 0.95)

        companion_metrics.record_emotional_interaction(
            user_id=user_id,
            detected_emotion=emotion,
            emotion_intensity=intensity,
            confidence=confidence,
            response_empathy_score=empathy_score,
            user_satisfaction=satisfaction,
            interaction_duration=random.uniform(30, 300),
            context={"analysis_duration": random.uniform(0.1, 0.5)},
        )

    # Simulate relationship milestones
    print("  🎯 Simulating relationship milestones...")

    milestones = list(MilestoneType)
    for user_id in users:
        milestone = random.choice(milestones)
        companion_metrics.record_relationship_milestone(
            user_id=user_id,
            milestone_type=milestone,
            significance_score=random.uniform(0.7, 1.0),
        )

    # Simulate emotion detection feedback
    print("  🎯 Simulating emotion detection feedback...")

    for i in range(10):
        user_id = random.choice(users)
        detected = random.choice(emotions)
        actual = (
            detected if random.random() > 0.2 else random.choice(emotions)
        )  # 80% accuracy

        companion_metrics.record_emotion_detection_feedback(
            user_id=user_id,
            detected_emotion=detected,
            actual_emotion=actual,
            confidence=random.uniform(0.6, 0.9),
        )

    # Get companion quality metrics
    quality_metrics = companion_metrics.get_companion_quality_metrics()
    insights = companion_metrics.generate_companion_insights()

    print(f"  ✅ Total users: {quality_metrics.total_users}")
    print(
        f"  ✅ Average satisfaction: {quality_metrics.average_satisfaction_score:.2f}"
    )
    print(f"  ✅ Empathy accuracy: {quality_metrics.empathy_accuracy_rate:.2f}")
    print(f"  ✅ Overall health: {insights['overall_health']['status']}")


async def demonstrate_health_monitoring(health_monitor: HealthMonitor):
    """Demonstrate health monitoring capabilities."""

    print("  🔍 Running health checks...")

    # Run all health checks
    results = await health_monitor.run_all_health_checks_once()

    print(f"  ✅ Completed {len(results)} health checks")

    # Get system health status
    health_status = health_monitor.get_system_health_status()

    print(f"  📊 Overall status: {health_status.overall_status.value}")
    print(f"  📊 Healthy checks: {health_status.healthy_checks}")
    print(f"  📊 Warning checks: {health_status.warning_checks}")
    print(f"  📊 Critical checks: {health_status.critical_checks}")
    print(f"  📊 Uptime: {health_status.uptime_seconds:.1f} seconds")

    # Show individual check results
    for result in health_status.check_results:
        status_emoji = (
            "✅"
            if result.status.value == "healthy"
            else "⚠️" if result.status.value == "warning" else "❌"
        )
        print(
            f"    {status_emoji} {result.name}: {result.message} ({result.duration:.3f}s)"
        )


async def demonstrate_alerting(
    alert_manager: AlertManager, performance_monitor: PerformanceMonitor
):
    """Demonstrate alerting system."""

    print("  🚨 Triggering test alerts...")

    # Simulate high memory usage alert
    from morgan.monitoring.performance_monitor import PerformanceAlert

    high_memory_alert = PerformanceAlert(
        alert_type="high_memory_usage",
        severity="critical",
        message="High memory usage detected: 92.5%",
        current_value=92.5,
        threshold_value=90.0,
        timestamp=datetime.now(),
        component="system",
        metadata={},
    )

    await alert_manager.check_performance_alert(high_memory_alert)

    # Simulate slow processing alert
    slow_processing_alert = PerformanceAlert(
        alert_type="slow_processing",
        severity="warning",
        message="Slow document processing: P95 = 6.2s",
        current_value=6.2,
        threshold_value=5.0,
        timestamp=datetime.now(),
        component="processing",
        metadata={},
    )

    await alert_manager.check_performance_alert(slow_processing_alert)

    # Wait for alert processing
    await asyncio.sleep(1.0)

    # Get alert statistics
    active_alerts = alert_manager.get_active_alerts()
    alert_stats = alert_manager.get_alert_statistics(hours=1)

    print(f"  ✅ Active alerts: {len(active_alerts)}")
    print(f"  ✅ Total alerts (1h): {alert_stats['total_alerts']}")
    print(f"  ✅ Critical alerts: {alert_stats['severity_breakdown']['critical']}")
    print(f"  ✅ Warning alerts: {alert_stats['severity_breakdown']['warning']}")


async def demonstrate_dashboard(dashboard: MonitoringDashboard):
    """Demonstrate dashboard capabilities."""

    print("  📊 Collecting dashboard data...")

    # Collect current dashboard data
    dashboard_data = await dashboard.collect_dashboard_data()

    print(f"  ✅ Dashboard data collected at {dashboard_data.timestamp}")

    # Get dashboard summary
    summary = dashboard.get_dashboard_summary()

    print(f"  📈 Overall status: {summary['overall_status']}")
    print(f"  📈 Uptime: {summary['uptime']}")
    print(
        f"  📈 Healthy checks: {summary['health_summary']['healthy_checks']}/{summary['health_summary']['total_checks']}"
    )
    print(
        f"  📈 Processing P95: {summary['performance_summary']['processing_p95']:.3f}s"
    )
    print(f"  📈 Search P95: {summary['performance_summary']['search_p95']:.3f}s")
    print(
        f"  📈 User satisfaction: {summary['companion_summary']['user_satisfaction']:.2f}"
    )
    print(f"  📈 Active alerts: {summary['alerts_summary']['active_alerts']}")

    # Show top recommendations
    if summary["top_recommendations"]:
        print("  💡 Top recommendations:")
        for i, rec in enumerate(summary["top_recommendations"], 1):
            print(f"    {i}. {rec}")

    # Show system health details
    print(f"  🏥 System health: {dashboard_data.system_health['overall_status']}")
    print(
        f"  🏥 CPU usage: {dashboard_data.system_health['system_resources']['current_cpu_percent']:.1f}%"
    )
    print(
        f"  🏥 Memory usage: {dashboard_data.system_health['system_resources']['current_memory_percent']:.1f}%"
    )


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(main())
