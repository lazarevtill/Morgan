"""
Tests for Morgan RAG monitoring and observability system.

Basic tests to verify the monitoring components work correctly.
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta

from morgan.monitoring import (
    MetricsCollector,
    PerformanceMonitor,
    CompanionMetrics,
    HealthMonitor,
    AlertManager,
    MonitoringDashboard
)
from morgan.monitoring.companion_metrics import EmotionType, MilestoneType
from morgan.monitoring.alerting import AlertingConfig


class TestMetricsCollector:
    """Test metrics collection functionality."""
    
    def test_metrics_collector_initialization(self):
        """Test metrics collector initializes correctly."""
        collector = MetricsCollector()
        assert collector is not None
        assert collector.registry is not None
    
    def test_record_processing_time(self):
        """Test recording processing time metrics."""
        collector = MetricsCollector()
        
        collector.record_processing_time(
            operation_type="test_processing",
            duration=1.5,
            document_type="pdf",
            success=True
        )
        
        # Verify metric was recorded
        summary = collector.get_metrics_summary("processing_time")
        assert summary is not None
        assert summary.total_count == 1
        assert summary.average == 1.5
    
    def test_record_search_time(self):
        """Test recording search time metrics."""
        collector = MetricsCollector()
        
        collector.record_search_time(
            search_type="semantic",
            duration=0.3,
            collection="test",
            success=True
        )
        
        summary = collector.get_metrics_summary("search_time")
        assert summary is not None
        assert summary.total_count == 1
        assert summary.average == 0.3
    
    def test_record_user_satisfaction(self):
        """Test recording user satisfaction metrics."""
        collector = MetricsCollector()
        
        collector.record_user_satisfaction(
            interaction_type="test_interaction",
            satisfaction_score=0.85
        )
        
        recent_metrics = collector.get_recent_metrics("user_satisfaction")
        assert len(recent_metrics) == 1
        assert recent_metrics[0].value == 0.85
    
    def test_prometheus_export(self):
        """Test Prometheus metrics export."""
        collector = MetricsCollector()
        
        # Record some metrics
        collector.record_processing_time("test", 1.0, "pdf", True)
        collector.record_search_time("semantic", 0.5, "test", True)
        
        # Export metrics
        prometheus_output = collector.export_prometheus_metrics()
        assert isinstance(prometheus_output, str)
        assert len(prometheus_output) > 0


class TestPerformanceMonitor:
    """Test performance monitoring functionality."""
    
    def test_performance_monitor_initialization(self):
        """Test performance monitor initializes correctly."""
        collector = MetricsCollector()
        monitor = PerformanceMonitor(collector)
        assert monitor is not None
        assert monitor.metrics_collector == collector
    
    @pytest.mark.asyncio
    async def test_track_operation_context_manager(self):
        """Test operation tracking context manager."""
        collector = MetricsCollector()
        monitor = PerformanceMonitor(collector)
        
        # Track an operation
        with monitor.track_operation("test_operation", "processing"):
            await asyncio.sleep(0.1)
        
        # Verify metrics were recorded
        summary = collector.get_metrics_summary("processing_time")
        assert summary is not None
        assert summary.total_count == 1
        assert summary.average >= 0.1
    
    def test_get_performance_summary(self):
        """Test getting performance summary."""
        collector = MetricsCollector()
        monitor = PerformanceMonitor(collector)
        
        # Record some metrics
        collector.record_processing_time("test", 1.0, "pdf", True)
        collector.record_search_time("semantic", 0.5, "test", True)
        
        summary = monitor.get_performance_summary()
        assert isinstance(summary, dict)
        assert "system_resources" in summary
        assert "application_performance" in summary


class TestCompanionMetrics:
    """Test companion metrics functionality."""
    
    def test_companion_metrics_initialization(self):
        """Test companion metrics initializes correctly."""
        collector = MetricsCollector()
        companion = CompanionMetrics(collector)
        assert companion is not None
        assert companion.metrics_collector == collector
    
    def test_record_emotional_interaction(self):
        """Test recording emotional interactions."""
        collector = MetricsCollector()
        companion = CompanionMetrics(collector)
        
        companion.record_emotional_interaction(
            user_id="test_user",
            detected_emotion=EmotionType.JOY,
            emotion_intensity=0.8,
            confidence=0.9,
            response_empathy_score=0.85,
            user_satisfaction=0.9
        )
        
        # Verify interaction was recorded
        relationship = companion.get_user_relationship_metrics("test_user")
        assert relationship is not None
        assert relationship.total_interactions == 1
        assert relationship.average_satisfaction == 0.9
    
    def test_record_relationship_milestone(self):
        """Test recording relationship milestones."""
        collector = MetricsCollector()
        companion = CompanionMetrics(collector)
        
        # First create a user relationship
        companion.record_emotional_interaction(
            user_id="test_user",
            detected_emotion=EmotionType.JOY,
            emotion_intensity=0.8,
            confidence=0.9,
            response_empathy_score=0.85
        )
        
        # Record milestone
        companion.record_relationship_milestone(
            user_id="test_user",
            milestone_type=MilestoneType.FIRST_CONVERSATION
        )
        
        relationship = companion.get_user_relationship_metrics("test_user")
        assert MilestoneType.FIRST_CONVERSATION in relationship.milestones_reached
    
    def test_get_companion_quality_metrics(self):
        """Test getting companion quality metrics."""
        collector = MetricsCollector()
        companion = CompanionMetrics(collector)
        
        # Record some interactions
        companion.record_emotional_interaction(
            user_id="user1",
            detected_emotion=EmotionType.JOY,
            emotion_intensity=0.8,
            confidence=0.9,
            response_empathy_score=0.85,
            user_satisfaction=0.9
        )
        
        quality_metrics = companion.get_companion_quality_metrics()
        assert quality_metrics.total_users == 1
        assert quality_metrics.average_satisfaction_score > 0


class TestHealthMonitor:
    """Test health monitoring functionality."""
    
    def test_health_monitor_initialization(self):
        """Test health monitor initializes correctly."""
        collector = MetricsCollector()
        monitor = HealthMonitor(collector)
        assert monitor is not None
        assert len(monitor._health_checks) > 0  # Should have default checks
    
    @pytest.mark.asyncio
    async def test_run_health_check_once(self):
        """Test running a single health check."""
        collector = MetricsCollector()
        monitor = HealthMonitor(collector)
        
        # Run memory usage check
        result = await monitor.run_health_check_once("memory_usage")
        assert result is not None
        assert result.name == "memory_usage"
        assert result.status is not None
    
    def test_get_system_health_status(self):
        """Test getting system health status."""
        collector = MetricsCollector()
        monitor = HealthMonitor(collector)
        
        status = monitor.get_system_health_status()
        assert status is not None
        assert status.overall_status is not None
        assert status.uptime_seconds >= 0


class TestAlertManager:
    """Test alerting functionality."""
    
    def test_alert_manager_initialization(self):
        """Test alert manager initializes correctly."""
        config = AlertingConfig()
        manager = AlertManager(config)
        assert manager is not None
        assert len(manager._alert_rules) > 0  # Should have default rules
    
    @pytest.mark.asyncio
    async def test_check_custom_metrics_alert(self):
        """Test checking custom metrics against alert rules."""
        config = AlertingConfig()
        manager = AlertManager(config)
        
        # Start processing
        await manager.start_processing()
        
        # Trigger high memory alert
        await manager.check_custom_metrics_alert({
            "memory_percent": 95.0
        })
        
        # Wait for processing
        await asyncio.sleep(0.1)
        
        # Check for alerts
        active_alerts = manager.get_active_alerts()
        
        # Stop processing
        await manager.stop_processing()
        
        # Should have triggered high memory alert
        assert len(active_alerts) > 0
        memory_alerts = [a for a in active_alerts if a.rule_name == "high_memory_usage"]
        assert len(memory_alerts) > 0
    
    def test_get_alert_statistics(self):
        """Test getting alert statistics."""
        config = AlertingConfig()
        manager = AlertManager(config)
        
        stats = manager.get_alert_statistics()
        assert isinstance(stats, dict)
        assert "total_alerts" in stats
        assert "severity_breakdown" in stats


class TestMonitoringDashboard:
    """Test monitoring dashboard functionality."""
    
    def test_dashboard_initialization(self):
        """Test dashboard initializes correctly."""
        collector = MetricsCollector()
        performance_monitor = PerformanceMonitor(collector)
        companion_metrics = CompanionMetrics(collector)
        health_monitor = HealthMonitor(collector)
        alert_manager = AlertManager()
        
        dashboard = MonitoringDashboard(
            metrics_collector=collector,
            performance_monitor=performance_monitor,
            companion_metrics=companion_metrics,
            health_monitor=health_monitor,
            alert_manager=alert_manager
        )
        
        assert dashboard is not None
    
    @pytest.mark.asyncio
    async def test_collect_dashboard_data(self):
        """Test collecting dashboard data."""
        collector = MetricsCollector()
        performance_monitor = PerformanceMonitor(collector)
        companion_metrics = CompanionMetrics(collector)
        health_monitor = HealthMonitor(collector)
        alert_manager = AlertManager()
        
        dashboard = MonitoringDashboard(
            metrics_collector=collector,
            performance_monitor=performance_monitor,
            companion_metrics=companion_metrics,
            health_monitor=health_monitor,
            alert_manager=alert_manager
        )
        
        # Collect dashboard data
        data = await dashboard.collect_dashboard_data()
        
        assert data is not None
        assert data.timestamp is not None
        assert isinstance(data.system_health, dict)
        assert isinstance(data.performance_metrics, dict)
        assert isinstance(data.companion_metrics, dict)
        assert isinstance(data.alerts, dict)
        assert isinstance(data.recommendations, list)
    
    def test_export_dashboard_json(self):
        """Test exporting dashboard as JSON."""
        collector = MetricsCollector()
        performance_monitor = PerformanceMonitor(collector)
        companion_metrics = CompanionMetrics(collector)
        health_monitor = HealthMonitor(collector)
        alert_manager = AlertManager()
        
        dashboard = MonitoringDashboard(
            metrics_collector=collector,
            performance_monitor=performance_monitor,
            companion_metrics=companion_metrics,
            health_monitor=health_monitor,
            alert_manager=alert_manager
        )
        
        json_output = dashboard.export_dashboard_json()
        assert isinstance(json_output, str)
        assert len(json_output) > 0


@pytest.mark.asyncio
async def test_integration_monitoring_workflow():
    """Test complete monitoring workflow integration."""
    # Initialize all components
    collector = MetricsCollector()
    performance_monitor = PerformanceMonitor(collector)
    companion_metrics = CompanionMetrics(collector)
    health_monitor = HealthMonitor(collector)
    alert_manager = AlertManager()
    
    dashboard = MonitoringDashboard(
        metrics_collector=collector,
        performance_monitor=performance_monitor,
        companion_metrics=companion_metrics,
        health_monitor=health_monitor,
        alert_manager=alert_manager
    )
    
    # Start services
    await health_monitor.start_monitoring()
    await alert_manager.start_processing()
    await dashboard.start_dashboard()
    
    # Simulate some activity
    collector.record_processing_time("test", 1.0, "pdf", True)
    collector.record_search_time("semantic", 0.5, "test", True)
    
    companion_metrics.record_emotional_interaction(
        user_id="test_user",
        detected_emotion=EmotionType.JOY,
        emotion_intensity=0.8,
        confidence=0.9,
        response_empathy_score=0.85,
        user_satisfaction=0.9
    )
    
    # Wait for processing
    await asyncio.sleep(0.5)
    
    # Collect dashboard data
    dashboard_data = await dashboard.collect_dashboard_data()
    
    # Verify data was collected
    assert dashboard_data is not None
    assert dashboard_data.performance_metrics["processing"]["total_operations"] > 0
    assert dashboard_data.companion_metrics["quality_metrics"]["total_users"] > 0
    
    # Stop services
    await health_monitor.stop_monitoring()
    await alert_manager.stop_processing()
    await dashboard.stop_dashboard()


if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])