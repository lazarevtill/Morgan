"""
Monitoring dashboard for Morgan RAG.

Provides a comprehensive dashboard for technical and companion metrics,
system health visualization, and real-time monitoring as specified in the requirements.
"""

import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import structlog

from .metrics_collector import MetricsCollector, MetricsSummary
from .performance_monitor import PerformanceMonitor, SystemResourceUsage
from .companion_metrics import CompanionMetrics, CompanionQualityMetrics
from .health_monitor import HealthMonitor, SystemHealthStatus
from .alerting import AlertManager

logger = structlog.get_logger(__name__)


@dataclass
class DashboardData:
    """Complete dashboard data structure."""
    timestamp: datetime
    system_health: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    companion_metrics: Dict[str, Any]
    alerts: Dict[str, Any]
    recommendations: List[str]


@dataclass
class DashboardConfig:
    """Dashboard configuration."""
    refresh_interval_seconds: int = 30
    data_retention_hours: int = 24
    enable_real_time_updates: bool = True
    export_prometheus: bool = True
    export_json: bool = True


class MonitoringDashboard:
    """
    Comprehensive monitoring dashboard for Morgan RAG.
    
    Provides real-time visualization of system health, performance metrics,
    companion experience quality, and alerting status.
    """
    
    def __init__(self, 
                 metrics_collector: MetricsCollector,
                 performance_monitor: PerformanceMonitor,
                 companion_metrics: CompanionMetrics,
                 health_monitor: HealthMonitor,
                 alert_manager: AlertManager,
                 config: Optional[DashboardConfig] = None):
        
        self.metrics_collector = metrics_collector
        self.performance_monitor = performance_monitor
        self.companion_metrics = companion_metrics
        self.health_monitor = health_monitor
        self.alert_manager = alert_manager
        self.config = config or DashboardConfig()
        
        # Dashboard state
        self._dashboard_data_history: List[DashboardData] = []
        self._update_task: Optional[asyncio.Task] = None
        self._update_active = False
        
        logger.info("MonitoringDashboard initialized")
    
    async def start_dashboard(self):
        """Start the dashboard with real-time updates."""
        if self._update_active:
            logger.warning("Dashboard already active")
            return
        
        self._update_active = True
        
        if self.config.enable_real_time_updates:
            self._update_task = asyncio.create_task(self._update_loop())
        
        logger.info("Started monitoring dashboard")
    
    async def stop_dashboard(self):
        """Stop the dashboard."""
        self._update_active = False
        
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped monitoring dashboard")
    
    async def _update_loop(self):
        """Main dashboard update loop."""
        while self._update_active:
            try:
                # Collect current dashboard data
                dashboard_data = await self.collect_dashboard_data()
                
                # Store in history
                self._dashboard_data_history.append(dashboard_data)
                
                # Clean up old data
                self._cleanup_old_data()
                
                # Sleep until next update
                await asyncio.sleep(self.config.refresh_interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in dashboard update loop", error=str(e))
                await asyncio.sleep(self.config.refresh_interval_seconds)
    
    async def collect_dashboard_data(self) -> DashboardData:
        """Collect comprehensive dashboard data."""
        timestamp = datetime.now()
        
        # Collect system health
        system_health = await self._collect_system_health_data()
        
        # Collect performance metrics
        performance_metrics = self._collect_performance_data()
        
        # Collect companion metrics
        companion_metrics = self._collect_companion_data()
        
        # Collect alerts
        alerts = self._collect_alerts_data()
        
        # Generate recommendations
        recommendations = self._generate_recommendations()
        
        return DashboardData(
            timestamp=timestamp,
            system_health=system_health,
            performance_metrics=performance_metrics,
            companion_metrics=companion_metrics,
            alerts=alerts,
            recommendations=recommendations
        )
    
    async def _collect_system_health_data(self) -> Dict[str, Any]:
        """Collect system health data."""
        health_status = self.health_monitor.get_system_health_status()
        
        # Get recent system resource usage
        resource_history = self.performance_monitor.get_system_stats_history(hours=1)
        
        # Calculate resource averages
        if resource_history:
            avg_cpu = sum(s.cpu_percent for s in resource_history) / len(resource_history)
            avg_memory = sum(s.memory_percent for s in resource_history) / len(resource_history)
            current_stats = resource_history[-1]
        else:
            avg_cpu = avg_memory = 0
            current_stats = None
        
        return {
            "overall_status": health_status.overall_status.value,
            "uptime_seconds": health_status.uptime_seconds,
            "uptime_formatted": self._format_uptime(health_status.uptime_seconds),
            "health_checks": {
                "healthy": health_status.healthy_checks,
                "warning": health_status.warning_checks,
                "critical": health_status.critical_checks,
                "unknown": health_status.unknown_checks,
                "total": len(health_status.check_results)
            },
            "check_details": [
                {
                    "name": result.name,
                    "status": result.status.value,
                    "message": result.message,
                    "duration": result.duration,
                    "timestamp": result.timestamp.isoformat()
                }
                for result in health_status.check_results
            ],
            "system_resources": {
                "current_cpu_percent": current_stats.cpu_percent if current_stats else 0,
                "current_memory_percent": current_stats.memory_percent if current_stats else 0,
                "average_cpu_percent_1h": avg_cpu,
                "average_memory_percent_1h": avg_memory,
                "memory_used_bytes": current_stats.memory_used_bytes if current_stats else 0,
                "memory_available_bytes": current_stats.memory_available_bytes if current_stats else 0
            },
            "last_updated": health_status.last_updated.isoformat()
        }
    
    def _collect_performance_data(self) -> Dict[str, Any]:
        """Collect performance metrics data."""
        performance_summary = self.performance_monitor.get_performance_summary()
        
        # Get recent metrics summaries
        processing_summary = self.metrics_collector.get_metrics_summary("processing_time")
        search_summary = self.metrics_collector.get_metrics_summary("search_time")
        embedding_summary = self.metrics_collector.get_metrics_summary("embedding_time")
        
        # Get active operations
        active_operations = self.performance_monitor.get_active_operations()
        
        return {
            "processing": {
                "average_duration": processing_summary.average if processing_summary else 0,
                "p95_duration": processing_summary.percentile_95 if processing_summary else 0,
                "p99_duration": processing_summary.percentile_99 if processing_summary else 0,
                "error_rate": processing_summary.error_rate if processing_summary else 0,
                "total_operations": processing_summary.total_count if processing_summary else 0,
                "throughput_per_minute": self._calculate_throughput(processing_summary)
            },
            "search": {
                "average_duration": search_summary.average if search_summary else 0,
                "p95_duration": search_summary.percentile_95 if search_summary else 0,
                "p99_duration": search_summary.percentile_99 if search_summary else 0,
                "error_rate": search_summary.error_rate if search_summary else 0,
                "total_operations": search_summary.total_count if search_summary else 0,
                "throughput_per_minute": self._calculate_throughput(search_summary)
            },
            "embedding": {
                "average_duration": embedding_summary.average if embedding_summary else 0,
                "p95_duration": embedding_summary.percentile_95 if embedding_summary else 0,
                "p99_duration": embedding_summary.percentile_99 if embedding_summary else 0,
                "total_operations": embedding_summary.total_count if embedding_summary else 0
            },
            "active_operations": {
                "count": len(active_operations),
                "longest_running": max(active_operations.values()) if active_operations else 0,
                "operations": active_operations
            },
            "cache_performance": self._get_cache_performance(),
            "thresholds": performance_summary.get("thresholds", {}),
            "recommendations": self.performance_monitor.generate_optimization_recommendations()
        }
    
    def _collect_companion_data(self) -> Dict[str, Any]:
        """Collect companion experience metrics."""
        companion_quality = self.companion_metrics.get_companion_quality_metrics()
        companion_insights = self.companion_metrics.generate_companion_insights()
        
        # Get emotion detection accuracy
        emotion_accuracy = self.companion_metrics.get_emotion_detection_accuracy()
        
        # Get milestone statistics
        milestone_stats = self.companion_metrics.get_milestone_statistics()
        
        return {
            "quality_metrics": {
                "total_users": companion_quality.total_users,
                "active_users_24h": companion_quality.active_users_24h,
                "active_users_7d": companion_quality.active_users_7d,
                "average_satisfaction": companion_quality.average_satisfaction_score,
                "empathy_accuracy": companion_quality.empathy_accuracy_rate,
                "emotional_support_success": companion_quality.emotional_support_success_rate,
                "relationship_retention": companion_quality.relationship_retention_rate,
                "milestone_completion": companion_quality.milestone_completion_rate,
                "personalization_score": companion_quality.response_personalization_score
            },
            "emotional_intelligence": {
                "overall_accuracy": companion_insights["emotional_intelligence"]["empathy_effectiveness"],
                "emotion_detection_accuracy": emotion_accuracy,
                "improvement_areas": companion_insights["emotional_intelligence"]["improvement_areas"]
            },
            "relationship_building": {
                "milestone_stats": milestone_stats,
                "trust_building_success": companion_insights["relationship_building"]["trust_building_success"],
                "personalization_effectiveness": companion_insights["relationship_building"]["personalization_effectiveness"]
            },
            "user_engagement": companion_insights["user_engagement"],
            "health_status": companion_insights["overall_health"]["status"],
            "recommendations": companion_insights["recommendations"]
        }
    
    def _collect_alerts_data(self) -> Dict[str, Any]:
        """Collect alerts and alerting statistics."""
        active_alerts = self.alert_manager.get_active_alerts()
        alert_history = self.alert_manager.get_alert_history(hours=24)
        alert_stats = self.alert_manager.get_alert_statistics(hours=24)
        
        return {
            "active_alerts": [
                {
                    "alert_id": alert.alert_id,
                    "rule_name": alert.rule_name,
                    "severity": alert.severity.value,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat(),
                    "channels": [c.value for c in alert.channels]
                }
                for alert in active_alerts
            ],
            "recent_alerts_24h": len(alert_history),
            "statistics": alert_stats,
            "alert_trends": self._calculate_alert_trends(alert_history)
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate comprehensive system recommendations."""
        recommendations = []
        
        # Get recommendations from various components
        performance_recommendations = self.performance_monitor.generate_optimization_recommendations()
        companion_insights = self.companion_metrics.generate_companion_insights()
        companion_recommendations = companion_insights.get("recommendations", [])
        
        # Combine and prioritize recommendations
        recommendations.extend(performance_recommendations)
        recommendations.extend(companion_recommendations)
        
        # Add system-level recommendations based on dashboard data
        system_health = self.health_monitor.get_system_health_status()
        
        if system_health.critical_checks > 0:
            recommendations.insert(0, 
                f"CRITICAL: {system_health.critical_checks} critical health checks failing. "
                "Immediate attention required."
            )
        
        if system_health.warning_checks > 2:
            recommendations.append(
                f"Multiple warning conditions detected ({system_health.warning_checks}). "
                "Review system health and consider preventive maintenance."
            )
        
        # Limit to top 10 recommendations
        return recommendations[:10]
    
    def _get_cache_performance(self) -> Dict[str, Any]:
        """Get cache performance metrics."""
        # Get recent cache metrics
        cache_metrics = self.metrics_collector.get_recent_metrics("cache_hit_rate", limit=10)
        
        if cache_metrics:
            recent_hit_rates = [m.value for m in cache_metrics]
            avg_hit_rate = sum(recent_hit_rates) / len(recent_hit_rates)
            min_hit_rate = min(recent_hit_rates)
            max_hit_rate = max(recent_hit_rates)
        else:
            avg_hit_rate = min_hit_rate = max_hit_rate = 0
        
        return {
            "average_hit_rate": avg_hit_rate,
            "min_hit_rate": min_hit_rate,
            "max_hit_rate": max_hit_rate,
            "cache_health": "healthy" if avg_hit_rate > 80 else "degraded"
        }
    
    def _calculate_throughput(self, summary: Optional[MetricsSummary]) -> float:
        """Calculate operations per minute from metrics summary."""
        if not summary or summary.total_count == 0:
            return 0.0
        
        # Assume metrics are from last hour, calculate per minute
        return summary.total_count / 60.0
    
    def _calculate_alert_trends(self, alert_history: List) -> Dict[str, Any]:
        """Calculate alert trends over time."""
        if not alert_history:
            return {"trend": "stable", "change_percent": 0}
        
        # Split into two halves to compare
        mid_point = len(alert_history) // 2
        first_half = alert_history[:mid_point]
        second_half = alert_history[mid_point:]
        
        if len(first_half) == 0:
            return {"trend": "stable", "change_percent": 0}
        
        change_percent = ((len(second_half) - len(first_half)) / len(first_half)) * 100
        
        if change_percent > 20:
            trend = "increasing"
        elif change_percent < -20:
            trend = "decreasing"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "change_percent": change_percent
        }
    
    def _format_uptime(self, uptime_seconds: float) -> str:
        """Format uptime in human-readable format."""
        days = int(uptime_seconds // 86400)
        hours = int((uptime_seconds % 86400) // 3600)
        minutes = int((uptime_seconds % 3600) // 60)
        
        if days > 0:
            return f"{days}d {hours}h {minutes}m"
        elif hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m"
    
    def _cleanup_old_data(self):
        """Clean up old dashboard data."""
        cutoff_time = datetime.now() - timedelta(hours=self.config.data_retention_hours)
        
        self._dashboard_data_history = [
            data for data in self._dashboard_data_history
            if data.timestamp >= cutoff_time
        ]
    
    def get_current_dashboard_data(self) -> Optional[DashboardData]:
        """Get the most recent dashboard data."""
        if self._dashboard_data_history:
            return self._dashboard_data_history[-1]
        return None
    
    def get_dashboard_history(self, hours: int = 24) -> List[DashboardData]:
        """Get dashboard data history for specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            data for data in self._dashboard_data_history
            if data.timestamp >= cutoff_time
        ]
    
    def export_dashboard_json(self) -> str:
        """Export current dashboard data as JSON."""
        current_data = self.get_current_dashboard_data()
        
        if current_data:
            return json.dumps(asdict(current_data), indent=2, default=str)
        else:
            return json.dumps({"error": "No dashboard data available"}, indent=2)
    
    def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        return self.metrics_collector.export_prometheus_metrics()
    
    def get_dashboard_summary(self) -> Dict[str, Any]:
        """Get a concise dashboard summary."""
        current_data = self.get_current_dashboard_data()
        
        if not current_data:
            return {"status": "no_data", "message": "Dashboard data not available"}
        
        system_health = current_data.system_health
        performance = current_data.performance_metrics
        companion = current_data.companion_metrics
        alerts = current_data.alerts
        
        return {
            "timestamp": current_data.timestamp.isoformat(),
            "overall_status": system_health["overall_status"],
            "uptime": system_health["uptime_formatted"],
            "health_summary": {
                "healthy_checks": system_health["health_checks"]["healthy"],
                "total_checks": system_health["health_checks"]["total"],
                "critical_issues": system_health["health_checks"]["critical"]
            },
            "performance_summary": {
                "processing_p95": performance["processing"]["p95_duration"],
                "search_p95": performance["search"]["p95_duration"],
                "error_rate": performance["processing"]["error_rate"],
                "active_operations": performance["active_operations"]["count"]
            },
            "companion_summary": {
                "user_satisfaction": companion["quality_metrics"]["average_satisfaction"],
                "empathy_accuracy": companion["quality_metrics"]["empathy_accuracy"],
                "active_users_24h": companion["quality_metrics"]["active_users_24h"],
                "health_status": companion["health_status"]
            },
            "alerts_summary": {
                "active_alerts": len(alerts["active_alerts"]),
                "recent_alerts_24h": alerts["recent_alerts_24h"],
                "alert_trend": alerts["alert_trends"]["trend"]
            },
            "top_recommendations": current_data.recommendations[:3]
        }
    
    async def generate_health_report(self) -> Dict[str, Any]:
        """Generate a comprehensive health report."""
        # Collect fresh data
        dashboard_data = await self.collect_dashboard_data()
        
        # Get historical data for trends
        history = self.get_dashboard_history(hours=24)
        
        # Calculate trends
        performance_trend = self._calculate_performance_trend(history)
        companion_trend = self._calculate_companion_trend(history)
        
        return {
            "report_timestamp": datetime.now().isoformat(),
            "executive_summary": {
                "overall_health": dashboard_data.system_health["overall_status"],
                "key_metrics": {
                    "uptime_hours": dashboard_data.system_health["uptime_seconds"] / 3600,
                    "user_satisfaction": dashboard_data.companion_metrics["quality_metrics"]["average_satisfaction"],
                    "system_performance": "good" if dashboard_data.performance_metrics["processing"]["p95_duration"] < 5.0 else "degraded",
                    "active_alerts": len(dashboard_data.alerts["active_alerts"])
                }
            },
            "detailed_analysis": {
                "system_health": dashboard_data.system_health,
                "performance_metrics": dashboard_data.performance_metrics,
                "companion_metrics": dashboard_data.companion_metrics,
                "performance_trends": performance_trend,
                "companion_trends": companion_trend
            },
            "recommendations": {
                "immediate_actions": [r for r in dashboard_data.recommendations if "CRITICAL" in r],
                "optimization_opportunities": [r for r in dashboard_data.recommendations if "CRITICAL" not in r],
                "monitoring_focus_areas": self._identify_monitoring_focus_areas(dashboard_data)
            },
            "alerts_analysis": dashboard_data.alerts
        }
    
    def _calculate_performance_trend(self, history: List[DashboardData]) -> Dict[str, str]:
        """Calculate performance trends from historical data."""
        if len(history) < 2:
            return {"processing": "insufficient_data", "search": "insufficient_data"}
        
        # Compare first and last quarters
        quarter_size = len(history) // 4
        if quarter_size == 0:
            return {"processing": "insufficient_data", "search": "insufficient_data"}
        
        early_data = history[:quarter_size]
        recent_data = history[-quarter_size:]
        
        # Calculate average P95 times
        early_processing_p95 = sum(d.performance_metrics["processing"]["p95_duration"] for d in early_data) / len(early_data)
        recent_processing_p95 = sum(d.performance_metrics["processing"]["p95_duration"] for d in recent_data) / len(recent_data)
        
        early_search_p95 = sum(d.performance_metrics["search"]["p95_duration"] for d in early_data) / len(early_data)
        recent_search_p95 = sum(d.performance_metrics["search"]["p95_duration"] for d in recent_data) / len(recent_data)
        
        # Determine trends
        processing_trend = "improving" if recent_processing_p95 < early_processing_p95 * 0.9 else \
                          "degrading" if recent_processing_p95 > early_processing_p95 * 1.1 else "stable"
        
        search_trend = "improving" if recent_search_p95 < early_search_p95 * 0.9 else \
                      "degrading" if recent_search_p95 > early_search_p95 * 1.1 else "stable"
        
        return {"processing": processing_trend, "search": search_trend}
    
    def _calculate_companion_trend(self, history: List[DashboardData]) -> Dict[str, str]:
        """Calculate companion experience trends from historical data."""
        if len(history) < 2:
            return {"satisfaction": "insufficient_data", "empathy": "insufficient_data"}
        
        quarter_size = len(history) // 4
        if quarter_size == 0:
            return {"satisfaction": "insufficient_data", "empathy": "insufficient_data"}
        
        early_data = history[:quarter_size]
        recent_data = history[-quarter_size:]
        
        # Calculate averages
        early_satisfaction = sum(d.companion_metrics["quality_metrics"]["average_satisfaction"] for d in early_data) / len(early_data)
        recent_satisfaction = sum(d.companion_metrics["quality_metrics"]["average_satisfaction"] for d in recent_data) / len(recent_data)
        
        early_empathy = sum(d.companion_metrics["quality_metrics"]["empathy_accuracy"] for d in early_data) / len(early_data)
        recent_empathy = sum(d.companion_metrics["quality_metrics"]["empathy_accuracy"] for d in recent_data) / len(recent_data)
        
        # Determine trends
        satisfaction_trend = "improving" if recent_satisfaction > early_satisfaction * 1.05 else \
                           "degrading" if recent_satisfaction < early_satisfaction * 0.95 else "stable"
        
        empathy_trend = "improving" if recent_empathy > early_empathy * 1.05 else \
                       "degrading" if recent_empathy < early_empathy * 0.95 else "stable"
        
        return {"satisfaction": satisfaction_trend, "empathy": empathy_trend}
    
    def _identify_monitoring_focus_areas(self, dashboard_data: DashboardData) -> List[str]:
        """Identify areas that need monitoring attention."""
        focus_areas = []
        
        # Check system health
        if dashboard_data.system_health["health_checks"]["critical"] > 0:
            focus_areas.append("Critical system health checks")
        
        # Check performance
        if dashboard_data.performance_metrics["processing"]["error_rate"] > 0.02:
            focus_areas.append("Processing error rates")
        
        if dashboard_data.performance_metrics["search"]["p95_duration"] > 0.5:
            focus_areas.append("Search performance optimization")
        
        # Check companion metrics
        if dashboard_data.companion_metrics["quality_metrics"]["average_satisfaction"] < 0.7:
            focus_areas.append("User satisfaction improvement")
        
        if dashboard_data.companion_metrics["quality_metrics"]["empathy_accuracy"] < 0.8:
            focus_areas.append("Emotional intelligence accuracy")
        
        # Check alerts
        if len(dashboard_data.alerts["active_alerts"]) > 5:
            focus_areas.append("Alert volume reduction")
        
        return focus_areas