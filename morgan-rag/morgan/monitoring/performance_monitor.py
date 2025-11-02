"""
Performance monitoring system for Morgan RAG.

Provides real-time performance tracking, bottleneck detection, and optimization
recommendations as specified in Requirements 7.1, 7.3, and 7.5.
"""

import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from contextlib import contextmanager
import structlog

from .metrics_collector import MetricsCollector

logger = structlog.get_logger(__name__)


@dataclass
class PerformanceThresholds:
    """Performance thresholds for alerting."""
    processing_time_p95: float = 5.0  # seconds
    search_time_p95: float = 0.5  # seconds
    memory_usage_percent: float = 90.0  # percent
    cache_hit_rate_min: float = 80.0  # percent
    error_rate_max: float = 5.0  # percent
    cpu_usage_max: float = 85.0  # percent


@dataclass
class PerformanceAlert:
    """Performance alert information."""
    alert_type: str
    severity: str  # "warning", "critical"
    message: str
    current_value: float
    threshold_value: float
    timestamp: datetime
    component: str
    metadata: Dict[str, Any]


@dataclass
class SystemResourceUsage:
    """Current system resource usage."""
    cpu_percent: float
    memory_percent: float
    memory_used_bytes: int
    memory_available_bytes: int
    disk_usage_percent: float
    network_io_bytes_sent: int
    network_io_bytes_recv: int
    timestamp: datetime


class PerformanceMonitor:
    """
    Real-time performance monitoring system for Morgan RAG.
    
    Tracks system performance, detects bottlenecks, and provides
    optimization recommendations.
    """
    
    def __init__(self, metrics_collector: MetricsCollector, 
                 thresholds: Optional[PerformanceThresholds] = None):
        self.metrics_collector = metrics_collector
        self.thresholds = thresholds or PerformanceThresholds()
        
        # Performance tracking
        self._active_operations: Dict[str, float] = {}
        self._operation_lock = threading.RLock()
        
        # System monitoring
        self._system_monitor_active = False
        self._system_monitor_thread: Optional[threading.Thread] = None
        self._system_stats_history: List[SystemResourceUsage] = []
        
        # Alert tracking
        self._active_alerts: List[PerformanceAlert] = []
        self._alert_callbacks: List[Callable[[PerformanceAlert], None]] = []
        
        logger.info("PerformanceMonitor initialized", thresholds=self.thresholds)
    
    @contextmanager
    def track_operation(self, operation_name: str, operation_type: str = "general",
                       labels: Optional[Dict[str, str]] = None):
        """
        Context manager to track operation performance.
        
        Usage:
            with performance_monitor.track_operation("document_processing", "ingestion"):
                # Your operation code here
                process_document()
        """
        operation_id = f"{operation_name}_{int(time.time() * 1000000)}"
        start_time = time.time()
        
        with self._operation_lock:
            self._active_operations[operation_id] = start_time
        
        try:
            logger.debug("Started tracking operation", 
                        operation_name=operation_name,
                        operation_id=operation_id)
            yield operation_id
            
        except Exception as e:
            # Record error
            duration = time.time() - start_time
            self.metrics_collector.record_error(
                error_type=type(e).__name__,
                component=operation_type,
                error_details=str(e)
            )
            
            # Record timing even for failed operations
            if operation_type == "processing":
                self.metrics_collector.record_processing_time(
                    operation_type=operation_name,
                    duration=duration,
                    document_type=labels.get("document_type", "unknown") if labels else "unknown",
                    success=False
                )
            elif operation_type == "search":
                self.metrics_collector.record_search_time(
                    search_type=operation_name,
                    duration=duration,
                    collection=labels.get("collection", "default") if labels else "default",
                    success=False
                )
            
            logger.error("Operation failed", 
                        operation_name=operation_name,
                        duration=duration,
                        error=str(e))
            raise
            
        finally:
            # Calculate duration and record metrics
            duration = time.time() - start_time
            
            with self._operation_lock:
                self._active_operations.pop(operation_id, None)
            
            # Record successful operation
            if operation_type == "processing":
                self.metrics_collector.record_processing_time(
                    operation_type=operation_name,
                    duration=duration,
                    document_type=labels.get("document_type", "unknown") if labels else "unknown",
                    success=True
                )
            elif operation_type == "search":
                self.metrics_collector.record_search_time(
                    search_type=operation_name,
                    duration=duration,
                    collection=labels.get("collection", "default") if labels else "default",
                    success=True
                )
            elif operation_type == "embedding":
                self.metrics_collector.record_embedding_time(
                    model_type=labels.get("model_type", "unknown") if labels else "unknown",
                    scale=labels.get("scale", "unknown") if labels else "unknown",
                    duration=duration
                )
            
            logger.debug("Completed tracking operation",
                        operation_name=operation_name,
                        duration=duration,
                        operation_id=operation_id)
    
    def start_system_monitoring(self, interval_seconds: float = 30.0):
        """Start continuous system resource monitoring."""
        if self._system_monitor_active:
            logger.warning("System monitoring already active")
            return
        
        self._system_monitor_active = True
        self._system_monitor_thread = threading.Thread(
            target=self._system_monitor_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self._system_monitor_thread.start()
        
        logger.info("Started system monitoring", interval_seconds=interval_seconds)
    
    def stop_system_monitoring(self):
        """Stop system resource monitoring."""
        self._system_monitor_active = False
        if self._system_monitor_thread:
            self._system_monitor_thread.join(timeout=5.0)
        
        logger.info("Stopped system monitoring")
    
    def _system_monitor_loop(self, interval_seconds: float):
        """Main system monitoring loop."""
        while self._system_monitor_active:
            try:
                # Collect system stats
                stats = self._collect_system_stats()
                
                # Record metrics
                self.metrics_collector.record_memory_usage("system", stats.memory_used_bytes)
                
                # Store history (keep last 1000 entries)
                self._system_stats_history.append(stats)
                if len(self._system_stats_history) > 1000:
                    self._system_stats_history.pop(0)
                
                # Check for performance alerts
                self._check_performance_alerts(stats)
                
                time.sleep(interval_seconds)
                
            except Exception as e:
                logger.error("Error in system monitoring loop", error=str(e))
                time.sleep(interval_seconds)
    
    def _collect_system_stats(self) -> SystemResourceUsage:
        """Collect current system resource usage."""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        
        # Disk usage (for the current directory)
        disk = psutil.disk_usage('.')
        
        # Network I/O
        network = psutil.net_io_counters()
        
        return SystemResourceUsage(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_bytes=memory.used,
            memory_available_bytes=memory.available,
            disk_usage_percent=(disk.used / disk.total) * 100,
            network_io_bytes_sent=network.bytes_sent,
            network_io_bytes_recv=network.bytes_recv,
            timestamp=datetime.now()
        )
    
    def _check_performance_alerts(self, stats: SystemResourceUsage):
        """Check for performance threshold violations."""
        alerts = []
        
        # Memory usage alert
        if stats.memory_percent > self.thresholds.memory_usage_percent:
            alerts.append(PerformanceAlert(
                alert_type="high_memory_usage",
                severity="critical" if stats.memory_percent > 95 else "warning",
                message=f"High memory usage: {stats.memory_percent:.1f}%",
                current_value=stats.memory_percent,
                threshold_value=self.thresholds.memory_usage_percent,
                timestamp=stats.timestamp,
                component="system",
                metadata={"memory_used_bytes": stats.memory_used_bytes}
            ))
        
        # CPU usage alert
        if stats.cpu_percent > self.thresholds.cpu_usage_max:
            alerts.append(PerformanceAlert(
                alert_type="high_cpu_usage",
                severity="critical" if stats.cpu_percent > 95 else "warning",
                message=f"High CPU usage: {stats.cpu_percent:.1f}%",
                current_value=stats.cpu_percent,
                threshold_value=self.thresholds.cpu_usage_max,
                timestamp=stats.timestamp,
                component="system",
                metadata={}
            ))
        
        # Check application-specific metrics
        self._check_application_performance_alerts(alerts)
        
        # Process new alerts
        for alert in alerts:
            self._process_alert(alert)
    
    def _check_application_performance_alerts(self, alerts: List[PerformanceAlert]):
        """Check application-specific performance metrics."""
        # Check processing time performance
        processing_summary = self.metrics_collector.get_metrics_summary("processing_time")
        if processing_summary and processing_summary.percentile_95 > self.thresholds.processing_time_p95:
            alerts.append(PerformanceAlert(
                alert_type="slow_processing",
                severity="warning",
                message=f"Slow document processing: P95 = {processing_summary.percentile_95:.2f}s",
                current_value=processing_summary.percentile_95,
                threshold_value=self.thresholds.processing_time_p95,
                timestamp=datetime.now(),
                component="processing",
                metadata={"error_rate": processing_summary.error_rate}
            ))
        
        # Check search time performance
        search_summary = self.metrics_collector.get_metrics_summary("search_time")
        if search_summary and search_summary.percentile_95 > self.thresholds.search_time_p95:
            alerts.append(PerformanceAlert(
                alert_type="slow_search",
                severity="warning",
                message=f"Slow search performance: P95 = {search_summary.percentile_95:.2f}s",
                current_value=search_summary.percentile_95,
                threshold_value=self.thresholds.search_time_p95,
                timestamp=datetime.now(),
                component="search",
                metadata={"error_rate": search_summary.error_rate}
            ))
        
        # Check error rates
        if processing_summary and processing_summary.error_rate > self.thresholds.error_rate_max / 100:
            alerts.append(PerformanceAlert(
                alert_type="high_error_rate",
                severity="critical",
                message=f"High processing error rate: {processing_summary.error_rate * 100:.1f}%",
                current_value=processing_summary.error_rate * 100,
                threshold_value=self.thresholds.error_rate_max,
                timestamp=datetime.now(),
                component="processing",
                metadata={"total_operations": processing_summary.total_count}
            ))
    
    def _process_alert(self, alert: PerformanceAlert):
        """Process a new performance alert."""
        # Check if this is a duplicate of an existing alert
        existing_alert = next(
            (a for a in self._active_alerts 
             if a.alert_type == alert.alert_type and a.component == alert.component),
            None
        )
        
        if existing_alert:
            # Update existing alert
            existing_alert.current_value = alert.current_value
            existing_alert.timestamp = alert.timestamp
            existing_alert.message = alert.message
        else:
            # New alert
            self._active_alerts.append(alert)
            logger.warning("Performance alert triggered",
                          alert_type=alert.alert_type,
                          severity=alert.severity,
                          message=alert.message,
                          component=alert.component)
        
        # Notify alert callbacks
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error("Error in alert callback", error=str(e))
    
    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]):
        """Add a callback function to be called when alerts are triggered."""
        self._alert_callbacks.append(callback)
        logger.debug("Added alert callback", callback=callback.__name__)
    
    def get_active_alerts(self) -> List[PerformanceAlert]:
        """Get list of currently active performance alerts."""
        # Clean up old alerts (older than 1 hour)
        cutoff_time = datetime.now() - timedelta(hours=1)
        self._active_alerts = [
            alert for alert in self._active_alerts
            if alert.timestamp >= cutoff_time
        ]
        
        return self._active_alerts.copy()
    
    def get_system_stats_history(self, hours: int = 1) -> List[SystemResourceUsage]:
        """Get system resource usage history."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            stats for stats in self._system_stats_history
            if stats.timestamp >= cutoff_time
        ]
    
    def get_active_operations(self) -> Dict[str, float]:
        """Get currently active operations and their start times."""
        with self._operation_lock:
            current_time = time.time()
            return {
                op_id: current_time - start_time
                for op_id, start_time in self._active_operations.items()
            }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        # Get recent system stats
        recent_stats = self.get_system_stats_history(hours=1)
        
        if recent_stats:
            avg_cpu = sum(s.cpu_percent for s in recent_stats) / len(recent_stats)
            avg_memory = sum(s.memory_percent for s in recent_stats) / len(recent_stats)
            current_stats = recent_stats[-1]
        else:
            avg_cpu = avg_memory = 0
            current_stats = None
        
        # Get application metrics
        processing_summary = self.metrics_collector.get_metrics_summary("processing_time")
        search_summary = self.metrics_collector.get_metrics_summary("search_time")
        
        # Get active operations
        active_ops = self.get_active_operations()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "system_resources": {
                "current_cpu_percent": current_stats.cpu_percent if current_stats else 0,
                "current_memory_percent": current_stats.memory_percent if current_stats else 0,
                "average_cpu_percent_1h": avg_cpu,
                "average_memory_percent_1h": avg_memory,
                "memory_used_bytes": current_stats.memory_used_bytes if current_stats else 0
            },
            "application_performance": {
                "processing": {
                    "average_duration": processing_summary.average if processing_summary else 0,
                    "p95_duration": processing_summary.percentile_95 if processing_summary else 0,
                    "p99_duration": processing_summary.percentile_99 if processing_summary else 0,
                    "error_rate": processing_summary.error_rate if processing_summary else 0,
                    "total_operations": processing_summary.total_count if processing_summary else 0
                },
                "search": {
                    "average_duration": search_summary.average if search_summary else 0,
                    "p95_duration": search_summary.percentile_95 if search_summary else 0,
                    "p99_duration": search_summary.percentile_99 if search_summary else 0,
                    "error_rate": search_summary.error_rate if search_summary else 0,
                    "total_operations": search_summary.total_count if search_summary else 0
                }
            },
            "active_operations": {
                "count": len(active_ops),
                "longest_running_duration": max(active_ops.values()) if active_ops else 0,
                "operations": active_ops
            },
            "alerts": {
                "active_count": len(self._active_alerts),
                "critical_count": len([a for a in self._active_alerts if a.severity == "critical"]),
                "warning_count": len([a for a in self._active_alerts if a.severity == "warning"])
            },
            "thresholds": {
                "processing_time_p95": self.thresholds.processing_time_p95,
                "search_time_p95": self.thresholds.search_time_p95,
                "memory_usage_percent": self.thresholds.memory_usage_percent,
                "error_rate_max": self.thresholds.error_rate_max,
                "cpu_usage_max": self.thresholds.cpu_usage_max
            }
        }
    
    def generate_optimization_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations based on current metrics."""
        recommendations = []
        
        # Get performance summary
        summary = self.get_performance_summary()
        
        # Memory recommendations
        if summary["system_resources"]["current_memory_percent"] > 80:
            recommendations.append(
                "High memory usage detected. Consider increasing available RAM or "
                "implementing more aggressive caching cleanup."
            )
        
        # Processing performance recommendations
        processing_perf = summary["application_performance"]["processing"]
        if processing_perf["p95_duration"] > self.thresholds.processing_time_p95:
            recommendations.append(
                f"Document processing is slow (P95: {processing_perf['p95_duration']:.2f}s). "
                "Consider enabling batch processing or optimizing document chunking."
            )
        
        # Search performance recommendations
        search_perf = summary["application_performance"]["search"]
        if search_perf["p95_duration"] > self.thresholds.search_time_p95:
            recommendations.append(
                f"Search performance is slow (P95: {search_perf['p95_duration']:.2f}s). "
                "Consider optimizing vector database indices or enabling hierarchical search."
            )
        
        # Error rate recommendations
        if processing_perf["error_rate"] > self.thresholds.error_rate_max / 100:
            recommendations.append(
                f"High processing error rate ({processing_perf['error_rate'] * 100:.1f}%). "
                "Check logs for common error patterns and implement better error handling."
            )
        
        # Long-running operations
        if summary["active_operations"]["longest_running_duration"] > 300:  # 5 minutes
            recommendations.append(
                "Long-running operations detected. Consider implementing operation timeouts "
                "or breaking large operations into smaller chunks."
            )
        
        # CPU recommendations
        if summary["system_resources"]["current_cpu_percent"] > 85:
            recommendations.append(
                "High CPU usage detected. Consider scaling horizontally or optimizing "
                "CPU-intensive operations like embedding generation."
            )
        
        return recommendations