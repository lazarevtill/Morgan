"""
Alerting system for Morgan RAG monitoring.

Provides intelligent alerting for system health, performance issues, and
companion experience quality as specified in the requirements.
"""

import asyncio
import smtplib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import structlog

# Email imports with fallback
try:
    from email.mime.text import MimeText
    from email.mime.multipart import MimeMultipart
except ImportError:
    # Fallback for older Python versions or missing email modules
    MimeText = None
    MimeMultipart = None

from .performance_monitor import PerformanceAlert
from .health_monitor import HealthCheckResult, HealthStatus
from .companion_metrics import CompanionQualityMetrics

logger = structlog.get_logger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertChannel(Enum):
    """Alert delivery channels."""
    LOG = "log"
    EMAIL = "email"
    WEBHOOK = "webhook"
    CONSOLE = "console"


@dataclass
class AlertRule:
    """Configuration for an alert rule."""
    name: str
    condition: Callable[[Any], bool]
    severity: AlertSeverity
    channels: List[AlertChannel]
    message_template: str
    cooldown_minutes: int = 15
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """An alert instance."""
    rule_name: str
    severity: AlertSeverity
    message: str
    timestamp: datetime
    source_data: Dict[str, Any]
    channels: List[AlertChannel]
    alert_id: str
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertingConfig:
    """Alerting system configuration."""
    email_smtp_host: Optional[str] = None
    email_smtp_port: int = 587
    email_username: Optional[str] = None
    email_password: Optional[str] = None
    email_from: Optional[str] = None
    email_to: List[str] = field(default_factory=list)
    webhook_urls: List[str] = field(default_factory=list)
    webhook_timeout: int = 10
    alert_retention_hours: int = 168  # 7 days


class AlertManager:
    """
    Intelligent alerting system for Morgan RAG.
    
    Manages alert rules, processes alerts, and delivers notifications
    through multiple channels with intelligent deduplication and cooldowns.
    """
    
    def __init__(self, config: Optional[AlertingConfig] = None):
        self.config = config or AlertingConfig()
        
        # Alert management
        self._alert_rules: Dict[str, AlertRule] = {}
        self._active_alerts: Dict[str, Alert] = {}
        self._alert_history: List[Alert] = []
        self._last_alert_times: Dict[str, datetime] = {}
        
        # Alert processing
        self._alert_queue: asyncio.Queue = asyncio.Queue()
        self._processing_task: Optional[asyncio.Task] = None
        self._processing_active = False
        
        # Initialize default alert rules
        self._register_default_alert_rules()
        
        logger.info("AlertManager initialized")
    
    def _register_default_alert_rules(self):
        """Register default alert rules for common issues."""
        
        # High memory usage alert
        self.register_alert_rule(
            name="high_memory_usage",
            condition=lambda data: data.get("memory_percent", 0) > 90,
            severity=AlertSeverity.CRITICAL,
            channels=[AlertChannel.LOG, AlertChannel.EMAIL],
            message_template="High memory usage detected: {memory_percent:.1f}%",
            cooldown_minutes=10
        )
        
        # High CPU usage alert
        self.register_alert_rule(
            name="high_cpu_usage",
            condition=lambda data: data.get("cpu_percent", 0) > 85,
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.LOG, AlertChannel.EMAIL],
            message_template="High CPU usage detected: {cpu_percent:.1f}%",
            cooldown_minutes=15
        )
        
        # Slow processing alert
        self.register_alert_rule(
            name="slow_processing",
            condition=lambda data: data.get("processing_p95", 0) > 5.0,
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.LOG, AlertChannel.EMAIL],
            message_template="Slow document processing detected: P95 = {processing_p95:.2f}s",
            cooldown_minutes=20
        )
        
        # Slow search alert
        self.register_alert_rule(
            name="slow_search",
            condition=lambda data: data.get("search_p95", 0) > 0.5,
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.LOG, AlertChannel.EMAIL],
            message_template="Slow search performance detected: P95 = {search_p95:.2f}s",
            cooldown_minutes=15
        )
        
        # High error rate alert
        self.register_alert_rule(
            name="high_error_rate",
            condition=lambda data: data.get("error_rate", 0) > 0.05,
            severity=AlertSeverity.CRITICAL,
            channels=[AlertChannel.LOG, AlertChannel.EMAIL, AlertChannel.WEBHOOK],
            message_template="High error rate detected: {error_rate:.1%}",
            cooldown_minutes=10
        )
        
        # Low cache hit rate alert
        self.register_alert_rule(
            name="low_cache_hit_rate",
            condition=lambda data: data.get("cache_hit_rate", 100) < 70,
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.LOG],
            message_template="Low cache hit rate: {cache_hit_rate:.1f}%",
            cooldown_minutes=30
        )
        
        # Database connectivity alert
        self.register_alert_rule(
            name="database_connectivity",
            condition=lambda data: data.get("database_status") == "unhealthy",
            severity=AlertSeverity.CRITICAL,
            channels=[AlertChannel.LOG, AlertChannel.EMAIL, AlertChannel.WEBHOOK],
            message_template="Database connectivity issue detected: {database_message}",
            cooldown_minutes=5
        )
        
        # Low user satisfaction alert
        self.register_alert_rule(
            name="low_user_satisfaction",
            condition=lambda data: data.get("user_satisfaction", 1.0) < 0.6,
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.LOG, AlertChannel.EMAIL],
            message_template="Low user satisfaction detected: {user_satisfaction:.1%}",
            cooldown_minutes=60
        )
        
        # Poor empathy accuracy alert
        self.register_alert_rule(
            name="poor_empathy_accuracy",
            condition=lambda data: data.get("empathy_accuracy", 1.0) < 0.7,
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.LOG, AlertChannel.EMAIL],
            message_template="Poor empathy accuracy detected: {empathy_accuracy:.1%}",
            cooldown_minutes=120
        )
        
        # Low relationship retention alert
        self.register_alert_rule(
            name="low_relationship_retention",
            condition=lambda data: data.get("relationship_retention", 1.0) < 0.5,
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.LOG, AlertChannel.EMAIL],
            message_template="Low relationship retention rate: {relationship_retention:.1%}",
            cooldown_minutes=240
        )
    
    def register_alert_rule(self, name: str, condition: Callable[[Any], bool],
                          severity: AlertSeverity, channels: List[AlertChannel],
                          message_template: str, cooldown_minutes: int = 15,
                          enabled: bool = True, metadata: Optional[Dict[str, Any]] = None):
        """Register a new alert rule."""
        rule = AlertRule(
            name=name,
            condition=condition,
            severity=severity,
            channels=channels,
            message_template=message_template,
            cooldown_minutes=cooldown_minutes,
            enabled=enabled,
            metadata=metadata or {}
        )
        
        self._alert_rules[name] = rule
        
        logger.info("Registered alert rule",
                   name=name,
                   severity=severity.value,
                   channels=[c.value for c in channels])
    
    def unregister_alert_rule(self, name: str):
        """Unregister an alert rule."""
        if name in self._alert_rules:
            del self._alert_rules[name]
            logger.info("Unregistered alert rule", name=name)
    
    def enable_alert_rule(self, name: str):
        """Enable an alert rule."""
        if name in self._alert_rules:
            self._alert_rules[name].enabled = True
            logger.info("Enabled alert rule", name=name)
    
    def disable_alert_rule(self, name: str):
        """Disable an alert rule."""
        if name in self._alert_rules:
            self._alert_rules[name].enabled = False
            logger.info("Disabled alert rule", name=name)
    
    async def start_processing(self):
        """Start alert processing."""
        if self._processing_active:
            logger.warning("Alert processing already active")
            return
        
        self._processing_active = True
        self._processing_task = asyncio.create_task(self._processing_loop())
        
        logger.info("Started alert processing")
    
    async def stop_processing(self):
        """Stop alert processing."""
        self._processing_active = False
        
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped alert processing")
    
    async def _processing_loop(self):
        """Main alert processing loop."""
        while self._processing_active:
            try:
                # Wait for alerts with timeout
                try:
                    alert = await asyncio.wait_for(self._alert_queue.get(), timeout=1.0)
                    await self._process_alert(alert)
                except asyncio.TimeoutError:
                    continue
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in alert processing loop", error=str(e))
                await asyncio.sleep(1.0)
    
    async def check_performance_alert(self, performance_alert: PerformanceAlert):
        """Check performance alert against rules."""
        data = {
            "alert_type": performance_alert.alert_type,
            "severity": performance_alert.severity,
            "current_value": performance_alert.current_value,
            "threshold_value": performance_alert.threshold_value,
            "component": performance_alert.component,
            "timestamp": performance_alert.timestamp
        }
        
        # Map performance alert data to rule-specific data
        if performance_alert.alert_type == "high_memory_usage":
            data["memory_percent"] = performance_alert.current_value
        elif performance_alert.alert_type == "high_cpu_usage":
            data["cpu_percent"] = performance_alert.current_value
        elif performance_alert.alert_type == "slow_processing":
            data["processing_p95"] = performance_alert.current_value
        elif performance_alert.alert_type == "slow_search":
            data["search_p95"] = performance_alert.current_value
        elif performance_alert.alert_type == "high_error_rate":
            data["error_rate"] = performance_alert.current_value / 100.0
        
        await self._check_rules_and_queue_alerts(data)
    
    async def check_health_alert(self, health_result: HealthCheckResult):
        """Check health check result against rules."""
        data = {
            "health_check_name": health_result.name,
            "health_status": health_result.status.value,
            "health_message": health_result.message,
            "health_duration": health_result.duration,
            "timestamp": health_result.timestamp
        }
        
        # Map health status to database status for rules
        if health_result.name == "vector_database":
            data["database_status"] = "unhealthy" if health_result.status != HealthStatus.HEALTHY else "healthy"
            data["database_message"] = health_result.message
        
        await self._check_rules_and_queue_alerts(data)
    
    async def check_companion_metrics_alert(self, companion_metrics: CompanionQualityMetrics):
        """Check companion metrics against rules."""
        data = {
            "user_satisfaction": companion_metrics.average_satisfaction_score,
            "empathy_accuracy": companion_metrics.empathy_accuracy_rate,
            "relationship_retention": companion_metrics.relationship_retention_rate,
            "emotional_support_success": companion_metrics.emotional_support_success_rate,
            "personalization_score": companion_metrics.response_personalization_score,
            "timestamp": companion_metrics.timestamp
        }
        
        await self._check_rules_and_queue_alerts(data)
    
    async def check_custom_metrics_alert(self, metrics_data: Dict[str, Any]):
        """Check custom metrics data against rules."""
        await self._check_rules_and_queue_alerts(metrics_data)
    
    async def _check_rules_and_queue_alerts(self, data: Dict[str, Any]):
        """Check data against all alert rules and queue matching alerts."""
        for rule_name, rule in self._alert_rules.items():
            if not rule.enabled:
                continue
            
            try:
                # Check cooldown
                if self._is_in_cooldown(rule_name, rule.cooldown_minutes):
                    continue
                
                # Check condition
                if rule.condition(data):
                    # Create alert
                    alert_id = f"{rule_name}_{int(datetime.now().timestamp())}"
                    message = rule.message_template.format(**data)
                    
                    alert = Alert(
                        rule_name=rule_name,
                        severity=rule.severity,
                        message=message,
                        timestamp=datetime.now(),
                        source_data=data.copy(),
                        channels=rule.channels,
                        alert_id=alert_id,
                        metadata=rule.metadata.copy()
                    )
                    
                    # Queue for processing
                    await self._alert_queue.put(alert)
                    
                    # Update last alert time
                    self._last_alert_times[rule_name] = datetime.now()
                    
            except Exception as e:
                logger.error("Error checking alert rule",
                           rule_name=rule_name,
                           error=str(e))
    
    def _is_in_cooldown(self, rule_name: str, cooldown_minutes: int) -> bool:
        """Check if alert rule is in cooldown period."""
        if rule_name not in self._last_alert_times:
            return False
        
        last_alert_time = self._last_alert_times[rule_name]
        cooldown_period = timedelta(minutes=cooldown_minutes)
        
        return datetime.now() - last_alert_time < cooldown_period
    
    async def _process_alert(self, alert: Alert):
        """Process and deliver an alert."""
        try:
            # Store alert
            self._active_alerts[alert.alert_id] = alert
            self._alert_history.append(alert)
            
            # Deliver through channels
            for channel in alert.channels:
                try:
                    await self._deliver_alert(alert, channel)
                except Exception as e:
                    logger.error("Failed to deliver alert through channel",
                               alert_id=alert.alert_id,
                               channel=channel.value,
                               error=str(e))
            
            logger.info("Processed alert",
                       alert_id=alert.alert_id,
                       rule_name=alert.rule_name,
                       severity=alert.severity.value,
                       message=alert.message)
            
        except Exception as e:
            logger.error("Error processing alert",
                        alert_id=alert.alert_id,
                        error=str(e))
    
    async def _deliver_alert(self, alert: Alert, channel: AlertChannel):
        """Deliver alert through specific channel."""
        if channel == AlertChannel.LOG:
            await self._deliver_log_alert(alert)
        elif channel == AlertChannel.EMAIL:
            await self._deliver_email_alert(alert)
        elif channel == AlertChannel.WEBHOOK:
            await self._deliver_webhook_alert(alert)
        elif channel == AlertChannel.CONSOLE:
            await self._deliver_console_alert(alert)
    
    async def _deliver_log_alert(self, alert: Alert):
        """Deliver alert to logs."""
        log_level = "error" if alert.severity == AlertSeverity.CRITICAL else "warning"
        
        logger.log(log_level, "ALERT",
                  alert_id=alert.alert_id,
                  rule_name=alert.rule_name,
                  severity=alert.severity.value,
                  message=alert.message,
                  timestamp=alert.timestamp.isoformat())
    
    async def _deliver_console_alert(self, alert: Alert):
        """Deliver alert to console."""
        severity_colors = {
            AlertSeverity.INFO: "\033[94m",      # Blue
            AlertSeverity.WARNING: "\033[93m",   # Yellow
            AlertSeverity.CRITICAL: "\033[91m"   # Red
        }
        reset_color = "\033[0m"
        
        color = severity_colors.get(alert.severity, "")
        print(f"{color}[ALERT {alert.severity.value.upper()}]{reset_color} {alert.message}")
    
    async def _deliver_email_alert(self, alert: Alert):
        """Deliver alert via email."""
        if not self.config.email_smtp_host or not self.config.email_to:
            logger.warning("Email configuration incomplete, skipping email alert")
            return
        
        if MimeText is None or MimeMultipart is None:
            logger.warning("Email modules not available, skipping email alert")
            return
        
        try:
            # Create email message
            msg = MimeMultipart()
            msg['From'] = self.config.email_from or self.config.email_username
            msg['To'] = ', '.join(self.config.email_to)
            msg['Subject'] = f"Morgan RAG Alert: {alert.rule_name} ({alert.severity.value.upper()})"
            
            # Email body
            body = f"""
Morgan RAG Alert

Rule: {alert.rule_name}
Severity: {alert.severity.value.upper()}
Time: {alert.timestamp.isoformat()}
Message: {alert.message}

Alert ID: {alert.alert_id}

Source Data:
{json.dumps(alert.source_data, indent=2, default=str)}
"""
            
            msg.attach(MimeText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(self.config.email_smtp_host, self.config.email_smtp_port)
            server.starttls()
            
            if self.config.email_username and self.config.email_password:
                server.login(self.config.email_username, self.config.email_password)
            
            server.send_message(msg)
            server.quit()
            
            logger.debug("Sent email alert", alert_id=alert.alert_id)
            
        except Exception as e:
            logger.error("Failed to send email alert",
                        alert_id=alert.alert_id,
                        error=str(e))
    
    async def _deliver_webhook_alert(self, alert: Alert):
        """Deliver alert via webhook."""
        if not self.config.webhook_urls:
            logger.warning("No webhook URLs configured, skipping webhook alert")
            return
        
        # Prepare webhook payload
        payload = {
            "alert_id": alert.alert_id,
            "rule_name": alert.rule_name,
            "severity": alert.severity.value,
            "message": alert.message,
            "timestamp": alert.timestamp.isoformat(),
            "source_data": alert.source_data,
            "metadata": alert.metadata
        }
        
        # Send to all configured webhooks
        for webhook_url in self.config.webhook_urls:
            try:
                import httpx
                
                async with httpx.AsyncClient(timeout=self.config.webhook_timeout) as client:
                    response = await client.post(
                        webhook_url,
                        json=payload,
                        headers={"Content-Type": "application/json"}
                    )
                    response.raise_for_status()
                
                logger.debug("Sent webhook alert",
                           alert_id=alert.alert_id,
                           webhook_url=webhook_url)
                
            except Exception as e:
                logger.error("Failed to send webhook alert",
                           alert_id=alert.alert_id,
                           webhook_url=webhook_url,
                           error=str(e))
    
    def resolve_alert(self, alert_id: str):
        """Manually resolve an alert."""
        if alert_id in self._active_alerts:
            alert = self._active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = datetime.now()
            
            del self._active_alerts[alert_id]
            
            logger.info("Resolved alert", alert_id=alert_id)
    
    def get_active_alerts(self) -> List[Alert]:
        """Get list of active (unresolved) alerts."""
        return list(self._active_alerts.values())
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            alert for alert in self._alert_history
            if alert.timestamp >= cutoff_time
        ]
    
    def get_alert_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get alert statistics for specified time period."""
        recent_alerts = self.get_alert_history(hours)
        
        # Count by severity
        severity_counts = {severity.value: 0 for severity in AlertSeverity}
        for alert in recent_alerts:
            severity_counts[alert.severity.value] += 1
        
        # Count by rule
        rule_counts = {}
        for alert in recent_alerts:
            rule_counts[alert.rule_name] = rule_counts.get(alert.rule_name, 0) + 1
        
        # Calculate resolution stats
        resolved_alerts = [alert for alert in recent_alerts if alert.resolved]
        resolution_rate = len(resolved_alerts) / len(recent_alerts) if recent_alerts else 0
        
        return {
            "total_alerts": len(recent_alerts),
            "active_alerts": len(self._active_alerts),
            "severity_breakdown": severity_counts,
            "rule_breakdown": rule_counts,
            "resolution_rate": resolution_rate,
            "time_period_hours": hours
        }
    
    def cleanup_old_alerts(self):
        """Clean up old alerts from history."""
        cutoff_time = datetime.now() - timedelta(hours=self.config.alert_retention_hours)
        
        # Clean history
        self._alert_history = [
            alert for alert in self._alert_history
            if alert.timestamp >= cutoff_time
        ]
        
        # Clean last alert times
        self._last_alert_times = {
            rule_name: timestamp
            for rule_name, timestamp in self._last_alert_times.items()
            if timestamp >= cutoff_time
        }
        
        logger.debug("Cleaned up old alerts",
                    retention_hours=self.config.alert_retention_hours)