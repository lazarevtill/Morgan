"""
Core metrics collection system for Morgan RAG.

Provides centralized metrics collection for performance, quality, and operational metrics
as specified in Requirements 7.1-7.5.
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import structlog
from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry, generate_latest

logger = structlog.get_logger(__name__)


@dataclass
class MetricValue:
    """Represents a single metric measurement."""
    name: str
    value: Union[int, float]
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricsSummary:
    """Summary of metrics over a time period."""
    total_count: int
    average: float
    min_value: float
    max_value: float
    percentile_95: float
    percentile_99: float
    error_rate: float


class MetricsCollector:
    """
    Central metrics collection system for Morgan RAG.
    
    Collects and manages performance, quality, and operational metrics
    with support for Prometheus export and real-time monitoring.
    """
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()
        self._metrics_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self._lock = threading.RLock()
        
        # Initialize Prometheus metrics
        self._init_prometheus_metrics()
        
        # Performance tracking
        self._operation_timings: Dict[str, List[float]] = defaultdict(list)
        self._error_counts: Dict[str, int] = defaultdict(int)
        
        logger.info("MetricsCollector initialized", registry_collectors=len(self.registry._collector_to_names))
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics for export."""
        # Performance metrics
        self.processing_duration = Histogram(
            'morgan_processing_duration_seconds',
            'Time spent processing documents',
            ['operation_type', 'document_type'],
            registry=self.registry
        )
        
        self.search_duration = Histogram(
            'morgan_search_duration_seconds', 
            'Time spent on search operations',
            ['search_type', 'collection'],
            registry=self.registry
        )
        
        self.embedding_duration = Histogram(
            'morgan_embedding_duration_seconds',
            'Time spent generating embeddings',
            ['model_type', 'scale'],
            registry=self.registry
        )
        
        # Throughput metrics
        self.documents_processed = Counter(
            'morgan_documents_processed_total',
            'Total number of documents processed',
            ['document_type', 'status'],
            registry=self.registry
        )
        
        self.search_requests = Counter(
            'morgan_search_requests_total',
            'Total number of search requests',
            ['search_type', 'status'],
            registry=self.registry
        )
        
        # Quality metrics
        self.search_relevance = Histogram(
            'morgan_search_relevance_score',
            'Search result relevance scores',
            ['search_type'],
            registry=self.registry
        )
        
        self.user_satisfaction = Histogram(
            'morgan_user_satisfaction_score',
            'User satisfaction ratings',
            ['interaction_type'],
            registry=self.registry
        )
        
        # System health metrics
        self.cache_hit_rate = Gauge(
            'morgan_cache_hit_rate',
            'Cache hit rate percentage',
            ['cache_type'],
            registry=self.registry
        )
        
        # Git hash cache metrics (implements R1.3, R9.1)
        self.git_cache_requests = Counter(
            'morgan_git_cache_requests_total',
            'Total Git hash cache requests',
            ['result'],
            registry=self.registry
        )
        
        self.git_hash_calculations = Counter(
            'morgan_git_hash_calculations_total',
            'Total Git hash calculations performed',
            ['source_type'],
            registry=self.registry
        )
        
        self.cache_invalidations = Counter(
            'morgan_cache_invalidations_total',
            'Total cache invalidations',
            ['reason'],
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'morgan_memory_usage_bytes',
            'Memory usage in bytes',
            ['component'],
            registry=self.registry
        )
        
        self.database_connections = Gauge(
            'morgan_database_connections_active',
            'Active database connections',
            ['database_type'],
            registry=self.registry
        )
        
        # Error metrics
        self.error_count = Counter(
            'morgan_errors_total',
            'Total number of errors',
            ['error_type', 'component'],
            registry=self.registry
        )
        
        # Companion metrics
        self.emotional_analysis_duration = Histogram(
            'morgan_emotional_analysis_duration_seconds',
            'Time spent on emotional analysis',
            ['analysis_type'],
            registry=self.registry
        )
        
        self.relationship_milestones = Counter(
            'morgan_relationship_milestones_total',
            'Total relationship milestones reached',
            ['milestone_type'],
            registry=self.registry
        )
        
        self.empathy_score = Histogram(
            'morgan_empathy_score',
            'Empathy scores for responses',
            ['response_type'],
            registry=self.registry
        )
    
    def record_processing_time(self, operation_type: str, duration: float, 
                             document_type: str = "unknown", success: bool = True):
        """Record document processing timing."""
        with self._lock:
            self.processing_duration.labels(
                operation_type=operation_type,
                document_type=document_type
            ).observe(duration)
            
            self.documents_processed.labels(
                document_type=document_type,
                status="success" if success else "error"
            ).inc()
            
            # Store for internal analysis
            key = f"processing_{operation_type}_{document_type}"
            self._operation_timings[key].append(duration)
            
            if not success:
                self._error_counts[f"processing_{operation_type}"] += 1
            
            # Buffer for real-time monitoring
            metric = MetricValue(
                name="processing_time",
                value=duration,
                timestamp=datetime.now(),
                labels={
                    "operation_type": operation_type,
                    "document_type": document_type,
                    "success": str(success)
                }
            )
            self._metrics_buffer["processing_time"].append(metric)
        
        logger.debug("Recorded processing time", 
                    operation=operation_type, 
                    duration=duration,
                    document_type=document_type,
                    success=success)
    
    def record_search_time(self, search_type: str, duration: float, 
                          collection: str = "default", success: bool = True):
        """Record search operation timing."""
        with self._lock:
            self.search_duration.labels(
                search_type=search_type,
                collection=collection
            ).observe(duration)
            
            self.search_requests.labels(
                search_type=search_type,
                status="success" if success else "error"
            ).inc()
            
            # Store for internal analysis
            key = f"search_{search_type}_{collection}"
            self._operation_timings[key].append(duration)
            
            if not success:
                self._error_counts[f"search_{search_type}"] += 1
            
            # Buffer for real-time monitoring
            metric = MetricValue(
                name="search_time",
                value=duration,
                timestamp=datetime.now(),
                labels={
                    "search_type": search_type,
                    "collection": collection,
                    "success": str(success)
                }
            )
            self._metrics_buffer["search_time"].append(metric)
        
        logger.debug("Recorded search time",
                    search_type=search_type,
                    duration=duration,
                    collection=collection,
                    success=success)
    
    def record_embedding_time(self, model_type: str, scale: str, duration: float):
        """Record embedding generation timing."""
        with self._lock:
            self.embedding_duration.labels(
                model_type=model_type,
                scale=scale
            ).observe(duration)
            
            # Store for internal analysis
            key = f"embedding_{model_type}_{scale}"
            self._operation_timings[key].append(duration)
            
            # Buffer for real-time monitoring
            metric = MetricValue(
                name="embedding_time",
                value=duration,
                timestamp=datetime.now(),
                labels={
                    "model_type": model_type,
                    "scale": scale
                }
            )
            self._metrics_buffer["embedding_time"].append(metric)
        
        logger.debug("Recorded embedding time",
                    model_type=model_type,
                    scale=scale,
                    duration=duration)
    
    def record_search_relevance(self, search_type: str, relevance_score: float):
        """Record search result relevance score."""
        with self._lock:
            self.search_relevance.labels(search_type=search_type).observe(relevance_score)
            
            # Buffer for real-time monitoring
            metric = MetricValue(
                name="search_relevance",
                value=relevance_score,
                timestamp=datetime.now(),
                labels={"search_type": search_type}
            )
            self._metrics_buffer["search_relevance"].append(metric)
        
        logger.debug("Recorded search relevance",
                    search_type=search_type,
                    relevance_score=relevance_score)
    
    def record_user_satisfaction(self, interaction_type: str, satisfaction_score: float):
        """Record user satisfaction rating."""
        with self._lock:
            self.user_satisfaction.labels(interaction_type=interaction_type).observe(satisfaction_score)
            
            # Buffer for real-time monitoring
            metric = MetricValue(
                name="user_satisfaction",
                value=satisfaction_score,
                timestamp=datetime.now(),
                labels={"interaction_type": interaction_type}
            )
            self._metrics_buffer["user_satisfaction"].append(metric)
        
        logger.info("Recorded user satisfaction",
                   interaction_type=interaction_type,
                   satisfaction_score=satisfaction_score)
    
    def record_cache_hit_rate(self, cache_type: str, hit_rate: float):
        """Record cache hit rate."""
        with self._lock:
            self.cache_hit_rate.labels(cache_type=cache_type).set(hit_rate)
            
            # Buffer for real-time monitoring
            metric = MetricValue(
                name="cache_hit_rate",
                value=hit_rate,
                timestamp=datetime.now(),
                labels={"cache_type": cache_type}
            )
            self._metrics_buffer["cache_hit_rate"].append(metric)
        
        logger.debug("Recorded cache hit rate",
                    cache_type=cache_type,
                    hit_rate=hit_rate)
    
    def record_git_cache_request(self, cache_hit: bool, source_type: str = "unknown"):
        """Record Git hash cache request result."""
        with self._lock:
            result = "hit" if cache_hit else "miss"
            self.git_cache_requests.labels(result=result).inc()
            
            # Buffer for real-time monitoring
            metric = MetricValue(
                name="git_cache_request",
                value=1,
                timestamp=datetime.now(),
                labels={
                    "result": result,
                    "source_type": source_type
                }
            )
            self._metrics_buffer["git_cache_requests"].append(metric)
        
        logger.debug("Recorded Git cache request",
                    cache_hit=cache_hit,
                    source_type=source_type)
    
    def record_git_hash_calculation(self, source_type: str = "file"):
        """Record Git hash calculation."""
        with self._lock:
            self.git_hash_calculations.labels(source_type=source_type).inc()
            
            # Buffer for real-time monitoring
            metric = MetricValue(
                name="git_hash_calculation",
                value=1,
                timestamp=datetime.now(),
                labels={"source_type": source_type}
            )
            self._metrics_buffer["git_hash_calculations"].append(metric)
        
        logger.debug("Recorded Git hash calculation",
                    source_type=source_type)
    
    def record_cache_invalidation(self, reason: str = "manual"):
        """Record cache invalidation event."""
        with self._lock:
            self.cache_invalidations.labels(reason=reason).inc()
            
            # Buffer for real-time monitoring
            metric = MetricValue(
                name="cache_invalidation",
                value=1,
                timestamp=datetime.now(),
                labels={"reason": reason}
            )
            self._metrics_buffer["cache_invalidations"].append(metric)
        
        logger.info("Recorded cache invalidation",
                   reason=reason)
    
    def record_memory_usage(self, component: str, memory_bytes: int):
        """Record memory usage for a component."""
        with self._lock:
            self.memory_usage.labels(component=component).set(memory_bytes)
            
            # Buffer for real-time monitoring
            metric = MetricValue(
                name="memory_usage",
                value=memory_bytes,
                timestamp=datetime.now(),
                labels={"component": component}
            )
            self._metrics_buffer["memory_usage"].append(metric)
        
        logger.debug("Recorded memory usage",
                    component=component,
                    memory_bytes=memory_bytes)
    
    def record_database_connections(self, database_type: str, connection_count: int):
        """Record active database connections."""
        with self._lock:
            self.database_connections.labels(database_type=database_type).set(connection_count)
            
            # Buffer for real-time monitoring
            metric = MetricValue(
                name="database_connections",
                value=connection_count,
                timestamp=datetime.now(),
                labels={"database_type": database_type}
            )
            self._metrics_buffer["database_connections"].append(metric)
        
        logger.debug("Recorded database connections",
                    database_type=database_type,
                    connection_count=connection_count)
    
    def record_error(self, error_type: str, component: str, error_details: Optional[str] = None):
        """Record an error occurrence."""
        with self._lock:
            self.error_count.labels(
                error_type=error_type,
                component=component
            ).inc()
            
            self._error_counts[f"{component}_{error_type}"] += 1
            
            # Buffer for real-time monitoring
            metric = MetricValue(
                name="error",
                value=1,
                timestamp=datetime.now(),
                labels={
                    "error_type": error_type,
                    "component": component
                },
                metadata={"error_details": error_details} if error_details else {}
            )
            self._metrics_buffer["errors"].append(metric)
        
        logger.error("Recorded error",
                    error_type=error_type,
                    component=component,
                    error_details=error_details)
    
    def record_emotional_analysis_time(self, analysis_type: str, duration: float):
        """Record emotional analysis timing."""
        with self._lock:
            self.emotional_analysis_duration.labels(analysis_type=analysis_type).observe(duration)
            
            # Buffer for real-time monitoring
            metric = MetricValue(
                name="emotional_analysis_time",
                value=duration,
                timestamp=datetime.now(),
                labels={"analysis_type": analysis_type}
            )
            self._metrics_buffer["emotional_analysis_time"].append(metric)
        
        logger.debug("Recorded emotional analysis time",
                    analysis_type=analysis_type,
                    duration=duration)
    
    def record_relationship_milestone(self, milestone_type: str):
        """Record a relationship milestone."""
        with self._lock:
            self.relationship_milestones.labels(milestone_type=milestone_type).inc()
            
            # Buffer for real-time monitoring
            metric = MetricValue(
                name="relationship_milestone",
                value=1,
                timestamp=datetime.now(),
                labels={"milestone_type": milestone_type}
            )
            self._metrics_buffer["relationship_milestones"].append(metric)
        
        logger.info("Recorded relationship milestone",
                   milestone_type=milestone_type)
    
    def record_empathy_score(self, response_type: str, empathy_score: float):
        """Record empathy score for a response."""
        with self._lock:
            self.empathy_score.labels(response_type=response_type).observe(empathy_score)
            
            # Buffer for real-time monitoring
            metric = MetricValue(
                name="empathy_score",
                value=empathy_score,
                timestamp=datetime.now(),
                labels={"response_type": response_type}
            )
            self._metrics_buffer["empathy_score"].append(metric)
        
        logger.debug("Recorded empathy score",
                    response_type=response_type,
                    empathy_score=empathy_score)
    
    def get_metrics_summary(self, metric_name: str, time_window: timedelta = timedelta(hours=1)) -> Optional[MetricsSummary]:
        """Get summary statistics for a metric over a time window."""
        with self._lock:
            if metric_name not in self._metrics_buffer:
                return None
            
            cutoff_time = datetime.now() - time_window
            recent_metrics = [
                m for m in self._metrics_buffer[metric_name]
                if m.timestamp >= cutoff_time
            ]
            
            if not recent_metrics:
                return None
            
            values = [m.value for m in recent_metrics]
            error_count = sum(1 for m in recent_metrics if m.labels.get("success") == "False")
            
            values.sort()
            n = len(values)
            
            return MetricsSummary(
                total_count=n,
                average=sum(values) / n,
                min_value=min(values),
                max_value=max(values),
                percentile_95=values[int(0.95 * n)] if n > 0 else 0,
                percentile_99=values[int(0.99 * n)] if n > 0 else 0,
                error_rate=error_count / n if n > 0 else 0
            )
    
    def get_recent_metrics(self, metric_name: str, limit: int = 100) -> List[MetricValue]:
        """Get recent metrics for a specific metric name."""
        with self._lock:
            if metric_name not in self._metrics_buffer:
                return []
            
            return list(self._metrics_buffer[metric_name])[-limit:]
    
    def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        return generate_latest(self.registry).decode('utf-8')
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get overall system health summary."""
        with self._lock:
            # Calculate error rates
            total_operations = sum(len(timings) for timings in self._operation_timings.values())
            total_errors = sum(self._error_counts.values())
            overall_error_rate = total_errors / total_operations if total_operations > 0 else 0
            
            # Get recent performance metrics
            processing_summary = self.get_metrics_summary("processing_time")
            search_summary = self.get_metrics_summary("search_time")
            
            return {
                "timestamp": datetime.now().isoformat(),
                "overall_error_rate": overall_error_rate,
                "total_operations": total_operations,
                "total_errors": total_errors,
                "processing_performance": {
                    "average_duration": processing_summary.average if processing_summary else 0,
                    "p95_duration": processing_summary.percentile_95 if processing_summary else 0,
                    "error_rate": processing_summary.error_rate if processing_summary else 0
                },
                "search_performance": {
                    "average_duration": search_summary.average if search_summary else 0,
                    "p95_duration": search_summary.percentile_95 if search_summary else 0,
                    "error_rate": search_summary.error_rate if search_summary else 0
                },
                "metrics_buffer_sizes": {
                    name: len(buffer) for name, buffer in self._metrics_buffer.items()
                }
            }
    
    def clear_metrics(self, older_than: Optional[timedelta] = None):
        """Clear old metrics from buffers."""
        if older_than is None:
            older_than = timedelta(hours=24)  # Default: keep 24 hours
        
        cutoff_time = datetime.now() - older_than
        
        with self._lock:
            for metric_name, buffer in self._metrics_buffer.items():
                # Keep only recent metrics
                recent_metrics = deque(
                    (m for m in buffer if m.timestamp >= cutoff_time),
                    maxlen=buffer.maxlen
                )
                self._metrics_buffer[metric_name] = recent_metrics
            
            # Clear old operation timings
            for key in list(self._operation_timings.keys()):
                # Keep only recent timings (simplified - in production might want timestamps)
                if len(self._operation_timings[key]) > 1000:
                    self._operation_timings[key] = self._operation_timings[key][-1000:]
        
        logger.info("Cleared old metrics", cutoff_time=cutoff_time.isoformat())