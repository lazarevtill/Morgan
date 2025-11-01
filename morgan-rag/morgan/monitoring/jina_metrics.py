"""
Specialized metrics collection for Jina AI components in Morgan RAG.

Provides comprehensive metrics tracking for Jina AI models, web scraping,
multimodal processing, and background tasks with performance optimization insights.
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import structlog

from .metrics_collector import MetricsCollector

logger = structlog.get_logger(__name__)


class JinaModelType(Enum):
    """Jina AI model types for metrics tracking."""
    EMBEDDINGS_V4 = "jina-embeddings-v4"
    CODE_EMBEDDINGS = "jina-code-embeddings-1.5b"
    CLIP_V2 = "jina-clip-v2"
    RERANKER_V3 = "jina-reranker-v3"
    RERANKER_V2_MULTILINGUAL = "jina-reranker-v2-base-multilingual"
    READER_LM_V2 = "jina-reader-lm-v2"


class JinaOperationType(Enum):
    """Types of Jina AI operations."""
    EMBEDDING_GENERATION = "embedding_generation"
    RESULT_RERANKING = "result_reranking"
    WEB_SCRAPING = "web_scraping"
    MULTIMODAL_PROCESSING = "multimodal_processing"
    CODE_ANALYSIS = "code_analysis"


@dataclass
class JinaModelMetrics:
    """Metrics for a specific Jina AI model."""
    model_type: JinaModelType
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_processing_time: float = 0.0
    total_tokens_processed: int = 0
    total_batch_size: int = 0
    average_latency: float = 0.0
    p95_latency: float = 0.0
    p99_latency: float = 0.0
    error_rate: float = 0.0
    throughput_per_second: float = 0.0
    last_used: Optional[datetime] = None
    model_load_time: Optional[float] = None
    memory_usage_mb: Optional[float] = None


@dataclass
class JinaPerformanceMetrics:
    """Performance metrics for Jina AI operations."""
    operation_type: JinaOperationType
    model_type: JinaModelType
    batch_size: int
    input_size: int
    processing_time: float
    tokens_processed: int
    success: bool
    error_type: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JinaQualityMetrics:
    """Quality metrics for Jina AI operations."""
    operation_type: JinaOperationType
    model_type: JinaModelType
    accuracy_score: Optional[float] = None
    relevance_improvement: Optional[float] = None
    user_satisfaction: Optional[float] = None
    confidence_score: Optional[float] = None
    fallback_used: bool = False
    timestamp: datetime = field(default_factory=datetime.now)


class JinaMetricsCollector:
    """
    Specialized metrics collector for Jina AI components.
    
    Tracks performance, quality, and operational metrics for all Jina AI
    models and services with detailed analysis and optimization insights.
    """
    
    def __init__(self, base_metrics_collector: MetricsCollector):
        self.base_metrics_collector = base_metrics_collector
        
        # Model-specific metrics
        self.model_metrics: Dict[JinaModelType, JinaModelMetrics] = {}
        
        # Performance tracking
        self.performance_history: deque = deque(maxlen=10000)
        self.quality_history: deque = deque(maxlen=5000)
        
        # Latency tracking by model and batch size
        self.latency_by_model: Dict[str, List[float]] = defaultdict(list)
        self.latency_by_batch_size: Dict[int, List[float]] = defaultdict(list)
        
        # Throughput tracking
        self.throughput_windows: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Error tracking
        self.error_patterns: Dict[str, int] = defaultdict(int)
        self.model_health: Dict[JinaModelType, bool] = {}
        
        # Initialize model metrics
        self._initialize_model_metrics()
        
        logger.info("JinaMetricsCollector initialized")
    
    def _initialize_model_metrics(self):
        """Initialize metrics for all Jina models."""
        for model_type in JinaModelType:
            self.model_metrics[model_type] = JinaModelMetrics(model_type=model_type)
            self.model_health[model_type] = True
    
    def record_model_performance(self, 
                                operation_type: JinaOperationType,
                                model_type: JinaModelType,
                                processing_time: float,
                                batch_size: int = 1,
                                input_size: int = 0,
                                tokens_processed: int = 0,
                                success: bool = True,
                                error_type: Optional[str] = None,
                                metadata: Optional[Dict[str, Any]] = None):
        """Record performance metrics for a Jina AI operation."""
        
        # Create performance record
        performance_record = JinaPerformanceMetrics(
            operation_type=operation_type,
            model_type=model_type,
            batch_size=batch_size,
            input_size=input_size,
            processing_time=processing_time,
            tokens_processed=tokens_processed,
            success=success,
            error_type=error_type,
            metadata=metadata or {}
        )
        
        # Store in history
        self.performance_history.append(performance_record)
        
        # Update model metrics
        model_metrics = self.model_metrics[model_type]
        model_metrics.total_requests += 1
        model_metrics.total_processing_time += processing_time
        model_metrics.total_tokens_processed += tokens_processed
        model_metrics.total_batch_size += batch_size
        model_metrics.last_used = datetime.now()
        
        if success:
            model_metrics.successful_requests += 1
        else:
            model_metrics.failed_requests += 1
            if error_type:
                self.error_patterns[f"{model_type.value}_{error_type}"] += 1
        
        # Update calculated metrics
        self._update_model_calculated_metrics(model_type)
        
        # Update latency tracking
        model_key = f"{model_type.value}_{operation_type.value}"
        self.latency_by_model[model_key].append(processing_time)
        self.latency_by_batch_size[batch_size].append(processing_time)
        
        # Keep only recent latency data
        if len(self.latency_by_model[model_key]) > 1000:
            self.latency_by_model[model_key] = self.latency_by_model[model_key][-1000:]
        
        # Update throughput tracking
        throughput_key = f"{model_type.value}_{operation_type.value}"
        self.throughput_windows[throughput_key].append((datetime.now(), batch_size))
        
        # Update model health
        self._update_model_health(model_type)
        
        # Record in base metrics collector
        self.base_metrics_collector.record_embedding_time(
            model_type=model_type.value,
            scale=operation_type.value,
            duration=processing_time
        )
        
        if not success and error_type:
            self.base_metrics_collector.record_error(
                error_type=error_type,
                component=f"jina_{model_type.value}",
                error_details=f"Operation: {operation_type.value}"
            )
        
        logger.debug("Recorded Jina model performance",
                    model_type=model_type.value,
                    operation_type=operation_type.value,
                    processing_time=processing_time,
                    batch_size=batch_size,
                    success=success)
    
    def record_model_quality(self,
                           operation_type: JinaOperationType,
                           model_type: JinaModelType,
                           accuracy_score: Optional[float] = None,
                           relevance_improvement: Optional[float] = None,
                           user_satisfaction: Optional[float] = None,
                           confidence_score: Optional[float] = None,
                           fallback_used: bool = False):
        """Record quality metrics for a Jina AI operation."""
        
        quality_record = JinaQualityMetrics(
            operation_type=operation_type,
            model_type=model_type,
            accuracy_score=accuracy_score,
            relevance_improvement=relevance_improvement,
            user_satisfaction=user_satisfaction,
            confidence_score=confidence_score,
            fallback_used=fallback_used
        )
        
        self.quality_history.append(quality_record)
        
        # Record in base metrics collector
        if relevance_improvement is not None:
            self.base_metrics_collector.record_search_relevance(
                search_type=f"jina_{operation_type.value}",
                relevance_score=relevance_improvement
            )
        
        if user_satisfaction is not None:
            self.base_metrics_collector.record_user_satisfaction(
                interaction_type=f"jina_{operation_type.value}",
                satisfaction_score=user_satisfaction
            )
        
        logger.debug("Recorded Jina model quality",
                    model_type=model_type.value,
                    operation_type=operation_type.value,
                    accuracy_score=accuracy_score,
                    relevance_improvement=relevance_improvement)
    
    def record_model_load_time(self, model_type: JinaModelType, load_time: float):
        """Record model loading time."""
        self.model_metrics[model_type].model_load_time = load_time
        
        logger.info("Recorded model load time",
                   model_type=model_type.value,
                   load_time=load_time)
    
    def record_model_memory_usage(self, model_type: JinaModelType, memory_mb: float):
        """Record model memory usage."""
        self.model_metrics[model_type].memory_usage_mb = memory_mb
        
        self.base_metrics_collector.record_memory_usage(
            component=f"jina_{model_type.value}",
            memory_bytes=int(memory_mb * 1024 * 1024)
        )
        
        logger.debug("Recorded model memory usage",
                    model_type=model_type.value,
                    memory_mb=memory_mb)
    
    def _update_model_calculated_metrics(self, model_type: JinaModelType):
        """Update calculated metrics for a model."""
        metrics = self.model_metrics[model_type]
        
        if metrics.total_requests > 0:
            # Calculate error rate
            metrics.error_rate = metrics.failed_requests / metrics.total_requests
            
            # Calculate average latency
            if metrics.successful_requests > 0:
                metrics.average_latency = metrics.total_processing_time / metrics.successful_requests
            
            # Calculate throughput (requests per second over last minute)
            throughput_key = f"{model_type.value}_*"  # Aggregate across operations
            recent_requests = [
                (timestamp, batch_size) for timestamp, batch_size in self.throughput_windows.get(throughput_key, [])
                if timestamp >= datetime.now() - timedelta(minutes=1)
            ]
            
            if recent_requests:
                total_batch_size = sum(batch_size for _, batch_size in recent_requests)
                metrics.throughput_per_second = total_batch_size / 60.0
        
        # Calculate percentile latencies
        model_latencies = []
        for key, latencies in self.latency_by_model.items():
            if model_type.value in key:
                model_latencies.extend(latencies)
        
        if model_latencies:
            sorted_latencies = sorted(model_latencies)
            n = len(sorted_latencies)
            metrics.p95_latency = sorted_latencies[int(0.95 * n)] if n > 0 else 0
            metrics.p99_latency = sorted_latencies[int(0.99 * n)] if n > 0 else 0
    
    def _update_model_health(self, model_type: JinaModelType):
        """Update model health status based on recent performance."""
        metrics = self.model_metrics[model_type]
        
        # Consider model unhealthy if error rate > 10% and has recent requests
        if metrics.total_requests >= 10:
            self.model_health[model_type] = metrics.error_rate < 0.1
        else:
            # Not enough data, assume healthy
            self.model_health[model_type] = True
    
    def get_model_metrics(self, model_type: Optional[JinaModelType] = None) -> Dict[str, JinaModelMetrics]:
        """Get metrics for specific model or all models."""
        if model_type:
            return {model_type.value: self.model_metrics[model_type]}
        else:
            return {mt.value: metrics for mt, metrics in self.model_metrics.items()}
    
    def get_model_health_status(self) -> Dict[str, bool]:
        """Get health status for all models."""
        return {mt.value: health for mt, health in self.model_health.items()}
    
    def get_performance_summary(self, 
                              time_window: timedelta = timedelta(hours=1)) -> Dict[str, Any]:
        """Get performance summary for specified time window."""
        cutoff_time = datetime.now() - time_window
        
        # Filter recent performance data
        recent_performance = [
            record for record in self.performance_history
            if record.timestamp >= cutoff_time
        ]
        
        if not recent_performance:
            return {"message": "No performance data in time window"}
        
        # Aggregate by model type
        model_stats = defaultdict(lambda: {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_processing_time": 0.0,
            "total_tokens": 0,
            "latencies": []
        })
        
        for record in recent_performance:
            stats = model_stats[record.model_type.value]
            stats["total_requests"] += 1
            stats["total_processing_time"] += record.processing_time
            stats["total_tokens"] += record.tokens_processed
            stats["latencies"].append(record.processing_time)
            
            if record.success:
                stats["successful_requests"] += 1
            else:
                stats["failed_requests"] += 1
        
        # Calculate summary statistics
        summary = {}
        for model_type, stats in model_stats.items():
            if stats["latencies"]:
                sorted_latencies = sorted(stats["latencies"])
                n = len(sorted_latencies)
                
                summary[model_type] = {
                    "total_requests": stats["total_requests"],
                    "success_rate": stats["successful_requests"] / stats["total_requests"],
                    "error_rate": stats["failed_requests"] / stats["total_requests"],
                    "average_latency": stats["total_processing_time"] / len(stats["latencies"]),
                    "p50_latency": sorted_latencies[int(0.5 * n)] if n > 0 else 0,
                    "p95_latency": sorted_latencies[int(0.95 * n)] if n > 0 else 0,
                    "p99_latency": sorted_latencies[int(0.99 * n)] if n > 0 else 0,
                    "total_tokens_processed": stats["total_tokens"],
                    "tokens_per_second": stats["total_tokens"] / time_window.total_seconds() if time_window.total_seconds() > 0 else 0
                }
        
        return summary
    
    def get_quality_summary(self, 
                          time_window: timedelta = timedelta(hours=1)) -> Dict[str, Any]:
        """Get quality summary for specified time window."""
        cutoff_time = datetime.now() - time_window
        
        # Filter recent quality data
        recent_quality = [
            record for record in self.quality_history
            if record.timestamp >= cutoff_time
        ]
        
        if not recent_quality:
            return {"message": "No quality data in time window"}
        
        # Aggregate by model and operation type
        quality_stats = defaultdict(lambda: {
            "accuracy_scores": [],
            "relevance_improvements": [],
            "user_satisfactions": [],
            "confidence_scores": [],
            "fallback_count": 0,
            "total_operations": 0
        })
        
        for record in recent_quality:
            key = f"{record.model_type.value}_{record.operation_type.value}"
            stats = quality_stats[key]
            
            stats["total_operations"] += 1
            
            if record.accuracy_score is not None:
                stats["accuracy_scores"].append(record.accuracy_score)
            if record.relevance_improvement is not None:
                stats["relevance_improvements"].append(record.relevance_improvement)
            if record.user_satisfaction is not None:
                stats["user_satisfactions"].append(record.user_satisfaction)
            if record.confidence_score is not None:
                stats["confidence_scores"].append(record.confidence_score)
            if record.fallback_used:
                stats["fallback_count"] += 1
        
        # Calculate summary statistics
        summary = {}
        for key, stats in quality_stats.items():
            summary[key] = {
                "total_operations": stats["total_operations"],
                "fallback_rate": stats["fallback_count"] / stats["total_operations"] if stats["total_operations"] > 0 else 0,
                "average_accuracy": sum(stats["accuracy_scores"]) / len(stats["accuracy_scores"]) if stats["accuracy_scores"] else None,
                "average_relevance_improvement": sum(stats["relevance_improvements"]) / len(stats["relevance_improvements"]) if stats["relevance_improvements"] else None,
                "average_user_satisfaction": sum(stats["user_satisfactions"]) / len(stats["user_satisfactions"]) if stats["user_satisfactions"] else None,
                "average_confidence": sum(stats["confidence_scores"]) / len(stats["confidence_scores"]) if stats["confidence_scores"] else None
            }
        
        return summary
    
    def get_optimization_insights(self) -> Dict[str, Any]:
        """Generate optimization insights based on collected metrics."""
        insights = {
            "model_performance": {},
            "batch_size_optimization": {},
            "error_patterns": {},
            "recommendations": []
        }
        
        # Model performance insights
        for model_type, metrics in self.model_metrics.items():
            if metrics.total_requests > 0:
                insights["model_performance"][model_type.value] = {
                    "health_status": "healthy" if self.model_health[model_type] else "unhealthy",
                    "error_rate": metrics.error_rate,
                    "average_latency": metrics.average_latency,
                    "p95_latency": metrics.p95_latency,
                    "throughput": metrics.throughput_per_second,
                    "memory_usage_mb": metrics.memory_usage_mb,
                    "load_time": metrics.model_load_time
                }
                
                # Generate recommendations
                if metrics.error_rate > 0.05:
                    insights["recommendations"].append(
                        f"High error rate ({metrics.error_rate:.1%}) for {model_type.value}. "
                        "Consider implementing circuit breakers or fallback mechanisms."
                    )
                
                if metrics.p95_latency > 5.0:
                    insights["recommendations"].append(
                        f"High P95 latency ({metrics.p95_latency:.2f}s) for {model_type.value}. "
                        "Consider optimizing batch sizes or model configuration."
                    )
                
                if metrics.memory_usage_mb and metrics.memory_usage_mb > 2000:
                    insights["recommendations"].append(
                        f"High memory usage ({metrics.memory_usage_mb:.0f}MB) for {model_type.value}. "
                        "Consider model quantization or memory optimization."
                    )
        
        # Batch size optimization insights
        batch_performance = {}
        for batch_size, latencies in self.latency_by_batch_size.items():
            if len(latencies) >= 10:  # Sufficient data
                avg_latency = sum(latencies) / len(latencies)
                throughput = batch_size / avg_latency  # items per second
                batch_performance[batch_size] = {
                    "average_latency": avg_latency,
                    "throughput": throughput,
                    "sample_count": len(latencies)
                }
        
        if batch_performance:
            # Find optimal batch size (highest throughput)
            optimal_batch = max(batch_performance.items(), key=lambda x: x[1]["throughput"])
            insights["batch_size_optimization"] = {
                "optimal_batch_size": optimal_batch[0],
                "optimal_throughput": optimal_batch[1]["throughput"],
                "batch_performance": batch_performance
            }
            
            insights["recommendations"].append(
                f"Optimal batch size appears to be {optimal_batch[0]} "
                f"with throughput of {optimal_batch[1]['throughput']:.2f} items/second."
            )
        
        # Error pattern analysis
        if self.error_patterns:
            insights["error_patterns"] = dict(self.error_patterns)
            
            # Find most common errors
            most_common_error = max(self.error_patterns.items(), key=lambda x: x[1])
            insights["recommendations"].append(
                f"Most common error: {most_common_error[0]} ({most_common_error[1]} occurrences). "
                "Focus on addressing this error pattern first."
            )
        
        return insights
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time metrics for dashboard display."""
        current_time = datetime.now()
        
        # Get metrics from last 5 minutes
        recent_window = timedelta(minutes=5)
        recent_performance = [
            record for record in self.performance_history
            if record.timestamp >= current_time - recent_window
        ]
        
        # Calculate real-time statistics
        if recent_performance:
            total_requests = len(recent_performance)
            successful_requests = sum(1 for r in recent_performance if r.success)
            avg_latency = sum(r.processing_time for r in recent_performance) / total_requests
            
            # Requests per minute
            rpm = total_requests / 5.0  # 5-minute window
            
            # Current error rate
            error_rate = (total_requests - successful_requests) / total_requests
        else:
            total_requests = successful_requests = avg_latency = rpm = error_rate = 0
        
        return {
            "timestamp": current_time.isoformat(),
            "requests_per_minute": rpm,
            "success_rate": successful_requests / total_requests if total_requests > 0 else 0,
            "error_rate": error_rate,
            "average_latency_5min": avg_latency,
            "total_requests_5min": total_requests,
            "model_health": self.get_model_health_status(),
            "active_models": [
                model_type.value for model_type, metrics in self.model_metrics.items()
                if metrics.last_used and metrics.last_used >= current_time - timedelta(minutes=10)
            ]
        }
    
    def export_metrics_for_prometheus(self) -> Dict[str, Any]:
        """Export metrics in format suitable for Prometheus."""
        prometheus_metrics = {}
        
        for model_type, metrics in self.model_metrics.items():
            model_name = model_type.value.replace("-", "_")
            
            prometheus_metrics.update({
                f"jina_model_requests_total_{model_name}": metrics.total_requests,
                f"jina_model_requests_successful_{model_name}": metrics.successful_requests,
                f"jina_model_requests_failed_{model_name}": metrics.failed_requests,
                f"jina_model_error_rate_{model_name}": metrics.error_rate,
                f"jina_model_average_latency_seconds_{model_name}": metrics.average_latency,
                f"jina_model_p95_latency_seconds_{model_name}": metrics.p95_latency,
                f"jina_model_p99_latency_seconds_{model_name}": metrics.p99_latency,
                f"jina_model_throughput_per_second_{model_name}": metrics.throughput_per_second,
                f"jina_model_tokens_processed_total_{model_name}": metrics.total_tokens_processed,
                f"jina_model_health_{model_name}": 1 if self.model_health[model_type] else 0
            })
            
            if metrics.memory_usage_mb is not None:
                prometheus_metrics[f"jina_model_memory_usage_mb_{model_name}"] = metrics.memory_usage_mb
            
            if metrics.model_load_time is not None:
                prometheus_metrics[f"jina_model_load_time_seconds_{model_name}"] = metrics.model_load_time
        
        return prometheus_metrics
    
    def clear_old_metrics(self, older_than: timedelta = timedelta(hours=24)):
        """Clear old metrics to manage memory usage."""
        cutoff_time = datetime.now() - older_than
        
        # Clear old performance history
        recent_performance = deque(
            (record for record in self.performance_history if record.timestamp >= cutoff_time),
            maxlen=self.performance_history.maxlen
        )
        self.performance_history = recent_performance
        
        # Clear old quality history
        recent_quality = deque(
            (record for record in self.quality_history if record.timestamp >= cutoff_time),
            maxlen=self.quality_history.maxlen
        )
        self.quality_history = recent_quality
        
        # Clear old latency data
        for key in list(self.latency_by_model.keys()):
            # Keep only recent 1000 entries per model
            if len(self.latency_by_model[key]) > 1000:
                self.latency_by_model[key] = self.latency_by_model[key][-1000:]
        
        logger.info("Cleared old Jina metrics", cutoff_time=cutoff_time.isoformat())


if __name__ == "__main__":
    # Demo Jina metrics collection
    print("📊 Jina AI Metrics Collection Demo")
    print("=" * 40)
    
    # Create base metrics collector
    from .metrics_collector import MetricsCollector
    base_collector = MetricsCollector()
    
    # Create Jina metrics collector
    jina_collector = JinaMetricsCollector(base_collector)
    
    # Simulate some operations
    import random
    
    for i in range(50):
        # Simulate embedding operations
        processing_time = random.uniform(0.1, 2.0)
        batch_size = random.choice([1, 8, 16, 32])
        success = random.random() > 0.1  # 90% success rate
        
        jina_collector.record_model_performance(
            operation_type=JinaOperationType.EMBEDDING_GENERATION,
            model_type=JinaModelType.EMBEDDINGS_V4,
            processing_time=processing_time,
            batch_size=batch_size,
            tokens_processed=batch_size * 100,
            success=success,
            error_type="timeout" if not success else None
        )
        
        # Simulate quality metrics
        if success:
            jina_collector.record_model_quality(
                operation_type=JinaOperationType.EMBEDDING_GENERATION,
                model_type=JinaModelType.EMBEDDINGS_V4,
                accuracy_score=random.uniform(0.8, 0.95),
                confidence_score=random.uniform(0.7, 0.9)
            )
    
    # Get performance summary
    performance_summary = jina_collector.get_performance_summary()
    print("Performance Summary:")
    for model, stats in performance_summary.items():
        print(f"  {model}:")
        print(f"    Success Rate: {stats['success_rate']:.1%}")
        print(f"    Average Latency: {stats['average_latency']:.3f}s")
        print(f"    P95 Latency: {stats['p95_latency']:.3f}s")
        print(f"    Tokens/sec: {stats['tokens_per_second']:.1f}")
    
    # Get optimization insights
    insights = jina_collector.get_optimization_insights()
    print(f"\nOptimization Insights:")
    print(f"  Recommendations: {len(insights['recommendations'])}")
    for rec in insights['recommendations']:
        print(f"    - {rec}")
    
    # Get real-time metrics
    real_time = jina_collector.get_real_time_metrics()
    print(f"\nReal-time Metrics:")
    print(f"  Requests/min: {real_time['requests_per_minute']:.1f}")
    print(f"  Success Rate: {real_time['success_rate']:.1%}")
    print(f"  Average Latency: {real_time['average_latency_5min']:.3f}s")
    
    print("\n" + "=" * 40)
    print("Jina AI metrics collection demo completed!")