#!/usr/bin/env python3
"""
Performance Monitoring and Benchmarking Script
==============================================

This script monitors the performance of the incident classification system:
- Real-time performance metrics
- Benchmark testing with various load scenarios
- Resource usage monitoring
- Classification accuracy tracking
- Response time analysis
"""

import time
import psutil
import json
import threading
import statistics
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from config import Config
from incident_classifier import IncidentClassifier

class PerformanceMonitor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics = {
            'response_times': [],
            'throughput': [],
            'accuracy_scores': [],
            'memory_usage': [],
            'cpu_usage': [],
            'error_rates': [],
            'timestamps': []
        }
        self.monitoring = False
        
    def start_monitoring(self, interval: float = 1.0):
        """Start real-time monitoring"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=5.0)
        self.logger.info("Performance monitoring stopped")
    
    def _monitor_loop(self, interval: float):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                timestamp = datetime.now()
                
                # System metrics
                memory_percent = psutil.virtual_memory().percent
                cpu_percent = psutil.cpu_percent(interval=0.1)
                
                self.metrics['memory_usage'].append(memory_percent)
                self.metrics['cpu_usage'].append(cpu_percent)
                self.metrics['timestamps'].append(timestamp)
                
                # Keep only last 1000 measurements
                for key in self.metrics:
                    if len(self.metrics[key]) > 1000:
                        self.metrics[key] = self.metrics[key][-1000:]
                
                time.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                time.sleep(interval)
    
    def benchmark_classification_speed(self, 
                                     classifier: IncidentClassifier,
                                     test_descriptions: List[str],
                                     iterations: int = 3) -> Dict:
        """Benchmark classification speed"""
        self.logger.info(f"Benchmarking classification speed with {len(test_descriptions)} samples")
        
        results = {
            'single_classification': [],
            'batch_classification': [],
            'memory_usage': [],
            'detailed_timings': []
        }
        
        for iteration in range(iterations):
            self.logger.info(f"Running iteration {iteration + 1}/{iterations}")
            
            # Measure memory before
            memory_before = psutil.virtual_memory().used / 1024 / 1024  # MB
            
            # Single classification benchmark
            single_times = []
            for description in test_descriptions:
                start_time = time.time()
                result = classifier.classify_incident(description)
                classification_time = time.time() - start_time
                single_times.append(classification_time)
                
                # Track detailed timing
                results['detailed_timings'].append({
                    'description_length': len(description),
                    'processing_time': result['processing_time'],
                    'total_time': classification_time,
                    'confidence': result['confidence'],
                    'status': result['classification_status']
                })
            
            results['single_classification'].append(single_times)
            
            # Batch classification benchmark
            batch_input = [
                {'description': desc, 'incident_id': f'BENCH_{i}'}
                for i, desc in enumerate(test_descriptions)
            ]
            
            start_time = time.time()
            batch_results = classifier.classify_batch(batch_input)
            batch_time = time.time() - start_time
            
            results['batch_classification'].append(batch_time)
            
            # Memory after
            memory_after = psutil.virtual_memory().used / 1024 / 1024  # MB
            results['memory_usage'].append(memory_after - memory_before)
        
        # Calculate statistics
        all_single_times = [t for iteration in results['single_classification'] for t in iteration]
        
        benchmark_stats = {
            'single_classification': {
                'mean_time': statistics.mean(all_single_times),
                'median_time': statistics.median(all_single_times),
                'min_time': min(all_single_times),
                'max_time': max(all_single_times),
                'std_dev': statistics.stdev(all_single_times) if len(all_single_times) > 1 else 0,
                'throughput_per_second': len(test_descriptions) / statistics.mean(results['batch_classification'])
            },
            'batch_classification': {
                'mean_time': statistics.mean(results['batch_classification']),
                'median_time': statistics.median(results['batch_classification']),
                'min_time': min(results['batch_classification']),
                'max_time': max(results['batch_classification']),
                'throughput_per_second': len(test_descriptions) / statistics.mean(results['batch_classification'])
            },
            'memory_usage': {
                'mean_mb': statistics.mean(results['memory_usage']),
                'max_mb': max(results['memory_usage']),
                'min_mb': min(results['memory_usage'])
            },
            'performance_requirements': {
                'meets_response_time': statistics.mean(all_single_times) <= 3.0,  # 3 seconds requirement
                'meets_throughput': len(test_descriptions) / statistics.mean(results['batch_classification']) >= 1.0
            }
        }
        
        results['statistics'] = benchmark_stats
        return results
    
    def load_test_api(self, 
                     base_url: str = "http://localhost:5000",
                     concurrent_users: int = 10,
                     duration_seconds: int = 60,
                     test_descriptions: List[str] = None) -> Dict:
        """Load test the API service"""
        self.logger.info(f"Starting load test: {concurrent_users} users for {duration_seconds}s")
        
        if test_descriptions is None:
            test_descriptions = [
                "مشكلة في تسجيل الدخول",
                "البريد الإلكتروني لا يعمل",
                "مشكلة في الطباعة",
                "Password reset not working",
                "Application crashes frequently"
            ]
        
        results = {
            'response_times': [],
            'error_count': 0,
            'success_count': 0,
            'start_time': datetime.now(),
            'errors': []
        }
        
        def worker_thread():
            """Worker thread for load testing"""
            session = requests.Session()
            end_time = time.time() + duration_seconds
            
            while time.time() < end_time:
                try:
                    description = np.random.choice(test_descriptions)
                    
                    payload = {
                        'description': description,
                        'incident_id': f'LOAD_TEST_{int(time.time() * 1000)}'
                    }
                    
                    start_time = time.time()
                    response = session.post(f"{base_url}/classify", json=payload, timeout=30)
                    response_time = time.time() - start_time
                    
                    results['response_times'].append(response_time)
                    
                    if response.status_code == 200:
                        results['success_count'] += 1
                    else:
                        results['error_count'] += 1
                        results['errors'].append({
                            'status_code': response.status_code,
                            'response': response.text[:200],
                            'timestamp': datetime.now()
                        })
                    
                    # Small delay between requests
                    time.sleep(0.1)
                    
                except Exception as e:
                    results['error_count'] += 1
                    results['errors'].append({
                        'error': str(e),
                        'timestamp': datetime.now()
                    })
        
        # Start worker threads
        threads = []
        for _ in range(concurrent_users):
            thread = threading.Thread(target=worker_thread)
            thread.daemon = True
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Calculate statistics
        if results['response_times']:
            load_test_stats = {
                'total_requests': len(results['response_times']),
                'success_rate': results['success_count'] / (results['success_count'] + results['error_count']),
                'error_rate': results['error_count'] / (results['success_count'] + results['error_count']),
                'mean_response_time': statistics.mean(results['response_times']),
                'median_response_time': statistics.median(results['response_times']),
                'p95_response_time': np.percentile(results['response_times'], 95),
                'p99_response_time': np.percentile(results['response_times'], 99),
                'max_response_time': max(results['response_times']),
                'min_response_time': min(results['response_times']),
                'throughput_rps': len(results['response_times']) / duration_seconds,
                'concurrent_users': concurrent_users,
                'duration_seconds': duration_seconds
            }
        else:
            load_test_stats = {'error': 'No successful requests'}
        
        results['statistics'] = load_test_stats
        results['end_time'] = datetime.now()
        
        return results
    
    def create_performance_report(self, 
                                benchmark_results: Dict,
                                load_test_results: Dict = None,
                                output_dir: str = "results") -> str:
        """Create comprehensive performance report"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"{output_dir}/performance_report_{timestamp}.json"
        
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024,
                'python_version': f"{psutil.Process().memory_info()}",
            },
            'benchmark_results': benchmark_results,
            'monitoring_metrics': {
                'avg_memory_usage': statistics.mean(self.metrics['memory_usage']) if self.metrics['memory_usage'] else 0,
                'avg_cpu_usage': statistics.mean(self.metrics['cpu_usage']) if self.metrics['cpu_usage'] else 0,
                'peak_memory_usage': max(self.metrics['memory_usage']) if self.metrics['memory_usage'] else 0,
                'peak_cpu_usage': max(self.metrics['cpu_usage']) if self.metrics['cpu_usage'] else 0,
            }
        }
        
        if load_test_results:
            report['load_test_results'] = load_test_results
        
        # Save report
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        self.logger.info(f"Performance report saved to: {report_file}")
        return report_file
    
    def plot_performance_graphs(self, output_dir: str = "results"):
        """Create performance visualization graphs"""
        if not self.metrics['timestamps']:
            self.logger.warning("No monitoring data available for plotting")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        timestamps = self.metrics['timestamps']
        
        # Memory usage
        ax1.plot(timestamps, self.metrics['memory_usage'], 'b-', linewidth=1)
        ax1.set_title('Memory Usage Over Time')
        ax1.set_ylabel('Memory Usage (%)')
        ax1.grid(True)
        
        # CPU usage
        ax2.plot(timestamps, self.metrics['cpu_usage'], 'r-', linewidth=1)
        ax2.set_title('CPU Usage Over Time')
        ax2.set_ylabel('CPU Usage (%)')
        ax2.grid(True)
        
        # Response times (if available)
        if self.metrics['response_times']:
            ax3.hist(self.metrics['response_times'], bins=50, alpha=0.7)
            ax3.set_title('Response Time Distribution')
            ax3.set_xlabel('Response Time (seconds)')
            ax3.set_ylabel('Frequency')
            ax3.grid(True)
        
        # Throughput (if available)
        if self.metrics['throughput']:
            ax4.plot(timestamps[-len(self.metrics['throughput']):], self.metrics['throughput'], 'g-', linewidth=1)
            ax4.set_title('Throughput Over Time')
            ax4.set_ylabel('Requests/Second')
            ax4.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = f"{output_dir}/performance_graphs_{timestamp}.png"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Performance graphs saved to: {plot_file}")
        return plot_file

def run_comprehensive_benchmark():
    """Run comprehensive performance benchmark"""
    
    # Setup
    monitor = PerformanceMonitor()
    monitor.start_monitoring()
    
    # Test descriptions in Arabic and English
    test_descriptions = [
        "مشكلة في تسجيل الدخول إلى النظام",
        "البريد الإلكتروني لا يعمل بشكل صحيح", 
        "البرنامج يتوقف عند فتح ملف كبير",
        "مشكلة في الاتصال بالشبكة",
        "كلمة المرور لا تعمل",
        "Login system is not responding",
        "Password reset functionality broken",
        "Application crashes when opening large files",
        "Network connectivity issues",
        "Email server connection failed",
        "Printer driver installation problem",
        "Database connection timeout",
        "File upload feature not working",
        "System running very slowly",
        "Unable to access shared folders"
    ]
    
    try:
        print("🚀 Starting Comprehensive Performance Benchmark")
        print("=" * 60)
        
        # Initialize classifier
        print("Initializing classifier...")
        classifier = IncidentClassifier(use_existing_index=True)
        if not classifier.initialize(sample_size=1000):  # Use sample for faster testing
            print("❌ Failed to initialize classifier")
            return
        
        print("✅ Classifier initialized")
        
        # Run classification benchmark
        print("\n📊 Running classification benchmark...")
        benchmark_results = monitor.benchmark_classification_speed(
            classifier, test_descriptions, iterations=3
        )
        
        # Print benchmark results
        stats = benchmark_results['statistics']
        print(f"\n📈 BENCHMARK RESULTS:")
        print(f"Average response time: {stats['single_classification']['mean_time']:.3f}s")
        print(f"Throughput: {stats['single_classification']['throughput_per_second']:.1f} requests/second")
        print(f"Memory usage: {stats['memory_usage']['mean_mb']:.1f} MB")
        print(f"Meets requirements: {stats['performance_requirements']}")
        
        # Run API load test (if API is running)
        load_test_results = None
        try:
            print("\n🔄 Running API load test...")
            load_test_results = monitor.load_test_api(
                concurrent_users=5,
                duration_seconds=30,
                test_descriptions=test_descriptions[:5]
            )
            
            load_stats = load_test_results['statistics']
            print(f"Load test - Success rate: {load_stats['success_rate']:.1%}")
            print(f"Load test - Mean response time: {load_stats['mean_response_time']:.3f}s")
            print(f"Load test - Throughput: {load_stats['throughput_rps']:.1f} RPS")
            
        except Exception as e:
            print(f"⚠️ API load test skipped: {e}")
        
        # Generate report
        print("\n📋 Generating performance report...")
        report_file = monitor.create_performance_report(
            benchmark_results, load_test_results
        )
        
        # Create graphs
        print("📊 Creating performance graphs...")
        plot_file = monitor.plot_performance_graphs()
        
        print(f"\n✅ Benchmark complete!")
        print(f"📄 Report: {report_file}")
        print(f"📊 Graphs: {plot_file}")
        
    finally:
        monitor.stop_monitoring()

if __name__ == "__main__":
    import argparse
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description="Performance monitoring and benchmarking")
    parser.add_argument('--benchmark', action='store_true', help='Run comprehensive benchmark')
    parser.add_argument('--monitor', type=int, help='Monitor for specified seconds')
    parser.add_argument('--load-test', action='store_true', help='Run API load test only')
    
    args = parser.parse_args()
    
    if args.benchmark:
        run_comprehensive_benchmark()
    elif args.monitor:
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        print(f"Monitoring for {args.monitor} seconds...")
        time.sleep(args.monitor)
        monitor.stop_monitoring()
        monitor.create_performance_report({}, None)
    elif args.load_test:
        monitor = PerformanceMonitor()
        results = monitor.load_test_api()
        print("Load test results:", json.dumps(results['statistics'], indent=2))
    else:
        print("Use --benchmark, --monitor <seconds>, or --load-test")
