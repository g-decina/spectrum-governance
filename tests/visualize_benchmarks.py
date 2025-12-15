"""
Benchmark Visualization Script
===============================

Generates charts and plots from benchmark results JSON.

Usage:
    python tests/visualize_benchmarks.py results.json --output charts/
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any

try:
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available, visualization disabled")


def load_results(json_path: str) -> Dict[str, Any]:
    """Load benchmark results from JSON."""
    with open(json_path, 'r') as f:
        return json.load(f)


def plot_speedup_comparison(results: List[Dict], output_dir: Path):
    """Plot speedup comparison across attacks."""
    if not MATPLOTLIB_AVAILABLE:
        return

    attacks = [r['attack_name'] for r in results]
    onnx_gains = [r['onnx_speedup_art'] or 0 for r in results]
    rust_gains = [r['rust_1core_speedup'] or 0 for r in results]
    best_speedups = [r['best_speedup'] or 0 for r in results]

    x = np.arange(len(attacks))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, onnx_gains, width, label='ONNX Optimization', color='#2ecc71')
    ax.bar(x, rust_gains, width, label='Rust vs ART (1 core)', color='#3498db')
    ax.bar(x + width, best_speedups, width, label='Best (Rust 16c + ONNX)', color='#e74c3c')

    ax.set_xlabel('Attack', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup (x)', fontsize=12, fontweight='bold')
    ax.set_title('Performance Gains: ONNX + Rust Backend', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(attacks)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'speedup_comparison.png', dpi=300)
    print(f"Saved: {output_dir / 'speedup_comparison.png'}")
    plt.close()


def plot_wall_time_comparison(results: List[Dict], output_dir: Path):
    """Plot absolute wall times for all configurations."""
    if not MATPLOTLIB_AVAILABLE:
        return

    attacks = [r['attack_name'] for r in results]

    # Extract times
    art_times = [r['art_1core']['wall_time'] for r in results]
    art_onnx_times = [r['art_1core_onnx']['wall_time'] if r['art_1core_onnx'] else 0 for r in results]
    rust_1c_times = [r['rust_1core']['wall_time'] if r['rust_1core'] else 0 for r in results]
    rust_16c_onnx_times = [r['rust_16core_onnx']['wall_time'] if r['rust_16core_onnx'] else 0 for r in results]

    x = np.arange(len(attacks))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - 1.5*width, art_times, width, label='ART (1 core)', color='#95a5a6')
    ax.bar(x - 0.5*width, art_onnx_times, width, label='ART + ONNX (1 core)', color='#7f8c8d')
    ax.bar(x + 0.5*width, rust_1c_times, width, label='Rust (1 core)', color='#3498db')
    ax.bar(x + 1.5*width, rust_16c_onnx_times, width, label='Rust + ONNX (16 cores)', color='#e74c3c')

    ax.set_xlabel('Attack', fontsize=12, fontweight='bold')
    ax.set_ylabel('Wall Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Absolute Performance: Wall Time Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(attacks)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'wall_time_comparison.png', dpi=300)
    print(f"Saved: {output_dir / 'wall_time_comparison.png'}")
    plt.close()


def plot_parallel_efficiency(results: List[Dict], output_dir: Path):
    """Plot parallel efficiency for Rust backend."""
    if not MATPLOTLIB_AVAILABLE:
        return

    attacks = [r['attack_name'] for r in results if r['parallel_efficiency']]
    efficiencies = [r['parallel_efficiency'] * 100 for r in results if r['parallel_efficiency']]

    if not attacks:
        print("No parallel efficiency data available")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(attacks, efficiencies, color='#9b59b6')

    # Add horizontal line at 100%
    ax.axhline(y=100, color='r', linestyle='--', label='Perfect Scaling (100%)')

    # Color code bars: green if > 70%, yellow if 50-70%, red if < 50%
    for i, (bar, eff) in enumerate(zip(bars, efficiencies)):
        if eff >= 70:
            bar.set_color('#2ecc71')
        elif eff >= 50:
            bar.set_color('#f39c12')
        else:
            bar.set_color('#e74c3c')

    ax.set_xlabel('Attack', fontsize=12, fontweight='bold')
    ax.set_ylabel('Parallel Efficiency (%)', fontsize=12, fontweight='bold')
    ax.set_title('Parallel Efficiency: Rust (16 cores)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'parallel_efficiency.png', dpi=300)
    print(f"Saved: {output_dir / 'parallel_efficiency.png'}")
    plt.close()


def plot_queries_comparison(results: List[Dict], output_dir: Path):
    """Plot queries used comparison."""
    if not MATPLOTLIB_AVAILABLE:
        return

    attacks = [r['attack_name'] for r in results]
    art_queries = [r['art_1core']['queries_used'] for r in results]
    rust_queries = [r['rust_16core_onnx']['queries_used'] if r['rust_16core_onnx'] else 0 for r in results]

    x = np.arange(len(attacks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, art_queries, width, label='ART', color='#95a5a6')
    ax.bar(x + width/2, rust_queries, width, label='Rust + ONNX', color='#e74c3c')

    ax.set_xlabel('Attack', fontsize=12, fontweight='bold')
    ax.set_ylabel('Queries Used', fontsize=12, fontweight='bold')
    ax.set_title('Query Efficiency Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(attacks)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'queries_comparison.png', dpi=300)
    print(f"Saved: {output_dir / 'queries_comparison.png'}")
    plt.close()


def generate_summary_report(data: Dict[str, Any], output_dir: Path):
    """Generate markdown summary report."""
    results = data['results']
    system = data['system']
    dataset = data['dataset']

    report = []
    report.append("# Spectrum-Red Attack Performance Benchmark Results\n")
    report.append("## System Configuration\n")
    report.append(f"- CPU Cores: {system['cpu_count']}\n")
    report.append(f"- Rust Available: {'✓' if system['rust_available'] else '✗'}\n")
    report.append(f"- ART Available: {'✓' if system['art_available'] else '✗'}\n")
    report.append("\n## Dataset\n")
    report.append(f"- Features: {dataset['n_features']}\n")
    report.append(f"- Training Samples: {dataset['n_samples']}\n")
    report.append(f"- Test Samples: {dataset['n_test']}\n")

    report.append("\n## Performance Summary\n")
    report.append("| Attack | ART (1c) | ART+ONNX | Rust (1c) | Rust (16c)+ONNX | ONNX Gain | Rust Gain | Best Speedup |\n")
    report.append("|--------|----------|----------|-----------|-----------------|-----------|-----------|-------------|\n")

    for r in results:
        art_time = f"{r['art_1core']['wall_time']:.2f}s"
        art_onnx_time = f"{r['art_1core_onnx']['wall_time']:.2f}s" if r['art_1core_onnx'] else "N/A"
        rust_1c_time = f"{r['rust_1core']['wall_time']:.2f}s" if r['rust_1core'] else "N/A"
        rust_16c_onnx_time = f"{r['rust_16core_onnx']['wall_time']:.2f}s" if r['rust_16core_onnx'] else "N/A"
        onnx_gain = f"{r['onnx_speedup_art']:.1f}x" if r['onnx_speedup_art'] else "N/A"
        rust_gain = f"{r['rust_1core_speedup']:.1f}x" if r['rust_1core_speedup'] else "N/A"
        best_speedup = f"**{r['best_speedup']:.1f}x**" if r['best_speedup'] else "N/A"

        report.append(f"| {r['attack_name']} | {art_time} | {art_onnx_time} | {rust_1c_time} | {rust_16c_onnx_time} | {onnx_gain} | {rust_gain} | {best_speedup} |\n")

    report.append("\n## Detailed Metrics\n")
    for r in results:
        report.append(f"\n### {r['attack_name']}\n")
        report.append(f"- Success Rate: {r['art_1core']['success_rate']:.1%}\n")
        report.append(f"- Queries (ART): {r['art_1core']['queries_used']}\n")
        if r['rust_16core_onnx']:
            report.append(f"- Queries (Rust): {r['rust_16core_onnx']['queries_used']}\n")
        report.append(f"- L2 Perturbation: {r['art_1core']['l2_perturbation']:.4f}\n")
        if r['best_speedup']:
            report.append(f"- **Total Speedup: {r['best_speedup']:.2f}x**\n")

    report_path = output_dir / 'benchmark_report.md'
    with open(report_path, 'w') as f:
        f.writelines(report)

    print(f"Saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize benchmark results")
    parser.add_argument('results_json', help='Path to benchmark results JSON')
    parser.add_argument('--output', default='benchmark_charts', help='Output directory for charts')

    args = parser.parse_args()

    # Load results
    data = load_results(args.results_json)
    results = data['results']

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating visualizations to {output_dir}...")

    # Generate all plots
    if MATPLOTLIB_AVAILABLE:
        plot_speedup_comparison(results, output_dir)
        plot_wall_time_comparison(results, output_dir)
        plot_parallel_efficiency(results, output_dir)
        plot_queries_comparison(results, output_dir)
    else:
        print("Matplotlib not available, skipping chart generation")

    # Generate summary report
    generate_summary_report(data, output_dir)

    print("\nVisualization complete!")


if __name__ == '__main__':
    main()
