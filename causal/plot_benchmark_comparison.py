import json
from pathlib import Path
import matplotlib.pyplot as plt


def load_results(path):
    with open(path, 'r') as f:
        return json.load(f)


def extract_means(results):
    stats = results['statistics']
    return [
        stats['parse_success_rate']['mean'],
        stats['valid_rate']['mean'],
        stats['novelty_rate']['mean'],
        stats['recovery_rate']['mean'],
    ]


def plot_bars_multi(results_list, labels, out_path):
    metrics = ['Parse', 'Valid', 'Novelty', 'Recovery']
    k = len(results_list)
    data = [extract_means(r) for r in results_list]  # shape: k x 4
    x = range(len(metrics))
    width = min(0.8 / max(k, 1), 0.25)  # keep bars readable
    offsets = [((i - (k - 1) / 2) * width) for i in range(k)]

    plt.figure(figsize=(10, 6))
    for j in range(k):
        plt.bar([xi + offsets[j] for xi in x], data[j], width, label=labels[j])
    plt.xticks(list(x), metrics)
    plt.ylim(0, 1)
    plt.ylabel('Rate')
    plt.title('Causal Benchmark: Multi-run Comparison (seed-controlled)')
    plt.legend(ncol=min(k, 4))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_scatter_multi(results_list, labels, out_path):
    markers = ['o', 's', '^', 'D', 'x', '*', '+', 'v', '<', '>']
    plt.figure(figsize=(10, 6))
    for idx, r in enumerate(results_list):
        samples = r.get('per_sample_results', r.get('results', []))
        gx = [s.get('n_ground_truths', 0) for s in samples]
        ry = [s.get('recovery_rate', 0) for s in samples]
        plt.scatter(gx, ry, label=labels[idx], alpha=0.7, marker=markers[idx % len(markers)])
    plt.xlabel('Number of ground truths (GT)')
    plt.ylabel('Recovery rate')
    plt.title('Recovery vs GT: Multi-run Comparison')
    plt.legend(ncol=min(len(results_list), 4))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Compare causal benchmark result JSONs and plot charts')
    parser.add_argument('--baseline', help='Path to baseline results JSON')
    parser.add_argument('--improved', help='Path to improved results JSON')
    parser.add_argument('--inputs', nargs='+', help='Paths to multiple result JSON files')
    parser.add_argument('--labels', nargs='+', help='Labels for inputs; defaults to file stem')
    parser.add_argument('--out-bars', default='figs/causal_multi_comparison_bars.png', help='Output path for bars chart')
    parser.add_argument('--out-scatter', default='figs/causal_multi_comparison_scatter_recovery.png', help='Output path for recovery scatter')
    args = parser.parse_args()

    results_list = []
    labels = []

    if args.inputs:
        for p in args.inputs:
            results_list.append(load_results(p))
            labels.append(Path(p).stem if not args.labels else None)
        if args.labels:
            if len(args.labels) != len(args.inputs):
                raise ValueError('Number of labels must match number of inputs')
            labels = args.labels
        else:
            labels = [Path(p).stem for p in args.inputs]
    else:
        # Fallback to 2-run mode for backward compatibility
        if not (args.baseline and args.improved):
            raise ValueError('Provide either --inputs ... or both --baseline and --improved')
        results_list = [load_results(args.baseline), load_results(args.improved)]
        labels = [Path(args.baseline).stem, Path(args.improved).stem]

    plot_bars_multi(results_list, labels, args.out_bars)
    plot_scatter_multi(results_list, labels, args.out_scatter)
    print(f"Saved charts to: {args.out_bars} and {args.out_scatter}")


if __name__ == '__main__':
    main()