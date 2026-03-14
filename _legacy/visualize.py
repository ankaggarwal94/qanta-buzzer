"""
Visualization utilities for analyzing training results
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse


def plot_training_curves(history, save_dir):
    """Plot training curves from history"""
    
    # Extract data
    iterations = [h['iteration'] for h in history]
    train_rewards = [h['train_reward'] for h in history]
    val_accuracies = [h['val']['accuracy'] for h in history]
    val_rewards = [h['val'].get('average_reward', 0) for h in history]
    val_ece = [h['val'].get('ece', 0) for h in history]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Training reward
    axes[0, 0].plot(iterations, train_rewards, 'b-', linewidth=2, label='Train Reward')
    axes[0, 0].plot(iterations, val_rewards, 'r-', linewidth=2, label='Val Reward')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Average Reward')
    axes[0, 0].set_title('Reward Progress')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Validation accuracy
    axes[0, 1].plot(iterations, val_accuracies, 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Validation Accuracy')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: ECE (calibration)
    axes[1, 0].plot(iterations, val_ece, 'purple', linewidth=2)
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Expected Calibration Error')
    axes[1, 0].set_title('Calibration (ECE)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Policy and value loss
    policy_losses = [h.get('policy_loss', 0) for h in history]
    value_losses = [h.get('value_loss', 0) for h in history]
    
    axes[1, 1].plot(iterations, policy_losses, 'orange', linewidth=2, label='Policy Loss')
    axes[1, 1].plot(iterations, value_losses, 'cyan', linewidth=2, label='Value Loss')
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Training Losses')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
    print(f"Saved training curves to {save_dir / 'training_curves.png'}")
    plt.close()


def plot_reliability_diagram(metrics_data, save_dir):
    """Plot reliability diagram for calibration"""
    
    # Get reliability data
    bin_data = metrics_data.get('reliability_data', {})
    
    if not bin_data:
        print("No reliability data available")
        return
    
    bin_centers = bin_data['bin_centers']
    accuracies = bin_data['accuracies']
    confidences = bin_data['confidences']
    counts = bin_data['counts']
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')
    
    # Plot actual calibration
    ax.scatter(confidences, accuracies, s=np.array(counts)*5, 
              alpha=0.6, c='blue', edgecolors='black', linewidth=1.5,
              label='Model Calibration')
    
    # Plot bars
    for conf, acc, count in zip(confidences, accuracies, counts):
        ax.plot([conf, conf], [conf, acc], 'r-', alpha=0.5, linewidth=2)
    
    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Reliability Diagram', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_dir / 'reliability_diagram.png', dpi=300, bbox_inches='tight')
    print(f"Saved reliability diagram to {save_dir / 'reliability_diagram.png'}")
    plt.close()


def plot_buzzing_behavior(metrics_data, save_dir):
    """Plot buzzing position distribution"""
    
    buzz_stats = metrics_data.get('buzz_stats', {})
    
    if not buzz_stats:
        print("No buzzing statistics available")
        return
    
    position_accuracy = buzz_stats.get('position_accuracy', {})
    
    if not position_accuracy:
        return
    
    positions = sorted(position_accuracy.keys())
    accuracies = [position_accuracy[p] for p in positions]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Accuracy by position
    axes[0].bar(positions, accuracies, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0].set_xlabel('Buzz Position (Clue Number)')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy by Buzz Position')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Distribution of buzz positions
    mean_pos = buzz_stats.get('mean', 0)
    std_pos = buzz_stats.get('std', 0)
    
    axes[1].axvline(mean_pos, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_pos:.2f}')
    axes[1].axvline(mean_pos - std_pos, color='orange', linestyle=':', linewidth=1.5,
                   label=f'±1 Std: {std_pos:.2f}')
    axes[1].axvline(mean_pos + std_pos, color='orange', linestyle=':', linewidth=1.5)
    axes[1].set_xlabel('Buzz Position')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Buzzing Position Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'buzzing_behavior.png', dpi=300, bbox_inches='tight')
    print(f"Saved buzzing behavior to {save_dir / 'buzzing_behavior.png'}")
    plt.close()


def plot_category_performance(metrics_data, save_dir):
    """Plot per-category performance"""
    
    category_acc = metrics_data.get('category_accuracy', {})
    
    if not category_acc:
        print("No category-specific data available")
        return
    
    categories = list(category_acc.keys())
    accuracies = [category_acc[c] for c in categories]
    
    # Sort by accuracy
    sorted_indices = np.argsort(accuracies)
    categories = [categories[i] for i in sorted_indices]
    accuracies = [accuracies[i] for i in sorted_indices]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(categories)))
    bars = ax.barh(categories, accuracies, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for i, (cat, acc) in enumerate(zip(categories, accuracies)):
        ax.text(acc + 0.01, i, f'{acc:.3f}', va='center', fontsize=10)
    
    ax.set_xlabel('Accuracy', fontsize=12)
    ax.set_ylabel('Category', fontsize=12)
    ax.set_title('Performance by Category', fontsize=14, fontweight='bold')
    ax.set_xlim([0, max(accuracies) * 1.15])
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'category_performance.png', dpi=300, bbox_inches='tight')
    print(f"Saved category performance to {save_dir / 'category_performance.png'}")
    plt.close()


def create_summary_report(checkpoint_dir, output_dir):
    """Create comprehensive visualization report"""
    
    checkpoint_dir = Path(checkpoint_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load history
    history_path = checkpoint_dir / 'history.json'
    if history_path.exists():
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        print(f"Loaded training history from {history_path}")
        plot_training_curves(history, output_dir)
    else:
        print(f"No history file found at {history_path}")
    
    # Load test results
    test_results_path = checkpoint_dir / 'test_results.json'
    if test_results_path.exists():
        with open(test_results_path, 'r') as f:
            test_results = json.load(f)
        
        print(f"Loaded test results from {test_results_path}")
        
        # Plot reliability diagram
        # Note: This would need reliability data from a separate run
        # plot_reliability_diagram(test_results, output_dir)
        
        # Plot buzzing behavior
        if 'buzz_stats' in test_results:
            plot_buzzing_behavior(test_results, output_dir)
        
        # Plot category performance
        if 'category_accuracy' in test_results:
            plot_category_performance(test_results, output_dir)
        
        # Print summary
        print("\n" + "=" * 60)
        print("TEST RESULTS SUMMARY")
        print("=" * 60)
        print(f"Accuracy: {test_results.get('accuracy', 0):.4f}")
        print(f"Average Reward: {test_results.get('average_reward', 0):.4f}")
        print(f"ECE: {test_results.get('ece', 0):.4f}")
        print(f"Avg Buzz Position: {test_results.get('average_buzz_position', 0):.2f}")
        print("=" * 60)
    else:
        print(f"No test results found at {test_results_path}")
    
    print(f"\nAll visualizations saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Visualize training results')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='Directory containing checkpoints and history')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                       help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    create_summary_report(args.checkpoint_dir, args.output_dir)


if __name__ == "__main__":
    main()
