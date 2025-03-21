import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, entropy as kl_entropy
import json

import json
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_logit_difference_distribution(file1, file2, percentiles=[50, 75, 90, 95, 99, 99.5, 99.9], output_file='logit_difference_distribution_run10100.png'):
    """
    Plot graphs of cumulative number of logits against logit differences for each step, mark specified percentiles,
    and save each plot as a PNG file with the step number in the filename.

    Args:
        file1 (str): Path to the first JSON file containing logits.
        file2 (str): Path to the second JSON file containing logits.
        percentiles (list): List of percentiles to mark on the graph (default: [50, 75, 90, 95, 99, 99.5, 99.9]).
        output_file (str): Base name of the output PNG files (default: 'logit_difference_distribution_run10100.png').
                           The actual filenames will include the step number, e.g., 'logit_difference_distribution_run10100_step0.png'.
    """
    # Load logits from both files
    with open(file1, 'r') as f:
        data1 = json.load(f)
    with open(file2, 'r') as f:
        data2 = json.load(f)

    steps1 = data1['steps']
    steps2 = data2['steps']

    # Determine the number of steps to process
    num_steps = min(len(steps1), len(steps2))

    for step in range(num_steps):
        logits1 = np.array(steps1[step]['logits'])
        logits2 = np.array(steps2[step]['logits'])

        if len(logits1) == len(logits2):
            diffs = np.abs(logits1 - logits2)
            if len(diffs) == 0:
                print(f"Step {step}: No logits to compare.")
                continue

            # Sort differences and compute cumulative count
            sorted_diffs = np.sort(diffs)
            cumulative_count = np.arange(1, len(sorted_diffs) + 1)

            # Calculate percentiles for this step
            percentile_values = np.percentile(sorted_diffs, percentiles)

            # Create the plot
            plt.figure(figsize=(10, 6))
            plt.plot(sorted_diffs, cumulative_count, label='Cumulative Distribution', color='blue')
            for p, val in zip(percentiles, percentile_values):
                plt.axvline(x=val, color='red', linestyle='--', label=f'{p}th percentile: {val:.8f}')
            plt.xlabel('Logit Difference')
            plt.ylabel('Cumulative Number of Logits')
            plt.title(f'Cumulative Distribution of Logit Differences - Step {step}')
            plt.legend()
            plt.grid(True)

            # Save the plot with step-specific filename
            base, ext = os.path.splitext(output_file)
            output_file_step = f"{base}_step{step}{ext}"
            plt.savefig(output_file_step)
            plt.close()
            print(f"Plot for step {step} saved to {output_file_step}")

            # Create a new figure for the histogram
            plt.figure(figsize=(10, 6))

            # Combine logits from both files to determine bin edges
            all_logits = np.concatenate([logits1, logits2])
            bins = np.linspace(np.min(all_logits), np.max(all_logits), 51)  # 50 bins

            # Plot histograms for both files with transparency
            plt.hist(logits1, bins=bins, alpha=0.5, label='File 1', color='blue')
            plt.hist(logits2, bins=bins, alpha=0.5, label='File 2', color='red')

            # Add labels and styling
            plt.xlabel('Logit Value')
            plt.ylabel('Count')
            plt.title(f'Logit Distribution - Step {step}')
            plt.legend()
            plt.grid(True)

            # Save the histogram plot
            histogram_file_step = f"{base}_histogram_step{step}{ext}"
            plt.savefig(histogram_file_step)
            plt.close()

            # Confirm the plot has been saved
            print(f"Histogram plot for step {step} saved to {histogram_file_step}")

            # Print percentile values for this step
            print(f"Step {step} percentiles:")
            for p, val in zip(percentiles, percentile_values):
                print(f"  {p}th percentile: {val:.8f}")
        else:
            print(f"Step {step}: Number of logits do not match ({len(logits1)} vs {len(logits2)}). Skipping.")

def compare_logits(file1, file2, top_k_values=[5, 10, 20, 100, 1000]):
    """
    Compare the 'logits' field between two JSON files step by step and compute statistical differences.
    
    Args:
        file1 (str): Path to the first JSON file (e.g., 'logits_sglang.json').
        file2 (str): Path to the second JSON file (e.g., 'logits_4_run10100.json').
        top_k_values (list): List of K values for top-K agreement (default: [5, 10]).
    """
    # Load both JSON files
    try:
        with open(file1, 'r') as f:
            data1 = json.load(f)
        with open(file2, 'r') as f:
            data2 = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {e}")
        return

    # Extract the 'steps' list containing logits and selected_token_id
    try:
        steps1 = data1['steps']
        steps2 = data2['steps']
    except KeyError:
        print("Error: 'steps' key not found in one or both JSON files.")
        return

    # Determine the minimum number of steps to compare
    min_steps = min(len(steps1), len(steps2))
    print(f"Comparing up to {min_steps} steps (minimum of {len(steps1)} and {len(steps2)}).")

    # Iterate through each step and compare logits
    for step in range(min_steps):
        logits1 = np.array(steps1[step]['logits'])
        logits2 = np.array(steps2[step]['logits'])
        selected1 = steps1[step]['selected_token_id']
        selected2 = steps2[step]['selected_token_id']

        # Verify logits have the same length
        if len(logits1) != len(logits2):
            print(f"Step {step}: Error - Logits have different lengths ({len(logits1)} vs {len(logits2)}).")
            continue

        # Check if selected tokens are the same
        same_selected = selected1 == selected2

        # Compute statistical differences
        mad = np.mean(np.abs(logits1 - logits2))  # Mean Absolute Difference
        rmsd = np.sqrt(np.mean((logits1 - logits2) ** 2))  # Root Mean Squared Difference
        pearson_corr, _ = pearsonr(logits1, logits2)  # Pearson Correlation
        spearman_corr, _ = spearmanr(logits1, logits2)  # Spearman Rank Correlation
        cosine_sim = np.dot(logits1, logits2) / (np.linalg.norm(logits1) * np.linalg.norm(logits2))  # Cosine Similarity

        # Convert logits to probabilities
        probs1 = np.exp(logits1) / np.sum(np.exp(logits1))
        probs2 = np.exp(logits2) / np.sum(np.exp(logits2))

        # KL Divergence
        kl_div = kl_entropy(probs1, probs2)

        # Entropy Difference
        entropy1 = -np.sum(probs1 * np.log(probs1 + 1e-10))  # Add epsilon for stability
        entropy2 = -np.sum(probs2 * np.log(probs2 + 1e-10))
        entropy_diff = np.abs(entropy1 - entropy2)

        # Top-1 Agreement
        top1_1 = np.argmax(logits1)
        top1_2 = np.argmax(logits2)
        same_top1 = top1_1 == top1_2

        # Top-K Agreement
        top_k_agreements = {}
        for k in top_k_values:
            top_k_1 = set(np.argsort(logits1)[-k:])
            top_k_2 = set(np.argsort(logits2)[-k:])
            overlap = len(top_k_1.intersection(top_k_2))
            agreement_percentage = (overlap / k) * 100
            top_k_agreements[k] = agreement_percentage

        # Display results
        print(f"\nStep {step}:")
        print(f"  Selected Token IDs: {selected1} (file1) vs {selected2} (file2)")
        print(f"  Selected tokens same: {same_selected}")
        print(f"  Mean Absolute Difference: {mad:.8f}")
        print(f"  Root Mean Squared Difference: {rmsd:.8f}")
        print(f"  Pearson Correlation: {pearson_corr:.8f}")
        print(f"  Spearman Rank Correlation: {spearman_corr:.8f}")
        print(f"  Cosine Similarity: {cosine_sim:.8f}")
        print(f"  KL Divergence (file1 || file2): {kl_div:.8f}")
        print(f"  Entropy Difference: {entropy_diff:.8f}")
        print(f"  Top-1 Token IDs: {top1_1} (file1) vs {top1_2} (file2)")
        print(f"  Top-1 tokens same: {same_top1}")
        for k, agreement in top_k_agreements.items():
            print(f"  Top-{k} Agreement: {agreement:.2f}%")
        if not same_selected:
            print("  Note: Selected tokens differ, subsequent logits may be for different contexts.")

    if len(steps1) != len(steps2):
        print(f"\nNote: Number of steps differs: {len(steps1)} (file1) vs {len(steps2)} (file2)")

if __name__ == "__main__":
    file1 = "logits_sglang_run10100.json"
    file2 = "logits_4_run10100.json"
    print(f"Comparing logits between '{file1}' and '{file2}'.\n")
    # Assuming compare_logits is an existing function
    compare_logits(file1, file2, top_k_values=[5, 10, 20, 100, 1000])
    # Add the distribution plot
    plot_logit_difference_distribution(file1, file2, output_file='logit_diff_plot.png')