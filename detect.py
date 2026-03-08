#!/usr/bin/env python3
"""
CLI entry point for the Cattle Drinking Anomaly Detection System.

Usage:
    python detect.py --input data/processed/scanner_data_clean.csv
    python detect.py --input data/processed/scanner_data_clean.csv --output results.csv
"""

import argparse
import os
import sys

# Add project root to path so src package can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description='Detect anomalies in cattle drinking behavior from RFID scanner data.'
    )
    parser.add_argument(
        '--input', '-i', required=True,
        help='Path to CSV file with scanner data.'
    )
    parser.add_argument(
        '--output', '-o', default=None,
        help='Path to save results CSV (optional).'
    )
    parser.add_argument(
        '--model-dir', '-m', default='models',
        help='Path to trained model directory (default: models/).'
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: input file not found: {args.input}")
        sys.exit(1)

    if not os.path.exists(args.model_dir):
        print(f"Error: model directory not found: {args.model_dir}")
        sys.exit(1)

    print(f"Loading scanner data from: {args.input}")
    results, summary = run_pipeline(args.input, model_dir=args.model_dir)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"ANOMALY DETECTION RESULTS")
    print(f"{'=' * 60}")
    print(f"Animals:          {summary['animals']}")
    print(f"Days:             {summary['days']}")
    print(f"Animal-days:      {summary['animal_days']}")
    print(f"Sessions loaded:  {summary['loaded_sessions']}")
    print(f"Drinking bouts:   {summary['bouts']}")
    print(f"Anomalies flagged: {summary['anomalies_flagged']} ({summary['anomaly_rate']:.1%})")

    # Show flagged animals
    flagged = results[results['ae_anomaly'] == 1]
    if len(flagged) > 0:
        print(f"\n{'=' * 60}")
        print(f"FLAGGED ANIMALS (check these)")
        print(f"{'=' * 60}")
        animal_flags = flagged.groupby('tag_short').agg(
            anomaly_days=('ae_anomaly', 'sum'),
            total_days=('ae_anomaly', 'count'),
            max_score=('ae_score', 'max'),
            mean_score=('ae_score', 'mean')
        ).reset_index()
        # Include total_days from all results, not just flagged
        total_by_animal = results.groupby('tag_short')['ae_anomaly'].count().rename('total_days_all')
        animal_flags = animal_flags.merge(total_by_animal, left_on='tag_short', right_index=True)
        animal_flags['anomaly_rate'] = animal_flags['anomaly_days'] / animal_flags['total_days_all']
        animal_flags = animal_flags.sort_values('anomaly_days', ascending=False)

        for _, row in animal_flags.iterrows():
            print(f"  Animal {row['tag_short']}: {row['anomaly_days']:.0f}/{row['total_days_all']:.0f} days "
                  f"({row['anomaly_rate']:.0%}), max score={row['max_score']:.2f}")
    else:
        print("\nNo anomalies detected.")

    # Save results
    if args.output:
        results.to_csv(args.output, index=False)
        print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
