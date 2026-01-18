#!/usr/bin/env python3
"""Script to collect trajectory data from CARL environments"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import TrajectoryCollector, get_context_distributions
from src.utils import load_config


def main():
    parser = argparse.ArgumentParser(description="Collect trajectory data")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test'],
                        help='Data split to collect')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Get contexts for the split
    env_name = config['environment']['name']
    contexts = get_context_distributions(env_name, split=args.split)

    print(f"Collecting data for {env_name} ({args.split} split)")
    print(f"Number of contexts: {len(contexts)}")
    print(f"Segment length: {config['data']['segment_length']}")
    print(f"Segments per context: {config['data']['num_segments_per_context']}")

    # Create collector
    collector = TrajectoryCollector(
        env_name=env_name,
        contexts=contexts,
        segment_length=config['data']['segment_length'],
        num_segments_per_context=config['data']['num_segments_per_context'],
        policy=config['data']['policy'],
        seed=config['experiment']['seed']
    )

    # Collect data
    segments = collector.collect(verbose=True)

    # Save
    data_dir = Path(config['paths']['data_dir'])
    save_path = data_dir / f"{env_name}_{args.split}_segments.pkl"
    collector.save(segments, str(save_path))

    print(f"\nData collection complete!")
    print(f"Total segments collected: {len(segments)}")
    print(f"Saved to: {save_path}")


if __name__ == "__main__":
    main()
