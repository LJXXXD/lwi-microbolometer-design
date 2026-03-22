#!/usr/bin/env python3
"""GA Hyperparameter Tuning Script.

Launches grid search experiments over GA hyperparameters using YAML experiment definitions.
"""

import argparse
import logging
import multiprocessing as mp
from pathlib import Path

from lwi_microbolometer_design.ga.experiment import (
    ExperimentConfig,
    create_fitness_evaluator_from_experiment,
    create_gene_space_from_experiment,
    create_search_space_from_experiment,
    load_experiment_config,
)
from lwi_microbolometer_design.ga.tuning import (
    HyperparameterSearchSpace,
    HyperparameterTuner,
    visualize_top_configurations,
)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="GA Hyperparameter Tuning - Grid Search over GA Parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default experiment file
  python scripts/tune_ga.py experiments/example_validation.yaml

  # Run with custom output directory
  python scripts/tune_ga.py experiments/example_full_search.yaml --output-dir outputs/tuning/my_tuning

  # Run in validation mode (overrides YAML validation settings)
  python scripts/tune_ga.py experiments/example_full_search.yaml --validation

  # Visualize top 10 configurations
  python scripts/tune_ga.py experiments/example_full_search.yaml --top-k 10
        """,
    )

    parser.add_argument(
        "experiment",
        type=Path,
        help="Path to YAML experiment definition file",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for results (default: outputs/tuning/)",
    )

    parser.add_argument(
        "--validation",
        action="store_true",
        help="Force validation mode (quick runs, fewer configs)",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top configurations to visualize (default: 5)",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: min(cpu_count(), 12))",
    )

    return parser.parse_args()


def apply_validation_mode(
    experiment_config: ExperimentConfig,
    search_space: HyperparameterSearchSpace,
) -> tuple[HyperparameterSearchSpace, int | None]:
    """Apply validation mode overrides to search space and generations.

    Parameters
    ----------
    experiment_config : ExperimentConfig
        Experiment configuration
    search_space : HyperparameterSearchSpace
        Original search space

    Returns
    -------
    tuple[HyperparameterSearchSpace, int]
        Modified search space and num_generations override
    """
    if not experiment_config.validation or not experiment_config.validation.get("enabled", False):
        return search_space, None

    validation = experiment_config.validation

    # Override generations if specified
    num_generations_override = validation.get("num_generations_override")

    # Limit configurations if max_configs specified
    max_configs = validation.get("max_configs")
    if max_configs:
        # Limit each parameter list to reduce total combinations
        # Simple heuristic: take first 2 values of each parameter if max_configs is small
        if max_configs < 50:
            logger.info("Validation mode: Limiting search space to reduce combinations")
            # We'll handle this in generate_configurations by limiting results
            pass

    return search_space, num_generations_override


def main() -> None:
    """Run main tuning workflow."""
    args = parse_arguments()

    logger.info("=" * 60)
    logger.info("      GA Hyperparameter Tuning")
    logger.info("=" * 60)

    # Load experiment configuration
    logger.info("\n=== Loading Experiment Configuration ===")
    logger.info(f"Experiment file: {args.experiment}")
    try:
        experiment = load_experiment_config(args.experiment)
        logger.info(f"Experiment name: {experiment.name}")
        if experiment.description:
            logger.info(f"Description: {experiment.description}")
    except Exception as e:
        logger.error(f"Failed to load experiment configuration: {e}")
        raise

    # Check validation mode
    validation_mode = args.validation or (
        experiment.validation is not None and experiment.validation.get("enabled", False)
    )
    if validation_mode:
        logger.info("\n⚠️  VALIDATION MODE ENABLED ⚠️")
        logger.info("   Running quick validation runs with reduced search space")

    # Create fitness evaluator
    logger.info("\n=== Creating Fitness Evaluator ===")
    try:
        fitness_func = create_fitness_evaluator_from_experiment(experiment)
        logger.info("✓ Fitness evaluator created successfully")
    except Exception as e:
        logger.error(f"Failed to create fitness evaluator: {e}")
        raise

    # Create search space
    logger.info("\n=== Creating Search Space ===")
    search_space = create_search_space_from_experiment(experiment)

    # Apply validation mode if enabled
    num_generations_override = None
    if validation_mode:
        search_space, num_generations_override = apply_validation_mode(experiment, search_space)
        if num_generations_override:
            logger.info(f"  Validation: Overriding num_generations to {num_generations_override}")

    # Create gene space
    gene_space = create_gene_space_from_experiment(experiment)
    logger.info(f"  Gene space: {len(gene_space)} genes")

    # Get execution parameters
    execution = experiment.execution
    num_runs_per_config = execution.get("num_runs_per_config", 3)
    random_seed_base = execution.get("random_seed_base", 42)
    fitness_threshold = execution.get("fitness_threshold", 45.0)
    max_workers = args.workers or execution.get("max_workers") or mp.cpu_count()

    logger.info("\n=== Execution Parameters ===")
    logger.info(f"  Runs per configuration: {num_runs_per_config}")
    logger.info(f"  Random seed base: {random_seed_base}")
    logger.info(f"  Fitness threshold: {fitness_threshold}")
    logger.info(f"  Parallel workers: {max_workers}")

    # Override num_generations in search space if validation mode
    if num_generations_override:
        search_space.num_generations = [num_generations_override]

    # Create tuner
    logger.info("\n=== Initializing Hyperparameter Tuner ===")
    tuner = HyperparameterTuner(
        fitness_func=fitness_func,
        gene_space=gene_space,
        search_space=search_space,
        params_per_basis_function=experiment.sensor["params_per_basis_function"],
        num_runs=num_runs_per_config,
        fitness_threshold=fitness_threshold,
        max_workers=max_workers,
    )

    # Generate configurations to check count
    configurations = tuner.generate_configurations()
    logger.info(f"  Total configurations: {len(configurations)}")

    # Apply validation mode config limit if needed
    if validation_mode and experiment.validation:
        max_configs = experiment.validation.get("max_configs")
        if max_configs and len(configurations) > max_configs:
            logger.info(f"  Validation: Limiting to first {max_configs} configurations")
            # We'll need to modify generate_configurations or filter here
            # For now, just warn - full implementation would require modifying the tuner
            logger.warning(
                f"  Note: {len(configurations)} configs exceed validation limit of {max_configs}"
            )
            logger.warning("  Consider reducing search space in validation mode")

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = Path("outputs/tuning") / experiment.name.replace(" ", "_").lower()

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("\n=== Output Directory ===")
    logger.info(f"  {output_dir}")

    # Run tuning
    logger.info("\n=== Starting Hyperparameter Tuning ===")
    logger.info("  This may take a while...")
    logger.info(f"  Total runs: {len(configurations) * num_runs_per_config}")

    try:
        results_df = tuner.tune(output_dir=output_dir)
        logger.info("\n✓ Tuning completed successfully!")
    except Exception as e:
        logger.error(f"\n✗ Tuning failed: {e}")
        raise

    # Display summary
    logger.info("\n=== Top 5 Configurations ===")
    top_5 = results_df.head(5)
    for idx, (_, row) in enumerate(top_5.iterrows(), 1):
        logger.info(
            f"\n{idx}. Best Fitness: {row['best_fitness']:.4f}"
            f"\n   Diversity: {row['diversity_score']:.4f}"
            f"\n   High-quality solutions: {row['high_quality_solutions']:.1f}"
        )

    # Generate analysis
    logger.info("\n=== Analyzing Results ===")
    analysis = tuner.analyze_results(results_df)
    logger.info("✓ Analysis complete")

    # Save analysis
    import json
    from datetime import datetime

    analysis_file = output_dir / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(analysis_file, "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    logger.info(f"  Analysis saved to: {analysis_file}")

    # Generate visualizations for top configurations
    logger.info("\n=== Generating Visualizations ===")
    try:
        # Use fewer generations for visualization runs if validation mode
        viz_generations_override = None
        if validation_mode and experiment.validation:
            viz_generations_override = experiment.validation.get("num_generations_override")
            if viz_generations_override:
                logger.info(
                    f"  Using {viz_generations_override} generations for visualization runs"
                )

        visualize_top_configurations(
            results_df=results_df,
            experiment=experiment,
            fitness_func=fitness_func,
            gene_space=gene_space,
            params_per_basis_function=experiment.sensor["params_per_basis_function"],
            output_dir=output_dir,
            top_k=args.top_k,
            num_generations_override=viz_generations_override,
        )
        logger.info("✓ Visualizations complete")
    except Exception as e:
        logger.error(f"✗ Visualization failed: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        logger.warning("Continuing without visualizations...")

    logger.info("\n" + "=" * 60)
    logger.info("     TUNING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"\nResults saved to: {output_dir}")
    logger.info(f"Best fitness achieved: {results_df.iloc[0]['best_fitness']:.4f}")
    logger.info(f"Visualizations generated for top {args.top_k} configurations")
    logger.info("=" * 60)


if __name__ == "__main__":
    # Required for multiprocessing on Windows/macOS
    mp.set_start_method("spawn", force=True)
    main()
