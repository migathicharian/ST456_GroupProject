"""
PCam histopathology SSL project - source package.

Usage from notebook:
    from src import (
        load_all_data, DataBundle,
        SimCLRModel, MAEViTImproved,
        train_simclr, train_mae_improved,
        run_experiment_grid,
    )
"""

from .data import (
    DataBundle,
    LABEL_FRACTIONS,
    SimCLRTransform,
    SSLImageDataset,
    SSLPairDataset,
    ClassificationDataset,
    build_downstream_loaders,
    load_all_data,
    make_classification_loader,
    make_classification_transform,
    make_labelled_subset,
    make_mae_transform,
    stratified_subset_indices,
)

from .models import (
    MAEBinaryClassifier,
    MAEViT,
    MAEViTImproved,
    ResNetBinaryClassifier,
    SimCLRModel,
    build_mae_classifier,
    build_mae_improved_classifier,
    build_simclr_classifier,
    build_supervised_scratch_classifier,
    configure_trainable_parameters,
)

from .training import (
    DEFAULT_EXPERIMENT_CONFIG,
    FINETUNE_STRATEGIES,
    MAIN_METHODS,
    SEED,
    TrainConfig,
    device,
    nt_xent_loss,
    patchify,
    run_experiment_grid,
    run_single_experiment,
    set_seed,
    train_classifier,
    train_mae,
    train_mae_improved,
    train_simclr,
)

from .evaluation import (
    GradCAM,
    build_report_table,
    evaluate_model,
    extract_embeddings,
    format_mean_std,
    plot_confusion,
    plot_roc_pr_curves,
    plot_tsne,
    summarise_results,
)
