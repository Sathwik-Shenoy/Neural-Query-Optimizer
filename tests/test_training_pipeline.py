from neural_query_optimizer.training.pipeline import TrainingPipeline


def test_training_pipeline_end_to_end(tmp_path) -> None:
    cfg = {
        "seed": 7,
        "training": {
            "num_tables": 3,
            "rows_per_table": 120,
            "num_queries": 12,
            "train_split": 0.75,
            "model_path": str(tmp_path / "model.joblib"),
            "dataset_path": str(tmp_path / "dataset.csv"),
            "metrics_path": str(tmp_path / "metrics.json"),
        },
        "execution": {
            "index_scan_bonus": 0.65,
            "hash_join_bonus": 0.55,
            "nested_loop_penalty": 1.6,
        },
    }

    metrics = TrainingPipeline(cfg).run()

    assert "mae" in metrics
    assert (tmp_path / "model.joblib").exists()
    assert (tmp_path / "dataset.csv").exists()
    assert (tmp_path / "metrics.json").exists()
