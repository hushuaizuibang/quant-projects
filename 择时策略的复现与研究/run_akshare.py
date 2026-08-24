from pathlib import Path


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    (project_root / "outputs" / ".matplotlib").mkdir(parents=True, exist_ok=True)

    import os

    os.environ.setdefault("MPLCONFIGDIR", str(project_root / "outputs" / ".matplotlib"))

    from src.factor_timing.akshare_pipeline import run_akshare_pipeline

    run_akshare_pipeline(project_root)
