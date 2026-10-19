
import subprocess
import sys

print("=" * 70)
print("BRFSS STROKE - NEURAL FEATURE PIPELINE")
print("=" * 70)

steps = [
    ("preprocessing.py", "1/3"),
    ("neural_feature_learning.py", "2/3"),
    ("classification_neural.py", "3/3"),
]

for script, stage in steps:
    print("\n" + "=" * 70)
    print(f"RUNNING {stage}: {script}")
    print("=" * 70)

    result = subprocess.run(
        [sys.executable, script],
        check=False,
    )

    if result.returncode != 0:
        raise SystemExit(
            f"\nPipeline stopped because {script} failed."
        )

print("\n" + "=" * 70)
print("NEURAL PIPELINE COMPLETE")
print("=" * 70)

print("\nRun visualizations_neural.py next to generate the existing")
print("ROC, PR, confusion-matrix, and model-comparison plots.")
