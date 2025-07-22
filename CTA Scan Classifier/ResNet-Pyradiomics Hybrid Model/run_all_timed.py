# scripts/run_all.py
import subprocess
import time

def timed_step(description, command):
    print(f"\n🔧 {description}...")
    start = time.time()
    subprocess.run(command)
    end = time.time()
    print(f"⏱️ Completed in {end - start:.2f} seconds")

print("\n🚀 Starting full hybrid pipeline with timing...")

timed_step("Step 1: Extracting radiomics features", ["python", "scripts/extract_radiomics.py"])
timed_step("Step 2: Extracting deep features", ["python", "scripts/extract_deep_feats.py"])
timed_step("Step 3: Training hybrid model", ["python", "scripts/train_hybrid.py"])
timed_step("Step 4: Running inference", ["python", "scripts/infer_hybrid.py"])

print("\n✅ All steps completed successfully!")
