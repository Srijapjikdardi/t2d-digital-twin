"""One-command setup: python run_all.py
1) generate synthetic data  2) train models + plots (a few minutes)  3) tells you how to start the dashboard.
Run it from anywhere; it always works inside this project folder."""
import os
import subprocess
import sys
import time

root = os.path.dirname(os.path.abspath(__file__))
os.chdir(root)
for step, script in (("1/2 Generating synthetic data", "src/generate_data.py"),
                     ("2/2 Training models (about 2-6 minutes)", "src/train.py")):
    print(f"\n=== {step} ===", flush=True)
    t0 = time.time()
    rc = subprocess.call([sys.executable, script])
    if rc != 0:
        sys.exit(f"\nStep failed: {script}. Copy the error above and ask for help.")
    print(f"done in {time.time() - t0:.0f}s")
print("\nAll done. Start the dashboard with:\n    python -m streamlit run app.py")
