import subprocess
import sys

steps = ["src/update_data.py", "src/load_data.py", "src/build_features.py", "src/fetch_players.py", "src/fetch_fixtures.py"]

for step in steps:
    print(f"\n=== Running {step} ===")
    result = subprocess.run([sys.executable, step])
    if result.returncode != 0:
        print(f"\nStopped: {step} failed. Fix that before continuing.")
        break
else:
    print("\nAll done. Restart Streamlit to load the fresh data.")
