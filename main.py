import subprocess
import json
import os

num_gpus = 4  # Adjust based on your system
num_runs = 3  # Number of runs per GPU configuration

for n in range(3, num_gpus + 1):
    for run in range(num_runs):
        print(f"Testing with {n} GPU(s), run {run + 1} of {num_runs}...")
        subprocess.run(["python", "generate_with_n_gpus.py", str(n), str(run)])

# Collect and display results
results = {}
for n in range(3, num_gpus + 1):
    results[n] = []
    for run in range(num_runs):
        filename = f"results_{n}_run{run}.json"
        if os.path.exists(filename):
            with open(filename, "r") as f:
                res = json.load(f)
                if "error" in res:
                    results[n].append(f"Run {run}: Failed - {res['error']}")
                else:
                    results[n].append(f"Run {run}: {res['generated_text']}")

# Display collected results
if any(results.values()):
    print("\nGenerated outputs per GPU configuration:")
    for n in range(3, num_gpus + 1):
        if results[n]:
            print(f"- {n} GPU(s):")
            for result in results[n]:
                print(f"  {result}")
else:
    print("No successful runs to compare.")