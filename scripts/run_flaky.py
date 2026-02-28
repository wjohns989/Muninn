import subprocess, sys
import time

TEST='tests/test_concurrency.py::test_sqlite_concurrency'
N=10
failures=0
for i in range(N):
    print(f'Run {i+1}/{N}')
    r = subprocess.run([sys.executable, '-m', 'pytest', TEST, '-q'], capture_output=True, text=True)
    print(r.stdout)
    if r.returncode != 0:
        failures += 1
        print('---FAILED OUTPUT---')
        print(r.stderr)
    time.sleep(0.5)

print(f'Completed {N} runs: {failures} failures')
if failures>0:
    sys.exit(2)
