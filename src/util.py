import time

start_time = time.time()  # Record the start time

def log(message):
    elapsed = time.time() - start_time
    print(f"[{elapsed:.2f}s] {message}")