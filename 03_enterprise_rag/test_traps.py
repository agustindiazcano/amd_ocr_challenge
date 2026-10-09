import os
import subprocess
import time
import sys

# Try to import pypdf to generate encrypted pdf, else skip that specific file
try:
    from pypdf import PdfWriter
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

CORPUS_DIR = "hostile_corpus"

def setup_hostile_corpus():
    """Generates the 4 traps strictly mentioned by the evaluator."""
    print(f"[*] Assembling hostile corpus at '{CORPUS_DIR}'...")
    os.makedirs(CORPUS_DIR, exist_ok=True)
    
    # 1. Empty Directory
    empty_dir = os.path.join(CORPUS_DIR, "empty_dir")
    os.makedirs(empty_dir, exist_ok=True)
    print("    -> Created Trap 1: Empty directory")
    
    # 2. Unknown Format
    unknown_file = os.path.join(CORPUS_DIR, "unknown_format.xyz")
    with open(unknown_file, "w") as f:
        f.write("Binary noise or weird format data.")
    print("    -> Created Trap 2: Unknown format (.xyz)")
        
    # 3. chmod 000 File (No Read Permission)
    no_perm_file = os.path.join(CORPUS_DIR, "no_permission.txt")
    with open(no_perm_file, "w") as f:
        f.write("You should not be able to read this without DAC_OVERRIDE.")
    try:
        os.chmod(no_perm_file, 0o000)
        print("    -> Created Trap 3: File with chmod 000")
    except Exception as e:
        print(f"    -> Warning: chmod 000 failed (normal on Windows host): {e}")
        
    # 4. Encrypted PDF
    encrypted_file = os.path.join(CORPUS_DIR, "encrypted.pdf")
    if HAS_PYPDF:
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        writer.encrypt("supersecret")
        with open(encrypted_file, "wb") as f:
            writer.write(f)
        print("    -> Created Trap 4: Encrypted PDF (password protected)")
    else:
        print("    -> Warning: pypdf not installed, skipping encrypted PDF generation.")

    # 5. Valid file to prove indexer doesn't halt
    valid_file = os.path.join(CORPUS_DIR, "valid_doc.txt")
    with open(valid_file, "w") as f:
        f.write("This is a valid document. The TQ-40 maximum temperature is 94.")
    print("    -> Created Valid file to ensure walk order continues.")
    print("[*] Corpus assembly complete.\n")


def run_test():
    setup_hostile_corpus()
    
    print("[*] Starting Daemon in the background...")
    # Using subprocess to run the uvicorn daemon as it would in Docker
    daemon_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "daemon:app", "--host", "127.0.0.1", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Give daemon time to spin up FastAPI
    print("[*] Waiting 5 seconds for Daemon to initialize...")
    time.sleep(5)
    
    print("[*] Invoking app.py (Thin Client) to index the hostile corpus...")
    start_time = time.time()
    
    # Run the index pass
    index_process = subprocess.run(
        [sys.executable, "app.py", "--index", CORPUS_DIR],
        capture_output=True,
        text=True
    )
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*40)
    print("--- THIN CLIENT STDOUT ---")
    print(index_process.stdout)
    print("--- THIN CLIENT STDERR ---")
    print(index_process.stderr)
    print("="*40)
    
    print(f"[*] Index pass completed in {elapsed:.2f} seconds.")
    print(f"[*] Thin Client Exit Code: {index_process.returncode} (Should be 0)")
    
    # Optional: We could also run a query to verify /query logic doesn't crash
    print("[*] Invoking app.py to run a query...")
    query_process = subprocess.run(
        [sys.executable, "app.py", "--corpus", CORPUS_DIR, "--query-id", "test_01", "--query", "What is the maximum temperature of TQ-40?"],
        capture_output=True,
        text=True
    )
    print("--- QUERY CLIENT STDOUT ---")
    print(query_process.stdout)
    print("--- QUERY CLIENT STDERR ---")
    print(query_process.stderr)
    
    print("[*] Terminating Daemon...")
    daemon_process.terminate()
    daemon_process.wait()
    print("[*] Test simulation finished.")

if __name__ == "__main__":
    run_test()
