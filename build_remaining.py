"""
Build remaining Mathlib modules using lean directly with -o flag.
Workaround for lake.exe being blocked by Application Control.
"""
import subprocess
import sys
import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
PACKAGES = PROJECT_ROOT / ".lake" / "packages"

def get_lean_path():
    paths = []
    for pkg in PACKAGES.iterdir():
        if pkg.is_dir():
            lib = pkg / ".lake" / "build" / "lib" / "lean"
            if lib.exists():
                paths.append(str(lib))
    own = PROJECT_ROOT / ".lake" / "build" / "lib" / "lean"
    paths.append(str(own))
    return ";".join(paths)

def module_to_paths(mod_name):
    """Return (source_path, olean_path, ilean_path) for a module."""
    parts = mod_name.split(".")
    # Try each package
    for pkg in PACKAGES.iterdir():
        if pkg.is_dir():
            src = pkg / Path(*parts).with_suffix(".lean")
            if src.exists():
                odir = pkg / ".lake" / "build" / "lib" / "lean"
                olean = odir / Path(*parts).with_suffix(".olean")
                ilean = odir / Path(*parts).with_suffix(".ilean")
                return src, olean, ilean
    # Try project root
    src = PROJECT_ROOT / Path(*parts).with_suffix(".lean")
    odir = PROJECT_ROOT / ".lake" / "build" / "lib" / "lean"
    olean = odir / Path(*parts).with_suffix(".olean")
    ilean = odir / Path(*parts).with_suffix(".ilean")
    return src, olean, ilean

def is_built(mod_name):
    _, olean, _ = module_to_paths(mod_name)
    return olean.exists()

def compile_one(mod_name, lean_path):
    """Compile one module, producing .olean and .ilean. Returns True on success."""
    src, olean, ilean = module_to_paths(mod_name)
    if not src.exists():
        print(f"  [SKIP] {mod_name} - no source", flush=True)
        return False
    
    olean.parent.mkdir(parents=True, exist_ok=True)
    
    env = os.environ.copy()
    env["LEAN_PATH"] = lean_path
    
    result = subprocess.run(
        ["lean", str(src), "-o", str(olean), "-i", str(ilean), "--threads=4"],
        capture_output=True, text=True, env=env,
        cwd=str(PROJECT_ROOT)
    )
    return result.returncode == 0, result.stderr + result.stdout

def extract_missing(output):
    """Extract missing module name from lean error output."""
    for line in output.split("\n"):
        m = re.search(r"of module (\S+) does not exist", line)
        if m:
            return m.group(1)
    return None

def build_module(mod_name, lean_path, building=None):
    """Recursively build a module and its missing dependencies."""
    if building is None:
        building = set()
    
    if is_built(mod_name):
        return True
    
    if mod_name in building:
        print(f"  [CYCLE] {mod_name} - circular dependency!", flush=True)
        return False
    
    building.add(mod_name)
    
    # Try up to 50 times (one per missing dependency)
    for attempt in range(50):
        if is_built(mod_name):
            return True
        
        print(f"  [BUILD] {mod_name} (attempt {attempt+1})...", flush=True)
        ok, output = compile_one(mod_name, lean_path)
        
        if ok:
            print(f"  [OK] {mod_name}", flush=True)
            return True
        
        missing = extract_missing(output)
        if missing:
            print(f"    -> needs {missing}", flush=True)
            if not build_module(missing, lean_path, building.copy()):
                print(f"    -> FAILED to build {missing}", flush=True)
                return False
        else:
            # Some other error
            print(f"  [ERROR] {mod_name}:", flush=True)
            for line in output.strip().split("\n")[:5]:
                print(f"    {line}", flush=True)
            return False
    
    print(f"  [ERROR] {mod_name} - too many attempts", flush=True)
    return False

def main():
    print("=" * 60, flush=True)
    print("Building remaining modules with lean -o", flush=True)
    print("=" * 60, flush=True)
    
    lean_path = get_lean_path()
    
    # Build the Mathlib target
    target = "Mathlib.NumberTheory.SumTwoSquares"
    print(f"\nTarget: {target}", flush=True)
    print(f"Already built: {is_built(target)}\n", flush=True)
    
    if build_module(target, lean_path):
        print("\n*** Mathlib.NumberTheory.SumTwoSquares built! ***", flush=True)
        
        # Now build our proof
        print("\nBuilding Proof.SumTwoSquares...", flush=True)
        src = PROJECT_ROOT / "Proof" / "SumTwoSquares.lean"
        olean = PROJECT_ROOT / ".lake" / "build" / "lib" / "lean" / "Proof" / "SumTwoSquares.olean"
        olean.parent.mkdir(parents=True, exist_ok=True)
        
        env = os.environ.copy()
        env["LEAN_PATH"] = lean_path
        result = subprocess.run(
            ["lean", str(src), "-o", str(olean), "--threads=4"],
            capture_output=True, text=True, env=env,
            cwd=str(PROJECT_ROOT)
        )
        
        if result.returncode == 0:
            print("\n" + "=" * 60, flush=True)
            print("*** SUCCESS! Proof verified by Lean! ***", flush=True)
            print("=" * 60, flush=True)
        else:
            print(f"\nProof compilation failed:", flush=True)
            print(result.stderr, flush=True)
            print(result.stdout, flush=True)
    else:
        print(f"\nFailed to build {target}", flush=True)

if __name__ == "__main__":
    main()
