"""
Download Mathlib4 pre-built .olean cache from Azure Blob Storage.

Workaround for systems where `lake exe cache get` is blocked by
Windows Application Control (WDAC/AppLocker).

Usage:  python get_cache.py
"""

import json
import hashlib
import os
import sys
import tarfile
import tempfile
import urllib.request
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

AZURE_URL = "https://lakecache.blob.core.windows.net/mathlib4"
PROJECT_ROOT = Path(__file__).resolve().parent


def get_mathlib_commit() -> str:
    """Read the Mathlib commit hash from lake-manifest.json."""
    manifest = PROJECT_ROOT / "lake-manifest.json"
    with open(manifest) as f:
        data = json.load(f)
    for pkg in data.get("packages", []):
        if pkg.get("name") == "mathlib":
            rev = pkg.get("rev")
            if rev:
                return rev
    raise RuntimeError("Could not find mathlib revision in lake-manifest.json")


def get_lean_toolchain() -> str:
    """Read the Lean toolchain version."""
    tc = PROJECT_ROOT / "lean-toolchain"
    return tc.read_text().strip()


def list_olean_dirs(mathlib_pkg: Path) -> list[Path]:
    """Find all directories containing .olean files under the build dir."""
    build_lib = mathlib_pkg / ".lake" / "build" / "lib"
    if build_lib.exists():
        return [build_lib]
    return []


def get_all_lean_files(mathlib_pkg: Path) -> list[str]:
    """Get all .lean source file paths relative to mathlib root."""
    lean_files = []
    for root, dirs, files in os.walk(mathlib_pkg):
        # Skip hidden dirs and build dirs
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for f in files:
            if f.endswith('.lean'):
                rel = os.path.relpath(os.path.join(root, f), mathlib_pkg)
                lean_files.append(rel.replace('\\', '/'))
    return lean_files


def compute_file_hash(filepath: Path) -> str:
    """Compute the hash used by Mathlib cache for a source file."""
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()[:20]


def get_cache_url(toolchain: str, commit: str, module_path: str) -> str:
    """Construct the Azure Blob URL for a cached .olean archive.
    
    The cache key format used by Mathlib is:
      {toolchain_hash}/{commit}/{module_path}.tar.zst
    But the exact format can vary. We try the most common patterns.
    """
    # Module path: e.g., "Mathlib/Algebra/Basic" (no extension)
    return f"{AZURE_URL}/{commit}/file/{module_path}.tar.zst"


def download_file(url: str, dest: Path) -> bool:
    """Download a file from URL to dest. Returns True on success."""
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=30) as resp:
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, 'wb') as f:
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    f.write(chunk)
        return True
    except Exception:
        return False


def unpack_ltar(archive: Path, dest_dir: Path) -> bool:
    """Unpack a .tar.zst or .ltar file using leantar or tar."""
    try:
        # Try using leantar (from Lean toolchain)
        leantar = Path(os.environ.get("USERPROFILE", "")) / ".elan" / "toolchains"
        # Find the leantar executable
        for tc_dir in leantar.iterdir():
            lt = tc_dir / "bin" / "leantar.exe"
            if lt.exists():
                subprocess.run(
                    [str(lt), "-x", str(archive)],
                    cwd=str(dest_dir),
                    check=True,
                    capture_output=True
                )
                return True
    except Exception:
        pass
    
    try:
        # Fallback: use Python tarfile (doesn't handle .zst natively)
        import lzma
        with tarfile.open(archive) as tf:
            tf.extractall(dest_dir)
        return True
    except Exception:
        return False


def download_cache_manifest(commit: str) -> dict | None:
    """Try to download the cache manifest for a given commit."""
    url = f"{AZURE_URL}/{commit}/manifest.json"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def try_leantar_approach(commit: str, mathlib_pkg: Path) -> bool:
    """Download and unpack cache using leantar from Lean toolchain."""
    
    # Find leantar
    elan_dir = Path(os.environ.get("USERPROFILE", "")) / ".elan" / "toolchains"
    leantar_exe = None
    if elan_dir.exists():
        for tc_dir in elan_dir.iterdir():
            lt = tc_dir / "bin" / "leantar.exe"
            if lt.exists():
                leantar_exe = lt
                break
    
    if not leantar_exe:
        print("  leantar.exe not found in elan toolchains")
        return False
    
    print(f"  Found leantar: {leantar_exe}")
    
    # Get list of .lean files and build corresponding cache URLs
    build_lib = mathlib_pkg / ".lake" / "build" / "lib"
    build_lib.mkdir(parents=True, exist_ok=True)
    
    # The cache stores files keyed by their content hash
    # Let's try downloading the pack file instead
    pack_url = f"{AZURE_URL}/{commit}/pack.tar.zst"
    
    print(f"  Trying pack download: {pack_url}")
    tmp = Path(tempfile.mkdtemp(dir=str(PROJECT_ROOT)))
    pack_file = tmp / "pack.tar.zst"
    
    if download_file(pack_url, pack_file):
        print(f"  Downloaded pack ({pack_file.stat().st_size / 1024 / 1024:.1f} MB)")
        try:
            result = subprocess.run(
                [str(leantar_exe), "-x", str(pack_file)],
                cwd=str(build_lib),
                check=True,
                capture_output=True,
                text=True
            )
            print("  Unpacked successfully!")
            pack_file.unlink()
            tmp.rmdir()
            return True
        except subprocess.CalledProcessError as e:
            print(f"  leantar unpack failed: {e.stderr}")
    else:
        print("  Pack download failed, trying individual files...")
    
    # Clean up
    if pack_file.exists():
        pack_file.unlink()
    if tmp.exists():
        tmp.rmdir()
    
    return False


def main():
    print("=" * 60)
    print("Mathlib4 Cache Downloader")
    print("=" * 60)
    
    # Get Mathlib commit
    try:
        commit = get_mathlib_commit()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    toolchain = get_lean_toolchain()
    mathlib_pkg = PROJECT_ROOT / ".lake" / "packages" / "mathlib"
    
    print(f"  Toolchain: {toolchain}")
    print(f"  Mathlib commit: {commit}")
    print(f"  Mathlib path: {mathlib_pkg}")
    print()
    
    if not mathlib_pkg.exists():
        print("Error: Mathlib package not found. Run 'lake update' first.")
        sys.exit(1)
    
    # Strategy 1: Try downloading the manifest
    print("[1/3] Checking cache manifest...")
    manifest = download_cache_manifest(commit)
    if manifest:
        print(f"  Found manifest with {len(manifest)} entries")
    else:
        print("  No manifest found (this is normal)")
    
    # Strategy 2: Try the leantar pack approach
    print("[2/3] Trying leantar pack download...")
    if try_leantar_approach(commit, mathlib_pkg):
        print("\nCache downloaded successfully!")
        return
    
    # Strategy 3: Try individual file downloads
    print("[3/3] Trying individual .olean downloads...")
    
    # Get all .lean source files in Mathlib
    lean_files = get_all_lean_files(mathlib_pkg)
    print(f"  Found {len(lean_files)} Lean source files")
    
    # Try downloading a few files to test the URL pattern
    test_files = [f for f in lean_files if "Mathlib" in f][:5]
    build_lib = mathlib_pkg / ".lake" / "build" / "lib"
    
    success = 0
    fail = 0
    for lf in test_files:
        module = lf.replace('.lean', '').replace('/', '-')
        url = f"{AZURE_URL}/{commit}/file/{module}.ltar"
        tmp_file = Path(tempfile.mktemp(suffix='.ltar'))
        if download_file(url, tmp_file):
            print(f"  ✓ {module}")
            success += 1
            tmp_file.unlink(missing_ok=True)
        else:
            # Try .ltar.zst
            url2 = f"{AZURE_URL}/{commit}/file/{module}.ltar.zst"
            if download_file(url2, tmp_file):
                print(f"  ✓ {module} (.zst)")
                success += 1
                tmp_file.unlink(missing_ok=True)
            else:
                print(f"  ✗ {module}")
                fail += 1
                tmp_file.unlink(missing_ok=True)
    
    if success == 0:
        print("\n  Could not download any cache files.")
        print("  The cache URL format may have changed.")
        print("\n  Alternative: try building from source with 'lake build'")
        print("  (this will take several hours)")
    else:
        print(f"\n  Downloaded {success}/{success+fail} test files")
        print("  Full download would need to fetch all files...")
    
    print("\nDone.")


if __name__ == "__main__":
    main()
