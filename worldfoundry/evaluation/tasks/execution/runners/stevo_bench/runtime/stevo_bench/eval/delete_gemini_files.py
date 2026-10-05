#!/usr/bin/env python3
"""Delete all files uploaded to the Gemini Files API backend."""

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed


def main() -> None:
    from google import genai  # type: ignore

    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key:
        print("ERROR: GOOGLE_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    print("Listing uploaded files...")
    files = list(client.files.list())
    if not files:
        print("No files found.")
        return
    print(f"Found {len(files)} file(s). Deleting in parallel...")

    def delete(file) -> tuple[str, bool, str]:
        try:
            client.files.delete(name=file.name)
            return file.name, True, ""
        except Exception as e:
            return file.name, False, str(e)

    n_ok = n_err = 0
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(delete, f): f for f in files}
        for fut in as_completed(futures):
            name, ok, err = fut.result()
            if ok:
                n_ok += 1
            else:
                n_err += 1
                print(f"  FAILED {name}: {err}")
            done = n_ok + n_err
            if done % 50 == 0:
                print(f"  {done}/{len(files)} processed ({n_err} failed so far)...")

    print(f"\nDone: {n_ok} deleted, {n_err} failed.")


if __name__ == "__main__":
    main()
