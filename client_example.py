"""
client_example.py — VibeVoice API usage examples
=================================================
Shows how to call every endpoint from Python.

    python client_example.py                     # runs all examples
    python client_example.py --host myserver.io  # remote server
    python client_example.py --url http://localhost:8000

Requires: requests  (pip install requests)
"""

import argparse
import os
import sys
import time

try:
    import requests
except ImportError:
    print("Install requests:  pip install requests")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--url",  default="http://localhost:8000",
                   help="Base URL of the VibeVoice API server")
    p.add_argument("--voice", default="en-Alice_woman",
                   help="Voice to use in examples")
    p.add_argument("--out",  default=".",
                   help="Directory to save downloaded WAV files")
    return p.parse_args()


def hr(title=""):
    print()
    print("─" * 60)
    if title:
        print(f"  {title}")
        print("─" * 60)


# ---------------------------------------------------------------------------
# 1. Health check
# ---------------------------------------------------------------------------
def example_health(base: str):
    hr("GET /health")
    r = requests.get(f"{base}/health", timeout=10)
    r.raise_for_status()
    data = r.json()
    print(f"  Status : {data['status']}")
    print(f"  GPU    : {data['gpu']}")
    print(f"  VRAM   : {data['vram_used_gb']} / {data['vram_total_gb']} GB")


# ---------------------------------------------------------------------------
# 2. List voices
# ---------------------------------------------------------------------------
def example_voices(base: str):
    hr("GET /voices")
    r = requests.get(f"{base}/voices", timeout=10)
    r.raise_for_status()
    data = r.json()
    print(f"  {data['count']} voices available:")
    for v in data["voices"]:
        print(f"    {v}")


# ---------------------------------------------------------------------------
# 3. /generate  — stream WAV bytes directly
# ---------------------------------------------------------------------------
def example_generate(base: str, voice: str, out_dir: str):
    hr("POST /generate  (stream WAV bytes)")
    payload = {
        "text":       "Hello! This is a live test of the VibeVoice API server.",
        "voice":      voice,
        "cfg_scale":  1.32,
        "ddpm_steps": 10,
    }
    t0 = time.time()
    r  = requests.post(f"{base}/generate", json=payload, timeout=120)
    r.raise_for_status()
    elapsed = time.time() - t0

    out_path = os.path.join(out_dir, "example_generate.wav")
    with open(out_path, "wb") as f:
        f.write(r.content)

    dur = float(r.headers.get("X-Duration-S",    0))
    gen = float(r.headers.get("X-Generation-S",  elapsed))
    rtf = float(r.headers.get("X-RTF",           0))
    print(f"  Saved : {out_path}  ({len(r.content)//1024} KB)")
    print(f"  Audio : {dur:.1f}s  generated in {gen:.1f}s  RTF={rtf:.2f}x")


# ---------------------------------------------------------------------------
# 4. /generate_url  — get a download URL back
# ---------------------------------------------------------------------------
def example_generate_url(base: str, voice: str, out_dir: str):
    hr("POST /generate_url  (JSON + download URL)")
    payload = {
        "text":  "The quick brown fox jumps over the lazy dog.",
        "voice": voice,
    }
    r = requests.post(f"{base}/generate_url", json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()

    print(f"  File ID     : {data['file_id']}")
    print(f"  URL         : {data['url']}")
    print(f"  Duration    : {data['duration_s']}s")
    print(f"  Generated   : {data['generation_s']}s")
    print(f"  Size        : {data['size_kb']} KB")

    # Download the file
    dl = requests.get(f"{base}{data['url']}", timeout=30)
    dl.raise_for_status()
    out_path = os.path.join(out_dir, data["filename"])
    with open(out_path, "wb") as f:
        f.write(dl.content)
    print(f"  Downloaded  : {out_path}")


# ---------------------------------------------------------------------------
# 5. /batch_generate  — multiple texts in one call
# ---------------------------------------------------------------------------
def example_batch(base: str, voice: str, out_dir: str):
    hr("POST /batch_generate  (multiple texts → array of URLs)")
    payload = {
        "items": [
            {
                "text":            "Welcome to VibeVoice. This is item one.",
                "voice":           voice,
                "output_filename": "batch_item_01.wav",
            },
            {
                "text":            "Item two. The model stays loaded between requests.",
                "voice":           voice,
                "output_filename": "batch_item_02.wav",
            },
            {
                "text":            "Item three. Batch generation is fast because the model never reloads.",
                "voice":           voice,
                "cfg_scale":       1.30,   # custom per-item settings
                "ddpm_steps":      8,
                "output_filename": "batch_item_03.wav",
            },
        ]
    }

    t0 = time.time()
    r  = requests.post(f"{base}/batch_generate", json=payload, timeout=300)
    r.raise_for_status()
    data    = r.json()
    elapsed = time.time() - t0

    print(f"  {data['succeeded']} succeeded  {data['failed']} failed  total={data['total_s']}s")
    for item in data["results"]:
        if item.get("error"):
            print(f"  [{item['index']}] ERROR: {item['error']}")
        else:
            print(
                f"  [{item['index']}] {item['filename']}  "
                f"dur={item['duration_s']}s  gen={item['generation_s']}s  "
                f"{item['size_kb']}KB  → {item['url']}"
            )
            # Download each file
            dl = requests.get(f"{base}{item['url']}", timeout=30)
            dl.raise_for_status()
            out_path = os.path.join(out_dir, item["filename"])
            with open(out_path, "wb") as f:
                f.write(dl.content)
            print(f"             saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    base = args.url.rstrip("/")
    os.makedirs(args.out, exist_ok=True)

    print(f"VibeVoice API client — server: {base}")

    try:
        example_health(base)
        example_voices(base)
        example_generate(base, args.voice, args.out)
        example_generate_url(base, args.voice, args.out)
        example_batch(base, args.voice, args.out)
    except requests.exceptions.ConnectionError:
        print(f"\nERROR: Cannot connect to {base}")
        print("       Start the server first:  bash start_server.sh")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"\nHTTP error: {e}")
        print(f"Response: {e.response.text[:400]}")
        sys.exit(1)

    hr()
    print("  All examples completed successfully.")


if __name__ == "__main__":
    main()
