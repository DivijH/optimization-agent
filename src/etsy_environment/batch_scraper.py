import os
import json
import time
import argparse
import threading
from queue import Queue, Empty
from tqdm import tqdm

from webpage_downloader import EtsyPageDownloader


def format_ranges(numbers: list[int]) -> str:
    """Turn a list of numbers into a string of ranges.

    For example: [1, 2, 3, 5, 6, 8] -> "1-3, 5-6, 8"
    """
    if not numbers:
        return ""

    # Sort and remove duplicates
    nums = sorted(list(set(numbers)))

    ranges: list[str] = []
    range_start = nums[0]

    for i in range(1, len(nums)):
        # If the number is not consecutive, we close the current range
        if nums[i] != nums[i - 1] + 1:
            if range_start == nums[i - 1]:
                ranges.append(str(range_start))
            else:
                ranges.append(f"{range_start}-{nums[i-1]}")
            range_start = nums[i]

    # Add the final range
    if range_start == nums[-1]:
        ranges.append(str(range_start))
    else:
        ranges.append(f"{range_start}-{nums[-1]}")

    return ", ".join(ranges)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Scrape Etsy search-result pages for every query found in persona files."
        )
    )
    parser.add_argument(
        "--num-pages",
        type=int,
        default=1,
        help="How many result pages to download per search query (default: 1)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.5,
        help="Optional delay (in seconds) between processing each query.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="How many worker threads to use for downloading (default: 4)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="The starting number of the persona file to process (inclusive).",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="The ending number of the persona file to process (inclusive).",
    )
    args = parser.parse_args()

    if (args.start is None) != (args.end is None):
        raise SystemExit("Must specify both --start and --end, or neither.")

    if args.start is not None and args.end is not None and args.start > args.end:
        raise SystemExit(
            f"--start value ({args.start}) cannot be greater than --end value ({args.end})."
        )

    # === Locate the personas directory ===
    # This file lives in <repo>/src/etsy_environment/batch_scraper.py
    # Going three levels up lands us in the repository root (<repo>/)
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    personas_dir = os.path.join(repo_root, "data", "personas")

    if not os.path.isdir(personas_dir):
        raise SystemExit(f"Personas directory not found: {personas_dir}")

    # === Discover personas to process ===
    persona_numbers: list[int]
    if args.start is not None and args.end is not None:
        persona_numbers = list(range(args.start, args.end + 1))
    else:
        # Discover all available personas by scanning the directory
        persona_numbers = []
        for filename in sorted(os.listdir(personas_dir)):
            if not (
                filename.startswith("virtual customer ") and filename.endswith(".json")
            ):
                continue
            try:
                num_str = filename.replace("virtual customer ", "").replace(".json", "")
                persona_numbers.append(int(num_str))
            except ValueError:
                # Not a numbered persona file, so we skip it
                print(f"[WARN] Could not parse persona number from: {filename}")
                continue

    if not persona_numbers:
        print("No personas found to process - exiting.")
        return

    num_personas = len(persona_numbers)
    print(f"Discovered {num_personas} personas to process.")

    # === Set up multi-worker processing ===
    # Overall progress bar occupies the first row (position 0)
    overall_desc = "Overall"
    if args.start is not None and args.end is not None:
        overall_desc += f" (Personas {args.start}-{args.end})"
    overall_bar = tqdm(total=num_personas, desc=overall_desc, position=0, unit="persona")

    # A queue feeding persona numbers to workers
    q: Queue[int] = Queue()
    for p_num in persona_numbers:
        q.put(p_num)

    # Use a lock to safely update shared state from multiple threads
    completed_lock = threading.Lock()
    completed_personas: list[int] = []

    def worker(worker_id: int) -> None:
        """Worker thread that consumes *persona_numbers* from *q*."""
        downloader = EtsyPageDownloader(
            progress_position=worker_id, delay=args.delay
        )

        while True:
            try:
                persona_num = q.get_nowait()
            except Empty:
                break

            persona_filename = f"virtual customer {persona_num}.json"
            path = os.path.join(personas_dir, persona_filename)
            queries: list[str] = []
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                for query in data.get("search_queries", []):
                    if isinstance(query, str) and query.strip():
                        queries.append(query.strip())
            except Exception as exc:
                tqdm.write(f"[Worker {worker_id}][ERROR] Failed to read {path}: {exc}")
                # Mark as done even if it failed to be read
                q.task_done()
                continue

            for i, query in enumerate(queries):
                try:
                    # Prefix shown in the per-worker tqdm bar
                    prefix = f"Worker {worker_id} | Persona {persona_num} ({i+1}/{len(queries)}) | "
                    downloader.download_search_results(query, num_pages=args.num_pages, desc_prefix=prefix)
                except Exception as exc:
                    tqdm.write(f"[Worker {worker_id}][ERROR] Failed to download '{query}' for persona {persona_num}: {exc}")

            # Mark progress
            with completed_lock:
                completed_personas.append(persona_num)
                desc = f"Overall | Done: {format_ranges(completed_personas)}"
                if args.start is not None and args.end is not None:
                    desc += f" (Personas {args.start}-{args.end})"
                overall_bar.set_description(desc)

            overall_bar.update(1)
            q.task_done()

    # Spawn worker threads
    threads: list[threading.Thread] = []
    num_workers = max(1, args.workers)
    for wid in range(1, num_workers + 1):  # start positions at 1 (0 reserved for overall)
        t = threading.Thread(target=worker, args=(wid,), daemon=True)
        t.start()
        threads.append(t)

    # Wait for all queries to be processed
    q.join()

    overall_bar.close()


if __name__ == "__main__":
    main() 