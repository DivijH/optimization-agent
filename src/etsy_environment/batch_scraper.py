import os
import json
import time
import argparse
import threading
from queue import Queue, Empty
from tqdm import tqdm

from webpage_downloader import EtsyPageDownloader


def collect_unique_queries(personas_dir: str) -> list[str]:
    """Iterate over every JSON file in *personas_dir* and collect the
    values under the ``search_queries`` key.

    Returns a sorted list of unique queries.
    """
    queries: set[str] = set()

    for filename in os.listdir(personas_dir):
        if not filename.endswith(".json"):
            continue

        path = os.path.join(personas_dir, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for q in data.get("search_queries", []):
                if isinstance(q, str) and q.strip():
                    queries.add(q.strip())
        except Exception as exc:
            # Keep going even if one file is malformed
            print(f"[WARN] Could not process {path}: {exc}")

    return sorted(queries)


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
        default=0.0,
        help="Optional delay (in seconds) between processing each query.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="How many worker threads to use for downloading (default: 4)",
    )
    args = parser.parse_args()

    # === Locate the personas directory ===
    # This file lives in <repo>/src/etsy_environment/batch_scraper.py
    # Going three levels up lands us in the repository root (<repo>/)
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    personas_dir = os.path.join(repo_root, "data", "personas")

    if not os.path.isdir(personas_dir):
        raise SystemExit(f"Personas directory not found: {personas_dir}")

    queries = collect_unique_queries(personas_dir)
    num_queries = len(queries)
    print(f"Discovered {num_queries} unique search queries across persona files.")

    if not queries:
        print("No queries found - exiting.")
        return

    # === Set up multi-worker processing ===
    # Overall progress bar occupies the first row (position 0)
    overall_bar = tqdm(total=num_queries, desc="Overall", position=0, unit="query")

    # A queue feeding queries to workers
    q: Queue[str] = Queue()
    for query in queries:
        q.put(query)

    def worker(worker_id: int) -> None:
        """Worker thread that consumes *queries* from *q*."""
        downloader = EtsyPageDownloader(progress_position=worker_id)

        # Keep track of how many search queries this worker has processed so far
        processed_count = 0

        while True:
            try:
                query = q.get_nowait()
            except Empty:
                break

            try:
                # Prefix shown in the per-worker tqdm bar
                prefix = f"Worker {worker_id} | Processed: {processed_count} | "
                downloader.download_search_results(query, num_pages=args.num_pages, desc_prefix=prefix)
            except Exception as exc:
                tqdm.write(f"[Worker {worker_id}][ERROR] Failed to download '{query}': {exc}")

            # Mark progress
            overall_bar.update(1)
            processed_count += 1
            q.task_done()

            if args.delay:
                time.sleep(args.delay)

    # Spawn worker threads
    threads: list[threading.Thread] = []
    num_workers = max(1, args.workers)
    for wid in range(1, num_workers + 1):  # start positions at 1 (0 reserved for overall)
        t = threading.Thread(target=worker, args=(wid,), daemon=True)
        t.start()
        threads.append(t)

    # Wait for all queries to be processed
    for t in threads:
        t.join()

    overall_bar.close()


if __name__ == "__main__":
    main() 