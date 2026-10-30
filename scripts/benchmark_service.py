import argparse
import time

import numpy as np
import requests

DEFAULT_URL = "http://127.0.0.1:8000/search"
TEST_QUERIES = [
    "How to calculate age based on DateTime in C#?",
    "Execute multiple queries in single SQLStatement",
    "Resolve Java widening vs autoboxing conflict in method overload",
    "Docker container exit with status 137 out of memory",
    "Pandas dataframe convert column to datetime format",
    "Spring Boot bean creation exception autowired dependency failed",
    "React useEffect infinite loop trigger state update",
    "Git discard all local uncommitted changes permanently",
    "Python multiprocessing pool vs ThreadPoolExecutor benchmark",
    "Nginx proxy_pass websocket connection upgrade 502 bad gateway",
]
LATENCY_FIELDS = {
    "dense": "dense_encode_and_search_ms",
    "bm25": "bm25_search_ms",
    "rrf": "rrf_fusion_ms",
    "rerank": "cross_encoder_rerank_ms",
}


def run_benchmark(num_requests: int = 100, url: str = DEFAULT_URL) -> None:
    latencies = []
    stage_latencies = {name: [] for name in LATENCY_FIELDS}
    print(f"[*] Benchmarking {num_requests} requests at {url}...")

    with requests.Session() as session:
        for index in range(num_requests):
            query = TEST_QUERIES[index % len(TEST_QUERIES)]
            started = time.perf_counter()
            try:
                response = session.post(
                    url,
                    json={"query": query, "top_k": 10},
                    timeout=30,
                )
                response.raise_for_status()
                result = response.json()
            except (requests.RequestException, ValueError) as error:
                print(f"Request {index + 1} failed: {error}")
                continue

            latencies.append((time.perf_counter() - started) * 1000)
            for name, field in LATENCY_FIELDS.items():
                stage_latencies[name].append(result["latency"][field])

    if not latencies:
        print("No requests completed successfully.")
        return

    print("\n## End-to-end service latency")
    print(f"Successful requests: {len(latencies)} / {num_requests}")
    print(f"Throughput: {1000 / np.mean(latencies):.2f} req/s")
    print(f"Mean: {np.mean(latencies):.2f} ms")
    for percentile in (50, 95, 99):
        print(f"p{percentile}: {np.percentile(latencies, percentile):.2f} ms")

    print("\nMean stage latency")
    for name, values in stage_latencies.items():
        print(f"{name}: {np.mean(values):.2f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the retrieval API over HTTP.")
    parser.add_argument("--requests", type=int, default=100, help="Number of requests to send")
    parser.add_argument("--url", default=DEFAULT_URL, help="Search endpoint URL")
    args = parser.parse_args()
    if args.requests < 1:
        parser.error("--requests must be at least 1")
    run_benchmark(args.requests, args.url)


if __name__ == "__main__":
    main()
