# main.py
from benchmark import run_benchmark
from mutation_test import run_mutation_experiment


def main():
    print("Running Bharat Bazaar problem 1 pipeline...\n")

    # 1. benchmark and part 6 planner breaker
    run_benchmark()

    print("\n" + "-" * 80 + "\n")

    # 2. part 5 mutation test
    run_mutation_experiment()

    print("\nDone. All benchmarks and experiments finished.")


if __name__ == "__main__":
    main()