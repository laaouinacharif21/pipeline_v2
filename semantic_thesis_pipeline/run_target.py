import argparse

from src.pipeline.extract_embeddings_target import run_extraction_target
from src.pipeline.compute_metrics import run_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str,
                        default="data/raw/semantic_sentences/sentences.json")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    run_extraction_target(args.model, args.dataset, batch_size=args.batch_size)
    run_metrics(args.model)


if __name__ == "__main__":
    main()
