import argparse

from src.pipeline.extract_embeddings import run_extraction
from src.pipeline.compute_metrics import run_metrics
from src.pipeline.generate_plots import run_plots
from src.pipeline.select_layers import run_layer_selection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="data/raw/semantic_sentences/sentences.json")
    parser.add_argument("--stage", type=str, default="full",
                        choices=["extract", "metrics", "plots", "select_layers", "full"])

    args = parser.parse_args()

    if args.stage == "extract":
        run_extraction(args.model, args.dataset)

    elif args.stage == "metrics":
        run_metrics(args.model)

    elif args.stage == "plots":
        run_plots(args.model)

    elif args.stage == "select_layers":
        run_layer_selection(args.model)

    elif args.stage == "full":
        run_extraction(args.model, args.dataset)
        run_metrics(args.model)
        run_plots(args.model)
        run_layer_selection(args.model)


if __name__ == "__main__":
    main()
