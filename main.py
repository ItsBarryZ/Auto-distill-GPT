import argparse
from openai_util import *
from util import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "file_path", help="Path to the input file containing sentences."
    )
    parser.add_argument(
        "--augment", help="Apply data augmentation", action="store_true"
    )

    args = parser.parse_args()

    data = read_and_validate_file(args.file_path)

    if args.augment:
        data = augment_data(data)

    generate_and_write_responses(data)

    # File upload
    file_id = upload_training_data("generated_data.jsonl")

    # Fine-tuning
    fine_tuning_id = fine_tune_model(file_id, epochs=config["fine_tuning"]["epochs"])
    print(f"Fine-tuning job finished with new model: {fine_tuning_id}")


if __name__ == "__main__":
    main()
