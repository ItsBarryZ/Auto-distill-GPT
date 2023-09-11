import argparse
from tqdm import tqdm
from openai_util import upload_training_data, fine_tune_model, model_call
from util import (
    read_and_validate_file,
    augment_data,
    generate_and_write_responses,
    config,
)


# Load configuration from YAML file
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

    generated_data = []
    for sentence in tqdm(data, desc="Generating responses"):
        response = model_call(
            user_message=sentence,
            system_message=config["generation"]["system_prompt"],
            max_tokens=config["generation"]["max_tokens"],
            temperature=config["generation"]["temperature"],
        )
        generated_data.append({"input": sentence, "response": response})

    output_file = generate_and_write_responses(generated_data)

    # File upload
    file_id = upload_training_data(output_file)

    # Fine-tuning
    fine_tuning_id = fine_tune_model(file_id, epochs=config["fine_tuning"]["epochs"])
    print(f"Fine-tuning job finished with new model: {fine_tuning_id}")


if __name__ == "__main__":
    main()
