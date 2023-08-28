import json
from tqdm import tqdm
import yaml
from openai_util import model_call

# Load configuration from YAML file
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


def read_and_validate_file(file_path):
    """Read a text file and return a list of sentences.

    Args:
        file_path (str): Path to the text file.

    Returns:
        list: List of sentences.
    """
    with open(file_path, "r") as f:
        data = f.readlines()
    data = [line.strip() for line in data]
    return data


def augment_data(data):
    """Placeholder for data augmentation logic.

    Args:
        data (list): List of sentences.

    Raises:
        NotImplementedError: Function not implemented yet.
    """
    raise NotImplementedError


def generate_and_write_responses(data, output_file="generated_data.jsonl"):
    generated_responses = []
    data_repetition = config["fine_tuning"]["data_repetition"]

    for sentence in tqdm(data, desc="Generating responses"):
        assistant_response = model_call(
            user_message=sentence,
            system_message=config["generation"]["system_prompt"],
            max_tokens=config["generation"]["max_tokens"],
            temperature=config["generation"]["temperature"],
        )
        entry = {
            "messages": [
                {"role": "user", "content": sentence},
                {"role": "assistant", "content": assistant_response},
            ]
        }
        generated_responses.append(entry)

    with open(output_file, "w") as f:
        for entry in generated_responses * data_repetition:
            f.write(json.dumps(entry) + "\n")
