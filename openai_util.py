import os
import openai
import time
from multiprocessing import Process, Value
from util import animated_loading
import ctypes

# Initialize OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")


def upload_training_data(file_path):
    stop_event = Value(ctypes.c_bool, False)
    p = Process(target=animated_loading, args=(stop_event, "Uploading file"))
    p.start()

    with open(file_path, "rb") as f:
        data_file = openai.File.create(file=f, purpose="fine-tune")
    file_id = data_file.id

    while True:
        file_status = openai.File.retrieve(file_id).status
        if file_status in ["processed", "failed"]:
            stop_event.value = True
            p.join()
            print(f"\rFile upload {file_status} with ID: {file_id}      ")
            break
        time.sleep(1)

    return file_id


def fine_tune_model(file_id, epochs):
    stop_event = Value(ctypes.c_bool, False)
    p = Process(target=animated_loading, args=(stop_event, "Fine-tuning model"))
    p.start()

    training_job = openai.FineTuningJob.create(
        training_file=file_id,
        model="gpt-3.5-turbo",
        hyperparameters={"n_epochs": epochs},
    )
    job_id = training_job.id

    while True:
        job_status = openai.FineTuningJob.retrieve(job_id).status
        if job_status in ["succeeded", "failed"]:
            stop_event.value = True
            p.join()
            print(f"\rFine-tuning {job_status} with job ID: {job_id} ")
            break
        time.sleep(10)

    return job_id


def model_call(
    user_message,
    system_message="",
    model_id="gpt-3.5-turbo",
    max_tokens=None,
    temperature=None,
):
    """
    Generate a response using a language model.

    Parameters:
        user_message (str): The message from the user.
        system_message (str, optional): The system message to guide the model's behavior. Defaults to ''.
        model_id (str, optional): The ID of the model to use. Defaults to 'gpt-3.5-turbo'.
        max_tokens (int, optional): Maximum number of tokens for the generated response. Defaults to None.
        temperature (float, optional): Sampling temperature. Defaults to None.

    Returns:
        str: The generated message from the model.
    """
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]
    completion = openai.ChatCompletion.create(
        model=model_id,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return completion.choices[0].message.content
