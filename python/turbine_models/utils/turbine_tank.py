import numpy as np
import requests as req
import subprocess
import hashlib
import os
from tqdm.std import tqdm
from pathlib import Path
from google.cloud import storage
import json
import numpy as np

custom_path = os.getenv("SHARK_TANK_CACHE_DIR")
if custom_path is not None:
    if not os.path.exists(custom_path):
        os.mkdir(custom_path)

    WORKDIR = custom_path

    print(f"Using {WORKDIR} as local shark_tank cache directory.")
else:
    WORKDIR = os.path.join(str(Path.home()), ".local/shark_tank/")
    print(
        f"shark_tank local cache is located at {WORKDIR} . You may change this by assigning the SHARK_TANK_CACHE_DIR environment variable."
    )
os.makedirs(WORKDIR, exist_ok=True)

def download_public_file(full_gs_url, destination_folder_name) -> bool:
    """Downloads a public blob from the bucket, returns False if it fails"""
    storage_client = storage.Client.create_anonymous_client()
    bucket_name = full_gs_url.split("/")[2]
    desired_file = full_gs_url.split("/")[-1]
    source_blob_name = "/".join(full_gs_url.split("/")[3:-1])
    destination_folder_name, dest_filename = os.path.split(destination_folder_name)
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=source_blob_name)
    if not os.path.exists(destination_folder_name):
        os.mkdir(destination_folder_name)
    for blob in blobs:
        blob_name = blob.name.split("/")[-1]
        if blob_name == desired_file:
            destination_filename = os.path.join(destination_folder_name, dest_filename)
            with open(destination_filename, "wb") as f:
                with tqdm.wrapattr(f, "write", total=blob.size) as file_obj:
                    storage_client.download_blob_to_file(blob, file_obj)
            return True
    return False

def download_public_folder(full_gs_url, destination_folder_name):
    """Downloads a public blob from the bucket."""
    storage_client = storage.Client.create_anonymous_client()
    # typical full_gs_url = "gs://shark_tank/some_prefix/model_name"
    bucket_name = full_gs_url.split("/")[2]
    if bucket_name != "shark_tank":
        raise ValueError(
            f"Bucket name {bucket_name} does not match expected bucket name shark_tank."
        )
    source_blob_name = "/".join(full_gs_url.split("/")[3:])
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=source_blob_name)
    if not os.path.exists(destination_folder_name):
        os.mkdir(destination_folder_name)
    for blob in blobs:
        blob_name = blob.name.split("/")[-1]
        blob.download_to_filename(os.path.join(destination_folder_name, blob_name))



# Checks whether the directory and files exists.
def check_dir_exists(model_name) -> bool:
    model_dir = os.path.join(WORKDIR, model_name)

    if os.path.isdir(model_dir):
        if (
            os.path.isfile(os.path.join(model_dir, model_name + ".mlir"))
            and os.path.isfile(os.path.join(model_dir, "hash.npy"))
        ):
            print(
                f"""Model artifacts for {model_name} found at {WORKDIR}..."""
            )
            return True
    return False


def _internet_connected() -> bool:

    try:
        req.get("http://1.1.1.1")
        return True
    except:
        return False


def get_stable_blob_prefix() -> str:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    src = os.path.join(dir_path, "tank_version.json")
    with open(src, "r") as f:
        data = json.loads(f.read())
        prefix_kw = data["version"]
    print(f"Checking for updates from gs://shark_tank/{prefix_kw}")
    return prefix_kw


def get_sharktank_prefix()-> str:
    tank_prefix = ""
    if not _internet_connected():
        print(
            "No internet connection. Using the model already present in the tank."
        )
        tank_prefix = "none"
    else:
        desired_prefix = get_stable_blob_prefix()
        storage_client_a = storage.Client.create_anonymous_client()
        base_bucket_name = "shark_tank"
        base_bucket = storage_client_a.bucket(base_bucket_name)
        dir_blobs = base_bucket.list_blobs(prefix=f"{desired_prefix}")
        for blob in dir_blobs:
            dir_blob_name = blob.name.split("/")
            if desired_prefix in dir_blob_name[0]:
                tank_prefix = dir_blob_name[0]
                break
            else:
                continue
        if tank_prefix == "":
            print(
                f"shark_tank bucket not found matching ({desired_prefix}). Defaulting to nightly."
            )
            tank_prefix = "nightly"
    return tank_prefix


def download_model(
    model_name: str,
    tank_url: str = None,
) -> str:
    model_name = model_name.replace("/", "_")
    prefix = get_sharktank_prefix()
    model_dir = os.path.join(WORKDIR, model_name)

    if not tank_url:
        tank_url = "gs://shark_tank/" + prefix

    full_gs_url = tank_url.rstrip("/") + "/" + model_name
    if not check_dir_exists(model_name):
        print(
            f"Downloading artifacts for model {model_name} from: {full_gs_url}"
        )
        download_public_folder(full_gs_url, model_dir)

    else:
        if not _internet_connected():
            print(
                "No internet connection. Using the model already present in the tank."
            )
        else:
            local_hash = str(np.load(os.path.join(model_dir, "hash.npy")))
            gs_hash_url = (
                tank_url.rstrip("/") + "/" + model_name + "/hash.npy"
            )
            download_public_file(
                gs_hash_url,
                os.path.join(model_dir, "upstream_hash.npy"),
            )
            try:
                upstream_hash = str(
                    np.load(os.path.join(model_dir, "upstream_hash.npy"))
                )
            except FileNotFoundError:
                print(f"Model artifact hash not found at {model_dir}.")
                upstream_hash = None
            if local_hash != upstream_hash:
                print(f"Updating artifacts for model {model_name}...")
                download_public_folder(full_gs_url, model_dir)

            else:
                print(
                    "Local and upstream hashes match. Using cached model artifacts."
                )

    model_dir = os.path.join(WORKDIR, model_name)
    mlir_filename = os.path.join(model_dir, model_name)
    print(
        f"Verifying that model artifacts were downloaded successfully to {mlir_filename}..."
    )
    assert os.path.exists(mlir_filename), f"MLIR not found at {mlir_filename}"

    return mlir_filename

def create_hash(file_name):
    with open(file_name, "rb") as f:
        file_hash = hashlib.blake2b(digest_size=64)
        while chunk := f.read(2**10):
            file_hash.update(chunk)

    return file_hash.hexdigest()

def get_short_git_sha()->str:
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('utf-8').strip()
    except FileNotFoundError:
        return None

def upload_to_tank(filename:str, delete_local_files=True) -> None:
    """Uploads a file to the shark_tank. 
        filename: str, the name of the file to upload, should be the model_name (hf_model_name)
    """
    storage_client = storage.Client.create_anonymous_client()
    prefix = get_short_git_sha()
    assert prefix is not None, "Could not get git sha. Is git installed?"
    bucket_name = f"shark_tank/turbine/{prefix}"
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(filename)
    np.save(os.path.join(os.getcwd(), "hash"), np.array(create_hash(filename)))
    assert os.path.exists(os.path.join(os.getcwd(), "hash.npy")), "Could not create hash file."
    blob.upload_from_filename(filename)
    blob.upload_from_filename(os.path.join(os.getcwd(), "hash.npy"))
    if delete_local_files:
        #clean up
        os.remove(os.path.join(os.getcwd(), "hash.npy")) 
        os.remove(filename)
    