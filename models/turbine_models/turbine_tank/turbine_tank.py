# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from azure.storage.blob import BlobServiceClient

import subprocess
import datetime
import os
from pathlib import Path
from functools import cmp_to_key

custom_path = os.getenv("TURBINE_TANK_CACHE_DIR")
if custom_path is not None:
    if not os.path.exists(custom_path):
        os.mkdir(custom_path)

    WORKDIR = custom_path

    print(f"Using {WORKDIR} as local turbine_tank cache directory.")
else:
    WORKDIR = os.path.join(str(Path.home()), ".local/turbine_tank/")
    print(
        f"turbine_tank local cache is located at {WORKDIR} . You may change this by assigning the TURBINE_TANK_CACHE_DIR environment variable."
    )
os.makedirs(WORKDIR, exist_ok=True)

storage_account_key = os.environ.get("AZURE_STORAGE_ACCOUNT_KEY")
storage_account_name = os.environ.get("AZURE_STORAGE_ACCOUNT_NAME")
connection_string = os.environ.get("AZURE_CONNECTION_STRING")
container_name = os.environ.get("AZURE_CONTAINER_NAME")


def get_short_git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode("utf-8")
            .strip()
        )
    except FileNotFoundError:
        return None


def uploadToBlobStorage(file_path, file_name):
    # create our prefix (we use this to keep track of when and what version of turbine is being used)
    today = str(datetime.date.today())
    commit = get_short_git_sha()
    prefix = today + "_" + commit
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_blob_client(
        container=container_name, blob=prefix + "/" + file_name
    )
    blob = blob_client.from_connection_string(
        conn_str=connection_string,
        container_name=container_name,
        blob_name=blob_client.blob_name,
    )
    # we check to see if we already uploaded the blob (don't want to duplicate)
    if blob.exists():
        print(
            f"model artifacts have already been uploaded for {today} on the same github commit ({commit})"
        )
        return
    # upload to azure storage container tankturbine
    with open(file_path, "rb") as data:
        blob_client.upload_blob(data)
    print(f"Uploaded {file_name}.")


def checkAndRemoveIfDownloadedOld(model_name: str, model_dir: str, prefix: str):
    if os.path.isdir(model_dir) and len(os.listdir(model_dir)) > 0:
        for item in os.listdir(model_dir):
            item_path = os.path.join(model_dir, item)
            # model artifacts already downloaded and up to date
            # we check if model artifacts are behind using the prefix (day + git_sha)
            if os.path.isdir(item_path) and item == prefix:
                return True
            # model artifacts are behind, so remove for new download
            if os.path.isdir(item_path) and os.path.isfile(
                os.path.join(item_path, model_name + ".mlir")
            ):
                os.remove(os.path.join(item_path, model_name + ".mlir"))
                os.rmdir(item_path)
                return False
            if os.path.isdir(item_path) and os.path.isfile(
                os.path.join(item_path, model_name + "-param.mlir")
            ):
                os.remove(os.path.join(item_path, model_name + "-param.mlir"))
                os.rmdir(item_path)
                return False
    # did not downloaded this model artifacts yet
    return False


def download_public_folder(model_name: str, prefix: str, model_dir: str):
    """Downloads a folder of blobs in azure container."""
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(
        container=container_name
    )
    blob_list = container_client.list_blobs(name_starts_with=prefix)
    empty = True

    # go through the blobs with our target prefix
    # example prefix: "2024-02-13_26d6428/CompVis_stable-diffusion-v1-4-clip"
    for blob in blob_list:
        empty = False
        blob_client = blob_service_client.get_blob_client(
            container=container_name, blob=blob.name
        )
        # create path if directory doesn't exist locally
        dest_path = model_dir
        if not os.path.isdir(dest_path):
            os.makedirs(dest_path)
        # download blob into local turbine tank cache
        if "param" in blob.name:
            file_path = os.path.join(model_dir, model_name + "-param.mlir")
        else:
            file_path = os.path.join(model_dir, model_name + ".mlir")
        with open(file=file_path, mode="wb") as sample_blob:
            download_stream = blob_client.download_blob()
            sample_blob.write(download_stream.readall())

    if empty:
        print(f"Model ({model_name}) has not been uploaded yet")
        return True

    return False


# sort blobs by last modified
def compare(item1, item2):
    if item1.last_modified < item2.last_modified:
        return -1
    elif item1.last_modified < item2.last_modified:
        return 1
    else:
        return 0


def downloadModelArtifacts(model_name: str) -> str:
    model_name = model_name.replace("/", "_")
    container_client = BlobServiceClient.from_connection_string(
        connection_string
    ).get_container_client(container=container_name)
    blob_list = container_client.list_blobs()
    # get the latest blob uploaded to turbine tank (can't use [] notation for blob_list)
    blob_list = sorted(blob_list, key=cmp_to_key(compare))
    for blob in blob_list:
        latest_blob = blob
    # get the prefix for the latest blob (2024-02-13_26d6428)
    download_latest_prefix = latest_blob.name.split("/")[0]
    model_dir = os.path.join(WORKDIR, model_name)
    # check if we already downloaded the model artifacts for this day + commit
    exists = checkAndRemoveIfDownloadedOld(
        model_name=model_name, model_dir=model_dir, prefix=download_latest_prefix
    )
    if exists:
        print("Already downloaded most recent version")
        return "NA"
    # download the model artifacts (passing in the model name, path in azure storage to model artifacts, local directory to store)
    blobDNE = download_public_folder(
        model_name,
        download_latest_prefix + "/" + model_name,
        os.path.join(model_dir, download_latest_prefix),
    )
    if blobDNE:
        return
    model_dir = os.path.join(WORKDIR, model_name + "/" + download_latest_prefix)
    mlir_filename = os.path.join(model_dir, model_name + ".mlir")
    print(
        f"Verifying that model artifacts were downloaded successfully to {mlir_filename}..."
    )
    assert os.path.exists(mlir_filename), f"MLIR not found at {mlir_filename}"

    return mlir_filename
