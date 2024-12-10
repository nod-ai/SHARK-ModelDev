#!/bin/bash

STORAGE_ACCOUNT="osscicluster"
CONTAINER_NAME="images"
FILE_PATH="sdxl_output.png"
BLOB_NAME="sdxl_output_$(date +%Y%m%d_%H%M%S).png"

# Upload file
az storage blob upload --account-name $STORAGE_ACCOUNT \
  --account-key $STORAGE_KEY \
  --container-name $CONTAINER_NAME \
  --name $BLOB_NAME \
  --file $FILE_PATH

# Construct url
BLOB_URL="https://${STORAGE_ACCOUNT}.blob.core.windows.net/${CONTAINER_NAME}/${BLOB_NAME}"

echo "File uploaded successfully. You can access it at:"
echo $BLOB_URL
