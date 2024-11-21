import requests
import json
import os

GITHUB_TOKEN = os.getenv("IREE_TOKEN")

OWNER = "iree-org"
REPO = "iree"

API_URL = f"https://api.github.com/repos/{OWNER}/{REPO}/actions/workflows/pkgci.yml/runs"

# Function to get the latest workflow run ID for pkgci.yml
def get_latest_pkgci_workflow_run():
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    params = {"per_page": 1}
    response = requests.get(API_URL, headers=headers, params=params)

    if response.status_code == 200:
        data = response.json()
        if data["total_count"] > 0:
            latest_run = data["workflow_runs"][0]
            return latest_run["id"]
        else:
            print("No workflow runs found for pkgci.yml.")
            return None
    else:
        print(f"Error fetching workflow runs: {response.status_code}")
        return None

# Function to get the artifacts of a specific workflow run
def get_artifacts(workflow_run_id):
    artifacts_url = f"https://api.github.com/repos/{OWNER}/{REPO}/actions/runs/{workflow_run_id}/artifacts"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    response = requests.get(artifacts_url, headers=headers)

    if response.status_code == 200:
        artifacts = response.json()["artifacts"]
        if artifacts:
            print(f"Artifacts for pkgci.yml workflow run {workflow_run_id}:")
            for artifact in artifacts:
                print(f"- {artifact['name']} (Size: {artifact['size_in_bytes']} bytes)")
                download_artifact(artifact['archive_download_url'], artifact['name'])
        else:
            print("No artifacts found for the pkgci.yml workflow run.")
    else:
        print(f"Error fetching artifacts: {response.status_code}")

# Function to download an artifact
def download_artifact(download_url, artifact_name):
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    response = requests.get(download_url, headers=headers, stream=True)

    if response.status_code == 200:
        file_name = f"{artifact_name}.tar.gz"
        with open(file_name, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"Artifact '{artifact_name}' downloaded successfully as '{file_name}'.")
    else:
        print(f"Error downloading artifact '{artifact_name}': {response.status_code}")

if __name__ == "__main__":
    workflow_run_id = get_latest_pkgci_workflow_run()
    if workflow_run_id:
        get_artifacts(workflow_run_id)
