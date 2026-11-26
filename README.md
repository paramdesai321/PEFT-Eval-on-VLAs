# PEFT Evaluation

# Instructions Please Install the pixi conda package manager

**Link: [Pixi Documentation](https://pixi.prefix.dev/latest/installation/)**

```bash
# Install Pixi with the following command (Macos):
brew install pixi

# Linux Installation:
curl -fsSL https://pixi.sh/install.sh | sh

```

# Initialize Project Dependencies in Root Directory

pixi install in the root directory of the project.

```bash
# Pixi targets your system architecture (Macos, Linux, Linux - NVIDIA)
pixi install

# Nvidia
pixi install -e cuda

# Macos M-Series
pixi install -e macos
```

## Test that the project has been correctly initialized

pixi install in the root directory of the project.

For Nvidia based systems, you can run the following command to test that the project has been correctly initialized:


# Alternatively, use the mamba environment provided at:
```bash
environment.yml
```

## Testing Instructions

```bash
# Pixi targets your system architecture (Macos, Linux, Linux - NVIDIA)

# Test SmolVLA
pixi run -e {macos,cuda} test-smolvla

# Test Tensor Batching
pixi run -e {macos,cuda} test-tensor-batching

# Test Mock Action SmolVLA
pixi run -e {macos,cuda} test-mock-action-smolvla
```

For other systems

```bash
# Pixi targets your system architecture (Macos, Linux, Linux - NVIDIA)

# Test SmolVLA
pixi run test-smolvla

# Test Tensor Batching
pixi run test-tensor-batching

# Test Mock Action SmolVLA
pixi run test-mock-action-smolvla
```


## Generate Environment from Pixi
```bash
# Environment for cuda based systems (nvidia gpu with nvcc)
pixi workspace export conda-environment -e cuda > nvidia-environment.yml

# Macos
pixi workspace export conda-environment -e macos > environment.yml
```

## How to Synchronize Hive with External Storage(Box or SFTP):
Follow the tutorial at this link
**Link: [Data Transfer](https://docs.hpc.ucdavis.edu/data-transfer/)**

```bash
# Load rclone before usage
module load rclone

# Upload a file:
rclone copy ./localfile.txt mybox:/BackupFolder

#Download a file:
rclone copy mybox:/BackupFolder/file.txt ./localdir

# Sync a folder (mirror changes):
rclone sync ./localdir mybox:/RemoteDir --progress
```
