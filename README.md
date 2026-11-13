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
```

## Test that the project has been correctly initialized

pixi install in the root directory of the project.

For Nvidia based systems, you can run the following command to test that the project has been correctly initialized:

```bash
# Pixi targets your system architecture (Macos, Linux, Linux - NVIDIA)

# Test SmolVLA
pixi -e run test-smolvla

# Test Tensor Batching
pixi -e run test-tensor-batching

# Test Mock Action SmolVLA
pixi -e run test-mock-action-smolvla
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
