# PEFT Evaluation

# Instructions Please Install the pixi conda package manager.

**Link: [Pixi Documentation](https://pixi.prefix.dev/latest/installation/)**



```bash
# Install Pixi with the following command (Macos):
brew install pixi

# Linux Installation:
curl -fsSL https://pixi.sh/install.sh | sh
```

# Initialize Project Dependencies in Root Directory:
pixi install in the root directory of the project.
```bash
# Pixi targets your system architecture (Macos, Linux, Linux - NVIDIA)
pixi install
```

# Test that the project has been correctly initialized:
pixi install in the root directory of the project.
```bash
# Pixi targets your system architecture (Macos, Linux, Linux - NVIDIA)

# Test SmolVLA
pixi run test-smolvla

# Test Tensor Batching
pixi run test-tensor-batching

# Test Mock Action SmolVLA
pixi run test-mock-action-smolvla
```