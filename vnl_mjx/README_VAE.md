# VAE Imitation Learning for Track-MJX

This module implements VAE-based imitation learning where a student VAE network learns to imitate a frozen track-mjx teacher network through pure action distillation.

## Architecture

- **Teacher Network**: Frozen track-mjx network that maps `[proprioception, reference_obs] -> actions`
- **Student VAE**: Composed of three MLPs:
  - **Encoder**: `[proprioception, reference_obs] -> latent_distribution`
  - **Prior**: `[proprioception] -> prior_distribution`
  - **Decoder**: `[proprioception, latent] -> actions`

## Key Features

- **Pure Distillation**: Focuses solely on teacher-student action matching, independent of specific environments
- **Environment Agnostic**: Uses the same track-mjx environment configuration as the teacher, without being tied to bowl_escape or flat_arena
- **VAE Regularization**: KL divergence between encoder and prior encourages learning useful representations
- **Parallel Training**: Uses JAX vmap for efficient parallel training
- **Track-MJX Integration**: Seamlessly loads teacher networks and replicates their training environment

## Installation

Ensure you have the required dependencies:
- JAX with GPU support
- Flax
- Optax
- Hydra
- MuJoCo MJX
- Track-MJX (for teacher networks)
- Weights & Biases (optional, for logging)

## Usage

### 1. Configure Training

Edit `config/vae_distillation.yaml` to set:
- Path to your track-mjx teacher checkpoint
- Path to reference trajectory data
- VAE architecture parameters
- Training hyperparameters

```yaml
teacher_config:
  checkpoint_path: /path/to/your/track-mjx/checkpoint
  reference_data_path: /path/to/reference/data.h5

vae_config:
  latent_dim: 32
  encoder_hidden_sizes: [512, 512, 256]
  decoder_hidden_sizes: [512, 512, 256]
  prior_hidden_sizes: [512, 256]

training_config:
  num_envs: 1024
  batch_size: 512
  learning_rate: 3e-4
  kl_weight: 0.001
```

### 2. Train VAE

```bash
cd vnl_mjx
python train_vae.py
```

### 3. Evaluate Trained VAE

```bash
cd vnl_mjx
python eval_vae.py checkpoint_path=/path/to/vae/checkpoint
```

## File Structure

```
vnl_mjx/
├── models/
│   ├── __init__.py
│   └── vae.py                 # VAE network architectures
├── training/
│   ├── __init__.py
│   ├── losses.py              # Loss functions
│   └── vae_distillation.py    # Training loop
├── tasks/
│   └── vae_distillation.py    # Pure distillation task environment
├── utils/
│   ├── __init__.py
│   ├── track_mjx_loader.py    # Track-MJX integration utilities
│   └── vae_checkpointing.py   # Checkpointing utilities
├── config/
│   └── vae_distillation.yaml  # Configuration file
├── train_vae.py               # Main training script
└── eval_vae.py                # Evaluation script
```

## Key Components

### VAE Model (`models/vae.py`)
- Modular VAE implementation with separate encoder, decoder, and prior
- Supports configurable MLP architectures
- Includes reparameterization trick for gradient flow

### Loss Functions (`training/losses.py`)
- Action MSE loss between teacher and student
- KL divergence loss between encoder and prior
- Combined loss with configurable weights

### Distillation Task (`tasks/vae_distillation.py`)
- Pure teacher-student distillation environment
- Uses the same track-mjx environment configuration as the teacher
- Focuses solely on action prediction accuracy
- Environment-agnostic (not tied to specific tasks like bowl_escape)

### Track-MJX Integration (`utils/track_mjx_loader.py`)
- Loads and freezes track-mjx teacher networks
- Handles observation normalization
- Creates inference functions compatible with training loop

## Training Process

1. **Initialization**: Load frozen teacher network and recreate its training environment
2. **Data Collection**: 
   - Reset parallel distillation environments
   - Get observations in the same format as teacher training
   - Get teacher actions via frozen teacher network
3. **VAE Forward Pass**:
   - Encoder: `q(z|proprioception, reference_obs)`
   - Prior: `p(z|proprioception)`
   - Sample latent: `z ~ q(z|...)`
   - Decoder: `student_actions = decoder(proprioception, z)`
4. **Loss Computation**:
   - Action MSE: `||teacher_actions - student_actions||²`
   - KL divergence: `KL(q(z|...) || p(z|...))`
5. **Environment Step**: Step physics with student actions
6. **Parameter Update**: Update VAE parameters via gradient descent

## Monitoring and Evaluation

The training script logs:
- Total loss (action MSE + KL divergence)
- Individual loss components
- Action prediction errors
- Latent space statistics
- Training metrics via Weights & Biases

The evaluation script provides:
- Action prediction accuracy vs teacher
- Reward comparisons
- Latent space visualizations
- Performance across multiple episodes

## Configuration Options

### VAE Architecture
- `latent_dim`: Dimensionality of latent space
- `encoder_hidden_sizes`: Hidden layer sizes for encoder MLP
- `decoder_hidden_sizes`: Hidden layer sizes for decoder MLP
- `prior_hidden_sizes`: Hidden layer sizes for prior MLP

### Training Parameters
- `num_envs`: Number of parallel environments
- `batch_size`: Training batch size
- `learning_rate`: Learning rate for Adam optimizer
- `kl_weight`: Weight for KL divergence term
- `num_steps`: Total training steps

### Environment Configuration
- The distillation task automatically uses the same environment as the teacher
- No need to specify bowl_escape, flat_arena, or other specific environments
- Environment configuration is extracted from the teacher checkpoint

## Tips for Usage

1. **Teacher Checkpoint**: Ensure your track-mjx teacher checkpoint is compatible and trained on the same environment
2. **Memory Management**: Reduce `num_envs` if you encounter memory issues
3. **KL Weight Tuning**: Start with small KL weights (0.001) and adjust based on training dynamics
4. **Latent Dimension**: Experiment with different latent dimensions based on task complexity
5. **Monitoring**: Watch for mode collapse in latent space or poor action prediction

## Troubleshooting

- **CUDA Memory Issues**: Reduce `num_envs` or `batch_size`
- **Poor Performance**: Check teacher checkpoint compatibility and observation normalization
- **Training Instability**: Reduce learning rate or adjust KL weight
- **Import Errors**: Ensure track-mjx is properly installed and accessible
