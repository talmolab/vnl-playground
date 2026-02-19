# Spiking Policy Architectures: Rate-Coded vs Spike-Propagating

## Overview

Two LIF (Leaky Integrate-and-Fire) spiking neural network policies for mouse arm imitation, both trained with PPO and surrogate gradients. They share the same E/I population structure (Dale's law, 80/20 excitatory/inhibitory split), heterogeneous membrane time constants, and refractory dynamics. They differ in how information flows between layers.

| | Rate-Coded (`train_mouse_LIF.py`) | Spike-Propagating (`train_mouse_LIF_dynamic.py`) |
|---|---|---|
| **Inter-layer signal** | Mean spike rate (continuous float) | Binary spikes (per micro-step) |
| **Scan structure** | One scan per layer (independent) | Single scan over all layers (lockstep) |
| **Readout** | Rate from last layer's exc. population | Last micro-step's exc. spikes from last layer |
| **Temporal coupling** | Layers are temporally decoupled | Layers interact at every micro-step |

---

## 1. Rate-Coded LIF (`BiologicalSpikingPolicy`)

### Architecture

Each layer is a self-contained `BiologicalLIFLayer` module with its own internal `jax.lax.scan` over $K$ micro-steps. Layers are called sequentially in the parent module.

```
obs ──► [Layer 0: scan K steps → mean rate] ──► [Layer 1: scan K steps → mean rate] ──► ... ──► readout
              ↑                                        ↑
         continuous input                        continuous input
```

### Single Layer Dynamics

For layer $\ell$ with $N_E$ excitatory and $N_I$ inhibitory neurons ($N = N_E + N_I$):

**Input current** (constant across micro-steps, computed once before the scan):

$$I_{\text{input}} = W_{\text{in}}^{(\ell)} \cdot r^{(\ell-1)} + b_{\text{in}}^{(\ell)}$$

where $r^{(\ell-1)} \in \mathbb{R}^{N_E^{(\ell-1)}}$ is the mean excitatory spike rate from the previous layer (or the observation for $\ell = 0$).

**Micro-step $k$ within layer $\ell$** ($k = 0, \ldots, K-1$):

$$I_{\text{lat},E} = -s_I^{(k-1)} \cdot |W_{IE}|, \quad I_{\text{lat},I} = s_E^{(k-1)} \cdot |W_{EI}|$$

$$I_{\text{total}} = I_{\text{input}} + [I_{\text{lat},E};\; I_{\text{lat},I}]$$

$$v^{(k)} = \alpha \odot v^{(k-1)} + (1 - \alpha) \odot I_{\text{total}}$$

$$v^{(k)} \leftarrow v^{(k)} \cdot (1 - \mathbb{1}[\text{refrac} > 0])$$

$$s^{(k)} = \Theta(v^{(k)} - v_{\text{th}}) \cdot (1 - \mathbb{1}[\text{refrac} > 0])$$

where $\alpha_j = \exp(-\Delta t / \tau_{m,j})$ and $\Theta$ is the Heaviside step function (with surrogate gradient $\sigma(\beta(v - v_{\text{th}}))$ in the backward pass).

**Output** (passed to next layer):

$$r_E^{(\ell)} = \frac{1}{K} \sum_{k=0}^{K-1} s_E^{(k)} \in [0, 1]^{N_E}$$

The next layer receives this continuous rate vector as its input. Information between layers is a **time-averaged summary** — the temporal spike pattern within each layer is discarded.

### Readout

$$\text{logits} = W_{\text{read}} \cdot r_E^{(L-1)} + b_{\text{read}}$$

---

## 2. Spike-Propagating LIF (`SpikePropagatingPolicy`)

### Architecture

A single `jax.lax.scan` over $K$ micro-steps drives all layers simultaneously. At each micro-step, spikes from layer $\ell$ immediately feed into layer $\ell+1$.

```
            micro-step k
            ┌──────────────────────────────────────────────┐
obs ──►  I₀ ──► Layer 0 ──► s₀ᴱ(k) ──► Layer 1 ──► s₁ᴱ(k) ──► Layer 2 ──► s₂ᴱ(k)
            └──────────────────────────────────────────────┘
                              ↑ repeated for k = 0 ... K-1

readout ◄── s₂ᴱ(K-1)   (last micro-step only)
```

### Dynamics at Micro-step $k$

All parameters ($W_{\text{in}}^{(\ell)}, b_{\text{in}}^{(\ell)}, W_{IE}^{(\ell)}, W_{EI}^{(\ell)}, \tau_m^{(\ell)}$) are created before the scan. Layer 0's input current $I_0 = W_{\text{in}}^{(0)} \cdot \text{obs} + b_{\text{in}}^{(0)}$ is also pre-computed (constant across micro-steps since the observation doesn't change within one control step).

**For each layer $\ell = 0, 1, \ldots, L-1$ sequentially within micro-step $k$:**

Feed-forward input:
$$I_{\text{ff}}^{(\ell, k)} = \begin{cases} I_0 & \text{if } \ell = 0 \\ W_{\text{in}}^{(\ell)} \cdot s_E^{(\ell-1, k)} + b_{\text{in}}^{(\ell)} & \text{if } \ell > 0 \end{cases}$$

Note: $s_E^{(\ell-1, k)}$ is the excitatory spike vector produced by layer $\ell-1$ **at this same micro-step $k$** — not a rate, not from a previous micro-step.

Lateral E/I currents (from previous micro-step's spikes within this layer):

$$I_{\text{lat},E}^{(\ell, k)} = -s_I^{(\ell, k-1)} \cdot |W_{IE}^{(\ell)}|, \quad I_{\text{lat},I}^{(\ell, k)} = s_E^{(\ell, k-1)} \cdot |W_{EI}^{(\ell)}|$$

Membrane update, spike generation, reset, refractory — identical equations to the rate-coded model.

### Readout

$$\text{logits} = W_{\text{read}} \cdot s_E^{(L-1,\, K-1)} + b_{\text{read}}$$

The readout uses **only the last micro-step's spikes** from the final layer's excitatory population. No averaging. The linear readout converts the binary spike vector (410 excitatory neurons with default config) into continuous action logits.

---

## 3. Key Differences

### 3.1 Information Flow

**Rate-coded**: Each layer processes its full $K$ micro-steps in isolation before handing off a scalar rate per neuron. Temporal dynamics within a layer are invisible to the next layer. A change in observation propagates instantly to all layers (each sees the same pre-computed input current), but the inter-layer signal is a lossy temporal average.

**Spike-propagating**: Information ripples through the network in real time. A spike in layer 0 at micro-step $k$ can cause a spike in layer 1 at micro-step $k$ (same step), which can cause a spike in layer 2 at micro-step $k$. Temporal correlations are preserved across layers. However, the first few micro-steps produce output based on incomplete propagation (layer 2 hasn't received signal from layer 0 yet at $k=0$), so the network needs enough micro-steps for signal to fully propagate.

### 3.2 Scan Structure

| | Rate-Coded | Spike-Propagating |
|---|---|---|
| Number of scans | $L$ (one per layer) | 1 (single outer scan) |
| Scan length | $K$ each | $K$ total |
| Total LIF steps | $L \times K$ | $L \times K$ (same) |
| Parameters inside scan | None (pre-computed) | Weight matmuls for layers $\ell > 0$ |

### 3.3 Readout Signal

| | Rate-Coded | Spike-Propagating |
|---|---|---|
| Signal type | Continuous rate $\in [0, 1]$ | Binary spike $\in \{0, 1\}$ |
| Temporal window | Average over all $K$ steps | Last micro-step only |
| Quantization | $1/K$ (rate resolution) | Binary (but 410 neurons sum in readout) |

### 3.4 Gradient Path

**Rate-coded**: Gradients flow through the surrogate gradient within each layer's scan, then through the rate computation (`mean`), then through the next layer's input projection. The `mean` over micro-steps acts as a low-pass filter on the gradient signal.

**Spike-propagating**: Gradients flow through the surrogate gradient at every micro-step, across all layers, in a single unrolled computation graph. The gradient path is deeper ($L \times K$ steps end-to-end vs $K$ steps per layer) but preserves temporal structure. The readout gradient flows back through only the last micro-step's spike, through all layers.

---

## 4. Shared Components

Both architectures share:

- **E/I populations**: Dale's law enforced via `|W|` on lateral weights. 80% excitatory, 20% inhibitory per layer. Only excitatory spikes propagate to the next layer.
- **Heterogeneous $\tau_m$**: Log-uniform initialization, fixed via `stop_gradient`. Each neuron has a unique membrane time constant.
- **Refractory period**: After firing, a neuron is clamped for $n_{\text{refrac}}$ micro-steps.
- **Surrogate gradient**: Forward pass uses hard threshold $\Theta(v - v_{\text{th}})$; backward pass substitutes $\sigma(\beta(v - v_{\text{th}}))$.
- **Persistent carry**: Membrane voltage and refractory state persist across environment steps (reset on episode done).
- **Value network**: Standard MLP with `swish` activation (no spiking).
- **Training**: PPO with GAE, same optimizer (Adam + grad clipping).

---

## 5. Hyperparameters

| Parameter | Rate-Coded | Spike-Propagating |
|---|---|---|
| `policy_hidden_layer_sizes` | (512, 512, 512) | (512, 512) |
| `n_micro_steps` | 16 | 16 |
| `tau_min` | 3.0 | 2.0 |
| `tau_max` | 15.0 | 8.0 |
| `v_th` | 0.3 | 0.3 |
| `v_reset` | 0.0 | 0.0 |
| `beta_surrogate` | 5.0 | 5.0 |
| `n_refractory` | 2 | 2 |
| `exc_ratio` | 0.8 | 0.8 |

The spike-propagating model uses fewer layers (2 vs 3) and faster membrane dynamics (`tau_max` 8 vs 15) to reduce propagation delay and response lag.

---

## 6. Surrogate Gradient: The Training Problem and How Both Models Solve It

### The Core Problem

LIF neurons fire when membrane voltage crosses a threshold. The firing function is the Heaviside step:

$$s = \Theta(v - v_{\text{th}}) = \begin{cases} 1 & \text{if } v \geq v_{\text{th}} \\ 0 & \text{otherwise} \end{cases}$$

This is not differentiable. $\frac{\partial s}{\partial v} = 0$ everywhere except at $v = v_{\text{th}}$ where it is undefined. Standard backpropagation through this function produces zero gradients — the network cannot learn.

### The Surrogate Gradient Trick

Both models use the same solution. In the **forward pass**, the hard threshold produces real binary spikes (0 or 1). In the **backward pass**, the derivative of a smooth surrogate function is substituted:

$$\tilde{s}(v) = \sigma(\beta(v - v_{\text{th}})) = \frac{1}{1 + e^{-\beta(v - v_{\text{th}})}}$$

This is implemented via the straight-through estimator pattern:

```python
spike_hard = (v >= v_th).astype(v.dtype)           # forward: binary {0, 1}
spike_soft = jax.nn.sigmoid(beta * (v - v_th))      # surrogate: smooth (0, 1)
spike = stop_gradient(spike_hard - spike_soft) + spike_soft
```

During the forward pass, `stop_gradient(spike_hard - spike_soft)` evaluates to `spike_hard - spike_soft`, so `spike = spike_hard` (the true binary spike). During the backward pass, the `stop_gradient` term has zero gradient, so the gradient flows through `spike_soft` only — the smooth sigmoid. This gives:

$$\frac{\partial \tilde{s}}{\partial v} = \beta \cdot \sigma(\beta(v - v_{\text{th}})) \cdot (1 - \sigma(\beta(v - v_{\text{th}})))$$

With $\beta = 5.0$ and $v_{\text{th}} = 0.3$, this is a bell-shaped curve centered at $v = 0.3$ with width $\approx 1/\beta = 0.2$. Neurons with voltage near the threshold get strong gradient signal; neurons far from threshold get near-zero gradient (they "know" they're clearly firing or clearly silent and don't need to change).

### How the Surrogate Gradient Differs Between the Two Models

**Rate-coded model — gradients are simpler:**

Each layer runs its $K$ micro-steps independently. The output is $r = \frac{1}{K}\sum_k s^{(k)}$, so:

$$\frac{\partial r}{\partial \theta} = \frac{1}{K} \sum_{k=0}^{K-1} \frac{\partial s^{(k)}}{\partial \theta}$$

The gradient for each layer's parameters comes from the average of $K$ surrogate gradient contributions. This averaging smooths the gradient signal — even if individual micro-step gradients are noisy (because spikes are binary events), the mean over $K$ steps produces a stable gradient. The `mean` operation acts as a built-in variance reducer.

Between layers, the gradient flows through: $\frac{\partial \mathcal{L}}{\partial r^{(\ell-1)}} = \frac{\partial \mathcal{L}}{\partial r^{(\ell)}} \cdot \frac{\partial r^{(\ell)}}{\partial I_{\text{input}}^{(\ell)}} \cdot W_{\text{in}}^{(\ell)}$. This is a standard dense gradient — the continuous rate $r$ makes inter-layer gradients well-behaved.

**Spike-propagating model — gradients are deeper and more structured:**

All layers are coupled within a single scan. The loss gradient must flow from the readout, through the last micro-step's spike in the final layer, backward through the surrogate at that layer, through the input projection weights, through the spike of the previous layer at the same micro-step, and so on — all the way back through $L$ layers $\times$ $K$ micro-steps.

Since only the last micro-step ($k = K-1$) connects to the readout, the gradient for earlier micro-steps comes entirely from how they influence the membrane state that eventually produces the final spike. The gradient must propagate through the membrane dynamics:

$$\frac{\partial v^{(k)}}{\partial v^{(k-1)}} = \alpha \cdot (1 - \mathbb{1}[\text{refrac}]) \cdot (1 - s^{(k)}) + \ldots$$

This chain of $\alpha$ multiplications can cause vanishing gradients over many micro-steps (since $\alpha < 1$), analogous to the vanishing gradient problem in RNNs. The higher $\tau$ (larger $\alpha$, closer to 1) helps mitigate this by keeping the gradient multiplier close to 1.

---

## 7. From Spikes to Muscle Activations: The Full Chain

The mouse arm model (`akira_muscle.xml`) has 9 muscle actuators:

| Muscle | Joint | Force |
|--------|-------|-------|
| Pec_C | Shoulder | 0.80 |
| Lat | Shoulder | 0.80 |
| PD (posterior deltoid) | Shoulder | 0.87 |
| AD (anterior deltoid) | Shoulder | 1.20 |
| MD (middle deltoid) | Shoulder | 0.90 |
| Triceps_Lateral | Elbow | 0.60 |
| Triceps_Long | Elbow | 0.90 |
| Brachialis | Elbow | 0.60 |
| Biceps_Long | Elbow | 0.52 |

Each muscle's control input is clamped to $[0, 1]$ (`ctrlrange="0 1"`), where 0 = no activation and 1 = full activation. The chain from network output to muscle force is:

### Step 1: Spiking Network → Logits

The policy network outputs `logits` of size $2 \times 9 = 18$ (the `param_size` from `NormalTanhDistribution`): a mean $\mu_j$ and log-std $\log \sigma_j$ for each of the 9 muscles.

**Rate-coded:**
$$\text{logits} = W_{\text{read}} \cdot r_E^{(L-1)} + b_{\text{read}} \in \mathbb{R}^{18}$$

The input $r_E^{(L-1)}$ is a 410-dimensional continuous vector of spike rates $\in [0, 1]$. Each logit is a weighted sum of 410 rates — a smooth function of the network activity.

**Spike-propagating:**
$$\text{logits} = W_{\text{read}} \cdot s_E^{(L-1, K-1)} + b_{\text{read}} \in \mathbb{R}^{18}$$

The input $s_E^{(L-1, K-1)}$ is a 410-dimensional binary vector — each entry is 0 (neuron silent) or 1 (neuron fired at the last micro-step). Each logit is a sum of a subset of readout weights, corresponding to whichever neurons happened to fire. With 410 neurons contributing, the logit space is still rich ($2^{410}$ possible output patterns), but each individual output is a discrete sum rather than a continuously-weighted average.

### Step 2: Logits → Stochastic Action (Training) or Deterministic Action (Eval)

The `NormalTanhDistribution` interprets the 18 logits as 9 means and 9 log-standard-deviations:

$$\mu_j = \text{logits}[j], \quad \sigma_j = e^{\text{logits}[9+j]} \quad \text{for } j = 0, \ldots, 8$$

**During training** (stochastic, for exploration):
$$z_j \sim \mathcal{N}(\mu_j, \sigma_j^2), \quad a_j = \tanh(z_j)$$

**During evaluation** (deterministic):
$$a_j = \tanh(\mu_j)$$

The $\tanh$ squashes the action to $[-1, 1]$, which MuJoCo then rescales to the actuator's control range $[0, 1]$:

$$u_j = \frac{a_j + 1}{2} \in [0, 1]$$

### Step 3: Muscle Activation → Joint Torque

Each muscle actuator in MuJoCo computes force using the Hill muscle model. The control signal $u_j$ sets the muscle activation level, and the muscle model computes force based on:

- **Activation** $u_j \in [0, 1]$: how much the brain is commanding this muscle
- **Force-length relationship**: force depends on current muscle length relative to its optimal range
- **Force-velocity relationship**: force depends on contraction speed (`vmax=15`)
- **Peak force**: the `force` parameter scales maximum isometric force

The muscle force is transmitted through tendons to produce joint torques. Antagonist muscle pairs (e.g., Biceps vs Triceps) create opposing torques at the same joint, giving the network bidirectional control.

### Step 4: Concrete Example

Consider the spike-propagating model deciding to flex the elbow:

1. **Observation** enters the network: current joint angles, velocities, and reference target indicating "elbow should flex"
2. **Layer 0** (micro-steps 0–15): observation current charges up neurons; some fire, some don't. By step ~3, a population of excitatory neurons in layer 0 are producing spikes
3. **Layer 1** (micro-steps ~3–15): receives binary spikes from layer 0's excitatory neurons. These drive currents in layer 1's neurons. The E/I balance within layer 1 shapes which neurons fire
4. **Last micro-step (15)**: layer 1's excitatory population has a particular spike pattern — say 180 of 410 neurons are firing
5. **Readout**: $\text{logits} = W_{\text{read}} \cdot s + b_{\text{read}}$. The 180 active neurons each contribute their readout weight column. Suppose the resulting logits produce:
   - $\mu_{\text{Biceps}} = 1.2, \quad \mu_{\text{Triceps\_Lat}} = -0.8, \quad \mu_{\text{Triceps\_Long}} = -0.6$
6. **Tanh + rescale**: $u_{\text{Biceps}} = (\tanh(1.2)+1)/2 \approx 0.92$, $u_{\text{Triceps\_Lat}} \approx 0.18$, $u_{\text{Triceps\_Long}} \approx 0.24$
7. **Muscle forces**: Biceps contracts strongly (92% activation), both Triceps heads are mostly relaxed → net elbow flexion torque
8. **Physics**: MuJoCo integrates the torques over 2 simulation substeps ($\text{sim\_dt} = 0.00125$s, $\text{ctrl\_dt} = 0.0025$s), producing joint movement

### How Rate-Coding Simplifies This

In the rate-coded model, step 5 looks different. Instead of 180 binary spikes, the readout receives 410 continuous rates like $[0.43, 0.12, 0.87, \ldots]$. The logit for each muscle is a smoothly-weighted sum:

$$\mu_{\text{Biceps}} = \sum_{j=1}^{410} W_{\text{read},j} \cdot r_j + b$$

Small changes in network activity produce small changes in muscle commands. The output is inherently smooth because the rates are continuous and averaged over 16 micro-steps.

In the spike-propagating model, the output is a sum of a **discrete subset** of weights. A single neuron flipping from 0→1 at the last micro-step adds its entire weight column to the logit. This makes the output more sensitive to individual neuron states but also gives the network access to sharper, more precise temporal signals — a neuron that fires at exactly the right moment contributes to the action, while one that fires too early or too late does not (since only the last micro-step matters for the readout).
