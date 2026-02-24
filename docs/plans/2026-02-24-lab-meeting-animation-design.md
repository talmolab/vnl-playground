# Lab Meeting 2/23/26 Animation Suite Design

## Goal
Generate a comprehensive set of pre-rendered MP4 videos from the `mouse-propriospinal-mn-20260223-202959` checkpoint that show synchronized MuJoCo behavior + neural activity panels (spike rasters, membrane voltages, PCA, actions) for baseline, stimulation, and ablation conditions.

## Checkpoint
- Path: `/root/vast/eric/vnl-playground/checkpoints/mouse-propriospinal-mn-20260223-202959`
- Architecture: ProprioSpinal (512 = 410E + 102I) -> Motor Neurons (128E) -> Dense readout (18 actions)
- LIF neurons: v_th=0.3, 8 micro-steps/env step, tau_m in [1,5]

## Script: `lab_meeting_2_23_26_animation.py`

### Architecture (Approach B - Modular)
```
lab_meeting_2_23_26_animation.py
  |
  +-- Data Collection Phase
  |     collect_baseline_data()      -> dict with frames, spikes, voltages, actions
  |     collect_stimulation_data()   -> dict per dose level
  |     collect_ablation_data()      -> dict per ablation level
  |
  +-- Panel Rendering Utilities
  |     render_raster_panel()        -> pre-rendered raster image
  |     render_voltage_heatmap()     -> pre-rendered voltage heatmap
  |     render_action_heatmap()      -> pre-rendered action heatmap
  |     render_pca_panel()           -> pre-rendered PCA traces
  |     render_drive_panel()         -> MN exc/inh drive traces
  |     draw_time_cursor()           -> overlay red vertical line
  |
  +-- Video Generators
  |     make_baseline_full_video()
  |     make_baseline_membrane_video()
  |     make_stimulation_comparison_video()
  |     make_ablation_comparison_video()
  |     make_spike_diagnostics_video()
  |
  +-- main()
        Parse args, load checkpoint, run data collection, generate all videos
```

### Data Collection

Each rollout collects per environment timestep:
| Signal | Shape | Source |
|--------|-------|--------|
| MuJoCo frames | (T, H, W, 3) | eval_env.render(render_ghost=True) |
| PS-E spikes | (T, K, 410) | diagnostics['propriospinal']['spikes_exc'] |
| PS-I spikes | (T, K, 102) | diagnostics['propriospinal']['spikes_inh'] |
| MN spikes | (T, K, 128) | diagnostics['motor_neurons']['spikes'] |
| PS-E voltages | (T, K, 410) | diagnostics['propriospinal']['voltages_exc'] |
| PS-I voltages | (T, K, 102) | diagnostics['propriospinal']['voltages_inh'] |
| MN voltages | (T, K, 128) | diagnostics['motor_neurons']['voltages'] |
| MN exc drive | (T, 128) | diagnostics['mn_input_exc'] |
| MN inh drive | (T, 128) | diagnostics['mn_input_inh'] |
| Actions | (T, 18) | from policy logits |
| Rewards | (T,) | env step reward |

### Video Catalog

#### 1. `baseline_full.mp4` — 2x4 grid
| MuJoCo + ghost | PS-E raster (100 neurons) | PS-I raster (102 neurons) | MN raster (128 neurons) |
| Actions heatmap | PCA spike rates (PS+MN) | PCA voltages (PS+MN) | Membrane voltage heatmap (MN) |

All temporal panels have synchronized red time cursor.

#### 2. `baseline_membrane_potentials.mp4` — 2x2 grid
| PS-E voltage heatmap | PS-I voltage heatmap |
| MN voltage heatmap | MN drive (exc vs inh traces) |

#### 3. `stim_{dose}x.mp4` — Left/Right comparison (one per dose: 2x, 5x, 10x, 20x, 50x)
Left column = normal, Right column = stimulated. Each side:
| MuJoCo + ghost |
| PS-E raster |
| PS-I raster |
| MN raster |
| Actions heatmap |

#### 4. `ablation_{pct}pct.mp4` — Left/Right comparison (one per level: 25%, 50%, 75%, 100%)
Same layout as stimulation comparison.

#### 5. `spike_diagnostics_baseline.mp4` — 3x2 neural focus
| PS-E raster (large) | PS-I raster (large) |
| MN raster (large) | Firing rate traces (PS-E mean, PS-I mean, MN mean) |
| PCA spike rates | PCA voltages |

### Panel Specifications

- **Raster panels**: Neuron index (y) vs time (x). Sorted by mean firing rate descending. Subsampled to 100 for PS-E. Blue colormap for E, orange for I. Binary spike display.
- **Voltage heatmaps**: Same layout as rasters but continuous colormap (viridis or RdBu). v_th=0.3 annotated.
- **Action heatmap**: 18 muscle dims (y) vs time (x), RdBu_r colormap centered at 0.
- **PCA panels**: Top 3 PCs as colored traces. Computed on spike rates or voltages averaged across micro-steps. Variance explained annotated.
- **MN drive panel**: Two overlaid traces (exc in blue, inh in orange) showing mean drive across MN pool.
- **Time cursor**: Red vertical line at current timestep, spans full height of each panel.

### Output
- All videos saved to: `outputs/lab_meeting_2_23_26/`
- Video format: MP4, imageio writer
- Playback: 50 fps (slowed from 400Hz physics → each second of video ~ 125ms sim time)
- Resolution: Individual panels ~480x320, grids scaled accordingly

### Color Scheme (consistent with existing codebase)
- Excitatory: `#1f77b4` (blue), colormap white→blue
- Inhibitory: `#ff7f0e` (orange), colormap white→orange
- Motor neurons: `#2ca02c` (green)
- Threshold: dashed green line
- Time cursor: red vertical line
- Stimulation/ablation annotations: red shaded regions or text labels
