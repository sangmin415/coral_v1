"""
Main controller orchestrating the inverse design workflow.
Connects the UI to the backend modules.
"""
import io
import numpy as np
import torch
import optuna
import matplotlib.pyplot as plt
from pathlib import Path
import gdstk

# Backend module imports
from ml_core.unet import UNetSmall
from geometry.layout_manager import params_to_binary_mask, binary_mask_to_gds
from drc.checker import load_pdk, aggregate_report, DRCReport
from pdk.helpers import get_pdk_spec, get_available_pdks

# Suppress Optuna's informational messages
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- Core Inverse Design Logic ---

def _plot_sparameters(target_s_params, predicted_s_params, pdk_spec, title="S-Parameter Comparison"):
    """
    Plots S-parameters (S11, S21) in dB scale against a frequency axis derived from PDK spec.
    
    ASSUMPTION: The 804-point input vectors are structured as:
    - Points 0-200: S11 (real)
    - Points 201-401: S11 (imag)
    - Points 402-602: S21 (real)
    - Points 603-803: S21 (imag)
    This corresponds to 201 frequency points.
    """
    if target_s_params is None or predicted_s_params is None or target_s_params.shape[0] != 804 or predicted_s_params.shape[0] != 804:
        # Return an empty figure if data is invalid
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Invalid S-parameter data for plotting.", ha='center', va='center')
        return fig

    # Extract frequency sweep parameters from PDK spec
    # Default to 1-100 GHz with 201 points if not found, for robustness
    f_start_GHz = pdk_spec.get("sweep", {}).get("f_start_GHz", 1.0)
    f_stop_GHz = pdk_spec.get("sweep", {}).get("f_stop_GHz", 100.0)
    n_points_sweep = pdk_spec.get("sweep", {}).get("n_points", 402) # Total points in sweep
    
    num_freq_points = n_points_sweep // 2 # Assuming real and imag for each freq point
    if num_freq_points == 0: # Avoid division by zero or invalid points
        num_freq_points = 201 # Fallback to default

    # --- Data Reshaping and Conversion ---
    def reshape_and_convert_to_db(s_params):
        s11_real = s_params[0 * num_freq_points : 1 * num_freq_points]
        s11_imag = s_params[1 * num_freq_points : 2 * num_freq_points]
        s21_real = s_params[2 * num_freq_points : 3 * num_freq_points]
        s21_imag = s_params[3 * num_freq_points : 4 * num_freq_points]
        
        s11_mag_db = 20 * np.log10(np.sqrt(s11_real**2 + s11_imag**2))
        s21_mag_db = 20 * np.log10(np.sqrt(s21_real**2 + s21_imag**2))
        
        return s11_mag_db, s21_mag_db

    target_s11_db, target_s21_db = reshape_and_convert_to_db(target_s_params)
    predicted_s11_db, predicted_s21_db = reshape_and_convert_to_db(predicted_s_params)

    # --- Plotting ---
    freq_ghz = np.linspace(f_start_GHz, f_stop_GHz, num_freq_points)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    fig.suptitle(title, fontsize=16)

    # S11 Plot
    ax1.plot(freq_ghz, target_s11_db, label="Target S11")
    ax1.plot(freq_ghz, predicted_s11_db, label="Predicted S11", linestyle='--')
    ax1.set_title("S11 (Return Loss)")
    ax1.set_xlabel("Frequency (GHz)")
    ax1.set_ylabel("Magnitude (dB)")
    ax1.legend()
    ax1.grid(True)

    # S21 Plot
    ax2.plot(freq_ghz, target_s21_db, label="Target S21")
    ax2.plot(freq_ghz, predicted_s21_db, label="Predicted S21", linestyle='--')
    ax2.set_title("S21 (Insertion Loss)")
    ax2.set_xlabel("Frequency (GHz)")
    ax2.set_ylabel("Magnitude (dB)")
    ax2.legend()
    ax2.grid(True)
    
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    return fig

def _load_surrogate_model(model_path, device="cpu"):
    """
    Loads the pre-trained surrogate model, handling different checkpoint formats.
    The model is expected to be a UNetSmall with 1 input channel.
    """
    model = UNetSmall(in_ch=1, out_dim=804).to(device)
    try:
        checkpoint = torch.load(model_path, map_location=device)
        # Check if the checkpoint is a dictionary from PyTorch Lightning
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            # Remove 'model.' prefix if it exists in the keys
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
        else:
            # Assume it's a raw state_dict
            model.load_state_dict(checkpoint)
        
        model.eval()
        return model
    except FileNotFoundError:
        raise ValueError(f"Model file not found at: {model_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")

def _get_prediction(model, params, device="cpu"):
    """
    Generates a prediction from the surrogate model given geometry parameters.
    The model expects a 1-channel input.
    """
    mask = params_to_binary_mask(params)
    # Create a 1-channel tensor
    mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        prediction = model(mask_tensor)
    return prediction.cpu().numpy().flatten()

def _approximate_params_from_mask(mask: np.ndarray, pdk_spec: dict) -> dict:
    """
    Approximates high-level geometry parameters from a binary mask.
    """
    if mask.ndim != 2:
        return {}

    h, w = mask.shape
    um_per_pixel = pdk_spec.get("units", {}).get("um_per_pixel", 1.0)

    # Estimate finger length by finding the most common vertical extent
    vertical_runs = np.sum(mask, axis=0)
    # Filter out small columns that are likely noise
    significant_cols = vertical_runs[vertical_runs > h * 0.2]
    if len(significant_cols) > 0:
        avg_finger_len_px = np.mean(significant_cols)
    else:
        avg_finger_len_px = 0

    # Estimate num_fingers, width, and spacing from a horizontal slice
    mid_slice = mask[h // 2, :]
    transitions = np.diff(mid_slice.astype(int))
    
    # Find runs of 1s (fingers) and 0s (spaces)
    finger_runs = np.where(transitions == 1)[0]
    space_runs = np.where(transitions == -1)[0]

    num_fingers = len(finger_runs)

    widths = []
    if len(finger_runs) > 0 and len(space_runs) > 0:
        # Align runs to calculate widths
        if space_runs[0] < finger_runs[0]:
             # Starts with a space, so drop first space run
             space_runs = space_runs[1:]
        
        min_len = min(len(finger_runs), len(space_runs))
        for i in range(min_len):
            widths.append(space_runs[i] - finger_runs[i])

    avg_width_px = np.mean(widths) if widths else 0

    spaces = []
    if len(finger_runs) > 1:
        for i in range(len(finger_runs) - 1):
            spaces.append(finger_runs[i+1] - (finger_runs[i] + widths[i] if i < len(widths) else 0))

    avg_spacing_px = np.mean(spaces) if spaces else 0

    return {
        "num_fingers": num_fingers,
        "finger_length_um": avg_finger_len_px * um_per_pixel,
        "finger_width_um": avg_width_px * um_um_per_pixel,
        "finger_spacing_um": avg_spacing_px * um_per_pixel,
    }


def run_inverse_design_freeform(
    target_path: str,
    model_path: str,
    pdk_name: str,
    n_trials: int, # Here, n_trials is used as optimization steps
    log_callback,
):
    """
    Main function to run the freeform, pixel-based inverse design process.
    """
    log_callback("Starting FREEFORM inverse design process...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_callback(f"Using device: {device}")

    # 1. Load data, model, and PDK
    try:
        target_s_params = np.load(target_path)
        target_tensor = torch.from_numpy(target_s_params).float().to(device)
        model = _load_surrogate_model(model_path, device)
        pdk_spec = get_pdk_spec(pdk_name)
        pdk_drc = load_pdk(pdk_spec['path'])
        log_callback("Successfully loaded target, model, and PDK.")
    except Exception as e:
        log_callback(f"Error during setup: {e}")
        return None, DRCReport(passed=False, violations=[]), {}, None, None

    # 2. Setup the optimization tensor and optimizer
    steps = n_trials 
    lr = 0.05
    
    p = torch.randn(1, 1, 64, 64, device=device) * 0.01
    p.requires_grad_(True)
    optimizer = torch.optim.Adam([p], lr=lr)
    log_callback(f"Running Adam optimization for {steps} steps...")

    # 3. Run the optimization loop
    for t in range(steps):
        m = torch.sigmoid(p)
        x = (m > 0.5).float() if t % 10 == 0 else m

        # Feed 1-channel mask to the model
        prediction = model(x)
        loss = torch.nn.functional.mse_loss(prediction, target_tensor)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (t + 1) % 50 == 0:
            log_callback(f"Step [{t+1}/{steps}]: loss={loss.item():.6f}")

    log_callback("Optimization finished.")
    
    # 4. Get final mask and approximate parameters
    final_mask = (torch.sigmoid(p).detach().cpu().numpy() > 0.5).astype(float)[0, 0]
    best_params = _approximate_params_from_mask(final_mask, pdk_spec)
    log_callback(f"Approximated parameters: {best_params}")

    # 5. Generate final results
    final_mask_tensor = torch.from_numpy(final_mask).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        # Feed 1-channel mask to the model
        predicted_s_params = model(final_mask_tensor).cpu().numpy().flatten()

    fig = _plot_sparameters(target_s_params, predicted_s_params, pdk_spec, title="S-Parameter Comparison (Freeform)")
    
    # DRC check on approximated parameters
    rect_for_drc = {"width_um": best_params.get('finger_length_um', 0), "height_um": best_params.get('num_fingers', 0) * (best_params.get('finger_width_um', 0) + best_params.get('finger_spacing_um', 0))}
    drc_report = aggregate_report(rect_for_drc, pdk_drc)
    log_callback("DRC check complete.")

    # GDS and layout image generation
    gds_cell = binary_mask_to_gds(final_mask.astype(np.uint8), cell_name="optimized_idc_freeform")
    lib = gdstk.Library()
    lib.add(gds_cell)
    gds_path = Path("out") / "optimized_design_freeform.gds"
    gds_path.parent.mkdir(exist_ok=True)
    lib.write_gds(str(gds_path))
    log_callback(f"GDS file saved to {gds_path}")

    fig_layout, ax_layout = plt.subplots(figsize=(4, 4))
    ax_layout.imshow(final_mask, cmap='gray_r', origin='lower')
    ax_layout.set_title("Layout Preview (Freeform)")
    ax_layout.axis('off')
    layout_img_path = Path("out") / "optimized_layout_freeform.png"
    fig_layout.savefig(layout_img_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig_layout)

    return fig, drc_report, best_params, str(gds_path), str(layout_img_path)


def run_inverse_design_parametric(
    target_path: str,
    model_path: str,
    pdk_name: str,
    n_trials: int,
    log_callback,
):
    """
    Main function to run the Optuna-based inverse design process.
    """
    log_callback("Starting PARAMETRIC inverse design process...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_callback(f"Using device: {device}")

    # 1. Load data, model, and PDK
    try:
        target_s_params = np.load(target_path)
        target_tensor = torch.from_numpy(target_s_params).float().to(device)
        model = _load_surrogate_model(model_path, device)
        pdk_spec = get_pdk_spec(pdk_name)
        pdk_drc = load_pdk(pdk_spec['path'])
        log_callback("Successfully loaded target, model, and PDK.")
    except Exception as e:
        log_callback(f"Error during setup: {e}")
        return None, DRCReport(passed=False, violations=[]), {}, None, None

    # 2. Define the Optuna objective function
    def objective(trial):
        param_space = pdk_spec['design_space']['idc']
        params = {
            "num_fingers": trial.suggest_int("num_fingers", *param_space["num_fingers"]),
            "finger_length_um": trial.suggest_float("finger_length_um", *param_space["finger_length_um"]),
            "finger_width_um": trial.suggest_float("finger_width_um", *param_space["finger_width_um"]),
            "finger_spacing_um": trial.suggest_float("finger_spacing_um", *param_space["finger_spacing_um"]),
        }

        mask = params_to_binary_mask(params)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float().to(device)
        
        with torch.no_grad():
            prediction = model(mask_tensor)
        loss = torch.nn.functional.mse_loss(prediction, target_tensor)
        return loss.item()

    # 3. Run the optimization
    log_callback(f"Running Optuna optimization for {n_trials} trials...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, callbacks=[lambda s, t: log_callback(f"Trial {t.number}: loss={t.value:.6f}")])
    
    best_params = study.best_params
    log_callback("Optimization finished.")
    log_callback(f"Best parameters found: {best_params}")

    # 4. Generate final results
    predicted_s_params = _get_prediction(model, best_params, device)
    fig = _plot_sparameters(target_s_params, predicted_s_params, pdk_spec, title="S-Parameter Comparison (Parametric)")
    
    # DRC check
    layout_width = best_params['finger_length_um']
    layout_height = best_params['num_fingers'] * (best_params['finger_width_um'] + best_params['finger_spacing_um'])
    rect_for_drc = {"width_um": layout_width, "height_um": layout_height}
    drc_report = aggregate_report(rect_for_drc, pdk_drc)
    log_callback("DRC check complete.")

    # GDS and layout image generation
    final_mask = params_to_binary_mask(best_params)
    
    gds_cell = binary_mask_to_gds(final_mask, cell_name="optimized_idc_parametric")
    lib = gdstk.Library()
    lib.add(gds_cell)
    gds_path = Path("out") / "optimized_design_parametric.gds"
    gds_path.parent.mkdir(exist_ok=True)
    lib.write_gds(str(gds_path))
    log_callback(f"GDS file saved to {gds_path}")

    fig_layout, ax_layout = plt.subplots(figsize=(4, 4))
    ax_layout.imshow(final_mask, cmap='gray_r', origin='lower')
    ax_layout.set_title("Layout Preview (Parametric)")
    ax_layout.axis('off')
    layout_img_path = Path("out") / "optimized_layout_parametric.png"
    fig_layout.savefig(layout_img_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig_layout)

    return fig, drc_report, best_params, str(gds_path), str(layout_img_path)
