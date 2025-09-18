"""
Gradio web interface for the RF Capacitor Inverse Design workflow.

This UI provides an interface to the core inverse design functionality
controlled by the `main_controller.py`.
"""
import gradio as gr
from pathlib import Path
import time
import sys

# Add project root to Python path to allow sibling imports
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# Import the controller and helper functions
from main_controller import run_inverse_design_parametric, run_inverse_design_freeform
from pdk.helpers import get_available_pdks
from drc.checker import DRCReport, DRCViolation

# --- Constants ---
# ROOT_DIR is already defined above
DEFAULT_MODEL_PATH = str(ROOT_DIR / "checkpoints" / "surrogate-epoch=20-val_loss=0.0000.ckpt")
DEFAULT_TARGET_PATH = str(ROOT_DIR / "target.npy")

# --- UI Helper Functions ---
def _format_drc_report(report: DRCReport, params: dict, mode: str) -> str:
    """Formats the DRC report object into a readable string."""
    header = "--- DRC Report ---"
    
    param_header = "--- Final Parameters ---"
    if "Freeform" in mode:
        param_header = "--- Final Parameters (Approximated) ---"

    if not params:
        params_str = f"\n{param_header}\nNo parameters could be estimated."
    else:
        params_str = f"\n{param_header}\n" + "\n".join([f"- {k}: {v:.4f}" for k, v in params.items()])

    if report.passed:
        status = "Status: PASS"
        violations = "No violations found."
    else:
        status = "Status: FAIL"
        if not report.violations:
            violations = "An unknown DRC error occurred."
        else:
            violations_list = [f"- {v.rule}: {v.message}" for v in report.violations]
            violations = "Violations:\n" + "\n".join(violations_list)
    
    return f"{header}\n{status}\n{violations}{params_str}"

# --- Gradio App Definition ---
def create_gradio_app():
    """Creates and configures the Gradio application."""

    with gr.Blocks(theme=gr.themes.Soft(), title="RF Capacitor Inverse Designer") as demo:
        gr.Markdown("# RF Capacitor Inverse Designer")
        gr.Markdown("Find optimal capacitor geometry from target S-parameters using a surrogate model.")

        with gr.Row():
            # --- Left Column: Inputs & Controls ---
            with gr.Column(scale=1):
                gr.Markdown("### 1. Inputs")
                
                target_file = gr.File(
                    label="Target S-Parameters (.npy)", 
                    value=DEFAULT_TARGET_PATH,
                    file_count="single"
                )
                model_file = gr.File(
                    label="Surrogate Model (.pt)", 
                    value=DEFAULT_MODEL_PATH,
                    file_count="single"
                )
                
                gr.Markdown("### 2. Configuration")
                opt_mode = gr.Radio(
                    label="Optimization Mode",
                    choices=["Parametric (Optuna)", "Freeform (Pixel-based)"],
                    value="Parametric (Optuna)"
                )

                pdk_name = gr.Dropdown(
                    label="Select Process Design Kit (PDK)", 
                    choices=get_available_pdks(), 
                    value=get_available_pdks()[0]
                )
                n_trials = gr.Slider(
                    minimum=10, maximum=1000, 
                    value=100, 
                    step=10, 
                    label="Optimization Trials"
                )

                run_button = gr.Button("Run Inverse Design", variant="primary")

            # --- Right Column: Outputs ---
            with gr.Column(scale=2):
                gr.Markdown("### 3. Results")
                
                output_log = gr.Textbox(label="Log", lines=12, interactive=False, autoscroll=True)
                
                with gr.Tabs():
                    with gr.TabItem("S-Parameter Plot"):
                        output_plot = gr.Plot(label="S-Parameter Comparison")
                    with gr.TabItem("Layout & GDS"):
                        with gr.Row():
                            output_layout = gr.Image(label="Optimal Layout Preview", type="filepath", interactive=False)
                            output_gds = gr.File(label="Download GDS File", interactive=False)

        # --- Event Handlers ---
        def handle_mode_change(mode):
            if "Parametric" in mode:
                return gr.Slider(label="Optimization Trials (Optuna)", value=100, minimum=10, maximum=500, step=10)
            else:
                return gr.Slider(label="Optimization Steps (Adam)", value=200, minimum=50, maximum=1000, step=50)

        def handle_run_design(target, model, mode, pdk, trials):
            log_stream = []
            def logger(message):
                log_stream.append(message)
                # This is not a true stream, but yields updates for Gradio
                yield {output_log: "\n".join(log_stream)}

            yield {output_log: "Initializing..."}

            if "Parametric" in mode:
                run_function = run_inverse_design_parametric
            else:
                run_function = run_inverse_design_freeform

            # Run the main backend function
            fig, drc_report, best_params, gds_path, layout_path = run_function(
                target_path=target.name, 
                model_path=model.name,
                pdk_name=pdk,
                n_trials=trials,
                log_callback=lambda msg: log_stream.append(msg)
            )
            
            # Format the final DRC report
            drc_string = _format_drc_report(drc_report, best_params, mode)
            final_log = "\n".join(log_stream) + "\n\n" + drc_string

            # Final update to all components
            yield {
                output_log: final_log,
                output_plot: fig,
                output_layout: layout_path,
                output_gds: gds_path
            }

        opt_mode.change(fn=handle_mode_change, inputs=opt_mode, outputs=n_trials)

        run_button.click(
            fn=handle_run_design,
            inputs=[target_file, model_file, opt_mode, pdk_name, n_trials],
            outputs=[output_log, output_plot, output_layout, output_gds]
        )

    return demo

if __name__ == "__main__":
    app = create_gradio_app()
    app.launch(share=True)