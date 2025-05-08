# app.py (Corrected index_route)
from flask import Flask, render_template, request, jsonify
import subprocess
import os
import threading
import uuid # For unique job IDs

# Import necessary items from your project
try:
    # This is the structure from your 15-param dataset_generator.py:
    # TRANSFORMED_FEATURE_MAPPING = {'p1': 'T1_power_noisy', ...}
    from dataset_generator import TRANSFORMED_FEATURE_MAPPING, RAW_PARAMETER_NAMES 
    from models.mlp_model import ACTIVATION_MAP as MLP_ACTIVATION_MAP, DEFAULT_ACTIVATION as MLP_DEFAULT_ACTIVATION
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Please ensure app.py is in the project root and other .py files are accessible.")
    exit()


app = Flask(__name__)
app.secret_key = 'your_very_secret_key_for_flask_sessions' 

JOBS = {}


def run_main_script_in_background(job_id, selected_features_cli_arg, train_fraction_cli_arg, epochs_cli_arg, activation_cli_arg):
    global JOBS
    try:
        report_filename_for_main_py = f"report_{job_id}.html"
        
        command = [
            "python", "main.py",
            "--features", selected_features_cli_arg,
            "--fraction", train_fraction_cli_arg,
            "--epochs", epochs_cli_arg,
            "--activation", activation_cli_arg,
            "--output_report_name", report_filename_for_main_py
        ]
        
        JOBS[job_id]['status'] = 'running'
        JOBS[job_id]['command'] = ' '.join(command)
        print(f"Executing for Job {job_id}: {JOBS[job_id]['command']}")

        process = subprocess.run(
            command, 
            capture_output=True, 
            text=True, 
            check=False,
            timeout=1800 
        )

        if process.returncode == 0:
            JOBS[job_id]['status'] = 'completed'
            JOBS[job_id]['report_path'] = os.path.join('reports', report_filename_for_main_py)
            JOBS[job_id]['stdout'] = process.stdout
            print(f"Job {job_id} completed successfully. Report: {JOBS[job_id]['report_path']}")
        else:
            JOBS[job_id]['status'] = 'failed'
            error_message = f"Main.py script failed with return code {process.returncode}.\n"
            error_message += f"STDOUT:\n{process.stdout}\n"
            error_message += f"STDERR:\n{process.stderr}\n"
            JOBS[job_id]['error'] = error_message
            print(f"Job {job_id} failed. Error:\n{error_message}")

    except subprocess.TimeoutExpired:
        JOBS[job_id]['status'] = 'failed'
        JOBS[job_id]['error'] = 'Analysis script (main.py) timed out.'
        print(f"Job {job_id} timed out.")
    except FileNotFoundError:
        JOBS[job_id]['status'] = 'failed'
        JOBS[job_id]['error'] = 'Error: main.py not found. Ensure it is in the correct path.'
        print(f"Job {job_id} failed: main.py not found.")
    except Exception as e:
        JOBS[job_id]['status'] = 'failed'
        JOBS[job_id]['error'] = f'An unexpected error occurred while running the script: {str(e)}'
        print(f"Job {job_id} unexpected error: {str(e)}")


@app.route('/', methods=['GET'])
def index_route():
    param_info_list = []
    # TRANSFORMED_FEATURE_MAPPING from dataset_generator.py is like: {'p1': 'T1_power_noisy', ...}
    # The value is the transformed feature name (a string).
    for p_raw_name in RAW_PARAMETER_NAMES: # e.g., p_raw_name = 'p1'
        transformed_name = TRANSFORMED_FEATURE_MAPPING.get(p_raw_name, "N/A - Check Mapping") # This is a string, e.g., 'T1_power_noisy'
        
        # Derive nature_desc based on the transformed_name string
        nature_desc = "Signal with noise" # Default
        if "FULLY_NOISY" in transformed_name:
            nature_desc = "Fully Noisy (Decoy)"
        elif "strong_signal" in transformed_name:
            nature_desc = "Strong Signal"
        elif "high_noise" in transformed_name or "unstable" in transformed_name:
            nature_desc = "Signal with High/Unstable Noise"
        elif "interaction" in transformed_name.lower():
            nature_desc = "Interaction-based Feature"
        elif "complex" in transformed_name.lower() or "trig" in transformed_name.lower():
            nature_desc = "Complex/Trigonometric Feature"
            
        param_info_list.append({
            "raw": p_raw_name, 
            "transformed": transformed_name, # This is the string name
            "description": nature_desc
        })
    
    activation_function_choices = []
    for key, display_val_from_map in MLP_ACTIVATION_MAP.items(): # displayVal is not used from map, just key
         activation_function_choices.append({"value": key, "text": key.capitalize()}) # Use key as text for consistency
    
    custom_activations = ["signum", "leaky_relu", "softmax"] 
    for act in custom_activations:
        activation_function_choices.append({"value": act, "text": f"{act.capitalize()} (Custom/Mapped)"})

    return render_template('index.html', # Or 'configurator.html' if you named it that in templates/
                           param_info=param_info_list,
                           raw_parameter_names=RAW_PARAMETER_NAMES,
                           activation_functions=activation_function_choices,
                           default_mlp_activation=MLP_DEFAULT_ACTIVATION)

@app.route('/run_analysis', methods=['POST'])
def run_analysis_route():
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {'status': 'pending', 'id': job_id}
    try:
        form_data = request.form
        selected_raw_params = form_data.getlist('parameters')
        
        features_arg_for_main_py = "all_but_noisy" 
        if selected_raw_params:
            if 'all' in selected_raw_params and len(selected_raw_params) == 1 : # only 'all' is selected
                features_arg_for_main_py = "all"
            elif 'all_but_noisy' in selected_raw_params and len(selected_raw_params) == 1:
                features_arg_for_main_py = "all_but_noisy"
            else: 
                specific_p_indices = [p.replace('p','') for p in selected_raw_params if p.startswith('p')]
                if specific_p_indices:
                    features_arg_for_main_py = ",".join(specific_p_indices)
                # If after filtering, specific_p_indices is empty but selected_raw_params was not,
                # it means only 'all' or 'all_but_noisy' might have been selected along with invalid stuff.
                # The default 'all_but_noisy' would still apply if specific_p_indices is empty.
        
        train_fraction_arg = form_data.get('train_fraction', '1.0')
        epochs_arg = form_data.get('epochs', '100')
        activation_arg = form_data.get('mlp_activation', MLP_DEFAULT_ACTIVATION)
        
        analysis_thread = threading.Thread(
            target=run_main_script_in_background, 
            args=(job_id, features_arg_for_main_py, train_fraction_arg, epochs_arg, activation_arg)
        )
        analysis_thread.start()
        return jsonify({"status": "success", "message": "Analysis job started.", "job_id": job_id})
    except Exception as e:
        JOBS[job_id]['status'] = 'failed'; JOBS[job_id]['error'] = f'Error processing request: {str(e)}'
        print(f"Error in /run_analysis for job {job_id}: {str(e)}")
        return jsonify({"status": "error", "message": f'Error processing request: {str(e)}'}), 500

@app.route('/job_status/<job_id>', methods=['GET'])
def job_status_route(job_id):
    job_info = JOBS.get(job_id)
    if not job_info: return jsonify({"status": "error", "message": "Job ID not found"}), 404
    response_data = {"job_id": job_id, "status": job_info['status']}
    if job_info['status'] == 'completed':
        response_data['report_content_url'] = f"/report_content/{job_id}" 
    elif job_info['status'] == 'failed':
        response_data['error_message'] = job_info.get('error', 'Unknown error occurred.')
    return jsonify(response_data)

@app.route('/report_content/<job_id>', methods=['GET'])
def serve_report_content_route(job_id):
    job_info = JOBS.get(job_id)
    if not job_info or job_info['status'] != 'completed' or 'report_path' not in job_info:
        return "Report not found, job not completed, or report path missing.", 404
    report_file_path = job_info['report_path']
    if os.path.exists(report_file_path):
        try:
            with open(report_file_path, 'r', encoding='utf-8') as f:
                report_html_content = f.read()
            return report_html_content, 200, {'Content-Type': 'text/html'}
        except Exception as e:
            print(f"Error reading report file {report_file_path} for job {job_id}: {e}")
            return f"Error reading report file: {str(e)}", 500
    else:
        print(f"Report file {report_file_path} missing on server for job {job_id}.")
        return "Report file is missing on the server.", 404

if __name__ == '__main__':
    if not os.path.exists('reports'): os.makedirs('reports'); print("Created 'reports' directory.")
    print("Flask app starting... Ensure 'templates/index.html' exists.")
    print("Open http://localhost:8080 in your browser.")
    app.run(debug=True, host='127.0.0.1', port=8080)