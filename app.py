import os
import time
import subprocess
import sys
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder='UI')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@app.route('/')
def index():
    return send_from_directory('UI', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('UI', path)

@app.route('/process', methods=['POST'])
def process_image():
    file = request.files.get('image')
    mode = request.form.get('mode')

    if not file or file.filename == '':
        return jsonify({"error": "No file uploaded"}), 400

    name, ext = os.path.splitext(file.filename)
    timestamp = int(time.time())
    stored_base_name = f"{name}_{timestamp}"  # We need this to track the file!
    stored_name = f"{stored_base_name}{ext}"

    try:
        if mode == 'decompress':
            save_dir = os.path.join(BASE_DIR, 'Enhancement', 'input_images')
            os.makedirs(save_dir, exist_ok=True)
            filepath = os.path.join(save_dir, stored_name)
            file.save(filepath)

            script_path = os.path.join(BASE_DIR, 'Enhancement', 'gui.py')
            subprocess.Popen([sys.executable, script_path, "--auto", filepath])

        elif mode == 'compress':
            save_dir = os.path.join(BASE_DIR, 'Compression', 'input_images')
            os.makedirs(save_dir, exist_ok=True)
            filepath = os.path.join(save_dir, stored_name)
            file.save(filepath)

            script_path = os.path.join(BASE_DIR, 'Compression', 'gui.py')
            subprocess.Popen([sys.executable, script_path, "--auto", filepath])
            
        else:
            return jsonify({"error": "Invalid mode"}), 400

        # Return the base_name so the JS knows what to look for
        return jsonify({"success": True, "base_name": stored_base_name})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- NEW: Check if the final image is ready ---
@app.route('/check_ready/<mode>/<base_name>')
def check_ready(mode, base_name):
    folder = 'Enhancement' if mode == 'decompress' else 'Compression'
    output_dir = os.path.join(BASE_DIR, folder, 'final_merged_output')
    
    # Check if a file starting with our base_name exists in the final folder
    if os.path.exists(output_dir):
        for filename in os.listdir(output_dir):
            if filename.startswith(base_name) and filename.endswith('.png'):
                return jsonify({"ready": True, "url": f"/serve_output/{mode}/{filename}"})
    
    return jsonify({"ready": False})

# --- NEW: Serve the final image to the UI ---
@app.route('/serve_output/<mode>/<filename>')
def serve_output(mode, filename):
    folder = 'Enhancement' if mode == 'decompress' else 'Compression'
    directory = os.path.join(BASE_DIR, folder, 'final_merged_output')
    return send_from_directory(directory, filename)

if __name__ == '__main__':
    print("Bridge Server running! Open http://127.0.0.1:5000 in your browser.")
    app.run(port=5000, debug=True)