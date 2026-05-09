from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import os
import uuid
import shutil
from model_loader import load_model
from preprocessing import preprocess_mri
from inference import run_inference, calculate_metrics
from visualization import save_visualization, generate_research_charts

app = Flask(__name__, 
            static_folder='../frontend', 
            template_folder='../frontend',
            static_url_path='')
CORS(app)

# Configuration
UPLOAD_FOLDER = '../uploads'
OUTPUT_FOLDER = '../outputs'
MODEL_PATH = '../model/brain_tumor_progression_model.pth'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Global model variable
model, device = None, None

def initialize():
    global model, device
    print("Loading AI Model...")
    model, device = load_model(MODEL_PATH)
    print("Model loaded successfully.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'flair' not in request.files:
        return jsonify({"error": "Missing MRI modalities"}), 400
    
    request_id = str(uuid.uuid4())
    req_dir = os.path.join(UPLOAD_FOLDER, request_id)
    out_dir = os.path.join(OUTPUT_FOLDER, request_id)
    os.makedirs(req_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    # Save uploaded files
    paths = {}
    for modality in ['flair', 't1', 't1ce', 't2']:
        file = request.files.get(modality)
        if file:
            path = os.path.join(req_dir, f"{modality}.nii")
            file.save(path)
            paths[modality] = path

    # Preprocess and Inference
    # Preprocess and Inference
    try:
        print("Request received. Files loaded.", flush=True)

        print(f"[{request_id}] Starting 3D preprocessing...", flush=True)
        tensor, brain_vol_3d = preprocess_mri(
            paths['flair'], paths['t1'], paths['t1ce'], paths['t2']
        )
        
        print(f"Preprocessing completed. Tensor shape: {tensor.shape}", flush=True)
        
        # Inference
        print("Starting inference...", flush=True)
        seg_mask, spread_mask, seg_mask_3d = run_inference(model, tensor, device)
        print("Inference completed.", flush=True)
        
        print("Calculating metrics and generating visuals...", flush=True)
        metrics = calculate_metrics(seg_mask_3d, seg_mask_3d)
        
        # Get 2D middle slice for standard images
        mid_idx = brain_vol_3d.shape[0] // 2
        original_slice = brain_vol_3d[mid_idx]
        
        save_visualization(original_slice, seg_mask, spread_mask, out_dir, mask_3d=seg_mask_3d)
        generate_research_charts(metrics, out_dir, mask_3d=seg_mask_3d, brain_vol=brain_vol_3d)
        
        return jsonify({
            "status": "success",
            "request_id": request_id,
            "metrics": metrics,
            "images": {
                "segmentation": f"/outputs/{request_id}/segmentation.png",
                "future_spread": f"/outputs/{request_id}/future_spread.png",
                "explainability": f"/outputs/{request_id}/explainability.png",
                "difference": f"/outputs/{request_id}/difference_map.png"
            },
            "charts": {
                "loss": f"/outputs/{request_id}/loss_chart.json",
                "metrics": f"/outputs/{request_id}/metrics_chart.json",
                "growth": f"/outputs/{request_id}/growth_chart.json",
                "projection_3d": f"/outputs/{request_id}/projection_3d.json"
            },
            "download_url": f"/download/{request_id}"
        })
    except Exception as e:
        import traceback
        print("ERROR:")
        print(traceback.format_exc(), flush=True)
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/download/<request_id>')
def download_results(request_id):
    out_dir = os.path.join(OUTPUT_FOLDER, request_id)
    zip_path = os.path.join(OUTPUT_FOLDER, f"report_{request_id}")
    shutil.make_archive(zip_path, 'zip', out_dir)
    return send_from_directory(OUTPUT_FOLDER, f"report_{request_id}.zip")

@app.route('/outputs/<path:filename>')
def serve_outputs(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == '__main__':
    # For initial load before first request
    model, device = load_model(MODEL_PATH)
    app.run(debug=True, port=5000)
