// Handle file name displays
const modalities = ['flair', 't1', 't1ce', 't2'];
modalities.forEach(m => {
    document.getElementById(m).addEventListener('change', (e) => {
        const fileName = e.target.files[0]?.name || "";
        document.getElementById(`${m}-name`).innerText = fileName;
    });
});

async function processAnalysis() {
    const formData = new FormData();
    let filesSelected = true;

    modalities.forEach(m => {
        const file = document.getElementById(m).files[0];
        if (!file) filesSelected = false;
        formData.append(m, file);
    });

    if (!filesSelected) {
        alert("Please select all 4 MRI modalities (.nii files).");
        return;
    }

    // Show Loader
    document.getElementById('loader').style.display = 'flex';
    document.getElementById('results').style.display = 'none';

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.status === 'success') {
            displayResults(data);
        } else {
            alert("Error: " + (data.error || data.message || "Processing failed"));
        }
    } catch (err) {
        console.error(err);
        alert("Connection Error. Is the backend running?");
    } finally {
        document.getElementById('loader').style.display = 'none';
    }
}

function displayResults(data) {
    document.getElementById('results').style.display = 'block';

    // Set Images
    document.getElementById('img-seg').src = data.images.segmentation;
    document.getElementById('img-spread').src = data.images.future_spread;
    document.getElementById('img-explain').src = data.images.explainability;
    document.getElementById('img-diff').src = data.images.difference;

    // Set Metrics
    document.getElementById('val-vol').innerText = data.metrics.voxel_volume;
    document.getElementById('val-future').innerText = data.metrics.future_volume;
    document.getElementById('val-growth').innerText = data.metrics.growth_percentage + "%";
    document.getElementById('val-prob').innerText = data.metrics.recovery_probability + "%";
    
    document.getElementById('val-dice').innerText = data.metrics.dice_score;
    document.getElementById('val-iou').innerText = data.metrics.iou;
    document.getElementById('val-prec').innerText = data.metrics.precision;
    document.getElementById('val-rec').innerText = data.metrics.recall;

    // Load Charts
    loadPlotlyChart('chart-loss', data.charts.loss);
    loadPlotlyChart('chart-metrics', data.charts.metrics);
    loadPlotlyChart('chart-3d', data.charts.projection_3d);
    loadPlotlyChart('chart-growth', data.charts.growth);

    // Set Download Link
    document.getElementById('btn-download').onclick = () => {
        window.location.href = data.download_url;
    };

    // Smooth Scroll to results
    document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
}

// Footer Tab Switching Logic
function switchTab(evt, tabName) {
    const tabContent = document.getElementsByClassName("tab-content");
    for (let i = 0; i < tabContent.length; i++) {
        tabContent[i].classList.remove("active");
    }

    const tabButtons = document.getElementsByClassName("tab-btn");
    for (let i = 0; i < tabButtons.length; i++) {
        tabButtons[i].classList.remove("active");
    }

    document.getElementById(tabName).classList.add("active");
    evt.currentTarget.classList.add("active");
}

// Modal Toggle Logic
function toggleModal(modalId) {
    const modal = document.getElementById(modalId);
    const isVisible = modal.style.display === 'flex';
    modal.style.display = isVisible ? 'none' : 'flex';
}

// Close modal when clicking outside content
window.onclick = function(event) {
    const infoModal = document.getElementById('info-modal');
    if (event.target == infoModal) {
        infoModal.style.display = "none";
    }
}

async function loadPlotlyChart(elementId, jsonPath) {
    try {
        const response = await fetch(jsonPath);
        const chartData = await response.json();
        
        // Ensure responsive behavior
        const config = { responsive: true, displayModeBar: false };
        
        Plotly.newPlot(elementId, chartData.data, chartData.layout, config);
        
        // Force a resize check for 3D plots
        if (elementId === 'chart-3d') {
            setTimeout(() => Plotly.Plots.resize(elementId), 500);
        }
    } catch (err) {
        console.error("Chart load error for " + elementId + ":", err);
        document.getElementById(elementId).innerHTML = `<p style="color:red; text-align:center; padding:2rem;">Failed to load chart: ${err.message}</p>`;
    }
}
