document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const uploadContent = document.getElementById('upload-content');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const removeBtn = document.getElementById('remove-btn');

    const idleState = document.getElementById('idle-state');
    const loadingState = document.getElementById('loading-state');
    const resultsContainer = document.getElementById('results-container');
    const predictionList = document.getElementById('prediction-list');
    const unrecognizedState = document.getElementById('unrecognized-state');
    const unrecognizedMsg = document.getElementById('unrecognized-msg');
    const lowConfidenceResults = document.getElementById('low-confidence-results');
    const labelInput = document.getElementById('label-input');
    const btnAddDataset = document.getElementById('btn-add-dataset');
    const formHint = document.getElementById('form-hint');
    const classSuggestions = document.getElementById('class-suggestions');

    const btnRetrain = document.getElementById('btn-retrain');
    const retrainStatus = document.getElementById('retrain-status');

    // Store the current file for add-to-dataset
    let currentFile = null;

    // ─── Drag & Drop ────────────────────────────────────────────
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
    });

    dropZone.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }

    dropZone.addEventListener('click', (e) => {
        if (e.target !== removeBtn && !removeBtn.contains(e.target)) {
            fileInput.click();
        }
    });

    fileInput.addEventListener('change', function() {
        handleFiles(this.files);
    });

    function handleFiles(files) {
        if (files.length === 0) return;
        const file = files[0];
        
        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file (JPG, PNG, etc.)');
            return;
        }

        currentFile = file;
        previewImage(file);
        uploadAndAnalyze(file);
    }

    function previewImage(file) {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onloadend = function() {
            imagePreview.src = reader.result;
            uploadContent.classList.add('hidden');
            previewContainer.classList.remove('hidden');
        }
    }

    // ─── Reset UI ───────────────────────────────────────────────
    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.value = '';
        imagePreview.src = '';
        currentFile = null;
        previewContainer.classList.add('hidden');
        uploadContent.classList.remove('hidden');
        
        idleState.classList.remove('hidden');
        loadingState.classList.add('hidden');
        resultsContainer.classList.add('hidden');
        unrecognizedState.classList.add('hidden');
        predictionList.innerHTML = '';
        lowConfidenceResults.innerHTML = '';
        labelInput.value = '';
        formHint.textContent = '';
        formHint.className = 'form-hint';
    });

    // ─── Predict ────────────────────────────────────────────────
    async function uploadAndAnalyze(file) {
        idleState.classList.add('hidden');
        resultsContainer.classList.add('hidden');
        unrecognizedState.classList.add('hidden');
        loadingState.classList.remove('hidden');
        predictionList.innerHTML = '';
        lowConfidenceResults.innerHTML = '';

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || 'Failed to analyze image');
            }

            const data = await response.json();
            
            if (data.recognized) {
                displayResults(data.predictions);
            } else {
                displayUnrecognized(data);
            }
            
        } catch (error) {
            console.error('Error:', error);
            alert(`Analysis Error: ${error.message}`);
            loadingState.classList.add('hidden');
            idleState.classList.remove('hidden');
        }
    }

    // ─── Display recognized results ─────────────────────────────
    function displayResults(predictions) {
        loadingState.classList.add('hidden');
        resultsContainer.classList.remove('hidden');
        
        predictions.forEach((pred, index) => {
            const confidence = pred.confidence.toFixed(1);
            
            const item = document.createElement('div');
            item.className = 'prediction-item';
            
            item.innerHTML = `
                <div class="prediction-header">
                    <span class="class-name">${pred.class}</span>
                    <span class="confidence-value">${confidence}%</span>
                </div>
                <div class="bar-bg">
                    <div class="bar-fill" id="bar-${index}"></div>
                </div>
            `;
            
            predictionList.appendChild(item);
            
            setTimeout(() => {
                document.getElementById(`bar-${index}`).style.width = `${confidence}%`;
            }, 50 + (index * 100));
        });
    }

    // ─── Display unrecognized state ─────────────────────────────
    function displayUnrecognized(data) {
        loadingState.classList.add('hidden');
        unrecognizedState.classList.remove('hidden');
        
        unrecognizedMsg.textContent = data.message || 'The model could not confidently classify this image.';
        
        // Show low-confidence predictions
        if (data.predictions && data.predictions.length > 0) {
            lowConfidenceResults.innerHTML = '<p class="low-conf-label">Best guesses (low confidence):</p>';
            data.predictions.slice(0, 3).forEach((pred) => {
                const conf = pred.confidence.toFixed(1);
                const item = document.createElement('div');
                item.className = 'low-conf-item';
                item.innerHTML = `
                    <span class="low-conf-class">${pred.class}</span>
                    <span class="low-conf-value">${conf}%</span>
                    <div class="low-conf-bar-bg">
                        <div class="low-conf-bar-fill" style="width: ${conf}%"></div>
                    </div>
                `;
                lowConfidenceResults.appendChild(item);
            });
        }
        
        // Reset form
        labelInput.value = '';
        formHint.textContent = '';
        formHint.className = 'form-hint';
    }

    // ─── Add to Dataset ─────────────────────────────────────────
    btnAddDataset.addEventListener('click', async () => {
        const label = labelInput.value.trim().toLowerCase();
        
        if (!label) {
            formHint.textContent = 'Please enter a label for this image.';
            formHint.className = 'form-hint hint-error';
            return;
        }
        
        if (!currentFile) {
            formHint.textContent = 'No image loaded. Please upload an image first.';
            formHint.className = 'form-hint hint-error';
            return;
        }
        
        btnAddDataset.disabled = true;
        btnAddDataset.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Adding...';
        
        const formData = new FormData();
        formData.append('file', currentFile);
        formData.append('label', label);
        
        try {
            const response = await fetch('/add-to-dataset', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (response.ok && data.success) {
                formHint.textContent = `✓ ${data.message} (${data.total_custom_images} total custom images)`;
                formHint.className = 'form-hint hint-success';
                labelInput.value = '';
            } else {
                throw new Error(data.detail || 'Failed to add image.');
            }
        } catch (error) {
            formHint.textContent = `✗ Error: ${error.message}`;
            formHint.className = 'form-hint hint-error';
        } finally {
            btnAddDataset.disabled = false;
            btnAddDataset.innerHTML = '<i class="fa-solid fa-plus"></i> Add to Dataset';
        }
    });

    // ─── Retrain ────────────────────────────────────────────────
    btnRetrain.addEventListener('click', async () => {
        btnRetrain.disabled = true;
        btnRetrain.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Starting...';
        retrainStatus.textContent = '';
        retrainStatus.className = 'retrain-status';
        
        try {
            const response = await fetch('/retrain', { method: 'POST' });
            const data = await response.json();
            
            if (response.ok && data.success) {
                retrainStatus.textContent = data.message;
                retrainStatus.className = 'retrain-status status-running';
                btnRetrain.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Training...';
                
                // Poll for completion
                pollRetrainStatus();
            } else {
                throw new Error(data.detail || data.message || 'Failed to start retraining.');
            }
        } catch (error) {
            retrainStatus.textContent = `Error: ${error.message}`;
            retrainStatus.className = 'retrain-status status-error';
            btnRetrain.disabled = false;
            btnRetrain.innerHTML = '<i class="fa-solid fa-play"></i> Start Retraining';
        }
    });
    
    function pollRetrainStatus() {
        const interval = setInterval(async () => {
            try {
                const response = await fetch('/retrain-status');
                const data = await response.json();
                
                retrainStatus.textContent = data.message;
                
                if (!data.running) {
                    clearInterval(interval);
                    retrainStatus.className = 'retrain-status status-done';
                    btnRetrain.disabled = false;
                    btnRetrain.innerHTML = '<i class="fa-solid fa-play"></i> Start Retraining';
                    
                    // Reload chart data
                    loadAccuracyChart();
                }
            } catch (e) {
                // Ignore polling errors
            }
        }, 5000);
    }

    // ─── Load class suggestions ─────────────────────────────────
    async function loadClassSuggestions() {
        try {
            const response = await fetch('/classes');
            const data = await response.json();
            
            if (data.classes) {
                classSuggestions.innerHTML = '';
                data.classes.forEach(cls => {
                    const option = document.createElement('option');
                    option.value = cls;
                    classSuggestions.appendChild(option);
                });
            }
        } catch (e) {
            // Non-critical, silently fail
        }
    }
    
    loadClassSuggestions();

    // ─── Accuracy Chart (Chart.js) ──────────────────────────────
    let accuracyChart = null;

    async function loadAccuracyChart() {
        const chartMeta = document.getElementById('chart-meta');
        const chartSection = document.getElementById('chart-section');
        
        try {
            const response = await fetch('/training-history');
            
            if (!response.ok) {
                chartSection.classList.add('chart-empty');
                chartMeta.innerHTML = '<span class="chart-no-data">No training data yet — run train.py first</span>';
                return;
            }
            
            const history = await response.json();
            chartSection.classList.remove('chart-empty');
            
            const epochs = history.train_acc.map((_, i) => `Epoch ${i + 1}`);
            
            // Show best accuracy badge
            chartMeta.innerHTML = `
                <span class="best-acc-badge">
                    <i class="fa-solid fa-trophy"></i> Best: ${history.best_val_acc}%
                </span>
            `;
            
            const ctx = document.getElementById('accuracy-chart').getContext('2d');
            
            // Destroy existing chart if any
            if (accuracyChart) {
                accuracyChart.destroy();
            }
            
            accuracyChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: epochs,
                    datasets: [
                        {
                            label: 'Train Accuracy (%)',
                            data: history.train_acc,
                            borderColor: '#4169e1',
                            backgroundColor: 'rgba(65, 105, 225, 0.1)',
                            borderWidth: 2.5,
                            pointRadius: 5,
                            pointHoverRadius: 8,
                            pointBackgroundColor: '#4169e1',
                            pointBorderColor: '#fff',
                            pointBorderWidth: 2,
                            pointHoverBackgroundColor: '#fff',
                            pointHoverBorderColor: '#4169e1',
                            pointHoverBorderWidth: 3,
                            fill: true,
                            tension: 0.3,
                        },
                        {
                            label: 'Validation Accuracy (%)',
                            data: history.val_acc,
                            borderColor: '#00ffaa',
                            backgroundColor: 'rgba(0, 255, 170, 0.08)',
                            borderWidth: 2.5,
                            pointRadius: 5,
                            pointHoverRadius: 8,
                            pointBackgroundColor: '#00ffaa',
                            pointBorderColor: '#fff',
                            pointBorderWidth: 2,
                            pointHoverBackgroundColor: '#fff',
                            pointHoverBorderColor: '#00ffaa',
                            pointHoverBorderWidth: 3,
                            fill: true,
                            tension: 0.3,
                        },
                        {
                            label: 'Validation Loss',
                            data: history.val_loss,
                            borderColor: '#ff3366',
                            backgroundColor: 'rgba(255, 51, 102, 0.05)',
                            borderWidth: 1.5,
                            borderDash: [5, 5],
                            pointRadius: 3,
                            pointHoverRadius: 6,
                            pointBackgroundColor: '#ff3366',
                            pointBorderColor: '#fff',
                            pointBorderWidth: 1,
                            pointHoverBackgroundColor: '#fff',
                            pointHoverBorderColor: '#ff3366',
                            fill: false,
                            tension: 0.3,
                            yAxisID: 'y1',
                        },
                    ],
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {
                        mode: 'index',
                        intersect: false,
                    },
                    plugins: {
                        legend: {
                            display: false, // Using custom legend
                        },
                        tooltip: {
                            enabled: true,
                            backgroundColor: 'rgba(15, 15, 25, 0.95)',
                            titleColor: '#f0f0f0',
                            bodyColor: '#ccc',
                            borderColor: 'rgba(255,255,255,0.15)',
                            borderWidth: 1,
                            padding: 14,
                            cornerRadius: 10,
                            titleFont: {
                                family: 'Inter',
                                size: 13,
                                weight: '600',
                            },
                            bodyFont: {
                                family: 'Inter',
                                size: 12,
                            },
                            displayColors: true,
                            boxPadding: 6,
                            callbacks: {
                                title: function(context) {
                                    const idx = context[0].dataIndex;
                                    const phase = history.phase ? history.phase[idx] : null;
                                    let title = context[0].label;
                                    if (phase) {
                                        title += phase === 1 ? '  •  Phase 1 (FC Head)' : '  •  Phase 2 (Fine-tune)';
                                    }
                                    return title;
                                },
                                label: function(context) {
                                    const label = context.dataset.label;
                                    const value = context.parsed.y;
                                    if (label.includes('Loss')) {
                                        return ` ${label}: ${value.toFixed(4)}`;
                                    }
                                    return ` ${label}: ${value.toFixed(1)}%`;
                                }
                            }
                        }
                    },
                    scales: {
                        x: {
                            grid: {
                                color: 'rgba(255,255,255,0.05)',
                            },
                            ticks: {
                                color: 'rgba(255,255,255,0.5)',
                                font: { family: 'Inter', size: 11 },
                            },
                        },
                        y: {
                            position: 'left',
                            title: {
                                display: true,
                                text: 'Accuracy (%)',
                                color: 'rgba(255,255,255,0.6)',
                                font: { family: 'Inter', size: 12 },
                            },
                            grid: {
                                color: 'rgba(255,255,255,0.05)',
                            },
                            ticks: {
                                color: 'rgba(255,255,255,0.5)',
                                font: { family: 'Inter', size: 11 },
                                callback: function(value) { return value + '%'; }
                            },
                            min: 0,
                            max: 100,
                        },
                        y1: {
                            position: 'right',
                            title: {
                                display: true,
                                text: 'Loss',
                                color: 'rgba(255,255,255,0.4)',
                                font: { family: 'Inter', size: 12 },
                            },
                            grid: {
                                drawOnChartArea: false,
                            },
                            ticks: {
                                color: 'rgba(255,255,255,0.4)',
                                font: { family: 'Inter', size: 11 },
                            },
                        },
                    },
                    animation: {
                        duration: 1200,
                        easing: 'easeOutQuart',
                    },
                },
            });
            
        } catch (error) {
            console.error('Failed to load accuracy chart:', error);
            chartMeta.innerHTML = '<span class="chart-no-data">Could not load training data</span>';
        }
    }
    
    // Load chart on page load
    loadAccuracyChart();
});
