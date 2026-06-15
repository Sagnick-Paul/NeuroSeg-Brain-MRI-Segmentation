document.addEventListener("DOMContentLoaded", () => {
    // =========================================================
    // STATE VARIABLES
    // =========================================================
    let sessionId = null;
    let mriFile = null;
    let gtFile = null;
    let updateTimeout = null;

    // =========================================================
    // DOM ELEMENTS
    // =========================================================
    // Status
    const systemStatusText = document.getElementById("system-status");
    const systemStatusDot = systemStatusText.previousElementSibling;
    const deviceTypeVal = document.getElementById("device-type");

    // File Inputs & Upload Zones
    const mriUploadZone = document.getElementById("mri-upload-zone");
    const mriFileInput = document.getElementById("mri-file-input");
    const mriFilename = document.getElementById("mri-filename");

    const gtUploadZone = document.getElementById("gt-upload-zone");
    const gtFileInput = document.getElementById("gt-file-input");
    const gtFilename = document.getElementById("gt-filename");

    // Controls
    const thresholdSlider = document.getElementById("threshold-slider");
    const thresholdVal = document.getElementById("threshold-val");
    const opacitySlider = document.getElementById("opacity-slider");
    const opacityVal = document.getElementById("opacity-val");
    const gradcamToggle = document.getElementById("gradcam-toggle");

    // Stat Cards
    const statArea = document.getElementById("stat-area");
    const statPercentage = document.getElementById("stat-percentage");
    const statConfidence = document.getElementById("stat-confidence");
    const statDice = document.getElementById("stat-dice");
    const statIoU = document.getElementById("stat-iou");
    const statSensitivity = document.getElementById("stat-sensitivity");
    const inferenceTimeVal = document.getElementById("inference-time-val");

    // Tabs
    const tabButtons = document.querySelectorAll(".tab-btn");
    const tabPanes = document.querySelectorAll(".tab-pane");
    const downloadBtn = document.getElementById("download-btn");

    // Viewers
    const imgOriginal = document.getElementById("img-original");
    const imgOverlay = document.getElementById("img-overlay");
    const imgGradcam = document.getElementById("img-gradcam");
    const imgRefMri = document.getElementById("img-ref-mri");

    // Comparison Slider Elements
    const sliderContainer = document.getElementById("slider-comp-container");
    const sliderImgOriginal = document.getElementById("slider-img-original");
    const sliderImgOverlay = document.getElementById("slider-img-overlay");
    const sliderOverlayWrapper = document.getElementById("slider-overlay-wrapper");
    const sliderDivider = document.getElementById("slider-divider");

    // Loader
    const loadingSpinner = document.getElementById("loading-spinner");

    // =========================================================
    // FILE UPLOAD HANDLERS (DRAG & DROP)
    // =========================================================
    
    // Set status helper
    function setSystemStatus(status, type = "ready") {
        systemStatusText.innerText = status;
        systemStatusDot.className = "status-dot";
        if (type === "processing") {
            systemStatusDot.classList.add("orange");
        } else if (type === "ready") {
            systemStatusDot.classList.add("green");
        }
    }

    // Set active style for drag and drop
    function setupDragAndDrop(zone, input, onFileSelect) {
        // Prevent clicks on the input from bubbling up to the zone
        input.addEventListener("click", (e) => e.stopPropagation());

        zone.addEventListener("click", () => input.click());

        input.addEventListener("change", (e) => {
            if (e.target.files.length > 0) {
                onFileSelect(e.target.files[0]);
            }
        });

        zone.addEventListener("dragover", (e) => {
            e.preventDefault();
            zone.classList.add("dragover");
        });

        zone.addEventListener("dragleave", () => {
            zone.classList.remove("dragover");
        });

        zone.addEventListener("drop", (e) => {
            e.preventDefault();
            zone.classList.remove("dragover");
            if (e.dataTransfer.files.length > 0) {
                onFileSelect(e.dataTransfer.files[0]);
            }
        });
    }

    // MRI File Setup
    setupDragAndDrop(mriUploadZone, mriFileInput, (file) => {
        mriFile = file;
        mriFilename.innerText = file.name;
        mriUploadZone.classList.add("active");
        uploadFiles();
    });

    // GT File Setup
    setupDragAndDrop(gtUploadZone, gtFileInput, (file) => {
        gtFile = file;
        gtFilename.innerText = file.name;
        gtUploadZone.classList.add("active");
        // Only trigger upload if MRI is already present
        if (mriFile) {
            uploadFiles();
        }
    });

    // =========================================================
    // API ACTIONS
    // =========================================================

    // 1. Initial Upload & Prediction Run
    async function uploadFiles() {
        if (!mriFile) return;

        setSystemStatus("Processing...", "processing");
        loadingSpinner.classList.add("active");

        const formData = new FormData();
        formData.append("file", mriFile);
        if (gtFile) {
            formData.append("gt_file", gtFile);
        }

        try {
            const response = await fetch("/api/upload", {
                method: "POST",
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || "Server error running segmentation");
            }

            const data = await response.json();
            sessionId = data.session_id;

            // Update DOM images
            imgOriginal.src = data.original;
            imgOriginal.classList.remove("placeholder");

            imgOverlay.src = data.overlay;
            imgOverlay.classList.remove("placeholder");

            imgGradcam.src = data.gradcam;
            imgGradcam.classList.remove("placeholder");

            imgRefMri.src = data.original;
            imgRefMri.classList.remove("placeholder");

            // Update split slider images
            sliderImgOriginal.src = data.original;
            sliderImgOverlay.src = data.overlay;
            sliderContainer.classList.add("active");

            // Update stats
            updateStatsUI(data.stats);

            // Update specs
            if (data.device) {
                deviceTypeVal.innerText = data.device;
            }

            // Enable download
            downloadBtn.removeAttribute("disabled");
            setSystemStatus("System Ready", "ready");

        } catch (error) {
            console.error("Upload failed:", error);
            alert("Error: " + error.message);
            setSystemStatus("Upload Error", "processing");
        } finally {
            loadingSpinner.classList.remove("active");
        }
    }

    // 2. Fast Parameter Update (Threshold / Opacity / GradCAM)
    async function updateParameters() {
        if (!sessionId) return;

        setSystemStatus("Updating Filter...", "processing");

        const payload = {
            session_id: sessionId,
            threshold: parseFloat(thresholdSlider.value),
            overlay_alpha: parseFloat(opacitySlider.value),
            show_gradcam: gradcamToggle.checked
        };

        try {
            const response = await fetch("/api/update", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || "Server error updating parameters");
            }

            const data = await response.json();

            // Update Overlay & Grad-CAM images
            imgOverlay.src = data.overlay;
            sliderImgOverlay.src = data.overlay;
            
            if (data.gradcam) {
                imgGradcam.src = data.gradcam;
            }

            // Update stats
            updateStatsUI(data.stats);
            setSystemStatus("System Ready", "ready");

        } catch (error) {
            console.error("Update failed:", error);
            setSystemStatus("Update Error", "processing");
        }
    }

    // Debounce slider updates to avoid flooding API
    function queueParameterUpdate() {
        clearTimeout(updateTimeout);
        updateTimeout = setTimeout(() => {
            updateParameters();
        }, 50);
    }

    // =========================================================
    // UI UPDATES & EVENT LISTENERS
    // =========================================================

    // Update statistics card values
    function updateStatsUI(stats) {
        statArea.innerText = stats.tumor_area.toLocaleString();
        statPercentage.innerText = stats.tumor_percentage.toFixed(2) + "%";
        statConfidence.innerText = (stats.confidence * 100).toFixed(2) + "%";
        
        if (gtFile) {
            statDice.innerText = stats.dice.toFixed(4);
            statIoU.innerText = stats.iou.toFixed(4);
            statSensitivity.innerText = stats.sensitivity.toFixed(4);
        } else {
            statDice.innerText = "-";
            statIoU.innerText = "-";
            statSensitivity.innerText = "-";
        }

        if (stats.inference_time) {
            inferenceTimeVal.innerText = stats.inference_time;
        }
    }

    // Slider inputs visual indicators
    thresholdSlider.addEventListener("input", (e) => {
        thresholdVal.innerText = parseFloat(e.target.value).toFixed(2);
        queueParameterUpdate();
    });

    opacitySlider.addEventListener("input", (e) => {
        opacityVal.innerText = parseFloat(e.target.value).toFixed(2);
        queueParameterUpdate();
    });

    gradcamToggle.addEventListener("change", () => {
        queueParameterUpdate();
    });

    // Tab buttons switching logic
    tabButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            tabButtons.forEach(b => b.classList.remove("active"));
            tabPanes.forEach(p => p.classList.remove("active"));

            btn.classList.add("active");
            const tabId = btn.getAttribute("data-tab");
            document.getElementById(`tab-${tabId}`).classList.add("active");
            
            // Adjust slider divider on tab switch to prevent misalignments
            if (tabId === "compare-slider" && sessionId) {
                setSliderPosition(50);
            }
        });
    });

    // Download overlay button
    downloadBtn.addEventListener("click", () => {
        if (!imgOverlay.src || imgOverlay.classList.contains("placeholder")) return;

        const link = document.createElement("a");
        link.href = imgOverlay.src;
        link.download = `segmentation_threshold_${thresholdSlider.value}.png`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    });

    // =========================================================
    // COMPARISON SPLIT SLIDER INTERACTION
    // =========================================================
    let isDraggingSlider = false;

    function setSliderPosition(percentage) {
        // Clamp between 0 and 100
        percentage = Math.max(0, Math.min(100, percentage));
        sliderDivider.style.left = `${percentage}%`;
        sliderOverlayWrapper.style.clipPath = `polygon(0 0, ${percentage}% 0, ${percentage}% 100%, 0 100%)`;
    }

    function handleSliderMove(clientX) {
        const rect = sliderContainer.getBoundingClientRect();
        const offsetX = clientX - rect.left;
        const percentage = (offsetX / rect.width) * 100;
        setSliderPosition(percentage);
    }

    // Mouse events
    sliderDivider.addEventListener("mousedown", (e) => {
        isDraggingSlider = true;
        e.preventDefault();
    });

    window.addEventListener("mousemove", (e) => {
        if (!isDraggingSlider) return;
        handleSliderMove(e.clientX);
    });

    window.addEventListener("mouseup", () => {
        isDraggingSlider = false;
    });

    // Touch events for mobile support
    sliderDivider.addEventListener("touchstart", (e) => {
        isDraggingSlider = true;
    });

    window.addEventListener("touchmove", (e) => {
        if (!isDraggingSlider) return;
        if (e.touches.length > 0) {
            handleSliderMove(e.touches[0].clientX);
        }
    });

    window.addEventListener("touchend", () => {
        isDraggingSlider = false;
    });
});
