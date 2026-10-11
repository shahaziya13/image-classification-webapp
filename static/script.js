const dropArea = document.getElementById("dropArea");
const fileInput = document.getElementById("fileInput");
const browseButton = document.getElementById("browseButton");

const previewSection = document.getElementById("previewSection");
const imagePreview = document.getElementById("imagePreview");
const predictButton = document.getElementById("predictButton");

const loading = document.getElementById("loading");
const resultSection = document.getElementById("resultSection");
const predictionList = document.getElementById("predictionList");

const errorMessage = document.getElementById("errorMessage");

const historyList = document.getElementById("historyList");
const clearHistoryButton = document.getElementById("clearHistory");

let selectedFile = null;


// -------------------------
// Browse Image
// -------------------------

browseButton.addEventListener("click", () => {
    fileInput.click();
});

fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) {
        handleFile(fileInput.files[0]);
    }
});


// -------------------------
// Drag and Drop
// -------------------------

dropArea.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropArea.classList.add("dragover");
});

dropArea.addEventListener("dragleave", () => {
    dropArea.classList.remove("dragover");
});

dropArea.addEventListener("drop", (event) => {
    event.preventDefault();

    dropArea.classList.remove("dragover");

    if (event.dataTransfer.files.length > 0) {
        handleFile(event.dataTransfer.files[0]);
    }
});


// -------------------------
// Validate + Preview Image
// -------------------------

function handleFile(file) {

    hideError();

    const allowedTypes = [
        "image/jpeg",
        "image/png",
        "image/webp"
    ];

    if (!allowedTypes.includes(file.type)) {
        showError(
            "Invalid file type. Please upload JPG, JPEG, PNG or WEBP."
        );
        return;
    }

    const maxSize = 5 * 1024 * 1024;

    if (file.size > maxSize) {
        showError(
            "Image is too large. Maximum file size is 5 MB."
        );
        return;
    }

    selectedFile = file;

    const reader = new FileReader();

    reader.onload = (event) => {
        imagePreview.src = event.target.result;
        previewSection.classList.remove("hidden");
    };

    reader.readAsDataURL(file);

    resultSection.classList.add("hidden");
}


// -------------------------
// Classify Image
// -------------------------

predictButton.addEventListener("click", async () => {

    if (!selectedFile) {
        showError("Please select an image first.");
        return;
    }

    hideError();

    resultSection.classList.add("hidden");
    loading.classList.remove("hidden");

    predictButton.disabled = true;
    predictButton.textContent = "Classifying...";

    const formData = new FormData();

    formData.append(
        "file",
        selectedFile
    );

    try {

        const response = await fetch(
            "/api/predict",
            {
                method: "POST",
                body: formData
            }
        );

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.detail || "Prediction failed."
            );
        }

        displayPredictions(data.predictions);

        saveHistory(
            data.filename,
            data.predictions[0]
        );

    } catch (error) {

        showError(error.message);

    } finally {

        loading.classList.add("hidden");

        predictButton.disabled = false;
        predictButton.textContent = "Classify Image";
    }
});


// -------------------------
// Display Top 5 Results
// -------------------------

function displayPredictions(predictions) {

    predictionList.innerHTML = "";

    predictions.forEach((prediction, index) => {

        const item = document.createElement("div");
        item.className = "prediction";

        const info = document.createElement("div");
        info.className = "prediction-info";

        const name = document.createElement("span");
        name.className = "prediction-name";

        name.textContent =
            `${index + 1}. ${prediction.class}`;

        const confidence = document.createElement("span");
        confidence.className = "confidence";

        confidence.textContent =
            `${prediction.confidence.toFixed(2)}%`;

        info.appendChild(name);
        info.appendChild(confidence);


        const barContainer =
            document.createElement("div");

        barContainer.className =
            "bar-container";


        const bar =
            document.createElement("div");

        bar.className = "bar";

        bar.style.width =
            `${prediction.confidence}%`;


        barContainer.appendChild(bar);

        item.appendChild(info);
        item.appendChild(barContainer);

        predictionList.appendChild(item);
    });

    resultSection.classList.remove("hidden");
}


// -------------------------
// Prediction History
// -------------------------

function saveHistory(filename, topPrediction) {

    let history = JSON.parse(
        localStorage.getItem("predictionHistory")
    ) || [];

    const record = {
        filename: filename,
        prediction: topPrediction.class,
        confidence: topPrediction.confidence,
        time: new Date().toLocaleString()
    };

    history.unshift(record);

    // Keep latest 10 predictions
    history = history.slice(0, 10);

    localStorage.setItem(
        "predictionHistory",
        JSON.stringify(history)
    );

    displayHistory();
}


function displayHistory() {

    const history = JSON.parse(
        localStorage.getItem("predictionHistory")
    ) || [];

    historyList.innerHTML = "";

    if (history.length === 0) {

        historyList.innerHTML =
            '<p class="empty-history">No predictions yet.</p>';

        return;
    }

    history.forEach((record) => {

        const item =
            document.createElement("div");

        item.className =
            "history-item";


        const prediction =
            document.createElement("div");

        prediction.innerHTML =
            `<span class="history-name">${record.prediction}</span>
             — 
             <span class="history-confidence">${Number(record.confidence).toFixed(2)}%</span>`;


        const file =
            document.createElement("div");

        file.className =
            "history-file";

        file.textContent =
            `${record.filename} • ${record.time}`;


        item.appendChild(prediction);
        item.appendChild(file);

        historyList.appendChild(item);
    });
}


// -------------------------
// Clear History
// -------------------------

clearHistoryButton.addEventListener(
    "click",
    () => {

        localStorage.removeItem(
            "predictionHistory"
        );

        displayHistory();
    }
);


// -------------------------
// Error Handling
// -------------------------

function showError(message) {

    errorMessage.textContent = message;

    errorMessage.classList.remove(
        "hidden"
    );
}


function hideError() {

    errorMessage.classList.add(
        "hidden"
    );
}


// Load saved history
displayHistory();