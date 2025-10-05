async function uploadImage() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];

    if (!file) {
        alert("Please select an image!");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch("/predict", {  // use relative URL if frontend served by FastAPI
            method: "POST",
            body: formData
        });

        const data = await response.json();

        // Show uploaded image
        const preview = document.getElementById('preview');
        preview.src = URL.createObjectURL(file);
        preview.style.display = "block"; // show uploaded image

        // Show prediction result
        const resultDiv = document.getElementById('result');
        resultDiv.style.display = "block"; // show the prediction box
        resultDiv.innerText = `Class: ${data.class}\nConfidence: ${data.confidence.toFixed(2)}`;

    } catch (error) {
        console.error("Error:", error);
    }
}
