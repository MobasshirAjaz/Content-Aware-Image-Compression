const input = document.getElementById("imageInput")
const originalImage = document.getElementById("originalImage")
const processedImage = document.getElementById("processedImage")

let uploadedImage = null

input.addEventListener("change", function(){

const file = this.files[0]

if(file){

const reader = new FileReader()

reader.onload = function(e){
originalImage.src = e.target.result
uploadedImage = e.target.result
}

reader.readAsDataURL(file)

}

})

function processImage() {
    const input = document.getElementById("imageInput");
    const processedImage = document.getElementById("processedImage");

    if (!input.files[0]) {
        alert("Please upload an image first");
        return;
    }

    const file = input.files[0];
    const mode = document.querySelector('input[name="mode"]:checked').value;

    const formData = new FormData();
    formData.append("image", file);
    formData.append("mode", mode);

    alert(`Sending to Python. Look out for the Tkinter popups!`);
    
    // Show loading state in UI
    processedImage.src = "";
    processedImage.alt = "Processing in the background... Please complete the Tkinter popups.";

    fetch('/process', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Start polling for the result using the base_name Python generated
            pollForResult(mode, data.base_name);
        } else {
            alert("Error: " + data.error);
        }
    })
    .catch(error => {
        console.error("Error:", error);
        alert("Failed to connect to the local Python server.");
    });
}

function pollForResult(mode, base_name) {
    // Check the server every 3 seconds (3000 milliseconds)
    const interval = setInterval(() => {
        fetch(`/check_ready/${mode}/${base_name}`)
        .then(response => response.json())
        .then(data => {
            if (data.ready) {
                // IT IS READY! Stop checking and show the image!
                clearInterval(interval);
                const processedImage = document.getElementById("processedImage");
                processedImage.src = data.url;
                processedImage.alt = "Final Processed Image";
            }
        });
    }, 3000);
}