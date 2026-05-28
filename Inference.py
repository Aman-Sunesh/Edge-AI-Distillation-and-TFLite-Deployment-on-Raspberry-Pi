from tflite_runtime.interpreter import Interpreter
from PIL import Image
import numpy as np
import time
import os
import psutil
import shutil

# List of models to test
models = [
    "student_model_float16_weights_only.tflite",
    "student_model_int8_weights_only_cifar.tflite"
]

# Path to images folder
image_folder = "images/"
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))])

# Ensure images exist
if not image_files:
    print("No images found in the 'images/' folder. Please add images before running inference.")
    exit()

# Function to get CPU temperature
def get_cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return int(f.read().strip()) / 1000.0  # Convert from millidegrees to Celsius
    except Exception:
        return None

# Function to preprocess images
def preprocess_image(image_path, input_shape, dtype=np.float32):
    """Load and preprocess an image to match the model's expected input shape."""
    img = Image.open(image_path).resize((input_shape[1], input_shape[2]))
    img_array = np.array(img).astype(dtype) / 255.0  # Normalize

    # INT8 models require int8 inputs, so scale correctly
    if dtype == np.int8:
        img_array = (img_array * 255 - 128).astype(np.int8)

    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# -------------------------------
# Function to run inference
# -------------------------------

def run_inference(model_path, image_path):
    """Run inference on a quantized TFLite model and collect performance data."""
    try:
        # 1) Load the TFLite model using the Interpreter.
        interpreter = Interpreter(model_path=model_path)

        # 2) Allocate memory for model tensors.
        interpreter.allocate_tensors()

        # 3) Retrieve model input and output tensor details.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # 4) Extract input shape and data type.
        input_shape = input_details[0]['shape']
        input_dtype = input_details[0]['dtype']

        # 5) Load and preprocess the image to match the model’s input format.
        sample_input = preprocess_image(image_path, input_shape, input_dtype)

        # 6) Measure CPU and memory usage before inference using psutil
        cpu_before = psutil.cpu_percent(interval=None)
        mem_before = psutil.virtual_memory().percent
        temp_before = get_cpu_temp()

        # 7) Perform inference
        start_time = time.time()
        interpreter.set_tensor(input_details[0]["index"], sample_input)
        interpreter.invoke()
        end_time = time.time()

        # 8) Retrieve the output tensor from the model.
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # 9) Determine the predicted class from the output.
        predicted_class = int(np.argmax(output_data, axis=1)[0])

        # 10) Measure CPU and memory usage after inference.
        cpu_after = psutil.cpu_percent(interval=None)
        mem_after = psutil.virtual_memory().percent
        temp_after = get_cpu_temp()

        return {
            "model": model_path,
            "image": os.path.basename(image_path),
            "inference_time": round(end_time - start_time, 4),
            "predicted_class": predicted_class,
            "cpu_before": cpu_before,
            "cpu_after": cpu_after,
            "mem_before": mem_before,
            "mem_after": mem_after,
            "temp_before": temp_before,
            "temp_after": temp_after
        }

    except Exception as e:
        print(f"Inference failed for {model_path} on {image_path}: {e}")
        return None


# Store results for final summary
results = []

# Run inference on all images using both models
for model in models:
    print(f"\nRunning inference with model: {model}\n" + "="*60)
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        result = run_inference(model, image_path)
        if result:
            results.append(result)

# Final Summary Table
shutil.get_terminal_size((80, 20))  # Ensure table formatting works well
print("\nFinal Inference Summary")
print("=" * 80)
print(f"{'Model':<40} {'Image':<20} {'Time (s)':<10} {'Pred Class':<12} {'CPU (%)':<12} {'Mem (%)':<10} {'Temp (°C)':<10}")
print("=" * 80)

for res in results:
    print(f"{res['model']:<40} {res['image']:<20} {res['inference_time']:<10} {res['predicted_class']:<12} {res['cpu_after']:<12} {res['mem_after']:<10} {res['temp_after']:<10}")

print("=" * 80)
print("Inference Completed.")