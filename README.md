!pip install -q ultralytics opencv-python matplotlib tqdm
# Import libraries
from IPython.display import display, HTML
import cv2
import numpy as np
from google.colab import files
from ultralytics import YOLO
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# Custom CSS styling for nice output
STYLE = """
<style>
.results-box {
 border: 2px solid #4CAF50;
 border-radius: 10px;
 padding: 15px;
 margin: 10px 0;
 background: #f8f9fa;
 box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}
.count-badge {
 display: inline-block;
 padding: 5px 10px;
 border-radius: 15px;
 background: #4CAF50;
 color: white;
 font-weight: bold;
 margin: 4px;
}
.header {
 color: #2c3e50;
 text-align: center;
 font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
</style>
"""
display(HTML(STYLE))
display(HTML('<h1 class="header">Enhanced Object Detection and Clustering</h1>'))
display(HTML('<p style="text-align:center">Upload an image to detect and group objects using YOLOv8 and
KMeans</p>'))
# Show loading bar during model initialization
def show_loading():
 for _ in tqdm(range(100), desc="Initializing model", ncols=70):
 time.sleep(0.01)
 print("Model ready!")
show_loading()
# Load YOLOv8 model
model = YOLO('yolov8n.pt')
# Function to detect objects and cluster them
def detect_and_cluster(image_path, n_clusters=3):
 # Read the input image
 img = cv2.imread(image_path)
 img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 print("\nDetecting objects...")
 results = model(image_path)[0]
 boxes = results.boxes
 names = results.names
 counts = {}
 centers = []
 box_data = []
 # Step 1: Collect bounding box data
 for box in boxes:
 x1, y1, x2, y2 = map(int, box.xyxy[0])
 cls_id = int(box.cls[0])
 conf = float(box.conf[0])
 class_name = names[cls_id]
 center_x = (x1 + x2) // 2
 center_y = (y1 + y2) // 2
 centers.append([center_x, center_y])
 box_data.append((x1, y1, x2, y2, class_name, conf))
 counts[class_name] = counts.get(class_name, 0) + 1
 # Step 2: Apply KMeans clustering
 if len(centers) >= n_clusters:
 kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(centers)
 labels = kmeans.labels_
 else:
 labels = [0] * len(centers)
 # Colors for each cluster
 COLORS = [(255, 0, 0), (0, 255, 0), (0, 255, 255), (255, 255, 0), (255, 0, 255)]
 # Step 3: Draw boxes with cluster colors
 for (x1, y1, x2, y2, class_name, conf), label in zip(box_data, labels):
 color = COLORS[label % len(COLORS)]
 label_text = f"{class_name} {conf:.2f} (C{label})"
 cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
 cv2.rectangle(img, (x1, y1 - 20), (x1 + 150, y1), color, -1)
 cv2.putText(img, label_text, (x1, y1 - 5),
 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
 # Save clustered output
 output_path = 'clustered_' + os.path.basename(image_path)
 cv2.imwrite(output_path, img)
 # Display detection counts
 count_html = '<div class="results-box"><h3>Detection Counts:</h3><div>'
 for obj, count in counts.items():
 count_html += f'<span class="count-badge">{obj}: {count}</span>'
 count_html += '</div></div>'
 display(HTML(count_html))
 # Read original and clustered images for side-by-side display
 original_img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
 clustered_img = cv2.cvtColor(cv2.imread(output_path), cv2.COLOR_BGR2RGB)
 # Display side-by-side using Matplotlib
 fig, axs = plt.subplots(1, 2, figsize=(18, 9))
 axs[0].imshow(original_img)
 axs[0].set_title("Original Image", fontsize=16)
 axs[0].axis('off')
 axs[1].imshow(clustered_img)
 axs[1].set_title("Clustered Detection", fontsize=16)
 axs[1].axis('off')
 plt.tight_layout()
 plt.show()
 # Print summary
 print("\nTotal Objects Detected:", sum(counts.values()))
 print("Clusters Assigned (KMeans):", n_clusters)
 print("Detected Objects:", ", ".join([f"{k} ({v})" for k, v in counts.items()]))
# Upload interface
print("\nPlease upload an image file:")
uploaded = files.upload()
# Process each uploaded image
for filename in uploaded.keys():
 print(f"\nProcessing {filename} with clustering...")
 detect_and_cluster(filename, n_clusters=3)
print("\nAll done! Run again to upload and process more images.")
