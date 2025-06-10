This project introduces a hybrid intelligent system that combines advanced object detection using YOLOv8 with
clustering based on the KMeans algorithm. Object detection is carried out using the powerful YOLOv8n model
from Ultralytics, which identifies objects in an uploaded image, locates their bounding boxes, and assigns class
labels with confidence scores. To enhance the analysis, KMeans clustering is applied to the spatial centers of the
detected objects, allowing users to observe groupings and distributions of objects visually. The final results are
presented with customized color-coded annotations, highlighting both object categories and their respective
clusters.
The system is developed using Python and designed to run seamlessly on Google Colab, ensuring ease of use
without the need for heavy local installations. A user-friendly interface is built using IPython display tools, and
enhanced with custom HTML and CSS to deliver a professional and interactive experience. Besides object
detection and visualization, the application also provides detailed statistics, such as detection counts and
clustering summaries, helping users gain insights into object patterns within the image. It is highly extensible,
making it an excellent base for future developments in visual analytics and machine learning research.
In the future, this project can be expanded to support real-time video processing, more sophisticated clustering
techniques such as DBSCAN, and even 3D spatial clustering using depth information. With minor
modifications, the system can be tailored for specific applications like traffic management, wildlife monitoring,
retail analytics, or smart city surveillance. By fusing detection and clustering into a single automated pipeline,
this project not only demonstrates the potential of integrating computer vision and machine learning but also
opens up multiple avenues for applied research and industrial solutions
