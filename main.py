from ultralytics import YOLO # Import YOLO class from ultralytics module
import numpy as np
import cv2

#Training the model
def train():
    # Load YOLOv11 model
    model = YOLO('yolov11n.pt')
    model.train(
        data="E:\\proiect_ai\\datasets\\weapon_detection\\data.yaml",  # Path to the dataset YAML file
        epochs=15,  # Number of training epochs
        imgsz=640,  # Image size
        batch_size=16,  # Batch size
    )

def run_predict(image_array, model):
    if not isinstance(image_array, np.ndarray):
        raise ValueError("Image must be a numpy array")
    if image_array.dtype != np.uint8:
        image_array = (image_array * 255).astype(np.uint8)
    results = model.predict(
        source = image_array,   # Image input
        conf = 0.25,            # Confidence threshold
        save_txt = False,       # Do not save the results
        verbose = False,        # Do not print the results
    )

    return results

def process_video(video_path, output_path, model):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # MP4 video codec
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Process video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Perform YOLO inference on the frame
        results = run_predict(frame, model)

        # Draw detection boxes and labels on the frame
        for result in results:
            for box in result.boxes:
                # Extract box coordinates, class, and confidence
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box (x1, y1, x2, y2)
                confidence = box.conf[0]  # Confidence score
                class_id = int(box.cls[0])  # Class ID
                label = f"{model.names[class_id]} {confidence:.2f}"  # Class name and confidence

                # Draw the bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw the label
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write the processed frame to the output video
        out.write(frame)

    # Release resources
    cap.release()
    out.release()
    print(f"Processed video saved to {output_path}")



if __name__ == "__main__":
    input_path = "video.mp4"
    output_path = "output.mp4"
    model = YOLO("runs/detect/train7/weights/best.pt")
    process_video(input_path, output_path, model)