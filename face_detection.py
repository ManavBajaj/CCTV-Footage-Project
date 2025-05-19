import cv2
import os
import re
from insightface.app import FaceAnalysis

def extract_frames_and_detect_faces(video_path, output_folder, fps_interval=1):
    # Prepare face detector
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=-1)  # Use 0 for GPU, -1 for CPU

    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = fps * fps_interval
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            timestamp = frame_count // fps
            print(f"⏱ Processing frame at {timestamp} sec...")

            # Detect faces
            faces = app.get(frame)
            if not faces:
                print("   No faces detected.")
                frame_count += 1
                continue

            # Create a folder for this frame
            frame_output_dir = os.path.join(output_folder, f"frame_{timestamp}")
            os.makedirs(frame_output_dir, exist_ok=True)

            log_lines = [f"Timestamp: {timestamp} sec", f"Detected {len(faces)} face(s):"]

            for idx, face in enumerate(faces, start=1):
                x1, y1, x2, y2 = face.bbox.astype(int)
                face_crop = frame[y1:y2, x1:x2]
                face_filename = os.path.join(frame_output_dir, f"face_{idx}.jpg")
                cv2.imwrite(face_filename, face_crop)
                log_lines.append(f"Face {idx}: BBox = ({x1}, {y1}) to ({x2}, {y2})")

            # Save log file
            log_path = os.path.join(frame_output_dir, "coordinates.txt")
            with open(log_path, "w") as f:
                f.write("\n".join(log_lines))

            print(f"   Saved {len(faces)} face(s) to {frame_output_dir}")

        frame_count += 1

    cap.release()
    print(f"\n All done! Faces saved in: '{output_folder}'")

if __name__ == "__main__":
    #  UPDATE THIS LINE ONLY as needed
    video_path = "factory_2.mp4"  # You can either use absolute or relative path

    output_dir = "faces_detected"
    extract_frames_and_detect_faces(video_path, output_dir)
