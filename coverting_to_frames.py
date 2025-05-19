import cv2
import os

def extract_frames(video_path, output_dir, fps_interval=1):
    os.makedirs(output_dir, exist_ok=True)
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
            filename = f"frame_{timestamp}.jpg"
            cv2.imwrite(os.path.join(output_dir, filename), frame)
            print(f"Saved frame at {timestamp} seconds: {filename}")
        frame_count += 1

    cap.release()
    print(f"\n Done extracting frames to '{output_dir}' folder.")

if __name__ == "__main__":
    video_filename = "factory_2.mp4"  
    
    frames_dir = "Frames"
    extract_frames(video_filename, frames_dir)
