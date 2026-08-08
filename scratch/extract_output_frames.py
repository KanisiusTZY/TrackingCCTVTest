import cv2
import os

video_path = "output_skynet_monitoring.mp4"
if not os.path.exists(video_path):
    print(f"Error: {video_path} does not exist.")
else:
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in {video_path}: {frame_count}")

    # Extract frame 30, 80, 150, 250, 400
    target_frames = [30, 80, 150, 250, 400]
    for frame_idx in target_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            out_name = f"output_frame_{frame_idx}.jpg"
            cv2.imwrite(out_name, frame)
            print(f"Saved {out_name}")
    cap.release()
