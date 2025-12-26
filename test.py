import os
import cv2
import numpy as np
import warnings
import time

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name

warnings.filterwarnings('ignore')

def run_multi_face_realtime(model_dir="./resources/anti_spoof_models", device_id=0):
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()
    
    model_list = []
    for model_name in os.listdir(model_dir):
        model_list.append((model_name, parse_model_name(model_name)))

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)

        faces_bbox = model_test.get_all_bbox(frame)

        for bbox in faces_bbox:
            bbox[0] = max(0, bbox[0])
            bbox[1] = max(0, bbox[1])
            
            prediction = np.zeros((1, 3))
            for model_name, model_info in model_list:
                h_input, w_input, model_type, scale = model_info
                param = {
                    "org_img": frame,
                    "bbox": bbox,
                    "scale": scale,
                    "out_w": w_input,
                    "out_h": h_input,
                    "crop": True,
                }
                if scale is None: param["crop"] = False
                
                img = image_cropper.crop(**param)
                prediction += model_test.predict(img, os.path.join(model_dir, model_name))

            label = np.argmax(prediction)
            value = prediction[0][label] / len(model_list)
            
            # --- แก้ไขตรงนี้ ---
            # เช็คว่าค่า confidence มากกว่าหรือเท่ากับ 0.8 หรือไม่
            if value >= 0.9:
                color = (0, 255, 0) if label == 1 else (0, 0, 255)
                txt = "Real" if label == 1 else "Fake"
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color, 2)
                cv2.putText(frame, f"{txt}: {value:.2f}", (bbox[0], bbox[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("Multi-Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_multi_face_realtime()