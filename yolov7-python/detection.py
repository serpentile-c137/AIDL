import cv2
import time
import random
import argparse
import numpy as np
import onnxruntime as ort

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), 
            auto=True, scaleup=True, stride=32):    
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]   

    # Scale ratio (new / old)
    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        ratio = min(ratio, 1.0)            

    # Compute padding
    new_unpad = int(round(shape[1] * ratio)), int(round(shape[0] * ratio))    
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    # Divide padding into 2 sides
    dw /= 2  
    dh /= 2    

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    # Padding Top and Bottom
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    # Padding Left and Right
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))    

    # Add padding to image
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

    # Transpose image
    image_ = im.transpose((2, 0, 1))    
    image_ = np.expand_dims(image_, 0)    
    image_ = np.ascontiguousarray(image_)

    im = image_.astype(np.float32)
    im /= 255
    
    outname = [i.name for i in session.get_outputs()]        
    inname = [i.name for i in session.get_inputs()]        
    inp = {inname[0]:im}

    # Inference
    outputs = session.run(outname, inp)[0]
    return im, outputs, ratio, (dw, dh)

def loadSource(source_file):
    img_formats = ['jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']
    key = 1 # 1 = Video, 0 = Image
    frame = None
    cap = None

    # Source from webcam
    if(source_file == "0"):
        image_type = False
        source_file = 0    
    else:
        image_type = source_file.split('.')[-1].lower() in img_formats

    # Open Image or Video
    if(image_type):
        frame = cv2.imread(source_file)
        key = 0
    else:
        cap = cv2.VideoCapture(source_file)

    return image_type, key, frame, cap

if __name__ == '__main__':
    # Add Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="data/videos/road.mp4", help="Video")
    parser.add_argument("--names", type=str, default="data/class.names", help="Object Names")
    parser.add_argument("--weights", type=str, default="yolov7.onnx", help="Pretrained Weights")
    parser.add_argument("--tresh", type=float, default=0.25, help="Confidence Threshold")
    args = parser.parse_args()    

    providers = ['CPUExecutionProvider']
    session = ort.InferenceSession(args.weights, providers=providers)
    NAMES = []
    with open(args.names, "r") as f:
        NAMES = [cname.strip() for cname in f.readlines()]
    COLORS = [[random.randint(0, 255) for _ in range(3)] for _ in NAMES]

    source_file = args.source    
    # Load Source
    image_type, key, frame, cap = loadSource(source_file)
    grabbed = True

    while(1):
        if not image_type:
            (grabbed, frame) = cap.read()

        if not grabbed:
            exit()

        image = frame.copy()
        image_, outputs, ratio, dwdh = letterbox(image, auto=False)
        ori_images = [frame.copy()]

        for batch_id, x0, y0, x1, y1, cls_id, score in outputs:
            image = ori_images[int(batch_id)]

            # Bounding Box
            box = np.array([x0, y0, x1, y1])
            box -= np.array(dwdh * 2)
            box /= ratio
            box = box.round().astype(np.int32).tolist()
            # Class id
            cls_id = int(cls_id)
            # Confidence Score
            score = round(float(score),3)
            # Class name
            name = NAMES[cls_id]    
            name += f' {str(score)}'

            if(score > args.tresh):
                # Draw Bounding box
                cv2.rectangle(image, box[:2], box[2:], COLORS[cls_id], 2)
                # Draw class name and confidence score
                cv2.putText(image, name, (box[0], box[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS[cls_id], thickness=2)

        grabbed = False
        cv2.imshow("Detected",image)
        if cv2.waitKey(key) ==  ord('q'):
            break        