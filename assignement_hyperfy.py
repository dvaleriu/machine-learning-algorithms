import cv2
import pafy
import time
import numpy as np
from yolo_utils import show_image, draw_labels_and_boxes
from coco_names import COCO_INSTANCE_CATEGORY_NAMES as coco_names
import torchvision.transforms as transforms
import torch
import argparse
import torchvision
#import stream2
from PIL import Image


# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='path to input video')
parser.add_argument('-m', '--min-size', dest='min_size', default=800, 
                    help='minimum input size for the RetinaNet network')
parser.add_argument('-t', '--threshold', default=0.5, type=float,
                    help='minimum confidence score for detection')
args = vars(parser.parse_args())




def predict(image, model, device, detection_threshold):
    # transform the image to tensor
    image = transforms(image).to(device)

    image = image.unsqueeze(0) # add a batch dimension
    with torch.no_grad():
        outputs = model(image) # get the predictions on the image

    # get all the scores
    scores = list(outputs[0]['scores'].detach().cpu().numpy())
    # index of those scores which are above a certain threshold
    thresholded_preds_inidices = [scores.index(i) for i in scores if i > detection_threshold]

    # get all the predicted bounding boxes
    bboxes = outputs[0]['boxes'].detach().cpu().numpy()
    # get boxes above the threshold score
    boxes = bboxes[np.array(scores) >= detection_threshold].astype(np.int32)

    # get all the predicited class names
    labels = outputs[0]['labels'].cpu().numpy()
    pred_classes = [coco_names[labels[i]] for i in thresholded_preds_inidices]
    return boxes, pred_classes

def draw_boxes(boxes, classes, image):
    for i, box in enumerate(boxes):
        color = COLORS[coco_names.index(classes[i])]
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 2
        )
        cv2.putText(image, classes[i], (int(box[0]), int(box[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, 
                    lineType=cv2.LINE_AA)
    return image


if __name__ == '__main__':


    model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True, 
                                                                num_classes=91, 
                                                                min_size=args['min_size'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load the model onto the computation device
    model = model.eval().to(device)

    cap = cv2.VideoCapture("http://pendelcam.kip.uni-heidelberg.de/mjpg/video.mjpg")

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # this will help us create a different color for each class
    COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))

    vcap = cv2.VideoCapture("http://pendelcam.kip.uni-heidelberg.de/mjpg/video.mjpg")


    transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(128),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  #standardizare + normalizare 0,1 
    ])

    IMG_WIDTH = 128
    IMG_HEIGHT = 128

    confidence_threshold = 0.1
    nms_threshold = 0.2

    model_config = 'D:/faia/cod/w/YOLOv3-Object-Detection-with-OpenCV/yolov3-coco/yolov31.cfg' #scris in darknet
    model_weights = 'D:/faia/cod/w/YOLOv3-Object-Detection-with-OpenCV/yolov3-coco/yolov3.weights' 

    classes = []
    folder = 'D:/faia/cod/w/YOLOv3-Object-Detection-with-OpenCV/yolov3-coco/coco-labels' 
    with open(folder, 'r') as f:
        classes = f.read().splitlines()

    net = cv2.dnn.readNetFromDarknet(model_config, model_weights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


    while True:

        ret, frame = vcap.read()
        ret1, frame1 = cap.read()
        height, width, channel = frame.shape

        if True:
            blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),
                                            [0, 0, 0], swapRB = True, crop=False)
            pil_image = Image.fromarray(frame1).convert('RGB')

            net.setInput(blob)
            outs = net.getUnconnectedOutLayersNames()
            layerOutputs = net.forward(outs)

            bbox = []
            boxes1, classes1 = predict(pil_image, model, device, args['threshold'])
            # draw boxes and show current frame on screen
            result = draw_boxes(boxes1, classes1, frame1)

           # cv2.imshow('image', result)
     
            if cv2.waitKey(27) & 0xFF == ord('q'):
             break

            confidences = []
            class_ids = []

            for output in layerOutputs:
                for detections in output:
                    scores = detections[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    if confidence > confidence_threshold:
                        center_x = int(detections[0] * width)
                        center_y = int(detections[1] * height)
                        w = int(detections[2] * width)
                        h = int(detections[3] * height)

                        x = int(center_x - (w/2))
                        y = int(center_y - (h/2))

                        bbox.append([x,y,w,h])
                        class_ids.append(class_id)
                        confidences.append(float(confidence))

            indices = cv2.dnn.NMSBoxes(bbox, confidences, confidence_threshold, nms_threshold)
            
            colors = np.random.uniform(0, 255, size = (len(bbox), 3))

            for i in indices:
                x, y, w, h = bbox[i]
                label = str(classes[class_ids[i]])
                confidence = str(round(confidences[i], 2))
                color = colors[i]

                cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
                cv2.putText(frame, label + " " + confidence, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("incercare", frame)
            cv2.imshow('image', result)
            
    
    vcap.release()
    cv2.destroyAllWindows()

    