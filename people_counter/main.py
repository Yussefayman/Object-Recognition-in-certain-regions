import cv2
import numpy as np

def frame_name_correction(frame_name):
    num_zeros = 0
    zeros = ""
    frame_num_str = frame_name[6:]
    nxt_frame_num = int(frame_num_str)+1
    if nxt_frame_num <=9:
        num_zeros = 3
    elif nxt_frame_num <=99:
        num_zeros = 2
    else:
        num_zeros = 1
    while num_zeros!=0:
        zeros+='0'
        num_zeros-=1
    return "frame_"+zeros+str(nxt_frame_num)

def add_ROIS_frame(frame,ROIs):
    frames = []
    for i in range(len(ROIS)):
        rect = cv2.rectangle(frame, ROIs[i][0], ROIs[i][1], (0, 0, 255), 5)
        frames.append(frame[ROIs[i][0][1]:ROIs[i][1][1], ROIs[i][0][0]:ROIs[i][1][0]])
    return frames


img_path = "E:\\A-eye-tech-task\\Crowd_PETS09\\S1\\L1\\Time_13-57\\View_001\\"
frame_name = "frame_0000"

ROIS = [ [(10,8),(757,566)] ,[(287,156),(711,431)], [(27,129),(230,289)] ]



pre_trained_model = 'E:\\A-eye-tech-task\\pre_trained_models\\MobileNetSSD_deploy.caffemodel'
pre_trained_config = 'E:\\A-eye-tech-task\\pre_trained_models\\MobileNetSSD_deploy.prototxt.txt'
model = cv2.dnn_DetectionModel(pre_trained_config, pre_trained_model)
model.setInputSize(770,700)
model.setInputMean((127.5,127.5,127.5))
model.setInputScale(1.0/127.5)

classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable",  "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

while True:
    path = img_path+frame_name+".jpg"
    frame = cv2.imread(path)
    frame = cv2.resize(frame, (770, 700))
    frame_num = int(frame_name[6:])
    frame_name = frame_name_correction(frame_name)

    Regions = add_ROIS_frame(frame, ROIS)
    reg_num = 1
    for reg in Regions:

        ClassIndex, confidence, bbox = model.detect(reg, confThreshold=0.5,nmsThreshold=0.5)
        #bbox = non_max_suppression_fast(bbox, overlapThresh=0.50)
        ppl_count = 0
        for i in ClassIndex:
            if i == [15]:
                ppl_count+=1

        cv2.putText(reg,"Region "+ str(reg_num)+" People: "+str(ppl_count), (10,reg.shape[0] -25), cv2.FONT_HERSHEY_TRIPLEX, 0.5,(255,255,0), 1)
        reg_num+=1
        if reg_num > 3:
            reg_num = 1
        font_scale = 3
        font = cv2.FONT_HERSHEY_PLAIN
        for class_ind, conf, box in zip(ClassIndex, confidence, bbox):
            if class_ind != [15]:
                continue
            if reg_num ==1:
                cv2.rectangle(reg, box, (255, 0, 0), 1)
            elif reg_num ==2:
                cv2.rectangle(reg, box, (0, 255, 0), 1)
            else:
                cv2.rectangle(reg, box, (0, 0, 255), 1)

    cv2.imshow("Output", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
