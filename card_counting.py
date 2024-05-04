import cv2
import numpy as np
import pyautogui
from inference import get_model
import supervision as sv
import time


# Initialize the model
model = get_model(model_id="karts-vc5wx/1",api_key="YOUR_API_KEY_ROBOWFLOW")

# Create supervision annotators
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()


# Define the region of the screen to capture (x, y, width, height)
region = (0, 550, 1920, 530)


def is_new_card(card_index,old_detection_list,new_detection_list):
    new_x1,new_y1= new_detection_list[card_index]
    for elem in old_detection_list:
        old_x1,old_y1 = elem
        if (new_x1-old_x1)**2 + (new_y1-old_y1)**2 < 15:
            return False
    
    return True

def hlo2(value):
    match value:
            case 2:
                return 1
            case 3:
                return 1
            case 4:
                return 2
            case 5:
                return 2
            case 6:
                return 1
            case 7:
                return 1
            case 8:
                return 0
            case 9:
                return 0
            case 10:
                return -2
            case 11:
                return -2
            case 12:
                return -2
            case 13:
                return -2
            case 1:
                return 0 


old_detections_position=[]
score=0
nb_cartes=0
while True:

    detections_position=[]
    # Capture a screenshot of the defined region
    screenshot = pyautogui.screenshot(region=region)

    # Convert the screenshot to a numpy array
    img = np.array(screenshot)

    # Convert the numpy array to a picture for cv2
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Run inference on the screenshot
    results = model.infer(img)

    # Load the results into the supervision Detections api
    detections = sv.Detections.from_inference(results[0].dict(by_alias=True, exclude_none=True))
    
    
    for elem in detections.xyxy:
        detections_position.append((elem[0], elem[1], elem[2], elem[3]))
    
    

    # Annotate the image with our inference results
    
    
    points_detections=[np.array([((x2-x1)/2)+x1,y1+((y2-y1)/2)]) for x1,y1,x2,y2 in detections_position]
    #print(points_detections)
    # Add the label to the list
    color1 = (0, 0, 255)
    color2 = (0, 255, 0)
    if len(points_detections)>len(old_detections_position):
        for _ in range(len(points_detections)-len(old_detections_position)):
            old_detections_position.append((0,0))
    
   
    annotated_image = bounding_box_annotator.annotate(scene=img, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
    
    for i in range(len(points_detections)):
    #permet de dessiner au milieu de la carte -> debug
        coordinates = (int(points_detections[i][0]), int(points_detections[i][1]))
        coordinates2 = (int(old_detections_position[i][0]), int(old_detections_position[i][1]))
        cv2.circle(annotated_image, coordinates, 3, color1, -1)
        cv2.circle(annotated_image, coordinates2, 5, color2, -1)
    
    class_int=[(''.join(c for c in class_name if c.isdigit()), np.where(np.array(detections.data['class_name']) == class_name)[0][0]) for class_name in detections.data['class_name']]
    class_int = [c for c in class_int if c[0] != '']
    #print(class_int) #11 = joker 11 q =12 k = 13 1=as
    
    if np.array_equal(detections.data['class_name'], np.array(['gg'])) or np.array_equal(detections.data['class_name'], np.array([])):
        print(f'[Etat]: Distribution')
        old_detections_position=[]
    
    #print(f'[Cartes detectées]: {class_int}')
    new_card_list=[]
    for card in class_int:
        if is_new_card(card[1],old_detections_position,points_detections):
            new_card_list.append(card)
    
    
    print(f'[Nouvelles Cartes detectées]: {[int(card[0]) for card in new_card_list]}')
    for new_card in new_card_list:
        value=int(new_card[0])
        score+=hlo2(value)
        nb_cartes+=1

    if nb_cartes<100:  
        print(f'[Score][LOW PRECISION]: {score}')
        if score>=2 and score<5:
            print(f'[BET][LOW PRECISION]: Bet LOW')
        if score>=5 and score<7:
            print(f'[BET][LOW PRECISION]: Bet MEDIUM')
        if score>=7 :
            print(f'[BET][LOW PRECISION]: Bet HIGH')
    else:
        print(f'[Score][HIGH PRECISION]: {score}')
        if score>=2 and score<5:
            print(f'[BET][HIGH PRECISION]: Bet LOW')
        if score>=5 and score<7:
            print(f'[BET][HIGH PRECISION]: Bet MEDIUM')
        if score>=7 :
            print(f'[BET][HIGH PRECISION]: Bet HIGH')
    
    print('\n')
    points_detections_list=[(x,y) for x,y in points_detections]
    set_points_detections=set(points_detections_list) #va permettre de garder en memoire toutes les cartes de la table deja joue + le reset quand distribue
    set_old_detections_position=set(old_detections_position)
    union_set = set_old_detections_position.union(set_points_detections)
    
    old_detections_position = list(union_set)
    
    
    # Display the image
    #sv.plot_image(annotated_image)
    
    time.sleep(1)


