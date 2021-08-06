import json
import math

f = open("C:/Users/aryam/Documents/FLIR ADAS dataset/FLIR_ADAS_1_3/train/thermal_annotations.json",)

data=json.load(f)

store={}

category_id=[0]*50
bbox=[0]*50
id=0
count=0

for i in data['annotations']:
    if(i['image_id']!=id or count==50):
        n=str(id)
        store.update({n: {"image_id": id}})
        store.update({n: {"category_id": category_id}})
        store.update({n: {"bbox": bbox}})
        id=i['image_id']
        category_id=[0]*50
        bbox=[0]*50
        count=0
    else:
        category_id[count]=i['category_id']
        height = i['bbox'][2]
        width = i['bbox'][3]
        HeightChange=math.floor(0.175*height)
        WidthChange=math.floor(0.175*width)
        Ypos=i['bbox'][0]
        Xpos=i['bbox'][1]
        Ypos=(Ypos-HeightChange)/640
        if Ypos<0:
            Ypos=0
        Xpos=(Xpos-WidthChange)/960
        if Xpos<0:
            Xpos=0
        height=(HeightChange+2*HeightChange)/640
        if height>640:
            height=1
        width=(width+2*WidthChange)/960
        if width>960:
            width=1
        bbox[count]=[Ypos, Xpos, height, width]
        count+=1
