import json
import pdb
import numpy as np
import matplotlib.pyplot as plt
import random
import mmdet.datasets.coco as coco
import mmdet.datasets.objects365 as obj
import matplotlib.patches as patches
from PIL import Image
from pycocotools.coco import COCO
import sys

COCO_TO_OBJECTS365 = {  'person':'Person',
                        'bicycle':'Bicycle',
                        'car':['Car', 'SUV','Sports Car','Formula 1 '],
                        'motorcycle':'Motorcycle',
                        'airplane':'Airplane',
                        'bus':'Bus',
                        'train':'Train',
                        'truck':['Truck','Pickup Truck', 'Fire Truck', 'Ambulance', 'Heavy Truck'], #
                        'boat':['Boat', 'Sailboat', 'Ship'], #
                        'traffic light':'Traffic Light',
                        'fire hydrant':'Fire Hydrant',
                        'stop sign':'Stop Sign',
                        'parking meter':'Parking meter',
                        'bench':'Bench',
                        'bird':['Wild Bird', 'Duck', 'Goose', 'Parrot', 'Chicken'], #
                        'cat':'Cat',
                        'dog':'Dog',
                        'horse':'Horse',
                        'sheep':'Sheep',
                        'cow':'Cow',
                        'elephant':'Elephant',
                        'bear':'Bear',
                        'zebra':'Zebra',
                        'giraffe':'Giraffe',
                        'backpack':'Backpack',
                        'umbrella':'Umbrella',
                        'handbag':'Handbag/Satchel',
                        'tie':['Tie', 'Bow Tie'],
                        'suitcase': 'Luggage', #
                        'frisbee':'Frisbee',
                        'skis':'Skiboard', #
                        'snowboard':'Snowboard',
                        'sports ball':['Baseball', 'Soccer', 'Basketball', 'Billards', 'American Football','Volleyball','Golf Ball','Table Tennis ','Tennis'], #
                        'kite':'Kite',
                        'baseball bat':'Baseball Bat',
                        'baseball glove':'Baseball Glove',
                        'skateboard':'Skateboard',
                        'surfboard':'Surfboard',
                        'tennis racket':'Tennis Racket',
                        'bottle':'Bottle',
                        'wine glass':'Wine Glass',
                        'cup':'Cup',
                        'fork':'Fork',
                        'knife':'Knife',
                        'spoon':'Spoon',
                        'bowl':'Bowl/Basin',
                        'banana':'Banana',
                        'apple':'Apple',
                        'sandwich':'Sandwich',
                        'orange':'Orange/Tangerine',
                        'broccoli':'Broccoli',
                        'carrot':'Carrot',
                        'hot dog':'Hot dog',
                        'pizza':'Pizza',
                        'donut':'Donut',
                        'cake':'Cake',
                        'chair':['Chair', 'Wheelchair'],
                        'couch':'Couch',
                        'potted plant':'Potted Plant',
                        'bed':'Bed',
                        'dining table':'Dinning Table',
                        'toilet':['Toilet', 'Urinal'],
                        'tv':'Moniter/TV',
                        'laptop':'Laptop',
                        'mouse':'Mouse',
                        'remote':'Remote',
                        'keyboard':'Keyboard',
                        'cell phone':'Cell Phone',
                        'microwave':'Microwave',
                        'oven':'Oven',
                        'toaster':'Toaster',
                        'sink':'Sink',
                        'refrigerator':'Refrigerator',
                        'book':'Book',
                        'clock':'Clock',
                        'vase':'Vase',
                        'scissors':'Scissors',
                        'teddy bear' : 'Stuffed Toy', 
                        'hair drier':'Hair Dryer',
                        'toothbrush':'Toothbrush'}


OOD_CLASSES  =  {'Sneakers', 'Other Shoes', 'Hat',
               'Lamp', 'Glasses', 'Street Lights',
               'Cabinet/shelf', 'Bracelet', 
               'Picture/Frame', 'Helmet', 'Gloves', 'Storage box',
               'Leather Shoes', 'Flag', 'Pillow', 'Boots', 'Microphone',
               'Necklace', 'Ring', 'Belt',
               'Speaker', 'Trash bin Can', 'Slippers',
               'Barrel/bucket', 'Sandals', 'Bakset', 'Drum',
               'Pen/Pencil', 'High Heels',
               'Guitar', 'Carpet', 'Bread', 'Camera', 'Canned',
               'Traffic cone', 'Cymbal', 'Lifesaver', 'Towel',
               'Candle', 'Awning',
               'Faucet', 'Tent', 'Mirror', 'Power outlet',
               'Air Conditioner', 'Hockey Stick', 'Paddle',
               'Ballon', 'Tripod',
               'Hanger', 'Blackboard/Whiteboard', 'Napkin',
               'Other Fish', 'Toiletry', 'Tomato', 'Lantern', 'Fan',
               'Pumpkin', 'Tea pot', 'Head Phone',
               'Scooter', 'Stroller', 'Crane', 'Lemon', 
               'Surveillance Camera', 'Jug',
               'Piano', 'Gun', 'Skating and Skiing shoes', 'Gas stove', 
               'Strawberry', 'Other Balls',
               'Shovel', 'Pepper', 'Computer Box', 'Toilet Paper',
               'Cleaning Products', 'Chopsticks', 'Pigeon',
               'Cutting/chopping Board', 'Marker', 'Ladder',
               'Radiator', 'Grape', 'Potato', 'Sausage',
               'Violin', 'Egg', 'Fire Extinguisher', 'Candy',
               'Converter', 'Bathtub', 
               'Golf Club', 'Cucumber', 'Cigar/Cigarette ',
               'Paint Brush', 'Pear', 'Hamburger',
               'Extention Cord', 'Tong', 'Folder',
               'earphone', 'Mask', 'Kettle', 
               'Swing', 'Coffee Machine', 'Slide', 'Onion',
               'Green beans', 'Projector', 
               'Washing Machine/Drying Machine', 'Printer',
               'Watermelon', 'Saxophone', 'Tissue', 'Ice cream',
               'Hotair ballon', 'Cello', 'French Fries', 'Scale', 'Trophy',
               'Cabbage', 'Blender', 'Peach', 'Rice',
               'Deer', 'Tape', 'Cosmetics', 'Trumpet', 'Pineapple', 
               'Mango', 'Key', 'Hurdle', 'Fishing Rod',
               'Medal', 'Flute', 'Brush', 'Penguin', 'Megaphone', 'Corn',
               'Lettuce', 'Garlic', 'Swan', 'Helicopter', 'Green Onion',
               'Nuts', 'Induction Cooker',
               'Broom', 'Trombone', 'Plum', 'Goldfish',
               'Kiwi fruit', 'Router/modem', 'Poker Card', 'Shrimp',
               'Sushi', 'Cheese', 'Notepaper', 'Cherry', 'Pliers', 'CD',
               'Pasta', 'Hammer', 'Cue', 'Avocado', 'Hamimelon', 
               'Mushroon', 'Screwdriver', 'Soap', 'Recorder', 
               'Eggplant', 'Board Eraser', 'Coconut', 'Tape Measur/ Ruler',
               'Pig', 'Showerhead', 'Globe', 'Chips', 'Steak',
               'Stapler', 'Campel', 'Pomegranate', 'Dishwasher', 'Crab', 
               'Meat ball', 'Rice Cooker', 'Tuba', 'Calculator', 'Papaya',
               'Antelope', 'Seal', 'Buttefly', 'Dumbbell', 'Donkey',
               'Lion', 'Dolphin', 'Electric Drill',
               'Jellyfish', 'Treadmill', 'Lighter', 'Grapefruit',
               'Game board', 'Mop', 'Radish', 'Baozi', 'Target', 'French',
               'Spring Rolls', 'Monkey', 'Rabbit', 'Pencil Case', 'Yak',
               'Red Cabbage', 'Binoculars', 'Asparagus', 'Barbell', 'Scallop',
               'Noddles', 'Comb', 'Dumpling', 'Oyster', 'Green Vegetables',
               'Cosmetics Brush/Eyeliner Pencil', 'Chainsaw', 'Eraser',
               'Lobster', 'Durian', 'Okra', 'Lipstick', 'Trolley',
               'Cosmetics Mirror', 'Curling', 'Hoverboard', 'Plate', 'Pot', 'Extractor',
               'Table Teniis paddle' # Checked aorund 1000 images in COCO, tennis racket includes court tennis or badminton rackets
               }


AMBIGUOUS_CLASSES = {'Nightstand', 'Desk', 'Coffee Table', 'Side Table', # Confusion with dining table, OOD
                    'Watch', # Confusion with clock, OOD 
                    'Stool', # Confision with chair, desk, OOD
                    'Machinery Vehicle', # Confusion with truck, OOD
                    'Tricycle','Carriage', 'Rickshaw', # Confusion with bicycle, OOD, also AV datasets may have such classes
                    'Van', # Confusion with car, truck
                    'Traffic Sign','Speed Limit Sign', 'Crosswalk Sign', # AV datasets may have such labels so not include those
                    'Flower',  # Confusion with potted plant, OOD
                    'Telephone',  # Confusion with cell phone, OOD
                    'Tablet', # Confusion with TV(which includes monitors), OOD
                    'Flask', # Confusion with bottle, OOD
                    'Briefcase', # Confusion with suitcase, OOD
                    'Egg tart', 'Pie', 'Dessert', 'Cookies', # Confusion with cake, OOD
                    'Wallet/Purse' # Confusion with handbag, OOD
                    }

def reverse_dict(COCO_TO_OBJECTS365):
    OBJECTS365_TO_COCO = {}
    for k, v in COCO_TO_OBJECTS365.items():
        if type(v) is list:
            for v_ in v:
                OBJECTS365_TO_COCO[v_] = k
        else:
            OBJECTS365_TO_COCO[v] = k
    return OBJECTS365_TO_COCO

def coco2robustod(data_path):

  obj_365 = obj.Objects365Dataset(data_path, [], test_mode=True)

  ## COCO is also required to match classes
  coco_path = 'data/coco/annotations/instances_val2017.json'
  coco_val = COCO(coco_path)

  # 2. Initialize data structures
  cov_shift_data = {'info': 'Natural Covariate Shift Split', 'licenses': obj_365.coco.dataset['licenses'], 'images': list(), 'annotations': list(), 'categories': coco_val.dataset['categories']}
  ood_data = {'info': 'OOD-general Split', 'licenses': obj_365.coco.dataset['licenses'], 'images': list(), 'annotations': list(), 'categories': coco_val.dataset['categories'], 'orig_categories': obj_365.coco.dataset['categories']}

  # 3. Create Natural Covariate Shift and OOD splits
  # Natural Covariate Shift: Include an image only if it has the same classes with COCO
  # OOD Split: Include an image if it has no classes with COCO
  OBJECTS365_ID_TO_NAME = {}
  for cat in obj_365.coco.dataset['categories']:
      OBJECTS365_ID_TO_NAME[cat['id']]= cat['name']

  OBJECTS365_TO_COCO = reverse_dict(COCO_TO_OBJECTS365)

  COCO_NAME_TO_ID = {}
  for cat in coco_val.dataset['categories']:
      COCO_NAME_TO_ID[cat['name']]= cat['id']

  nat_cov_sh_counter = 0
  ood_gen_counter = 0
  for i in range(len(obj_365.coco.dataset['images'])):
      # [{'height': 512, 'id': 420917, 'license': 5, 'width': 769, 'file_name': 'images/v1/patch8/objects365_v1_00420917.jpg', 'url': '', 'filename': 'images/v1/patch8/objects365_v1_00420917.jpg'}]
      img_id =  obj_365.coco.dataset['images'][i]['id']
      ann_ids = obj_365.coco.get_ann_ids(img_ids=[img_id])
      anns = obj_365.coco.load_anns(ann_ids)
      # Ex. ann: {'id': 51, 'iscrowd': 0, 'isfake': 0, 'area': 1584.6324365356109, 'isreflected': 0, 'bbox': [491.3955078011, 88.1856689664, 35.65588379410002, 44.442382796800004], 'image_id': 420917, 'category_id': 84}
      
      num_id, num_ood, num_amb = 0, 0, 0
      for ann in anns:
          ann_cat_name = OBJECTS365_ID_TO_NAME[ann['category_id']]
          if ann_cat_name in OBJECTS365_TO_COCO.keys():
              num_id += 1
          elif ann_cat_name in OOD_CLASSES:
              num_ood += 1
          elif ann_cat_name in AMBIGUOUS_CLASSES:
              num_amb += 1

      if num_amb > 0:
          continue

      if num_ood == len(anns):
          # Add to OOD-general
          ood_data['images'].append(obj_365.coco.dataset['images'][i])
          for ann in anns:
              ood_data['annotations'].append(ann)
          ood_gen_counter += 1

      elif num_id > 0:
          # Add to Natural Covariate Shift
          nat_cov_sh_counter += 1
          cov_shift_data['images'].append(obj_365.coco.dataset['images'][i])
          for ann in anns:
              ann_cat_name = OBJECTS365_ID_TO_NAME[ann['category_id']]
              if ann_cat_name in OBJECTS365_TO_COCO.keys():
                  coco_ann_cat_name = OBJECTS365_TO_COCO[ann_cat_name]
                  ann['category_id'] = COCO_NAME_TO_ID[coco_ann_cat_name]
                  cov_shift_data['annotations'].append(ann)

  
  if data_path[-8:-5] == 'val':
    out_file_paths = [data_path[:-5] + '_robust_od.json', data_path[:-5] + '_robust_od_ood.json']
    with open(out_file_paths[0], 'w') as outfile:
        json.dump(cov_shift_data, outfile)

    with open(out_file_paths[1], 'w') as outfile:
        json.dump(ood_data, outfile)
  else:
    out_file_paths = [data_path[:-5] + '_robust_od_ood.json']
    with open(out_file_paths[0], 'w') as outfile:
        json.dump(ood_data, outfile)

  return out_file_paths


def convert():
  obj365_paths = ['data/objects365/annotations/zhiyuan_objv2_train.json',
                  'data/objects365/annotations/zhiyuan_objv2_val.json']

  out_file_paths = []

  for path in obj365_paths:
    out_file_paths.extend(coco2robustod(path))

  return out_file_paths