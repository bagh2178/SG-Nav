import csv
import gzip
import json
import copy

# GLIP prompt
categories = [] # categories except doors 
categories_40 = []
categories_map = {}
categories_doors = []
with open('matterport_category_mappings.tsv') as file:
    tsv_file = csv.reader(file, delimiter="\t")
    for i, line in enumerate(tsv_file):
        line_ = [item for item in line[0].split('   ') if not item =='']
        if i == 0 or len(line_) < 4:
            continue
        if int(line_[3]) > 10:
            if 'door' in line_[-1] and line_[2] not in categories_doors:
                categories_doors.append(line_[2])
            else:
                categories.append(line_[2])
                categories_map[line_[2]] = line_[-1]
        if line_[-1] not in categories_40 and line_[-1] is not 'objects' and 'void' not in line_[-1]:
            categories_40.append(line_[-1])


categories_21 = ['chair', 'table', 'picture', 'cabinet', 'cushion', 'sofa',
'bed', 'chest_of_drawers', 'plant', 'sink', 'toilet', 'stool',
'towel', 'tv_monitor', 'shower', 'bathtub', 'counter', 'fireplace', 'gym_equipment', 'seating', 'clothes']

categories_21_origin = copy.deepcopy(categories_21)


categories_21.append('heater')
categories_21.append('window')
categories_21.append('treadmill')
categories_21.append('exercise machine')
object_captions = '. '.join(categories_21) +'.'# version 1
rooms = ['bedroom', 'living room', 'bathroom', 'kitchen', 'dining room', 'office room', 'gym', 'lounge', 'laundry room']
rooms_captions = '. '.join(rooms)+'.'
door_captions = 'doorway. hallway.'# v2
# object_captions = '. '.join(categories_21)+'.' # + '. wall. door.'

# pre_defined_captions = rooms + pre_defined_captions


# LLM reasoning prompt
# room_prompt = "In which room will you most likely to find a "

with gzip.open("habitat-challenge-data/data/val/val.json.gz", 'r') as fin:        # 4. gzip
    json_bytes = fin.read()                      # 3. bytes (i.e. UTF-8)

json_str = json_bytes.decode('utf-8')            # 2. string (i.e. JSON)
data = json.loads(json_str)

projection_reverse = data['category_to_task_category_id']
projection = {}
for key, item in projection_reverse.items():
    projection[item] = key

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    bb1 = {'x1': bb1[0], 'x2': bb1[2], 'y1': bb1[1], 'y2': bb1[3]}
    bb2 = {'x1': bb2[0], 'x2': bb2[2], 'y1': bb2[1], 'y2': bb2[3]}
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou
