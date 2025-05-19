import os
import math
import xml.etree.ElementTree as ET

def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    objects = []
    working_zones = []

    for obj in root.findall('object'):
        label = obj.find('name').text.strip()
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)

        cx = (xmin + xmax) // 2
        cy = (ymin + ymax) // 2

        box_data = {
            'label': label,
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax,
            'center': (cx, cy)
        }

        if label.lower() == 'people working zone':
            working_zones.append(box_data)
        else:
            objects.append(box_data)

    return objects, working_zones

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def map_zones_to_objects(objects, zones):
    mapping = {obj['label']: [] for obj in objects}

    for zone in zones:
        closest_obj = None
        min_dist = float('inf')
        for obj in objects:
            dist = euclidean_distance(zone['center'], obj['center'])
            if dist < min_dist:
                min_dist = dist
                closest_obj = obj

        if closest_obj:
            mapping[closest_obj['label']].append(zone)

    return mapping

def save_mapping(mapping, output_file, timestamp):
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(f"Timestamp: {timestamp:02d} sec\n")
        for obj_label, zones in mapping.items():
            f.write(f"{obj_label} →\n")
            for i, z in enumerate(zones, 1):
                f.write(f"  Zone {i}: Center = {z['center']}, Box = ({z['xmin']}, {z['ymin']}) to ({z['xmax']}, {z['ymax']})\n")
            f.write("\n")

if __name__ == "__main__":
    xml_filename = input("Enter annotated XML file name (e.g. frame_0.xml): ").strip()
    frames_dir = "frames"
    output_txt = "video_zone_mapping.txt"

    # Parse annotated XML once
    objects, working_zones = parse_xml(xml_filename)

    # Remove old output file if exists
    if os.path.exists(output_txt):
        os.remove(output_txt)

    # Loop through frames in 'frames' folder and map zones to objects for each timestamp
    frame_files = sorted(os.listdir(frames_dir))
    for frame_file in frame_files:
        if frame_file.endswith(".jpg") and frame_file.startswith("frame_"):
            timestamp_str = frame_file.split('_')[1].split('.')[0]
            timestamp = int(timestamp_str)

            # Mapping is always same because working zones and objects don't change
            mapping = map_zones_to_objects(objects, working_zones)

            save_mapping(mapping, output_txt, timestamp)

    print(f" All mappings saved in: {output_txt}")
