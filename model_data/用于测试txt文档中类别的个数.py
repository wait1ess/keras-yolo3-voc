# -*- coding: utf-8 -*-

classes_path = 'coco_classes.txt'
def get_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names
 
class_names = get_classes(classes_path)
print(class_names)
num_classes = len(class_names)
print(num_classes)
