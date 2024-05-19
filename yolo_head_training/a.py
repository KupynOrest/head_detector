from yolo_head.flame import get_indices

i = get_indices()
for k, v in i.items():
    print(k, len(v), len(set(v)))