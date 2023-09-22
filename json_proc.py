import json
import os
import glob
import tqdm

if __name__ == "__main__":
    json_file_list = list(glob.glob('/home/wangjunjie/mmrotate/data/*.json'))
    for json_file_path in tqdm.tqdm(json_file_list):
        with open(json_file_path, 'r') as f:
            infos = json.load(f)
        _, file_name = os.path.split(json_file_path)
        infos["imagePath"] = file_name.replace('json', 'tif')
        with open(json_file_path, 'w') as f:
            json.dump(infos, f)
    
