import pandas as pd
import os
from PIL import Image
from rock_classifier import BasicClassifier

# All images have same dimensions (720, 480)
IMG_WIDTH = 720
IMG_HEIGHT = 480


# RCNN: convert to x_min, y_min, x_max, y_max
def convert_to_rcnn(dataset_path: str, df: pd.DataFrame):
    new_df = pd.DataFrame()
    
    new_df['image'] = 'render' + df['Frame'].astype(int).astype(str).str.zfill(4) + '.png' # Image Name: 1 --> 00001
    new_df['x_min'] = df['TopLeftCornerX'].round(0).astype(int)
    new_df['y_min'] = df['TopLeftCornerY'].round(0).astype(int)
    new_df['x_max'] = (df['TopLeftCornerX'] + df['Length']).round(0).astype(int)
    new_df['y_max'] = (df['TopLeftCornerY'] + df['Height']).round(0).astype(int)
    
    classifier = BasicClassifier(df)
    new_df['class_id'] = classifier.classify_df()
    
    return new_df



# YOLO: convert to <class> <x center> <y center> <width> <length>
def convert_to_yolo(dataset_path: str, df: pd.DataFrame):
    new_df = pd.DataFrame()

    new_df['image'] = df['Frame'].astype(str).str.zfill(5)
    # convert and normalize dimensions
    new_df['x_center'] = (df['TopLeftCornerX'] + df['Length'] / 2.0) / IMG_WIDTH
    new_df['y_center'] = (df['TopLeftCornerY'] + df['Height'] / 2.0) / IMG_HEIGHT
    new_df['width'] = df['Length'] / IMG_WIDTH
    new_df['height'] = df['Height'] / IMG_HEIGHT

    classifier = BasicClassifier(df)
    new_df['class_id'] = classifier.classify_df()
    
    return new_df

    
# remove faulty images from dataset (October Kaggle update)
def remove_errors(dataset_path: str, df: pd.DataFrame):
    faulty_path = os.path.join(dataset_path, 'faulty_images')
    txt_files = [i for i in os.listdir(faulty_path) if i.endswith('.txt')]
    faulty_images = []
    
    for txt in txt_files:
        with open(os.path.join(faulty_path, txt)) as file:
            for line in file:
                faulty_images.append(f"render{line.strip()}.png")
                
    # print(len(faulty_images)) # 773 faulty images exist
    img_path = os.path.join(dataset_path, 'images/render')
    
    # removes faulty images from render folder
    def delete_images(img_path: str, faulty_images: list):
        data_images = [img for img in os.listdir(img_path) if img.endswith('.png')]
        for img in data_images:
            if img in faulty_images:
                os.remove(os.path.join(img_path, img))

    # removes rows of faulty images from dataframe
    def delete_boxes(new_df: pd.DataFrame, faulty_images: list):
        filtered_df = new_df[~new_df['image'].isin(faulty_images)]
        return filtered_df
    
    delete_images(img_path, faulty_images)
    filtered_df = delete_boxes(df, faulty_images)
    return filtered_df


# create YOLO-formatted label files: one file per image, one line per object
def create_txts(new_df: pd.DataFrame, dataset_path: str):
    labels_path = os.path.join(dataset_path, 'images/yolo_labels')
    images_path = os.path.join(dataset_path, 'images/render')
    os.makedirs(labels_path, exist_ok=True)


    # gather all images
    all_images = sorted([f for f in os.listdir(images_path) if f.endswith('.png')])
    
    image_ids = [
        int(os.path.splitext(f.replace("render", ""))[0])
        for f in all_images
    ]
    # Group annotations by image
    grouped = new_df.groupby("image")

    for img_id in image_ids:
        label_path = os.path.join(labels_path, f"render{img_id:04}.txt")

        if img_id in grouped.groups:
            # There ARE annotations for this image
            lines = []
            group = grouped.get_group(img_id)

            for _, row in group.iterrows():
                cls = int(row["class_id"])
                line = f"{cls} {row['x_center']:.6f} {row['y_center']:.6f} {row['width']:.6f} {row['height']:.6f}"
                lines.append(line)

            with open(label_path, "w") as f:
                f.write("\n".join(lines))

        else:
            # No annotations --> create an empty txt file
            open(label_path, "w").close()

# create yolo yaml
def create_yaml(dataset_path: str, class_names: list[str]):
    yaml_path = os.path.join(dataset_path, 'yolo_dataset/data.yaml')
    
    with open(yaml_path, "w") as file:
        file.write(f"path: {dataset_path}\n")
        file.write(f"train: images/train\n")
        file.write(f"val: images/val\n")
        file.write(f"test: images/test\n\n")
        
        file.write(f"names:\n")
        for i, name in enumerate(class_names):
            file.write(f"   {i}: {name}\n")
        file.close()



# ====================MAIN==================== 
if __name__ == "__main__":
    dataset_path = os.path.join(os.getcwd(), 'dataset/')
    
    
    # create_yaml(dataset_path, ["small_rock", "medium_rock", "large_rock"] )

    df_path = os.path.join(dataset_path, 'bounding_boxes.csv')
    df = pd.read_csv(df_path)
    
    new_df = convert_to_rcnn(dataset_path, df)
    
    new_df = remove_errors(dataset_path, new_df)
    
    new_df_path = os.path.join(dataset_path, 'rcnn_bounding_boxes_final.csv')
    new_df.to_csv(new_df_path, index=False)
    
    # create_txts(new_df, dataset_path)