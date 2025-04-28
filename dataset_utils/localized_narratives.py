import os
from logging import exception
import uuid
from datasets import load_dataset, Value
from PIL import Image

def save_img_to_disk(img_obj, exten):
    try:
        img_id = f"{uuid.uuid4()}.{exten}"
        cur_save_path = f"/media/vishal/datasets/ln/samples/{img_id}"
        img_obj.save(cur_save_path)
        return cur_save_path
    except exception as e:
        return None

def remove_corrupted_images():
    small_res_id = []
    corrupted_id = []

    for sample_id, sample in enumerate(processed_ds):
        try:
            curr_img = Image.open(sample["image_path"])
            w, h = curr_img.size
            mode = curr_img.mode
            if w < 50 or h < 50 or mode not in ['RGB', 'RGBA', 'YCbCr', "P", "L", "CMYK", "LA", "1"]:
                print(f"dropped image due to {w}x{h} size or {mode} mode")
                small_res_id.append(sample_id)
            # elif w * h >  high_res_limit:
            #     print(f"dropped image due to {w}x{h} size")
            #     high_res_ids.append(sample_id)

        except:  # if image is corrupted or cannot be opened
            print(f"Corrupted image: {sample['image_path']}")
            corrupted_id.append(sample_id)

    return small_res_id, corrupted_id


def reformat_loc_narr(example):

    try:
        exten = example["images"][0].format
    except exception as e:
        exten = "jpg"
    save_path = save_img_to_disk(example["images"][0], exten)

    if save_path:
        return {**example, 'image_path': str(save_path), 'status': 'success'}
    else:
        return {**example, 'image_path': None, 'status': 'failed'}


if __name__ == "__main__":
    # os.environ['HF_DATASETS_CACHE'] = '/media/vishal/datasets/hf_cache'
    NUM_PROC = os.cpu_count()
    save_path = "/media/vishal/datasets/ln/samples/"
    ds_save_path = "/media/vishal/datasets/ln/metadata"
    dataset_path = "/media/vishal/datasets/localized_narratives/HuggingFaceM4___the_cauldron/localized_narratives/0.0.0/847a98a779b1652d65111daf20c972dfcd333605/"

    dataset = load_dataset(dataset_path, split="train", cache_dir='/media/vishal/datasets/hf_cache')
    ds_features = dataset.features.copy()
    ds_features["image_path"] = Value('string')
    ds_features["status"] = Value('string')

    processed_ds = dataset.map(reformat_loc_narr, batched=False, features=ds_features, num_proc=NUM_PROC)

    # remove samples for which saving failed
    processed_ds = processed_ds.filter(
        lambda example: example['image_path'] is not None and example['status'] == 'success')

    small_res_id, corrupted_id = remove_corrupted_images()

    all_ids_to_remove = set(corrupted_id + small_res_id)
    # Remove images from the dataset
    final_ds = processed_ds.select(
        i for i in range(len(processed_ds)) if i not in all_ids_to_remove
    )

    final_ds = final_ds.remove_columns(['status'])
    final_ds = final_ds.remove_columns(['images'])
    final_ds.save_to_disk(ds_save_path)