from super_gradients.training.transforms import KeypointsBrightnessContrast, KeypointsHSV, KeypointsImageStandardize, KeypointsRemoveSmallObjects
from torch.utils.data import DataLoader
from tqdm import tqdm

from yolo_head import DAD3DHeadsDataset, MeshRandomAffineTransform, MeshRandomRotate90, MeshLongestMaxSize, \
    MeshPadIfNeeded, VGGHeadCollateFN


def main():
    dataset = DAD3DHeadsDataset(
        data_dir="VGGHead/large",
        num_joints=1,
        mode=None,
        transforms=[
            KeypointsBrightnessContrast(brightness_range=(0.8, 1.2), contrast_range=(0.8, 1.2), prob=0.5),
            KeypointsHSV(prob=0.5, hgain=20, sgain=20, vgain=20),
            MeshRandomAffineTransform(
                max_rotation=45,
                min_scale=0.5,
                max_scale=2.0,
                max_translate=0.1,
                image_pad_value=127,
                mask_pad_value=1,
                prob=0.35,
                interpolation_mode=[0, 1, 2, 3, 4],
            ),
            MeshRandomRotate90(prob=0.2),
            MeshLongestMaxSize(max_height=640, max_width=640),
            MeshPadIfNeeded(min_height=640, min_width=640, image_pad_value=(127, 127, 127), padding_mode="center"),
            KeypointsImageStandardize(max_value=255),
            KeypointsRemoveSmallObjects(min_instance_area=1, min_visible_keypoints=1),
        ],
        crop_bbox_to_visible_keypoints=True,
        splits=[
            "split_00000",
            "split_00001",
            "split_00002",
            "split_00003",
            "split_00004",
            "split_00005",
            "split_00006",
            "split_00007",
            "split_00008",
            "split_00009",
            "split_00010",
            "split_00011",
            "split_00012",
            "split_00013",
            "split_00014",
            "split_00015",
            "split_00016",
            "split_00017",
            "split_00018",
            "split_00019",
            "split_00020",
            "split_00021",
            "split_00022",
            "split_00023",
            "split_00024",
            "split_00025",
            "split_00026",
            "split_00027",
            "split_00028",
            "split_00029",
            "split_00030",
            "split_00031",
            "split_00032",
            "split_00033",
            "split_00034",
            "split_00035",
            "split_00036",
            "split_00037",
            "split_00038",
            "split_00039",
            "split_00040",
            "split_00041",
            "split_00042",
            "split_00043",
            "split_00044",
            "split_00045",
            "split_00046",
            "split_00047",
            "split_00048",
            "split_00049",
            "split_00050",
            "split_00051",
            "split_00052",
            "split_00053",
            "split_00054",
            "split_00055",
            "split_00056",
            "split_00057",
            "split_00058",
            "split_00059",
            "split_00060",
            "split_00061",
            "split_00062",
            "split_00063",
            "split_00064",
            "split_00065",
            "split_00066",
            "split_00067",
            "split_00068",
            "split_00069",
            "split_00070",
            "split_00071",
            "split_00072",
            "split_00073",
            "split_00074",
            "split_00075",
            "split_00076",
            "split_00077",
            "split_00078",
            "split_00079",
            "split_00080",
            "split_00081",
            "split_00082",
            "split_00083",
            "split_00084",
            "split_00085",
            "split_00086",
            "split_00087",
            "split_00088",
            "split_00089",
            "split_00090",
            "split_00091",
            "split_00092",
            "split_00093",
            "split_00094",
            "split_00095",
            "split_00096",
            "split_00097",
            "split_00098",
            "split_00099",
        ],
    )


    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=16, collate_fn=VGGHeadCollateFN())
    for batch in tqdm(loader):
        images, targets, extras = batch
        gt_samples = extras["gt_samples"]
        for sample in gt_samples:
            bboxes_xywh = sample.bboxes_xywh
            areas = bboxes_xywh[..., 2:4].prod(axis=-1)
            if (areas < 1).any():
                print("Area is less than 1")
                print("Sample index", sample.sample_index)
                print("Bboxes", bboxes_xywh)

if __name__ == "__main__":
    main()

