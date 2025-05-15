# RT-DETR-v2

RT-DETR-v2 (pytorch) with a collection of patches & fixes.

## Patches
- Support dynamic batch sizing in `dataloader.yaml` config via the `device_batch_split: []` parameter. When populated with a batch size for each gpu, each rank will be optimally populated with data which will result in better training times on mixed gpu systems.

- Fix for multi gpu training for datasets with a large number of background images.

- Per COCO class metrics logging in tensorboard

- NaN patch as suggested by: Peterande/D-FINE#199 NaN patch
