# MPLT for RGB-T Tracking

Implementation of the paper

## Environment Installation
```
conda create -n mplt python=3.8
conda activate mplt
bash install.sh
```

## Project Paths Setup
Run the following command to set paths for this project
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```
After running this command, you can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

## Data Preparation
Put the tracking datasets in `./data`. It should look like:
```
${PROJECT_ROOT}
  -- data
      -- lasher
          |-- trainingset
          |-- testingset
          |-- trainingsetList.txt
          |-- testingsetList.txt
          ...
```

## Training
Download [SOT](https://) pretrained weights and put them under `$PROJECT_ROOT$/pretrained_models`.

```
python tracking/train.py --script mplt_track --config vitb_256_mplt_32x4_4e4_lasher_15ep_in1k --save_dir ./output/vitb_256_mplt_32x4_4e4_lasher_15ep_in1k --mode multiple --nproc_per_node 4
```

Replace `--config` with the desired model config under `experiments/mplt_track`.

## Evaluation
Put the checkpoint into `$PROJECT_ROOT$/output/config_name/...` or modify the checkpoint path in testing code.

```
python tracking/test.py mplt_track vitb_256_mplt_32x4_4e4_lasher_15ep_in1k --dataset_name lasher_test --threads 6 --num_gpus 1

python tracking/analysis_results.py --tracker_name mplt_track --tracker_param vitb_256_mplt_32x4_4e4_lasher_15ep_in1k --dataset_name lasher_test
```

### Results on LasHeR testing set

Model | Backbone | Pretraining | Precision | Success | FPS |       Checkpoint      | Raw Result

MPLT  | ViT-Base |     SOT     |   72.0    |   57.1  | 22.8 | [download](https://) | [download](https://)

## Acknowledgments
Our project is developed upon [OSTrack](https://github.com/botaoye/OSTrack). Thanks for their contributions which help us to quickly implement our ideas.

## Citation
If our work is useful for your research, please consider cite:

```
@inproceedings{,
  title={},
  author={},
  booktitle={},
  pages={},
  year={2023}
}
```
# MPLT
