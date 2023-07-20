class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/home/young/Code/MPLT-main-prompt'  # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/home/young/Code/MPLT-main-prompt/tensorboard'  # Directory for tensorboard files.
        self.pretrained_networks = '/home/young/Code/MPLT-main-prompt/pretrained_networks'
        self.got10k_val_dir = '/media/jiawen/Datasets/Codes/ViPT/ViPT/data/got10k/val'
        self.lasot_lmdb_dir = '/media/young/TiPlus/Datasets/LaSOT/zip'
        self.got10k_lmdb_dir = '/media/jiawen/Datasets/Codes/ViPT/ViPT/data/got10k_lmdb'
        self.trackingnet_lmdb_dir = '/media/jiawen/Datasets/Codes/ViPT/ViPT/data/trackingnet_lmdb'
        self.coco_lmdb_dir = '/media/jiawen/Datasets/Codes/ViPT/ViPT/data/coco_lmdb'
        self.coco_dir = '/media/young/Data-SSD/Datasets/COCO2017'
        self.lasot_dir = '/media/young/TiPlus/Datasets/LaSOT/zip'
        self.got10k_dir = '/media/jiawen/Datasets/Codes/ViPT/ViPT/data/got10k/train'
        self.trackingnet_dir = '/media/young/TiPlus/Datasets/TrackingNet'
        self.depthtrack_dir = '/media/jiawen/Datasets/Codes/ViPT/ViPT/data/depthtrack/train'
        self.lasher_train_dir = '/media/young/Data-SSD/Datasets/LasHeR-Divided/TrainingSet/Trainingset'
        self.lasher_test_dir = '/media/young/Data-SSD/Datasets/LasHeR-Divided/TestingSet/testingset'
        self.visevent_dir = '/media/jiawen/Datasets/Codes/ViPT/ViPT/data/visevent/train'
