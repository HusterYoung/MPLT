from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.network_path = '/home/young/Code/MPLT-main-prompt/pretrained'    # Where tracking networks are stored.
    settings.gtot_dir = '/media/young/Data-SSD/Datasets/GTOT'
    settings.lasher_path = '/media/young/Data-SSD/Datasets/LasHeR-Divided/TestingSet'
    settings.UAV_RGBT_dir = '/media/young/Dataset/VTUAV-st/Test'
    settings.rgbt234_dir = '/media/young/Data-SSD/Datasets/RGBT234/'
    settings.got10k_path = '/media/young/Data-SSD/Datasets/got10k/'
    settings.rgbt210_dir = '/media/young/Dataset/RGBT210/RGBT210_part1'
    settings.got_reports_path = ''
    settings.lasot_lmdb_path = ''
    settings.lasot_path = '/media/young/TiPlus/Datasets/LaSOT/zip'
    settings.network_path = ''  # Where tracking networks are stored.
    settings.nfs_path = ''
    settings.otb_path = '/media/young/TiPlus/Datasets/OTB2015/OTB100'
    settings.prj_dir = ''
    settings.result_plot_path = '/home/young/Code/MPLT-main-prompt/RGBT_workspace/results_plot'
    settings.results_path = '/home/young/Code/MPLT-main-prompt/RGBT_workspace/results/LasHeR'
    settings.save_dir = './output'


    return settings

