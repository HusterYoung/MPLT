import importlib
import os
from collections import OrderedDict
from lib.test.evaluation.environment import env_settings
from lib.test.utils import kalman
import time
import cv2 as cv

from lib.utils.lmdb_utils import decode_img
from pathlib import Path
import numpy as np


def trackerlist(name: str, parameter_name: str, dataset_name: str, run_ids = None, display_name: str = None,
                result_only=False):
    """Generate list of trackers.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_ids: A single or list of run_ids.
        display_name: Name to be displayed in the result plots.
    """
    if run_ids is None or isinstance(run_ids, int):
        run_ids = [run_ids]
    return [Tracker(name, parameter_name, dataset_name, run_id, display_name, result_only) for run_id in run_ids]


class Tracker:
    """Wraps the tracker for evaluation and running purposes.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_id: The run id.
        display_name: Name to be displayed in the result plots.
    """

    def __init__(self, name: str, parameter_name: str, dataset_name: str, run_id: int = None, display_name: str = None,
                 result_only=False):
        assert run_id is None or isinstance(run_id, str) or isinstance(run_id, int)

        self.name = name
        self.parameter_name = parameter_name
        self.dataset_name = dataset_name
        self.run_id = run_id
        self.display_name = display_name
        self.confidence = []
        self.out_box = []

        env = env_settings()
        if self.run_id is None:
            self.results_dir = '{}/{}/{}'.format(env.results_path, self.name, self.parameter_name)
        else:
            self.results_dir = '{}/{}/{}_{}'.format(env.results_path, self.name, self.parameter_name, self.run_id)
        if result_only:
            self.results_dir = '{}/{}'.format(env.results_path, self.name)

        tracker_module_abspath = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                              '..', 'tracker', '%s.py' % self.name))
        if os.path.isfile(tracker_module_abspath):
            tracker_module = importlib.import_module('lib.test.tracker.{}'.format(self.name))
            self.tracker_class = tracker_module.get_tracker_class()
        else:
            self.tracker_class = None

    def create_tracker(self, params):
        tracker = self.tracker_class(params, self.dataset_name)
        return tracker

    def run_sequence(self, seq, debug=None, run_id=None, name=None):
        """Run tracker on sequence.
        args:
            seq: Sequence to run the tracker on.
            visualization: Set visualization flag (None means default value specified in the parameters).
            debug: Set debug level (None means default value specified in the parameters).
            multiobj_mode: Which mode to use for multiple objects.
        """
        params = self.get_parameters()

        from lib.test.evaluation.environment import env_settings
        # run_id = int(run_id)
        if run_id is None:
            run_id = params.cfg.TEST.EPOCH
        params.checkpoint = os.path.join(env_settings().save_dir, "%s/checkpoints/train/mplt_track/%s/MPLTTrack_ep%04d.pth.tar" %
                                     (name, name, run_id))

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)

        params.debug = debug_

        # Get init information
        init_info = seq.init_info()

        tracker = self.create_tracker(params)

        output = self._track_sequence(tracker, seq, init_info)
        return output

    def _kalman(self, out_box_list):

        return out_box_list

    def _track_sequence(self, tracker, seq, init_info):
        # Define outputs
        # Each field in output is a list containing tracker prediction for each frame.

        # In case of single object tracking mode:
        # target_bbox[i] is the predicted bounding box for frame i
        # time[i] is the processing time for frame i

        # In case of multi object tracking mode:
        # target_bbox[i] is an OrderedDict, where target_bbox[i][obj_id] is the predicted box for target obj_id in
        # frame i
        # time[i] is either the processing time for frame i, or an OrderedDict containing processing times for each
        # object in frame i

        output = {'target_bbox': [],
                  'time': []}
        if tracker.params.save_all_boxes:
            output['all_boxes'] = []
            output['all_scores'] = []

        def _store_outputs(tracker_out: dict, defaults=None):
            defaults = {} if defaults is None else defaults
            for key in output.keys():
                val = tracker_out.get(key, defaults.get(key, None))
                if key in tracker_out or val is not None:
                    output[key].append(val)

        # Initialize
        # import ipdb; ipdb.set_trace()
        self.image_v = self._read_image([seq.frames[0][0], seq.frames[0][0]])
        self.image_i = self._read_image([seq.frames[1][0], seq.frames[1][0]])
        if isinstance(self.image_v, list):
            self.image = [np.concatenate([self.image_v[0], self.image_i[0]], 2), np.concatenate([self.image_v[1], self.image_i[1]], 2)]
        else:
            self.image = np.concatenate([self.image_v, self.image_i], 2)
        start_time = time.time()
        out = tracker.initialize(self.image, init_info)
        if out is None:
            out = {}

        prev_output = OrderedDict(out)
        init_default = {'target_bbox': init_info.get('init_bbox'),
                        'time': time.time() - start_time}
        if tracker.params.save_all_boxes:
            init_default['all_boxes'] = out['all_boxes']
            init_default['all_scores'] = out['all_scores']

        _store_outputs(out, init_default)
        self.sum = 0
        self.mean = 0
        for frame_num, frame_path in enumerate(zip(seq.frames[0][1:], seq.frames[1][1:]), start=1):
            # import ipdb; ipdb.set_trace()
            if frame_num == 1:
                self.image_v[1] = self._read_image(frame_path[0])
                # self.image_v[0] = self._read_image(frame_path[0])
                self.image_i[1] = self._read_image(frame_path[1])
                # self.image_i[0] = self._read_image(frame_path[1])
            else:
                self.image_v[0] = self.image_v[1]
                self.image_v[1] = self._read_image(frame_path[0])
                self.image_i[0] = self.image_i[1]
                self.image_i[1] = self._read_image(frame_path[1])
            self.image = [np.concatenate([self.image_v[0], self.image_i[0]], 2), np.concatenate([self.image_v[1], self.image_i[1]], 2)]
            start_time = time.time()

            info = seq.frame_info(frame_num)
            info['previous_output'] = prev_output

            if len(seq.ground_truth_rect) > 1:
                info['gt_bbox'] = seq.ground_truth_rect[frame_num]
            out = tracker.track(self.image, info)
            if len(self.confidence) < 16 and len(self.out_box) < 16:
                self.confidence.append(out['best_score'])
                self.out_box.append(out['target_bbox'])
            else:
                del self.confidence[0]
                del self.out_box[0]
                self.confidence.append(out['best_score'])
                self.out_box.append(out['target_bbox'])
                if self.confidence[-1] < 0.25:
                    kalman_filter_params = kalman.kf_init(self.out_box[0][0], self.out_box[0][1], 0, 0)
                    kf_params_record = np.zeros((16, 4))
                    kf_params_p = np.zeros((16, 4))
                    for i in range(16):
                        if i == 0:
                            kalman_filter_params = kalman.kf_init(self.out_box[i][0], self.out_box[i][1], 0, 0)  # 初始化
                        else:
                            kalman_filter_params.z = np.transpose([self.out_box[i][0], self.out_box[i][1]])  # 设置当前时刻的观测位置
                            kalman_filter_params = kalman.kf_update(kalman_filter_params)  # 卡尔曼滤波
                        kf_params_record[i, ::] = np.transpose(kalman_filter_params.x)
                        kf_params_p[i, ::] = np.transpose(kalman_filter_params.G)
                    self.out_box[-1][0] = kf_params_record[-1][0]
                    self.out_box[-1][1] = kf_params_record[-1][1]
                    # self.out_box[-1][2] = self.out_box[-2][2]
                    # self.out_box[-1][3] = self.out_box[-2][3]
                    out["target_bbox"] = self.out_box[-1]

            # confidence = out['best_score']
            if self.confidence[-1] > 0.91:
                tracker.update_template(self.image, out['target_bbox'])
            # if confidence - self.mean > 0.1 and frame_num > 1 and confidence > 0.925:
            #     tracker.update_template(self.image, out['target_bbox'])
            # self.sum += confidence
            # self.mean = self.sum / frame_num
            prev_output = OrderedDict(out)
            _store_outputs(out, {'time': time.time() - start_time})

        for key in ['target_bbox', 'all_boxes', 'all_scores']:
            if key in output and len(output[key]) <= 1:
                output.pop(key)

        return output


    def run_video(self, videofilepath, optional_box=None, debug=None, visdom_info=None, save_results=False):
        """Run the tracker with the vieofile.
        args:
            debug: Debug level.
        """

        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.tracker_name = self.name
        params.param_name = self.parameter_name
        # self._init_visdom(visdom_info, debug_)

        multiobj_mode = getattr(params, 'multiobj_mode', getattr(self.tracker_class, 'multiobj_mode', 'default'))

        if multiobj_mode == 'default':
            tracker = self.create_tracker(params)

        elif multiobj_mode == 'parallel':
            tracker = MultiObjectWrapper(self.tracker_class, params, self.visdom, fast_load=True)
        else:
            raise ValueError('Unknown multi object mode {}'.format(multiobj_mode))

        assert os.path.isfile(videofilepath), "Invalid param {}".format(videofilepath)
        ", videofilepath must be a valid videofile"

        output_boxes = []

        cap = cv.VideoCapture(videofilepath)
        display_name = 'Display: ' + tracker.params.tracker_name
        cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        cv.resizeWindow(display_name, 960, 720)
        success, frame = cap.read()
        cv.imshow(display_name, frame)

        def _build_init_info(box):
            return {'init_bbox': box}

        if success is not True:
            print("Read frame from {} failed.".format(videofilepath))
            exit(-1)
        if optional_box is not None:
            assert isinstance(optional_box, (list, tuple))
            assert len(optional_box) == 4, "valid box's foramt is [x,y,w,h]"
            tracker.initialize(frame, _build_init_info(optional_box))
            output_boxes.append(optional_box)
        else:
            while True:
                # cv.waitKey()
                frame_disp = frame.copy()

                cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL,
                           1.5, (0, 0, 0), 1)

                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                tracker.initialize(frame, _build_init_info(init_state))
                output_boxes.append(init_state)
                break

        while True:
            ret, frame = cap.read()

            if frame is None:
                break

            frame_disp = frame.copy()

            # Draw box
            out = tracker.track(frame)
            state = [int(s) for s in out['target_bbox']]
            output_boxes.append(state)

            cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                         (0, 255, 0), 5)

            font_color = (0, 0, 0)
            cv.putText(frame_disp, 'Tracking!', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'Press r to reset', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'Press q to quit', (20, 80), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)

            # Display the resulting frame
            cv.imshow(display_name, frame_disp)
            key = cv.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('r'):
                ret, frame = cap.read()
                frame_disp = frame.copy()

                cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                           (0, 0, 0), 1)

                cv.imshow(display_name, frame_disp)
                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                tracker.initialize(frame, _build_init_info(init_state))
                output_boxes.append(init_state)

        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()

        if save_results:
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
            video_name = Path(videofilepath).stem
            base_results_path = os.path.join(self.results_dir, 'video_{}'.format(video_name))

            tracked_bb = np.array(output_boxes).astype(int)
            bbox_file = '{}.txt'.format(base_results_path)
            np.savetxt(bbox_file, tracked_bb, delimiter='\t', fmt='%d')


    def get_parameters(self):
        """Get parameters."""
        param_module = importlib.import_module('lib.test.parameter.{}'.format(self.name))
        params = param_module.parameters(self.parameter_name)
        return params

    def _read_image(self, image_file: str):
        if isinstance(image_file, str):
            im = cv.imread(image_file)
            return cv.cvtColor(im, cv.COLOR_BGR2RGB)
        elif isinstance(image_file, list) and len(image_file) == 2:
            im0 = cv.imread(image_file[0])
            im1 = cv.imread(image_file[1])
            im0 = cv.cvtColor(im0, cv.COLOR_BGR2RGB)
            im1 = cv.cvtColor(im1, cv.COLOR_BGR2RGB)
            # return decode_img(image_file[0], image_file[1])
            return [im0, im1]
        else:
            raise ValueError("type of image_file should be str or list")



