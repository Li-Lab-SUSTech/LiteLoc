import torch
import torch.utils.data
from torch.cuda.amp import autocast
import numpy as np
import copy
import os
import time
import tifffile
import natsort
import pathlib
import torch.multiprocessing as mp
import queue
import csv
from utils.help_utils import cpu, get_mean_percentile, write_csv_array

def split_fov(data, fov_xy=None, sub_fov_size=128, over_cut=8):
    """
    Divide the data into sub-FOVs with over cut.

    Args:
        data (np.ndarray): sequential images to analyze, shape (num_img, height, width), as the lateral size may be
            too large to cause GPU memory overflow, the data will be divided into sub-FOVs and analyzed separately
        fov_xy (tuple of int or None): (x_start, x_end, y_start, y_end), start from 0, in pixel unit, the FOV indicator
            for these images
        sub_fov_size (int): in pixel, size of the sub-FOVs, must be multiple of 4
        over_cut: must be multiple of 4, cut a slightly larger sub-FOV to avoid artifact from the incomplete PSFs at
            image edge.

    Returns:
        (list of np.ndarray, list of tuple, list of tuple):
            list of sub-FOV data with over cut, list of over cut sub-FOV indicator
            (x_start, x_end, y_start, y_end) and list of sub-FOV indicator without over cut
    """

    data = cpu(data)

    if fov_xy is None:
        fov_xy = (0, data.shape[-1] - 1, 0, data.shape[-2] - 1)
        fov_xy_start = [0, 0]
    else:
        fov_xy_start = [fov_xy[0], fov_xy[2]]

    num_img, h, w = data.shape

    assert h == fov_xy[3] - fov_xy[2] + 1 and w == fov_xy[1] - fov_xy[0] + 1, 'data shape does not match fov_xy'

    # enforce the image size to be multiple of 4, pad with estimated background adu. fov_xy_start should be modified
    # according to the padding size, and sub_fov_xy for sub-area images should be modified too.
    factor = 4
    if (h % factor != 0) or (w % factor != 0):
        empty_area_adu = get_mean_percentile(data, percentile=50)
        if h % factor != 0:
            new_h = (h // factor + 1) * factor
            pad_h = new_h - h
            data = np.pad(data, [[0, 0], [pad_h, 0], [0, 0]], mode='constant', constant_values=empty_area_adu)
            fov_xy_start[1] -= pad_h
            h += pad_h
        if w % factor != 0:
            new_w = (w // factor + 1) * factor
            pad_w = new_w - w
            data = np.pad(data, [[0, 0], [0, 0], [pad_w, 0]], mode='constant', constant_values=empty_area_adu)
            fov_xy_start[0] -= pad_w
            w += pad_w

    assert sub_fov_size % factor == 0 and over_cut % factor == 0, f'sub_fov_size and over_cut must be multiple of {factor}'

    # divide the data into sub-FOVs with over_cut
    row_sub_fov = int(np.ceil(h / sub_fov_size))
    col_sub_fov = int(np.ceil(w / sub_fov_size))

    sub_fov_data_list = []
    sub_fov_xy_list = []
    original_sub_fov_xy_list = []
    for row in range(row_sub_fov):  # 0 ~ row_sub_fov-1
        for col in range(col_sub_fov):  # 0 ~ col_sub_fov-1
            x_origin_start = col * sub_fov_size
            y_origin_start = row * sub_fov_size
            x_origin_end = w if x_origin_start + sub_fov_size > w else x_origin_start + sub_fov_size
            y_origin_end = h if y_origin_start + sub_fov_size > h else y_origin_start + sub_fov_size

            x_start = x_origin_start if x_origin_start - over_cut < 0 else x_origin_start - over_cut
            y_start = y_origin_start if y_origin_start - over_cut < 0 else y_origin_start - over_cut
            x_end = x_origin_end if x_origin_end + over_cut > w else x_origin_end + over_cut
            y_end = y_origin_end if y_origin_end + over_cut > h else y_origin_end + over_cut

            sub_fov_data_tmp = data[:, y_start:y_end, x_start:x_end] + .0

            sub_fov_data_list.append(sub_fov_data_tmp)
            sub_fov_xy_list.append((x_start + fov_xy_start[0],
                                    x_end - 1 + fov_xy_start[0],
                                    y_start + fov_xy_start[1],
                                    y_end - 1 + fov_xy_start[1]))
            original_sub_fov_xy_list.append((x_origin_start + fov_xy_start[0],
                                             x_origin_end - 1 + fov_xy_start[0],
                                             y_origin_start + fov_xy_start[1],
                                             y_origin_end - 1 + fov_xy_start[1]))

    return sub_fov_data_list, sub_fov_xy_list, original_sub_fov_xy_list

def filter_over_cut(sub_fov_molecule_list, sub_fov_xy_list, original_sub_fov_xy_list, pixel_size_xy):
    """
    filter the molecules that are out of the original sub-FOV due to the over-cut

    Args:
        sub_fov_molecule_list (list):
            [frame, x, y, z, photon,...], molecule list on each sub-FOV data
            with xy position in the whole FOV.
        sub_fov_xy_list (list of tuple):
            (x_start, x_end, y_start, y_end) unit pixel, the sub-FOV indicator with over cut
        original_sub_fov_xy_list (list of tuple):
            the sub-FOV indicator without over cut, unit pixel
        pixel_size_xy (tuple of int):
            pixel size in xy dimension, unit nm

    Returns:
        list: molecule list of the time block data, the sub-FOV molecules are concatenated together
            with correction of the over-cut.
    """

    molecule_list = []
    for i_fov in range(len(sub_fov_molecule_list)):
        # curr_sub_fov_xy = (sub_fov_xy_list[i_fov][0] * pixel_size_xy[0],
        #                    (sub_fov_xy_list[i_fov][1]+1) * pixel_size_xy[0],
        #                    sub_fov_xy_list[i_fov][2] * pixel_size_xy[1],
        #                    (sub_fov_xy_list[i_fov][3]+1) * pixel_size_xy[1])
        curr_ori_sub_fov_xy_nm = (original_sub_fov_xy_list[i_fov][0] * pixel_size_xy[0],
                                  (original_sub_fov_xy_list[i_fov][1]+1) * pixel_size_xy[0],
                                  original_sub_fov_xy_list[i_fov][2] * pixel_size_xy[1],
                                  (original_sub_fov_xy_list[i_fov][3]+1) * pixel_size_xy[1])

        curr_mol_array = np.array(sub_fov_molecule_list[i_fov])
        if len(curr_mol_array) > 0:
            valid_idx = np.where((curr_mol_array[:, 1] >= curr_ori_sub_fov_xy_nm[0]) &
                                 (curr_mol_array[:, 1] < curr_ori_sub_fov_xy_nm[1]) &
                                 (curr_mol_array[:, 2] >= curr_ori_sub_fov_xy_nm[2]) &
                                 (curr_mol_array[:, 2] < curr_ori_sub_fov_xy_nm[3]))

            molecule_list += curr_mol_array[valid_idx].tolist()

    return sorted(molecule_list, key=lambda x: x[0])


class CompetitiveSmlmDataAnalyzer_multi_producer:
    """
    This class uses the torch.multiprocessing to analyze data in parallel, it will create a read list in main process,
    and then create a public producer to load the tiff file and do the preprocess
    (rolling inference, fov segmentation, batch segmentation), put the batch data in a queue,
    then create multiple consumer processes with the same number of available GPUs.
    Consumers will competitively get and analyze the batch data from the shared memory queue.
    """

    def __init__(self,
                 loc_model,
                 tiff_path,
                 output_path,
                 time_block_gb=1,
                 batch_size=16,
                 sub_fov_size=256,
                 over_cut=8,
                 multi_GPU=False,
                 end_frame_num=None,
                 num_producers=4,
                 ):
        """
        Args:
            loc_model (ailoc.common.XXLoc): localization model object
            tiff_path (str): the path of the tiff file, can also be a directory containing multiple tiff files
            output_path (str): the path to save the analysis results
            time_block_gb (int or float): the size (GB) of the data block loaded into the RAM iteratively,
                to deal with the large data problem
            batch_size (int): batch size for analyzing the sub-FOVs data, the larger the faster, but more GPU memory
            sub_fov_size (int): in pixel, the data is divided into this size of the sub-FOVs, must be multiple of 4,
                the larger the faster, but more GPU memory
            over_cut (int): in pixel, must be multiple of 4, cut a slightly larger sub-FOV to avoid artifact from
                the incomplete PSFs at image edge.
            multi_GPU (bool): if True, use multiple GPUs to analyze the data in parallel
            camera (ailoc.simulation.Camera or None): camera object used to transform the data to photon unit, if None, use
                the default camera object in the loc_model
            fov_xy_start (list of int or None): (x_start, y_start) in pixel unit, If None, use (0,0).
                The global xy pixel position (not row and column) of the tiff images in the whole pixelated FOV,
                start from the top left. For example, (102, 41) means the top left pixel
                of the input images (namely data[:, 0, 0]) corresponds to the pixel xy (102,41) in the whole FOV. This
                parameter is normally (0,0) as we usually treat the input images as the whole FOV. However, when using
                an FD-DeepLoc model trained with pixel-wise field-dependent aberration, this parameter should be carefully
                set to ensure the consistency of the input data position relative to the training aberration map.
            end_frame_num (int or None): the end frame number to analyze, if None, analyze all frames
        """

        mp.set_start_method('spawn', force=True)  # 进程启动的同时启动一个资源追踪器进程，防止泄露

        self.loc_model = copy.deepcopy(loc_model.LiteLoc.cpu())
        self.tiff_path = pathlib.Path(tiff_path)
        self.output_path = output_path
        self.time_block_gb = time_block_gb
        self.batch_size = batch_size
        self.sub_fov_size = sub_fov_size
        self.over_cut = over_cut
        self.multi_GPU = multi_GPU
        self.end_frame_num = end_frame_num
        self.num_producers = num_producers
        self.z_scale = loc_model.params.PSF_model.z_scale
        self.max_photon = loc_model.params.Training.photon_range[1]
        if loc_model.params.PSF_model.simulate_method == 'spline':
            self.pixel_size_xy = [loc_model.EvalMetric.x_scale, loc_model.EvalMetric.y_scale]
        else:
            self.pixel_size_xy = [loc_model.params.PSF_model.vector_psf.pixelSizeX, loc_model.params.PSF_model.vector_psf.pixelSizeY]

        print(f'the file to save the predictions is: {self.output_path}')

        if os.path.exists(self.output_path):
            last_frame_num = 0
            print('the csv file exists but the analyzer will overwrite it and start from the first frame')
        else:
            last_frame_num = 0

        self.start_frame_num = last_frame_num
        # create the file name list and corresponding frame range, used for seamlessly slicing
        start_num = 0
        self.file_name_list = []
        self.file_range_list = []
        if self.tiff_path.is_dir():
            files_list = natsort.natsorted(self.tiff_path.glob('*.tif*'))
            files_list = [str(file_tmp) for file_tmp in files_list]
            for file_tmp in files_list:
                tiff_handle_tmp = tifffile.TiffFile(file_tmp, is_ome=False, is_lsm=False, is_ndpi=False)
                length_tmp = len(tiff_handle_tmp.pages)
                if start_num == 0:
                    single_frame_nbyte = tiff_handle_tmp.filehandle.size / length_tmp
                    self.time_block_n_img = int(np.ceil(self.time_block_gb * (1024 ** 3) / single_frame_nbyte))
                    self.tiff_shape = [tiff_handle_tmp.pages.first.imagewidth, tiff_handle_tmp.pages.first.imagelength]
                    self.fov_xy = (0, self.tiff_shape[0] - 1, 0, self.tiff_shape[1] - 1)
                    self.fov_xy_nm = (0, self.tiff_shape[0] * self.pixel_size_xy[0],
                                      0, self.tiff_shape[1] * self.pixel_size_xy[1])
                tiff_handle_tmp.close()
                self.file_name_list.append(file_tmp)
                self.file_range_list.append((start_num, start_num + length_tmp - 1, length_tmp))
                start_num += length_tmp
        else:
            tiff_handle_tmp = tifffile.TiffFile(self.tiff_path)
            length_tmp = len(tiff_handle_tmp.pages)
            single_frame_nbyte = tiff_handle_tmp.filehandle.size / length_tmp
            self.time_block_n_img = int(np.ceil(self.time_block_gb * (1024 ** 3) / single_frame_nbyte))
            self.tiff_shape = [tiff_handle_tmp.pages.first.imagewidth, tiff_handle_tmp.pages.first.imagelength]
            self.fov_xy = (0, self.tiff_shape[0] - 1, 0, self.tiff_shape[1] - 1)
            self.fov_xy_nm = (0, self.tiff_shape[0] * self.pixel_size_xy[0],
                              0, self.tiff_shape[1] * self.pixel_size_xy[1])
            tiff_handle_tmp.close()
            self.file_name_list.append(str(self.tiff_path))
            self.file_range_list.append((0, length_tmp - 1, length_tmp))

        print(f"frame ranges || filename: ")
        for i in range(len(self.file_range_list)):
            print(f"[{self.file_range_list[i][0]}-{self.file_range_list[i][1]}] || {self.file_name_list[i]}")

        self.sum_file_length = np.array(self.file_range_list)[:, 1].sum() - np.array(self.file_range_list)[:,
                                                                            0].sum() + len(self.file_range_list)  # todo: not understand
        self.end_frame_num = self.sum_file_length if end_frame_num is None or end_frame_num > self.sum_file_length else end_frame_num

        frame_slice = []
        i = 0
        while ((i + 1) * self.time_block_n_img + self.start_frame_num) <= self.end_frame_num:
            frame_slice.append(
                slice(i * self.time_block_n_img + self.start_frame_num, (i + 1) * self.time_block_n_img + self.start_frame_num))
            i += 1
        if i * self.time_block_n_img + self.start_frame_num < self.end_frame_num:
            frame_slice.append(slice(i * self.time_block_n_img + self.start_frame_num, self.end_frame_num))
        self.frame_slice = frame_slice

        # make the file read list queue, should include the frame number and the file name and the read plan
        self.file_read_list_queue = mp.Queue()
        for curr_frame_slice in self.frame_slice:
            slice_start = curr_frame_slice.start
            slice_end = curr_frame_slice.stop

            # for multiprocessing dataloader, the tiff handle cannot be shared by different processes, so we need to
            # get the relation between the frame number and the file name and corresponding frame range for each file,
            # and then use the imread function in each process
            files_to_read = []
            slice_for_files = []
            for file_name, file_range in zip(self.file_name_list, self.file_range_list):
                # situation 1: the first frame to get is in the current file
                # situation 2: the last frame to get is in the current file
                # situation 3: the current file is in the middle of the frame range
                if file_range[0] <= slice_start <= file_range[1] or \
                        file_range[0] < slice_end <= file_range[1] + 1 or \
                        (slice_start < file_range[0] and slice_end > file_range[1] + 1):
                    files_to_read.append(file_name)
                    slice_for_files.append(slice(max(0, slice_start - file_range[0]),
                                                 min(file_range[2], slice_end - file_range[0])))

            item = [slice_start, slice_end, files_to_read, slice_for_files]
            self.file_read_list_queue.put(item)

        # instantiate one producer and multiple consumer
        self.num_consumers = torch.cuda.device_count() if self.multi_GPU else 1
        self.batch_data_queue = mp.JoinableQueue()
        self.result_queue = mp.JoinableQueue()

        #self.loc_model.remove_gpu_attribute()
        self.print_lock = mp.Lock()
        self.total_result_item_num = 0
        for slice_tmp in self.frame_slice:
            self.total_result_item_num += int(np.ceil(self.tiff_shape[-1] / sub_fov_size)) * int(
            np.ceil(self.tiff_shape[-2] / sub_fov_size)) * np.ceil((slice_tmp.stop - slice_tmp.start) / batch_size)

        self.producer_list = []
        for producer_idx in range(self.num_producers):  # added by fy
            self.file_read_list_queue.put(None)
            self.producer_list.append(mp.Process(target=self.producer_func,
                                               args=(
                                                   self.file_read_list_queue,
                                                   self.batch_size,
                                                   self.sub_fov_size,
                                                   self.over_cut,
                                                   self.fov_xy,
                                                   self.batch_data_queue,
                                                   self.num_consumers,
                                                   self.print_lock,
                                                   producer_idx,
                                               )))

        self.consumer_list = []
        for i in range(self.num_consumers):
            consumer = mp.Process(
                target=self.consumer_func,
                args=(
                    self.loc_model,
                    self.batch_data_queue,
                    'cuda:' + str(i),
                    self.result_queue,
                    self.print_lock,
                )
            )
            self.consumer_list.append(consumer)

        self.saver = mp.Process(
            target=self.saver_func,
            args=(
                self.result_queue,
                self.output_path,
                self.num_consumers,
                self.pixel_size_xy,
                self.max_photon,
                self.z_scale,
                self.print_lock,
                self.total_result_item_num,
            )
        )

    def start(self):
        """
        start the data analysis
        """

        print(f'{mp.current_process().name}, id is {os.getpid()}')

        for producer in self.producer_list:
            producer.start()
        self.saver.start()
        for consumer in self.consumer_list:
            consumer.start()

        for producer in self.producer_list:
            producer.join()
        self.saver.join()
        for consumer in self.consumer_list:
            consumer.join()

    @staticmethod
    def producer_func(  # todo: one process to read data, n producers to pre-process data
            file_read_list_queue,
            batch_size,
            sub_fov_size,
            over_cut,
            fov_xy,
            batch_data_queue,
            num_consumers,
            print_lock,
            producer_idx,
    ):

        with print_lock:
            print(f'enter the producer {producer_idx} process: {os.getpid()}')

        rolling_inference = True
        extra_length = 1

        pre_process_time = 0
        while True:
            try:
                file_read_list = file_read_list_queue.get_nowait()
            except queue.Empty:
                continue

            if file_read_list is None:
                break

            t0 = time.monotonic()

            slice_start = file_read_list[0]
            slice_end = file_read_list[1]
            files_to_read = file_read_list[2]
            slice_for_files = file_read_list[3]

            data_block = []
            for file_name, slice_tmp in zip(files_to_read, slice_for_files):
                data_tmp = tifffile.imread(file_name, key=slice_tmp)
                data_block.append(data_tmp)

            data_block = np.concatenate(data_block, axis=0)
            num_img, h, w = data_block.shape

            # rolling inference strategy needs to pad the whole data with two more images at the beginning and end
            # to provide the context for the first and last image
            if rolling_inference:
                if num_img > extra_length:
                    data_block = np.concatenate(
                        [data_block[extra_length:0:-1], data_block, data_block[-2:-2 - extra_length:-1]], 0)
                else:
                    data_nopad = data_block.copy()
                    for i in range(extra_length):
                        data_block = np.concatenate([data_nopad[(i + 1) % num_img: (i + 1) % num_img + 1],
                                                     data_block,
                                                     data_nopad[num_img - 1 - ((i + 1) % num_img): num_img - 1 - (
                                                             (i + 1) % num_img) + 1],
                                                     ], 0)

            # sub-FOV segmentation
            sub_fov_data_list, sub_fov_xy_list, original_sub_fov_xy_list = split_fov(data=data_block,
                                                                                     fov_xy=fov_xy,
                                                                                     sub_fov_size=sub_fov_size,
                                                                                     over_cut=over_cut)

            pre_process_time += time.monotonic() - t0

            # put the sub-FOV batch data in the shared queue
            for i_fov in range(len(sub_fov_data_list)):
                # for each batch, rolling inference needs to take 2 more images at the beginning and end, but only needs
                # to return the molecule list for the middle images
                for i in range(int(np.ceil(num_img / batch_size))):
                    t1 = time.monotonic()
                    if rolling_inference:
                        item = {
                            'data': torch.tensor(
                                sub_fov_data_list[i_fov][i * batch_size: (i + 1) * batch_size + 2 * extra_length]),
                            'sub_fov_xy': sub_fov_xy_list[i_fov],
                            'original_sub_fov_xy': original_sub_fov_xy_list[i_fov],
                            'frame_num': slice_start + i * batch_size,
                        }
                    else:
                        item = {
                            'data': torch.tensor(sub_fov_data_list[i_fov][i * batch_size: (i + 1) * batch_size]),
                            'sub_fov_xy': sub_fov_xy_list[i_fov],
                            'original_sub_fov_xy': original_sub_fov_xy_list[i_fov],
                            'frame_num': slice_start + i * batch_size,
                        }
                    pre_process_time += time.monotonic() - t1
                    batch_data_queue.put(item)
        batch_data_queue.join()
        if producer_idx == 0:
            for i in range(int(num_consumers)):
                batch_data_queue.put(None)
            batch_data_queue.join()
        with print_lock:
            print(f'Producer {producer_idx}, \n'
                  f'    preprocess time {pre_process_time}')

    @staticmethod
    def consumer_func(
            loc_model,
            batch_data_queue,
            device,
            result_queue,
            print_lock,
    ):

        with print_lock:
            print(f'enter the comsumer process: {os.getpid()}, '
                  f'device: {device}')

        torch.cuda.set_device(device)
        loc_model.to(device)
        loc_model.eval()

        get_time = 0
        anlz_time = 0
        item_counts = 0

        while True:
            with torch.no_grad():
                with autocast():
                    t0 = time.monotonic()
                    try:
                        item = batch_data_queue.get(timeout=1)  # timeout is optional
                    except queue.Empty:
                        get_time += time.monotonic() - t0
                        continue
                    batch_data_queue.task_done()
                    get_time += time.monotonic() - t0

                    if item is None:
                        break

                    item_counts += 1

                    t1 = time.monotonic()
                    data = item['data']
                    data = data.to(device, non_blocking=True)
                    sub_fov_xy = item['sub_fov_xy']
                    original_sub_fov_xy = item['original_sub_fov_xy']
                    frame_num = item['frame_num']
                    get_time += time.monotonic() - t1

                    torch.cuda.synchronize(); t2 = time.monotonic()
                    molecule_array_tmp = loc_model.analyze(data, test=True)
                    torch.cuda.synchronize(); anlz_time += time.monotonic() - t2
                    molecule_array_tmp = cpu(molecule_array_tmp)

                    result_item = {
                        'molecule_array': molecule_array_tmp,
                        'sub_fov_xy': sub_fov_xy,
                        'original_sub_fov_xy': original_sub_fov_xy,
                        'frame_num': frame_num
                    }
                    # print(result_item)
                    result_queue.put(result_item)

        result_queue.put(None)
        result_queue.join()
        with print_lock:
            print(f'Consumer {os.getpid()}, '
                  f'device: {device}, \n'
                  f'    total data get time: {get_time}, \n'
                  f'    analyze time: {anlz_time}, \n'
                  f'    item counts: {item_counts}')

    @staticmethod
    def saver_func(
            result_queue,
            output_path,
            num_consumers,
            pixel_size_xy,
            max_photon,
            z_scale,
            print_lock,
            total_result_item_num,
    ):

        with print_lock:
            print(f'enter the saver process: {os.getpid()}')

        write_csv_array(input_array=np.array([]), filename=output_path,
                                     write_mode='write localizations')

        with open(output_path, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)

            patience = total_result_item_num//20
            finished_item_num = 0
            time_recent = time.monotonic()

            get_time = 0
            process_time = 0
            format_time = 0
            write_time = 0
            none_counts = 0
            while True:

                # compute approximate remaining time
                if finished_item_num > 0 and finished_item_num % patience == 0:
                    time_cost_recent = time.monotonic() - time_recent
                    time_recent = time.monotonic()
                    print(f'Analysis progress: {finished_item_num/total_result_item_num*100:.2f}%, '
                          f'ETA: {time_cost_recent/patience*(total_result_item_num-finished_item_num):.0f}s')
                    # print('finished_item_num:' + str(finished_item_num))
                    # print('total_result_item_num:' + str(total_result_item_num))

                t0 = time.monotonic()
                try:
                    result_item = result_queue.get(timeout=1)  # timeout is optional
                except queue.Empty:
                    get_time += time.monotonic() - t0
                    continue
                result_queue.task_done()
                get_time += time.monotonic() - t0

                if result_item is None:
                    none_counts += 1
                    if none_counts == num_consumers:
                        break
                else:
                    t1 = time.monotonic()
                    molecule_array_tmp = result_item['molecule_array']
                    sub_fov_xy_tmp = result_item['sub_fov_xy']
                    original_sub_fov_xy_tmp = result_item['original_sub_fov_xy']
                    frame_num = result_item['frame_num']
                    get_time += time.monotonic() - t1

                    if len(molecule_array_tmp) > 0:
                        t2 = time.monotonic()
                        molecule_array_tmp[:, 0] += frame_num
                        molecule_array_tmp[:, 1] = (molecule_array_tmp[:, 1] + sub_fov_xy_tmp[0]) * pixel_size_xy[0]
                        molecule_array_tmp[:, 2] = (molecule_array_tmp[:, 2] + sub_fov_xy_tmp[2]) * pixel_size_xy[1]
                        molecule_array_tmp[:, 3] = molecule_array_tmp[:, 3] * z_scale
                        molecule_array_tmp[:, 4] = molecule_array_tmp[:, 4] * max_photon

                        molecule_list_tmp = filter_over_cut([molecule_array_tmp],
                                                            [sub_fov_xy_tmp],
                                                            [original_sub_fov_xy_tmp],
                                                            pixel_size_xy, )
                        # molecule_list_tmp = np.array(molecule_list_tmp)
                        process_time += time.monotonic()-t2

                        if len(molecule_list_tmp) > 0:
                            t3 = time.monotonic()
                            # format the data to string with 2 decimal to save the disk space and csv writing time
                            # formatted_data = [['{:.2f}'.format(item) for item in row] for row in molecule_list_tmp]
                            format_time += time.monotonic() - t3

                            t4 = time.monotonic()
                            csvwriter.writerows(molecule_list_tmp)
                            write_time += time.monotonic()-t4

                    finished_item_num += 1

        with print_lock:
            print(f'Saver {os.getpid()}, \n'
                  f'    total result get time {get_time}, \n'
                  f'    process time {process_time}, \n'
                  f'    format time {format_time}, \n'
                  f'    write time {write_time}')