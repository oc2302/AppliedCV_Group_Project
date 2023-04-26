
#Modified from TSN and SlowOnly
import torch.utils.data as data
import decord
from PIL import Image
import os
import numpy as np
from numpy.random import randint

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[-1])


class DataSet(data.Dataset):
    def __init__(self, dataset, root_path, list_file,
                 num_segments=3, new_length=5, clip_index=0, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 random_shift=True, test_mode=False,
                 remove_missing=False, dense_sample=False, twice_sample=False):

        self.root_path = root_path
        self.list_file = list_file
        self.clip_index = clip_index
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.dataset = dataset
        self.remove_missing = remove_missing
        self.I3D_sample = False  #Dense
        self.dense_sample = dense_sample  
        self.twice_sample = twice_sample  

        self._parse_list()

    def _load_image(self, directory, idx):
        if self.modality == 'RGB':
            try:
                return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert('RGB')]
            except Exception:
                print('error loading image:', os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
                return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')]

    def _parse_list(self):
        #Check frame number
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        if len(tmp[0]) == 3: # skip remove_missin for decording "raw_video label" type dataset_config
            if not self.test_mode or self.remove_missing:
                tmp = [item for item in tmp if int(item[1]) >= 8]
        self.video_list = [VideoRecord(item) for item in tmp]

        if self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
            for v in self.video_list:
                v._data[1] = int(v._data[1]) / 2
        print('video number:%d' % (len(self.video_list)))

    def _sample_indices(self, video_list):
        if not self.I3D_sample : # Uniform sampling
            if((len(video_list) - self.new_length + 1) < self.num_segments):
                average_duration = (len(video_list) - 5 + 1) // (self.num_segments)
            else:
                average_duration = (len(video_list) - self.new_length + 1) // (self.num_segments)
            offsets = []
            if average_duration > 0:
                offsets += list(np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,size=self.num_segments))
            elif len(video_list) > self.num_segments:
                if((len(video_list) - self.new_length + 1) >= self.num_segments):
                    offsets += list(np.sort(randint(len(video_list) - self.new_length + 1, size=self.num_segments)))
                else:
                    offsets += list(np.sort(randint(len(video_list) - 5 + 1, size=self.num_segments)))
            else:
                offsets += list(np.zeros((self.num_segments,)))
            offsets = np.array(offsets)
            return offsets + 1
        elif self.dataset == 'kinetics' and self.I3D_sample and (not self.dense_sample) : # i3d type sampling for training
            sample_pos = max(1, 1 + len(video_list) - self.new_length - 64)
            t_stride = 64 // self.num_segments
            start_idx1 = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx1) % len(video_list) for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        elif self.dense_sample:  # i3d dense sample for test
            sample_pos = max(1, 1 + len(video_list) - self.new_length - 64)
            t_stride = 64 // self.num_segments
            start_idx1 = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            start_idx2 = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            start_idx3 = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            start_idx4 = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            start_idx5 = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            start_idx6 = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            start_idx7 = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            start_idx8 = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            start_idx9 = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            start_idx10 = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx1) % len(video_list) for idx in range(self.num_segments)]+[(idx * t_stride + start_idx2) % len(video_list) for idx in range(self.num_segments)]+[(idx * t_stride + start_idx3) % len(video_list) for idx in range(self.num_segments)]+[(idx * t_stride + start_idx4) % len(video_list) for idx in range(self.num_segments)]+[(idx * t_stride + start_idx5) % len(video_list) for idx in range(self.num_segments)]+[(idx * t_stride + start_idx6) % len(video_list) for idx in range(self.num_segments)]+[(idx * t_stride + start_idx7) % len(video_list) for idx in range(self.num_segments)]+[(idx * t_stride + start_idx8) % len(video_list) for idx in range(self.num_segments)]+[(idx * t_stride + start_idx9) % len(video_list) for idx in range(self.num_segments)]+[(idx * t_stride + start_idx10) % len(video_list) for idx in range(self.num_segments)]
            return np.array(offsets) + 1


    def _get_val_indices(self, video_list):
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + len(video_list) - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % len(video_list) for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        else:
            if len(video_list) > self.num_segments + self.new_length - 1:
                tick = (len(video_list) - self.new_length + 1) / float(self.num_segments)
                offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            else:
                offsets = np.zeros((self.num_segments,))
            return offsets + 1

    def _get_test_indices(self, video_list):
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + len(video_list) - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % len(video_list) for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        else:
            if len(video_list) > self.num_segments + self.new_length - 1:
                tick = (len(video_list) - self.new_length + 1) / float(self.num_segments)
                offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            else:
                offsets = np.zeros((self.num_segments,))
            return offsets + 1

    def __getitem__(self, index):
        record = self.video_list[index]
        
        if('something' in self.dataset): 
            decode_boo = False
            video_list = os.listdir(record.path)
        
        else:
            decode_boo = True
            try:
                directory = record.path
                if directory[-4:] != ".mp4":
                    video_path = directory+".mp4"
                else:
                    video_path = directory
                video_list = decord.VideoReader(video_path)
            except UnicodeDecodeError:
                decode_boo = False
                video_list = os.listdir(record.path)
        
        
        if not self.test_mode:
            if self.I3D_sample :
                segment_indices = self._sample_indices(video_list) 
            else:
                segment_indices = self._sample_indices(video_list) if self.random_shift else self._get_val_indices(video_list) 
        else:
            if self.dataset == 'kinetics':
                segment_indices = self._sample_indices(video_list)
            else:
                segment_indices = self._get_test_indices(video_list)

        
        return self.get(record,video_list, segment_indices,decode_boo)

    def get(self, record,video_list, indices,decode_boo):
        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(0,self.new_length,1):
                if(decode_boo):
                    seg_imgs = [Image.fromarray(video_list[p-1].asnumpy()).convert('RGB')]
                else:
                    seg_imgs = self._load_image(record.path,p)
                images.extend(seg_imgs)
                if((len(video_list)-self.new_length*1+1)>=8):
                    if p < (len(video_list)):
                        p += 1
                else:
                    if p < (len(video_list)):
                        p += 1

        process_data, record_label = self.transform((images,record.label))

        return process_data, record_label

    def __len__(self):
        return len(self.video_list)


#Configure the data
ROOT_DATASET = '/ssd/video/' 

def return_something(modality):
    filename_categories = 174
    if modality == 'RGB' or modality== 'RGBDiff':
        root_data = "/" 
        filename_imglist_train = "/your_path_to/something-something-v1-train_list.txt"
        filename_imglist_val = "/your_path_to/something-something-v1-val_list.txt"
        prefix = '{:05d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + '/your_path_to/something/v1/20bn-something-something-v1-flow'
        filename_imglist_train = '/your_path_to/something/v1/train_videofolder_flow.txt'
        filename_imglist_val = '/your_path_to/something/v1/val_videofolder_flow.txt'
        prefix = '{:06d}-{}_{:05d}.jpg'
    else:
        print('no such modality:'+modality)
        raise NotImplementedError
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_kinetics(modality):
    filename_categories = 400
    if modality == 'RGB':
        root_data = "/"
        filename_imglist_train = "/your_path_to/list_kinetics-400/train.csv"
        filename_imglist_val = "/your_path_to/list_kinetics-400/val.csv"
        prefix = 'image_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_dataset(dataset, modality):
    dict_single = {'jester': return_jester, 'something': return_something, 'somethingv2': return_somethingv2,
                   'ucf101': return_ucf101, 'hmdb51': return_hmdb51,
                   'kinetics': return_kinetics }
    if dataset in dict_single:
        file_categories, file_imglist_train, file_imglist_val, root_data, prefix = dict_single[dataset](modality)
    else:
        raise ValueError('Unknown dataset '+dataset)

    file_imglist_train = os.path.join(ROOT_DATASET, file_imglist_train)
    file_imglist_val = os.path.join(ROOT_DATASET, file_imglist_val)
    if isinstance(file_categories, str):
        file_categories = os.path.join(ROOT_DATASET, file_categories)
        with open(file_categories) as f:
            lines = f.readlines()
        categories = [item.rstrip() for item in lines]
    else:  
        categories = [None] * file_categories
    n_class = len(categories)
    print('{}: {} classes'.format(dataset, n_class))
    return n_class, file_imglist_train, file_imglist_val, root_data, prefix