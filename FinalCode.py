import time
import datetime
#initial = time.time()
import matplotlib
matplotlib.use('Agg')
import os, sys
import yaml
from argparse import ArgumentParser
from tqdm import tqdm
#from gtts import gTTS
import pyttsx3

import sda
import imageio
import os
import ffmpeg
import sys
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
import torch
from sync_batchnorm import DataParallelWithCallback

from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from animate import normalize_kp
from scipy.spatial import ConvexHull
from gTTS import synthesize_text

import warnings
warnings.filterwarnings("ignore")

#Loading the checkpoints and the config files
def load_checkpoints(config_path, checkpoint_path, cpu=True):

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    if not cpu:
        generator.cuda()

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    if not cpu:
        kp_detector.cuda()
    
    if cpu:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path)
 
    generator.load_state_dict(checkpoint['generator'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])
    
    if not cpu:
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)

    generator.eval()
    kp_detector.eval()
    
    return generator, kp_detector

#Making Animations
def make_animation(source_image, driving_video, generator, kp_detector, relative=True, adapt_movement_scale=True, cpu=True):
    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        if not cpu:
            source = source.cuda()
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
        kp_source = kp_detector(source)
        kp_driving_initial = kp_detector(driving[:, :, 0])

        for frame_idx in tqdm(range(driving.shape[2])):
            driving_frame = driving[:, :, frame_idx]
            if not cpu:
                driving_frame = driving_frame.cuda()
            kp_driving = kp_detector(driving_frame)
            kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                   kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                   use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
            out = generator(source, kp_source=kp_source, kp_driving=kp_norm)

            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
    return predictions

def tts(opt):
    f = open("C:/Users/Moiz/Desktop/Jubi Work Local/Project - Final/inputs/test.txt","r")
    if str(opt.voice).lower() == "male":
        synthesize_text(f.read(),"B")
    else:
        synthesize_text(f.read(),"A")

def facial_animation():
    va = sda.VideoAnimator(gpu=0, model_path="timit")# Instantiate the animator
    vid, aud = va("config/image.bmp", "inputs/audio.wav")
    va.save_video(vid, aud, "output/generated.mp4")
    os.remove('inputs/audio.wav')


def fomm(opt):
    source_image = imageio.imread(opt.source_image)
    config = 'config/vox-256.yaml'
    checkpoint = 'checkpoint/vox-cpk.pth.tar'
    result_video = 'output/intermediate.mp4'

    source_video = "output/generated.mp4"    
    reader = imageio.get_reader(source_video)
    input = ffmpeg.input(source_video)
    audioStream = input.audio
    fps = reader.get_meta_data()['fps']
    driving_video = []
    try:
        for im in reader:
            driving_video.append(im)
    except RuntimeError:
        pass
    reader.close()

    source_image = resize(source_image, (256, 256))[..., :3]
    driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]
    generator, kp_detector = load_checkpoints(config_path=config, checkpoint_path=checkpoint, cpu=True)
    #loading = time.time()
    predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=True, adapt_movement_scale=False, cpu=True)
    #predict = time.time()
    imageio.mimsave(result_video, [img_as_ubyte(frame) for frame in predictions], fps=fps)
    videoStream = ffmpeg.input(result_video)
    currenttime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    out = ffmpeg.output(audioStream, videoStream, 'output/final_' + currenttime + '.mp4')
    ffmpeg.run(out)
    os.remove('output/intermediate.mp4')
    os.remove('output/generated.mp4')
    #end = time.time()

#Main function
if __name__ == "__main__":
    #start = time.time()
    parser = ArgumentParser()
    parser.add_argument("--source_image", default='inputs/new_face_1.jpeg', help="path to source image")
    parser.add_argument("--voice", default='male', help="choose gender")
    opt = parser.parse_args()

    tts(opt)
    print("Text to Speech Complete - Audio Saved")
    facial_animation()
    print("Facial Animation Creation Complete")
    print("Morphing created Facial Animations on input image.......")
    fomm(opt)
    print("Process Complete, Video Ready!")

    # print("======================Time Statistics======================")
    # print("Importing Libraries and Modules: " + str(round(start-initial,2)) + "s")
    # print("Preprocessing and Loading Checkpoint: " + str(round(loading-start,2)) + "s")
    # print("Predictions: " + str(round(predict-loading,2)) + "s")
    # print("Video Editing: " + str(round(end-predict,2)) + "s")
    # print("Total Time: " + str(round(end-initial,2)) + "s")

        

    
