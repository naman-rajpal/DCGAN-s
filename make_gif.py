import os
import glob
from PIL import Image
from natsort import natsorted

def make_gif(frame_folder):
    frames = [Image.open(image) for image in glob.glob(f"{frame_folder}/*.png")]
    frame_one = frames[0]
    frame_one.save("./pokemons_result.gif", format="GIF", append_images=frames,
               save_all=True, duration=100, loop=0)

make_gif("./temp")