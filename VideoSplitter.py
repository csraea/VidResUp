# imports
import click
from os import path
from tqdm import tqdm
import cv2

# local imports
from src.video_editor import VideoEditor


@click.command()
@click.option('-i', '--input_path', required=True, type=click.Path(exists=True, readable=True),
              help='Path to input video')
@click.option('-o', '--output_path', default='../data/video_frames/', type=click.Path(writable=True),
              help='Path where video frames will be saved.')
@click.option('-c', '--count', default=1, type=click.Path(writable=True),
              help='Count of result frames: this param allows to limit count of result frames')
def main(input_path: str, output_path: str, count: int):
    """Split input video to frames"""

    print("Start reading input videos..")
    if not path.exists(input_path):
        print(f"File doesn't exist on a path {input_path}")
        return

    # obtain input video reader and it's params
    video_input, video_width, video_height, fps = VideoEditor.read_input_video_with_params(input_path)

    print(f"Start slicing video to frames to path {output_path}")

    # generate output img name
    output_img_name = input_path.split('/')[-1].split('.')[0]

    # proceed frames from input video.. display progress with progress-bar using 'tqdm'
    for iter in tqdm(range(int(count))):
        if iter >= int(count):
            break

        ret, frame_img = video_input.read()
        # skip current frame from input video if reader returns false
        if not ret:
            continue
        cv2.imwrite(output_path + output_img_name + "_frame%d.jpg" % (iter + 1), frame_img)  # save frame as JPG file
    print(f"Successfully obtain {count} frames from video.")


if __name__ == '__main__':
    main()
