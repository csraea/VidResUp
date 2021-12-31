# imports
import click
from os import path
from tqdm import tqdm
import cv2

# local imports
from src.video_editor import VideoEditor


@click.command()
@click.option('-i', '--input_path', required=True, type=click.Path(exists=True, readable=True),
              help='Path to input high resolution video')
@click.option('-o', '--output_path', default='../data/downscaled_videos/output_downscaled.mp4', type=click.Path(writable=True),
              help='Path where downscaled video will be saved.')
def main(input_path: str, output_path: str):
    """Downscale input video 4 times"""

    print("Start reading input videos..")
    if not path.exists(input_path):
        print(f"File doesn't exist on a path {input_path}")
        return

    # obtain input video reader and it's params
    video_input, video_width, video_height, fps = VideoEditor.read_input_video_with_params(input_path)

    video_output = VideoEditor.get_video_writer_for_output(
        path_to_output_video_file=output_path,
        fps=fps,
        resolution=(int(video_width/4), int(video_height/4))
    )

    print(f"Start writing output downscaled video to path {output_path}")
    # proceed all frames from input video.. display progress with progress-bar using 'tqdm'
    for i in tqdm(range(int(video_input.get(cv2.CAP_PROP_FRAME_COUNT)))):
        ret, frame_img = video_input.read()
        if not ret:
            continue

        # downscale
        downscaled = cv2.resize(frame_img, (int(video_width/4), int(video_height/4)), interpolation=cv2.INTER_AREA)
        # write output
        video_output.write(downscaled)


if __name__ == '__main__':
    main()
