# imports
import click
import os
from tqdm import tqdm
import cv2

# local imports
from src.video_editor import VideoEditor

@click.command()
@click.option('-i', '--input_path', required=True, type=click.Path(exists=True, readable=True),
              help='Path to to directory with input frames.')
@click.option('-o', '--output_path', default='../data/videos_from_frames/video_from_frames.mp4', type=click.Path(writable=True),
              help='Path to directory where result video will be saved.')
def main(input_path: str, output_path: str):
    """Union frames from input directory to a single video"""

    print("Start reading directory with frames..")
    if not os.path.exists(input_path):
        print(f"Directory doesn't exists for path {input_path}")
        return

    input_frames = [img for img in os.listdir(input_path) if img.endswith(".jpg") or img.endswith(".png")]
    frame = cv2.imread(os.path.join(input_path, input_frames[0]))
    # obtain output video params
    height, width, layers = frame.shape

    video_output = VideoEditor.get_video_writer_for_output(
        path_to_output_video_file=output_path,
        fps=20.0,
        resolution=(int(width), int(height))
    )

    print(f"Start writing output video to path {output_path}")
    for input_frame in input_frames:
        video_output.write(cv2.imread(os.path.join(input_path, input_frame)))

    cv2.destroyAllWindows()
    video_output.release()

    print("Done.")


if __name__ == '__main__':
    main()
