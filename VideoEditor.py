import cv2
from typing import Tuple


class VideoEditor:
    @classmethod
    def read_input_video_with_params(cls, path_to_input_video_file: str) \
            -> Tuple[cv2.VideoCapture, float, float, int]:
        video_input = cv2.VideoCapture(path_to_input_video_file)
        # read video params
        video_width = video_input.get(cv2.CAP_PROP_FRAME_WIDTH)
        video_height = video_input.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = video_input.get(cv2.CAP_PROP_FPS)

        return video_input, video_width, video_height, fps

    @classmethod
    def get_video_writer_for_output(cls, path_to_output_video_file: str, fps: int, codec: str = 'mp4v',
                                    resolution: Tuple = (1920, 1080)) -> cv2.VideoWriter:
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*codec)
        video_output_writer = cv2.VideoWriter(path_to_output_video_file, fourcc, fps,
                                              (resolution[0], resolution[1]))

        return video_output_writer
