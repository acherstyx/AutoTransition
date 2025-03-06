import argparse

from autotransition.data.io import glob_video_files, convert_tasker

parser = argparse.ArgumentParser(description="Convert all videos in a specified directory to rgb frames")

parser.add_argument("dir", help="video directory")
parser.add_argument("--extensions", "-e", help="file extension", default=("mp4", "avi"))
parser.add_argument("-o", "--overwrite", "--force", default=False, action="store_true",
                    help="overwrite existing results")
parser.add_argument("-d", "--delete", default=False, action="store_true",
                    help="delete videos after converting")
parser.add_argument("--resize", "-r", default=320, type=int, help="resize images before saving")
parser.add_argument("-P", "--process", default=1, type=int)

if __name__ == '__main__':
    args = parser.parse_args()

    video_files = glob_video_files(args.dir, extension=args.extensions)

    convert_tasker(video_files,
                   overwrite=args.overwrite,
                   delete_video=args.delete,
                   resize_size=args.resize,
                   num_workers=args.process)
