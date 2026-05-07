"""script to change a gif's loop count"""

import argparse

from PIL import Image
from PIL import ImageSequence



def set_gif_loop_forever(input_path: str, output_path: str, count: int = 0):
    gif = Image.open(input_path)

    frames = [
        frame.copy()
        for frame in ImageSequence.Iterator(gif)
    ]

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        loop=count,  # 0 = infinite loop
        duration=gif.info.get("duration", 40),
        disposal=2,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Change loop count of a gif")
    parser.add_argument("-c", "--count", required=True, type=int, help="Loop count (0 for infinite loop)")
    parser.add_argument("-i", "--input", required=True, help="Input gif path")
    parser.add_argument("-o", "--output", required=False, default=None, help="Output file path")
    args = parser.parse_args()

    
    set_gif_loop_forever(
        input_path=args.input,
        output_path=args.output,
        count=args.count
    )