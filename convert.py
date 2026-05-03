"""
convert mp4 to gif to be used in markdowns etc.
# CAREFUL: high ram usage in compression
"""


import os
import argparse
from scripts.mp4_to_gif import convert


if __name__ == "__main__":
    # read fn from arg -i and outfn from arg -o
    parser = argparse.ArgumentParser(description="Convert mp4 to gif")
    parser.add_argument("-i", "--input", required=True, help="Input mp4 file path")
    parser.add_argument("-o", "--output", required=False, default=None, help="Output gif file path")
    args = parser.parse_args()

    fn = args.input
    outfn = args.output if args.output else os.path.splitext(fn)[0] + ".gif"
    print(fn, outfn)
    convert(
        fn=fn,
        outfn=outfn,
        compression=75
    )