import argparse
from fractions import Fraction
from io import BytesIO
import json
from pathlib import Path
import struct
import sys

import av


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    header = struct.Struct('!Id')
    with av.open(settings['video'], 'w', options={'movflags': '+faststart'}) as output:
        stream = output.add_stream(settings['codec'], rate=settings['frame_rate'])
        stream.width = settings['viewport']['width']
        stream.height = settings['viewport']['height']
        stream.pix_fmt = settings['pixel_format']
        stream.options = settings['encoder_options']
        while data := sys.stdin.buffer.read(header.size):
            length, elapsed = header.unpack(data)
            with av.open(BytesIO(sys.stdin.buffer.read(length))) as image:
                frame = next(image.decode(video=0))
            frame.pts = round(elapsed * settings['frame_rate'])
            frame.time_base = Fraction(1, settings['frame_rate'])
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


if __name__ == '__main__':
    main()
