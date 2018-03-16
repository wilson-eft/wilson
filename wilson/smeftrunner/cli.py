import argparse
from smeftrunner import SMEFT

def main():
    parser = argparse.ArgumentParser(description="Command line interface "
        "for the SMEFTrunner Python package for the SMEFT RG evolution from a "
        "high (UV) scale to the electroweak (EW) scale.")
    parser.add_argument("highscale", help="UV scale = initial scale", type=float)
    parser.add_argument("lowscale", help="EW scale = output scale", type=float)
    parser.add_argument("file", help="Input file(s)", type=str, nargs='*')
    parser.add_argument("--output", help="Output file", type=str)
    args = parser.parse_args()
    smeft = SMEFT()
    smeft.load_initial(tuple(open(f, 'r') for f in args.file))
    smeft.scale_in = args.highscale
    smeft.scale_high = args.highscale
    C_out = smeft.rgevolve(scale_out=args.lowscale)
    if args.output is None:
        print(smeft.dump(C_out, scale_out=args.lowscale))
    else:
        with open(args.output, 'w') as f:
            smeft.dump(C_out, scale_out=args.lowscale, stream=f)
