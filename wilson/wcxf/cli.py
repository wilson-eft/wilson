import argparse
from wilson import wcxf
import sys
import logging
import os
import yaml
import pylha


def wcxf_cli():
    parser = argparse.ArgumentParser(description="Command line interface to manipulate WCxf files.")
    subparsers = parser.add_subparsers(title='subcommands')

    # convert

    parser_convert = subparsers.add_parser('convert',
                                           description="Command line script to convert WCxf files between YAML and JSON.",
                                           help="convert between YAML and JSON formats")
    parser_convert.add_argument("FORMAT", type=str,
                                help="Output format (should be yaml or json)")
    parser_convert.add_argument("FILE", nargs='?', type=argparse.FileType('r'),
                                default=sys.stdin,
                                help="Input file. If \"-\", read from standard input")
    parser_convert.add_argument("--output", nargs='?',
                                      type=argparse.FileType('w'),
                                      default=sys.stdout,
                                      help="Output file. If absent, print to standard output")
    parser_convert.set_defaults(func=convert)

    # translate

    parser_translate = subparsers.add_parser('translate',
                                             description="Command line script for basis translation of WCxf files.",
                                             help="Translate between different bases")
    parser_translate.add_argument("BASIS", help="Output basis", type=str)
    parser_translate.add_argument("FILE", nargs='?',
                                  type=argparse.FileType('r'),
                                  default=sys.stdin,
                                  help="Input file. If \"-\", read from standard input")
    parser_translate.add_argument("--output", nargs='?',
                                  type=argparse.FileType('w'), default=sys.stdout,
                                  help="Output file. If absent, print to standard output")
    parser_translate.add_argument("--format", type=str,
                                  default="json",
                                  help="Output format (default: json)")
    parser_translate.set_defaults(func=translate)

    # match

    parser_match = subparsers.add_parser('match',
                                         description="Command line script for matching of WCxf files.",
                                         help="Match between different EFTs")
    parser_match.add_argument("EFT", help="Output EFT", type=str)
    parser_match.add_argument("BASIS", help="Output basis", type=str)
    parser_match.add_argument("FILE", nargs='?',
                              type=argparse.FileType('r'), default=sys.stdin,
                              help="Input file. If \"-\", read from standard input")
    parser_match.add_argument("--output", nargs='?',
                              type=argparse.FileType('w'), default=sys.stdout,
                              help="Output file. If absent, print to standard output")
    parser_match.add_argument("--format", type=str, default="json",
                              help="Output format (default: json)")
    parser_match.set_defaults(func=match)

    # validate

    parser_validate = subparsers.add_parser('validate',
                                            description="Command line script for validation of WCxf files.",
                                            help="Validate basis or Wilson coefficient files")
    parser_validate.add_argument("TYPE", type=str,
                                       help="Type of file to validate: should be 'eft', 'basis', or 'wc'")
    parser_validate.add_argument("FILE", nargs='?',
                                 type=argparse.FileType('r'), default=sys.stdin,
                                 help="Input file. If \"-\", read from standard input")
    parser_validate.set_defaults(func=validate)

    args = parser.parse_args()
    try:
        args.func(args)
    except AttributeError:
        parser.print_help()


def convert(args):
    from wilson.wcxf.converters.yamljson import convert_json, convert_yaml
    if args.FORMAT.lower() == 'json':
        convert_json(args.FILE, args.output)
    if args.FORMAT.lower() == 'yaml':
        convert_yaml(args.FILE, args.output)


def translate(args):
    wc_in = wcxf.WC.load(args.FILE)
    wc_out = wc_in.translate(args.BASIS)
    wc_out.dump(stream=args.output, fmt=args.format)


def match(args):
    wc_in = wcxf.WC.load(args.FILE)
    wc_out = wc_in.match(args.EFT, args.BASIS)
    wc_out.dump(stream=args.output, fmt=args.format)


def validate(args):
    if args.TYPE == 'eft':
        eft = wcxf.EFT.load(args.FILE)
    elif args.TYPE == 'basis':
        basis = wcxf.Basis.load(args.FILE)
        basis.validate()
    elif args.TYPE == 'wc':
        wc = wcxf.WC.load(args.FILE)
        wc.validate()
    else:
        logging.error("TYPE should be 'eft', 'basis', or 'wc'")
        return 1
    print("Validation successful.")
    return 0


def eos():
    from wilson.wcxf.converters.eos import wcxf2eos, get_sm_wcs
    parser = argparse.ArgumentParser(description="""Command line script to convert a WCxf file to an EOS Wilson coefficient parameter file.""",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("FILE", nargs='?', help="Input file. If \"-\", read from standard input",
                        type=argparse.FileType('r'), default=sys.stdin)
    parser.add_argument("--eosprefix", help="Installation prefix for the EOS installation. Defaults to /usr",
                        default='/usr')
    parser.add_argument("--output", nargs='?', help="Output file. If absent, print to standard output",
                        type=argparse.FileType('w'), default=sys.stdout)
    parser.add_argument("--eoshome", help="EOS home directory. If specified, values will be written to EOSHOME/parameters/wcxf.yaml. Cannot be used simultaneously with output",
                        default=None)
    args = parser.parse_args()
    # check sanity of inputs
    if args.output != sys.stdout and args.eoshome is not None:
        logging.error("Cannot use --output and --eoshome arguments simultaneously")
        return 1
    elif args.output == sys.stdout and args.eoshome is not None:
        output_dir = os.path.join(args.eoshome, 'parameters')
        if not os.path.isdir(output_dir):
            logging.error("Output directory {} does not exist".format(output_dir))
            return 1
        f = open(os.path.join(output_dir, 'wcxf.yaml'), 'w')
    else:
        f = args.output
    # read in & validate WCxf file
    wc = wcxf.WC.load(args.FILE)
    wc.validate()
    # read EOS SM contributions
    sm_wc_dict = get_sm_wcs(os.path.join(args.eosprefix, 'share/eos', 'parameters'))
    # convert to EOS parameters
    eos_dict = wcxf2eos(wc, sm_wc_dict)
    yaml.dump(eos_dict, f, default_flow_style=False)
    f.close()
    return 0


def smeftsim():
    from wilson.wcxf.converters.smeftsim import initialize_smeftsim_card, smeftsim_card_fill, smeftsim_card_text
    parser = argparse.ArgumentParser(description="""Command line script to convert a WCxf file to a MadGraph param_card file for SMEFTsim.""",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("FILE", nargs='?', help="Input file. Must be specified.",
                        type=argparse.FileType('r'), default="{}")
    parser.add_argument("--output", nargs='?', help="Output file. Default is wcxf2smeftsim_param_card.dat.", default="wcxf2smeftsim_param_card.dat")
    parser.add_argument("--input-scheme", nargs='?', help="Input parameters set. Can be either alpha (alpha_ew, m_Z, G_F) or mw (m_W, m_Z, G_F). Default is alpha.", choices=['alpha','mw'], default='alpha')
    parser.add_argument("--cutoff-scale", nargs='?', help="Value of the EFT cutoff scale in GeV. Default is 1 TeV.", type=float, default=1000)
    parser.add_argument("--model-set", nargs='?', help="SMEFTsim model set to be used. Can be either A or B, default is A.", choices=['A','B'],default="A")

    args = parser.parse_args()


    # read in & validate WCxf file
    wc = wcxf.WC.load(args.FILE)
    wc.validate()
    # check that the input is in the Warsaw mass basis. quit otherwise.
    if wc.basis != "Warsaw mass":
      print('''
WARNING: The input file must be in the 'Warsaw mass' basis!
Please translate into 'Warsaw mass' before converting to SMEFTsim.

Press 'i' to ignore this warning or any other key to exit.''')

      key = input()
      if key != 'i' and key != 'I': quit()

    f = open(args.output, 'w')

    # initialize and fill the dictionary for the param_card
    card = initialize_smeftsim_card(args.model_set)
    card_filled = smeftsim_card_fill(card, wc, args.model_set, args.cutoff_scale, args.input_scheme)

    # write output file
    f.write(smeftsim_card_text(args.model_set, args.input_scheme)[0])
    pylha.dump(card_filled, fmt='lha', stream = f)
    f.write(smeftsim_card_text(args.model_set, args.input_scheme)[1])
    f.close()
    return 0


def wcxf2dsixtools():
    from wilson.wcxf.converters import dsixtools
    parser = argparse.ArgumentParser(description="""Command line script to convert a WCxf file to a DsixTools Wilson coefficient file.""",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("FILE", nargs='?', help="Input file. If \"-\", read from standard input",
                        type=argparse.FileType('r'), default=sys.stdin)
    parser.add_argument("--output", nargs='?', help="Output file. If absent, print to standard output",
                        type=argparse.FileType('w'), default=sys.stdout)
    args = parser.parse_args()
    wc = wcxf.WC.load(args.FILE)
    wc.validate()
    dsixtools.wcxf2dsixtools(wc, stream=args.output)
    return 0


def dsixtools2wcxf():
    from wilson.wcxf.converters import dsixtools
    parser = argparse.ArgumentParser(description="""Command line script to convert DsixTools output files to a WCxf file.""",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("FILE", nargs='+', help="Input file(s).",
                        type=argparse.FileType('r'), default=sys.stdin)
    parser.add_argument("--output", nargs='?', help="Output file. If absent, print to standard output",
                        type=argparse.FileType('w'), default=sys.stdout)
    args = parser.parse_args()
    dsixtools.dsixtools2wcxf(tuple(f for f in args.FILE), stream=args.output)
    return 0
