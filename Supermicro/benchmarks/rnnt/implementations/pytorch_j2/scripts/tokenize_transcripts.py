import argparse
import json

import sentencepiece as spm


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help='Path to the sentencepiece model')
    parser.add_argument("JSONS", nargs='+', help='Json files with .[].transcript field')
    parser.add_argument("--output_dir", default=None,
                        help='If set, output files will be in a different directory')
    parser.add_argument("--suffix", default='-tokenized', help='Suffix added to the output files')
    return parser

def get_model(model_path):
    return spm.SentencePieceProcessor(model_file=model_path)

def get_outputs(inputs, output_dir, suffix):
    fnames = (i[:-len('.json')].rsplit('/', maxsplit=1) for i in inputs)
    return [f'{output_dir or dirname}/{fname}{suffix}.json' for dirname, fname in fnames]

def transform(model, inputs, outputs):
    for i, o in zip(inputs, outputs):
        with open(i, 'r') as f:
            j = json.load(f)
        for entry in j:
            entry['tokenized_transcript'] = model.encode(entry['transcript'])
        with open(o, 'w') as f:
            json.dump(j, f)

def main():
    args = get_parser().parse_args()
    model = get_model(args.model)
    outputs = get_outputs(args.JSONS, args.output_dir, args.suffix)
    transform(model, args.JSONS, outputs)


if __name__ == '__main__':
    main()
