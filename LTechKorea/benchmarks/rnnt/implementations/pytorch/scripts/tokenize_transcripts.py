import argparse
import json
import pickle
import sentencepiece as spm


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help='Path to the sentencepiece model')
    parser.add_argument("JSONS", nargs='+', help='Json files with .[].transcript field')
    parser.add_argument("--output_dir", default=None,
                        help='If set, output files will be in a different directory')
    parser.add_argument("--suffix", default='-tokenized', help='Suffix added to the output files')
    parser.add_argument("--output_format", default='pkl', choices=['pkl', 'json'], help='Output format')
    return parser

def get_model(model_path):
    return spm.SentencePieceProcessor(model_file=model_path)

def get_outputs(inputs, output_dir, suffix, output_format):
    fnames = (i[:-len('.json')].rsplit('/', maxsplit=1) for i in inputs)
    return [f'{output_dir or dirname}/{fname}{suffix}.{output_format}' for dirname, fname in fnames]

def transform(model, inputs, outputs, output_format):
    for i, o in zip(inputs, outputs):
        with open(i, 'r') as f:
                j = json.load(f)
        if output_format == "json":
            for entry in j:
                entry['tokenized_transcript'] = model.encode(entry['transcript'])
            with open(o, 'w') as f:
                json.dump(j, f)
        else:
            pruned_j = []
            for entry in j:
                pruned_entry = {    "fname": entry['files'][-1]['fname'],
                                    "original_duration": entry["original_duration"],
                                    "tokenized_transcript": model.encode(entry['transcript'])}
                pruned_j.append(pruned_entry)
            with open(o, 'wb') as f:
                pickle.dump(pruned_j, f)


def main():
    args = get_parser().parse_args()
    model = get_model(args.model)
    outputs = get_outputs(args.JSONS, args.output_dir, args.suffix, args.output_format)
    transform(model, args.JSONS, outputs, args.output_format)


if __name__ == '__main__':
    main()
