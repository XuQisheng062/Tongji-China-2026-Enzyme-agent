import argparse
import pathlib
import string
import torch
from esm import  MSATransformer
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
import itertools
from typing import List, Tuple
from transformers import EsmTokenizer
from models.modeling_EnzGFM import EnzGFMForMaskedLM


def remove_insertions(sequence: str) -> str:
    """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
    # This is an efficient way to delete lowercase characters and insertion characters from a string
    deletekeys = dict.fromkeys(string.ascii_lowercase)
    deletekeys["."] = None
    deletekeys["*"] = None

    translation = str.maketrans(deletekeys)
    return sequence.translate(translation)


def read_msa(filename: str, nseq: int) -> List[Tuple[str, str]]:
    """ Reads the first nseq sequences from an MSA file, automatically removes insertions.

    The input file must be in a3m format (although we use the SeqIO fasta parser)
    for remove_insertions to work properly."""

    msa = [
        (record.description, remove_insertions(str(record.seq)))
        for record in itertools.islice(SeqIO.parse(filename, "fasta"), nseq)
    ]
    return msa


def create_parser():
    parser = argparse.ArgumentParser(
        description="Label a deep mutational scan with predictions from an ensemble of ESM-1v models."
    )
    parser.add_argument(
        "--model-location",
        default=['./EnzGFM-150M'],
        type=str,
        help="PyTorch model file OR name of pretrained model to download (see README for models)",
        nargs="+",
    )
    parser.add_argument(
        "--sequence",
        type=str,
        help="Base sequence to which mutations were applied",
    )
    parser.add_argument(
        "--input-dir",
        default='./DATA',
        type=pathlib.Path,
        help="Directory containing CSV files for deep mutational scan",
    )
    parser.add_argument(
        "--mutation-col",
        type=str,
        default="mutant",
        help="column in the deep mutational scan labeling the mutation as 'AiB'"
    )
    parser.add_argument(
        "--output-dir",
        default='./results',
        type=pathlib.Path,
        help="Directory to save output files",
    )
    parser.add_argument(
        "--offset-idx",
        type=int,
        default=0,
        help="Offset of the mutation positions in `--mutation-col`"
    )
    parser.add_argument(
        "--scoring-strategy",
        type=str,
        default="wt-marginals",
        choices=["wt-marginals", "pseudo-ppl", "masked-marginals"],
        help=""
    )
    parser.add_argument(
        "--msa-path",
        type=pathlib.Path,
        help="path to MSA in a3m format (required for MSA Transformer)"
    )
    parser.add_argument(
        "--msa-samples",
        type=int,
        default=400,
        help="number of sequences to select from the start of the MSA"
    )
    parser.add_argument("--nogpu", action="store_true", help="Do not use GPU even if available")
    return parser


def label_row(row, sequence, token_probs, tokenizer, offset_idx):
    wt, idx, mt = row[0], int(row[1:-1]) - offset_idx, row[-1]
    assert sequence[idx - 1] == wt, "The listed wildtype does not match the provided sequence"


    wt_encoded = tokenizer.convert_tokens_to_ids(wt)
    mt_encoded = tokenizer.convert_tokens_to_ids(mt)

    # add 1 for BOS
    score = token_probs[0, idx, mt_encoded] - token_probs[0, idx, wt_encoded]
    return score.item()


def compute_pppl(row, sequence, model, alphabet, offset_idx):
    wt, idx, mt = row[0], int(row[1:-1]) - offset_idx, row[-1]
    assert sequence[idx] == wt, "The listed wildtype does not match the provided sequence"

    # modify the sequence
    sequence = sequence[:idx] + mt + sequence[(idx + 1):]

    # encode the sequence
    data = [
        ("protein1", sequence),
    ]

    batch_converter = alphabet.get_batch_converter()

    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    wt_encoded, mt_encoded = alphabet.get_idx(wt), alphabet.get_idx(mt)

    # compute probabilities at each position
    log_probs = []
    for i in range(1, len(sequence) - 1):
        batch_tokens_masked = batch_tokens.clone()
        batch_tokens_masked[0, i] = alphabet.mask_idx
        with torch.no_grad():
            token_probs = torch.log_softmax(model(batch_tokens_masked.cuda())["logits"], dim=-1)
        log_probs.append(token_probs[0, i, alphabet.get_idx(sequence[i])].item())  # vocab size
    return sum(log_probs)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the deep mutational scan
    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = list(args.input_dir.glob('*.csv'))
    excel_files = list(args.input_dir.glob('*.xlsx'))
    input_files = csv_files + excel_files

    models = []
    tokenizer = EsmTokenizer.from_pretrained(
        './EsmTokenizer')

    for i, model_location in enumerate(args.model_location):
        model = EnzGFMForMaskedLM.from_pretrained(model_location)
        model.eval()
        if torch.cuda.is_available() and not args.nogpu:
            model = model.cuda()
        models.append(model)

    for input_file in input_files:
        output_file = args.output_dir / f"processed_{input_file.stem}.csv"

        if input_file.suffix.lower() == '.csv':
            df = pd.read_csv(input_file)
        else:  # .xlsx
            df = pd.read_excel(input_file)

        for i, model in enumerate(models):

            if isinstance(model, MSATransformer):
                print("MSA Transformer processing not implemented for multiple sequences")
                continue
            else:
                sequence_groups = df.groupby('Sequence')
                all_results = []
                for sequence, group in sequence_groups:
                    len_seq = len(sequence)
                    if len(sequence) > 1024:
                        continue
                    if args.scoring_strategy == "wt-marginals":
                        encoded_input = tokenizer.encode_plus(
                            sequence,
                            return_tensors="pt",
                        )
                        toks = encoded_input['input_ids'].to(device)
                        attention_mask = encoded_input['attention_mask'].to(device)

                        with torch.no_grad():
                            token_probs = torch.log_softmax(model(input_ids=toks, attention_mask=attention_mask).logits,
                                                            dim=-1)

                        # Process mutations for this sequence
                        results = group.apply(
                            lambda row: label_row(
                                row[args.mutation_col],
                                sequence,
                                token_probs,
                                tokenizer,
                                args.offset_idx,
                            ),
                            axis=1,
                        )

                    elif args.scoring_strategy == "masked-marginals":
                        encoded_input = tokenizer.encode_plus(
                            sequence,
                            return_tensors="pt",
                        )
                        batch_tokens = encoded_input['input_ids'].to(device)
                        attention_mask = encoded_input['attention_mask'].to(device)
                        all_token_probs = []

                        for j in tqdm(range(batch_tokens.size(1))):
                            batch_tokens_masked = batch_tokens.clone()
                            batch_tokens_masked[0, j] = tokenizer.mask_token_id
                            with torch.no_grad():
                                token_probs = torch.log_softmax(
                                    model(input_ids=batch_tokens_masked, attention_mask=attention_mask)["logits"],
                                    dim=-1
                                )
                            all_token_probs.append(token_probs[:, j])

                        token_probs = torch.cat(all_token_probs, dim=0).unsqueeze(0)

                        # Process mutations for this sequence
                        results = group.apply(
                            lambda row: label_row(
                                row[args.mutation_col],
                                sequence,
                                token_probs,
                                tokenizer,
                                args.offset_idx,
                            ),
                            axis=1,
                        )

                    # Store results for this sequence
                    idx_name = 'pred_' + str(i)
                    group[idx_name] = results
                    all_results.append(group)

                # Combine results from all sequences
                if all_results:
                    df = pd.concat(all_results, axis=0)

        # Save the results
        df.to_csv(output_file)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
