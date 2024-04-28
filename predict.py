"""
"""
import os
import argparse
import torch
import esm
from DeepTMpred.model import GAUCRFNet
from DeepTMpred.utils import tmh_predict
import logging
from torch.utils.data import DataLoader
from DeepTMpred.data import OPMDataset, collate_fn
from minlora import add_lora, merge_lora, LoRAParametrization
from functools import partial
import json



#####################################
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create a file handler
handler = logging.FileHandler('deeptmpred-2.log')
handler.setLevel(logging.INFO)

# create a logging format
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# add the file handler to the logger
logger.addHandler(handler)
#####################################


def get_params():
    parser = argparse.ArgumentParser('DeepTMpred V2')

    parser.add_argument("--input", type=str, help='input fasta')

    parser.add_argument("--output", type=str, help='json format TMH file')

    args, _ = parser.parse_known_args()
    return args


def predict(pretrain_model, model, batch_converter, loader, device):
    pretrain_model.eval()
    model.eval()
    with torch.no_grad():
        for item in loader:
            batch_labels, batch_strs, batch_tokens = batch_converter(item["seq"])
            seq_ids = [pair[0] for pair in item["seq"]]
            lengths = item["seq_len"].to(device)
            labels = item["label"].to(device)
            orientation = item["orientation"].to(device)
            batch_tokens = batch_tokens.to(device)
            with torch.autocast(device_type="cuda"):
                results = pretrain_model(batch_tokens, repr_layers=[33])
                token_representations = results["representations"][33][:,1:-1]
                out, prob, orientation = model(token_representations, lengths)
    tmh_dict = tmh_predict(seq_ids, out, prob, orientation.cpu().tolist())
    return tmh_dict

def main(args):

    ###############
    input_file = args["input"]
    output_file = args["output"]

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ###############

    ##############
    # setup_seed(args['seed'])
    ##############
    lora_config = {
        torch.nn.Embedding: {
            "weight": partial(LoRAParametrization.from_embedding, rank=4),
        },
        torch.nn.Linear: {
            "weight": partial(LoRAParametrization.from_linear, rank=4),
        },
    }

    logger.info("init model")
    model_state_dict = torch.load("./deeotmpred2.pt",map_location='cpu')
    pretrain_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    add_lora(pretrain_model, lora_config=lora_config)
    _ = pretrain_model.load_state_dict(model_state_dict["lora_state_dict"], strict=False)
    merge_lora(pretrain_model)
    pretrain_model = pretrain_model.to(device)

    model = GAUCRFNet(**model_state_dict["model_args"])
    model = model.to(device)

    # -------------
    logger.info("load dataset....")
    data = OPMDataset(input_file)
    data_loader = DataLoader(data, batch_size=1, collate_fn=collate_fn)
    tmh_dict = predict(pretrain_model, model, batch_converter, data_loader, device)
    json.dump(open(output_file, 'w+'), tmh_dict)

if __name__ == "__main__":
    try:
        # hyper_param = json.load(open("./best_hyper_paramter.json", 'r'))
        params = vars(get_params())
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise
