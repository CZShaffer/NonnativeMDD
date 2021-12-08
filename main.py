import torch
import speechbrain as sb
import models.sinc_ctc_att as sca
from hyperpyyaml import load_hyperpyyaml
import os
import torchaudio

def train():
    pass

def test():
    pass

def main():
    pass

def train_sincnet():
    # CLI:
    # hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    # print("timit prep")
    # from data import timit_prepare
    # timit_prepare.prepare_timit("data/timit/data", "timit_train.json", "timit_valid.json", "timit_test.json", uppercase=True)
    print("torchaudio test")
    audio, _ = torchaudio.load('data/timit/data/TRAIN/DR8/MEJS0/SX70.WAV')
    print(f"audio: {audio}")
    hparams_file = "models/sinc_ctc_att.yaml"
    overrides = ""
    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    print(os.getcwd())
    # Dataset prep (parsing TIMIT and annotation into csv files)
    # from timit_prepare import prepare_timit  # noqa

    # # Initialize ddp (useful only for multi-GPU DDP training)
    # sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        #overrides=overrides,
    )

    # multi-gpu (ddp) save data preparation
    # run_on_main(
    #     prepare_timit,
    #     kwargs={
    #         "data_folder": hparams["data_folder"],
    #         "save_json_train": hparams["train_annotation"],
    #         "save_json_valid": hparams["valid_annotation"],
    #         "save_json_test": hparams["test_annotation"],
    #         "skip_prep": hparams["skip_prep"],
    #     },
    # )

    print("prepping data")
    # Dataset IO prep: creating Dataset objects and proper encodings for phones
    train_data, valid_data, test_data, label_encoder = sca.dataio_prep(hparams)
    # print(train_data[0])
    for i in range(10):
        s = train_data[i]["sig"]
        print(f"train_data[{i}] shape: {s.shape}")
    # Trainer initialization
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_opts = {"device": 'cuda:0'}

    asr_brain = sca.sinc_ctc_att(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    asr_brain.label_encoder = label_encoder
    print("starting training loop")
    # Training/validation loop
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )
    print("starting test")
    # Test
    asr_brain.evaluate(
        test_data,
        min_key="PER",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )

if __name__ == "__main__":
    train_sincnet()