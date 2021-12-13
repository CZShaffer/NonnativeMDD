import torch
import speechbrain as sb
import models.sinc_ctc_att as sca
import models.cnn_rnn_ctc as crnn
from hyperpyyaml import load_hyperpyyaml


def train_cnnrnnctc()::

    hparams_file = "models/cnn_rnn_ctc.yaml"
    overrides = ""
    run_opts = {"device": 'cuda:0'}

    # Load hyperparameters file
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Create Dataset
    train_data, valid_data, test_data, label_encoder = dataio_prep(hparams)

    # Trainer initialization
    asr_brain = crnn(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
    )

    asr_brain.label_encoder = label_encoder

    # Training/validation loop
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # Test
    asr_brain.evaluate(
        test_data,
        min_key="PER",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )

def train_sincnet():
    torch.autograd.set_detect_anomaly(True)
    hparams_file = "models/sinc_ctc_att.yaml"
    overrides = ""
    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
    )

    print("prepping data")
    # Dataset IO prep: creating Dataset objects and proper encodings for phones
    train_data, valid_data, test_data, label_encoder = sca.dataio_prep(hparams)
    run_opts = {"device": 'cuda:0'}

    asr_brain = sca.sinc_ctc_att(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    asr_brain.label_encoder = label_encoder

    def conv16(model):
        if not hasattr(model, '__iter__'):
            model.half()
            return
        for module in model:
            if not isinstance(module, sb.processing.features.InputNormalization) and not isinstance(module, sb.nnet.linear.Linear) and not isinstance(module, sb.nnet.CNN.SincConv) and not isinstance(module, sb.nnet.RNN.AttentionalRNNDecoder):
                module.half()
                for child in module.children():
                    conv16(child)
    # conv16(asr_brain.modules.values())

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
    train_cnnrnnctc()
    train_sincnet()