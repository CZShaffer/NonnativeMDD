import torch
import torch.nn as nn
import speechbrain as sb
import speechbrain.nnet.CNN as cnn
import os
from hyperpyyaml import load_hyperpyyaml

# class sinc_ctc_att(nn.Module):
#     def __init__(self, device="cpu"):
#         super(sinc_ctc_att, self).__init__()
#         self.device = device
#         self.full_stack = nn.Sequential()
#         # https://speechbrain.readthedocs.io/en/latest/API/speechbrain.nnet.CNN.html?highlight=sincnet
#         # conv = SincConv(input_shape=inp_tensor.shape, out_channels=25, kernel_size=11)
#
#         '''
#         Encoder: an encoder module that is shared across CTC and the attention-based model. The
#         encoder module extracts S-length high-level encoder state
#         sequence H = (h_1, … , h_s") from a T-length acoustic features
#         X = (x_1, … , x_t) through a stack of convolutional and/or
#         recurrent networks, where S ≤ T is due to downsampling;
#
#         Attention: an attention module that calculates a fixed-length context
#         vector c_l by summarizing the output of the encoder module
#         at each output step for l ∈ [1, … , L], finding out relevant
#         parts of the encoder state sequence to be attended for
#         predicting an output phone symbol y_l , where the output
#         symbol sequence y = (y_1, … , y_L) belongs to a canonical
#         phone set U;
#
#         Decoder: Given the context vector c_l and the history of
#         partial diagnostic results y_1:l-1, a decoder module updates its
#         hidden state q_l autoregressively and estimates the next phone
#         symbol y_1 ;
#
#         CTC: The CTC module offers another diagnostic
#         results based on the frame-level alignment between the input
#         sequence X and the canonical phone symbol sequences y by
#         introducing a special <blank> token. It can substantially
#         reduce irregular alignments during the training and test
#         phases.
#         '''
#
#         #Encoder:
#         input_shape = inp_tensor.shape
#         out_channels = 25
#         kernel_size = 3
#
#         conv = cnn.SincConv(input_shape=input_shape, out_channels=25, kernel_size=11)
#
#         #Attention
#         embed_dim = 0
#         num_heads = 0
#         att_layer = torch.nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0, device=device)
#
#         #Decoder
#
#         #CTC
#
#     def forward(self, x):
#         #encoder
#         #attention
#         #decoder
#         #ctc
#         criterion = nn.CTCLoss(blank=40).to(self.device)
#         pass

class sinc_ctc_att(sb.Brain):
    def compute_forward(self, batch, stage):
        '''
        TODO: IMPLEMENT THIS

        Our baseline E2E MDD models built on the hybrid CTCATT model. The
        encoder network is composed of the VGGbased deep CNN component plus a bidirectional LSTM
        component with 1024 hidden units [19], which takes input the
        hand-crafted acoustic features, such as Mel-filterbank outputs
        (FBANK) or MFCCs. The decoder network consists of twolayer unidirectional-LSTM
        with 1024 cells. As to the handcrafted acoustic features, FBANK is 80-dimensional while
        MFCCs 40-dimensional. Both of them were extracted from
        waveform signals with a hop size of 10 ms and a window size
        of 25 ms, and further normalized with the global mean and
        variance. When taking input raw waveform signals
        alternatively, the SincNet module is tacked in front of the
        encoder network. SincNet module was based on the
        configuration suggested in [18], which is made of an array of
        parametrized sinc-functions in the first layer, followed by two
        one-dimensional convolutional layers. Further, each layer of
        SincNet has 80, 128 and 128 filters and kernel size are set to
        be 251, 3 and 3, respectively.
        '''
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        phns_bos, _ = batch.phn_encoded_bos
        # feats = self.hparams.compute_features(wavs)
        # feats = self.modules.normalize(feats, wav_lens)
        # x = self.modules.enc(feats)
        #
        # # output layer for ctc log-probabilities
        # logits = self.modules.ctc_lin(x)
        # p_ctc = self.hparams.log_softmax(logits)
        #
        # e_in = self.modules.emb(phns_bos)
        # #see yaml for specific attention + decoder mechanism
        # h, _ = self.modules.dec(e_in, x, wav_lens)

        #TODO: verify
        # feats = self.hparams.compute_features(wavs)
        # feats = self.modules.normalize(feats, wav_lens)
        # print(f"wavs: {wavs}")
        # print(f"wavs.shape: {wavs.shape}")
        # print(f"feats.shape: {feats.shape}")
        feats_sinc = self.modules.sincconv_raw(wavs)
        feats_conv1d_1 = self.modules.conv1d_1(feats_sinc)
        feats_conv1d_2 = self.modules.conv1d_2(feats_conv1d_1)

        x, _ = self.modules.encoder(feats_conv1d_2)
        # print(f"x: {x}")
        # print(f"len(x): {len(x)}")
        # print(f"x.shape: {x.shape}")
        # output layer for ctc log-probabilities
        logits = self.modules.ctc_lin(x)
        p_ctc = self.hparams.log_softmax(logits)

        e_in = self.modules.embed(phns_bos)
        # see yaml for specific attention + decoder mechanism
        h, _ = self.modules.decoder(e_in, x, wav_lens)
        #end TODO

        # output layer for seq2seq log-probabilities
        logits = self.modules.seq_lin(h)
        p_seq = self.hparams.log_softmax(logits)

        if stage == sb.Stage.VALID:
            hyps, scores = self.hparams.greedy_searcher(x, wav_lens)
            return p_ctc, p_seq, wav_lens, hyps

        elif stage == sb.Stage.TEST:
            hyps, scores = self.hparams.beam_searcher(x, wav_lens)
            return p_ctc, p_seq, wav_lens, hyps

        return p_ctc, p_seq, wav_lens

    def compute_objectives(self, predictions, batch, stage):
        "Given the network predictions and targets computed the NLL loss."
        if stage == sb.Stage.TRAIN:
            p_ctc, p_seq, wav_lens = predictions
        else:
            p_ctc, p_seq, wav_lens, hyps = predictions

        ids = batch.id
        phns_eos, phn_lens_eos = batch.phn_encoded_eos
        phns, phn_lens = batch.phn_encoded

        if hasattr(self.modules, "env_corrupt") and stage == sb.Stage.TRAIN:
            phns = torch.cat([phns, phns], dim=0)
            phn_lens = torch.cat([phn_lens, phn_lens], dim=0)
            phns_eos = torch.cat([phns_eos, phns_eos], dim=0)
            phn_lens_eos = torch.cat([phn_lens_eos, phn_lens_eos], dim=0)

        loss_ctc = self.hparams.ctc_cost(p_ctc, phns, wav_lens, phn_lens)
        loss_seq = self.hparams.seq_cost(p_seq, phns_eos, phn_lens_eos)
        loss = self.hparams.ctc_weight * loss_ctc
        loss += (1 - self.hparams.ctc_weight) * loss_seq

        # # Record losses for posterity
        # if stage != sb.Stage.TRAIN:
        #     self.ctc_metrics.append(ids, p_ctc, phns, wav_lens, phn_lens)
        #     self.seq_metrics.append(ids, p_seq, phns_eos, phn_lens_eos)
        #     self.per_metrics.append(
        #         ids, hyps, phns, None, phn_lens, self.label_encoder.decode_ndim,
        #     )

        return loss

#TODO: CITATION FROM https://github.com/speechbrain/speechbrain/blob/develop/recipes/TIMIT/ASR/seq2seq/train.py
def dataio_prep(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    data_folder = hparams["data_folder"]
    # 1. Declarations:
    print("reading train data")
    train_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["train_annotation"],
        replacements={"data_root": data_folder},
    )
    # if hparams["sorting"] == "ascending":
    #     # we sort training data to speed up training and get better results.
    train_data = train_data.filtered_sorted(sort_key="duration")
    #     # when sorting do not shuffle in dataloader ! otherwise is pointless
    #     hparams["train_dataloader_opts"]["shuffle"] = False
    #
    # elif hparams["sorting"] == "descending":
    #     train_data = train_data.filtered_sorted(
    #         sort_key="duration", reverse=True
    #     )
    #     # when sorting do not shuffle in dataloader ! otherwise is pointless
    #     hparams["train_dataloader_opts"]["shuffle"] = False
    #
    # elif hparams["sorting"] == "random":
    #     pass
    #
    # else:
    #     raise NotImplementedError(
    #         "sorting must be random, ascending or descending"
    #     )

    print("reading valid data")
    valid_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["valid_annotation"],
        replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    print("reading test data")
    test_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["test_annotation"],
        replacements={"data_root": data_folder},
    )
    test_data = test_data.filtered_sorted(sort_key="duration")

    datasets = [train_data, valid_data, test_data]
    label_encoder = sb.dataio.encoder.CTCTextEncoder()

    print("defining audio pipeline")
    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    print("defining text pipeline")
    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("phn")
    @sb.utils.data_pipeline.provides(
        "phn_list",
        "phn_encoded_list",
        "phn_encoded",
        "phn_encoded_eos",
        "phn_encoded_bos",
    )
    def text_pipeline(phn):
        phn_list = phn.strip().split()
        yield phn_list
        phn_encoded_list = label_encoder.encode_sequence(phn_list)
        yield phn_encoded_list
        phn_encoded = torch.LongTensor(phn_encoded_list)
        yield phn_encoded
        phn_encoded_eos = torch.LongTensor(
            label_encoder.append_eos_index(phn_encoded_list)
        )
        yield phn_encoded_eos
        phn_encoded_bos = torch.LongTensor(
            label_encoder.prepend_bos_index(phn_encoded_list)
        )
        yield phn_encoded_bos

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    print("defining encoder")
    # 3. Fit encoder:
    # Load or compute the label encoder
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    special_labels = {
        "bos_label": hparams["bos_index"],
        "eos_label": hparams["eos_index"],
        "blank_label": hparams["blank_index"],
    }
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[train_data],
        output_key="phn_list",
        special_labels=special_labels,
        sequence_input=True,
    )

    print("defining output")
    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "sig", "phn_encoded", "phn_encoded_eos", "phn_encoded_bos"],
    )

    return train_data, valid_data, test_data, label_encoder

