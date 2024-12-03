from pathlib import Path

import pandas as pd
import torch

from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer



class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """
    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]


        initial_wav = batch['wav']
        initial_melspec = batch['melspec']
        wav_fake = self.model.generator(initial_melspec)
        batch["generated_wav"] = wav_fake
        mel_spec_fake = self.create_mel_spec(wav_fake).squeeze(1)
        if self.is_train:
            self.disc_optimizer.zero_grad()

        mpd_gt_out, _, mpd_fake_out, _ = self.model.mpd(initial_wav, wav_fake.detach())

        msd_gt_out, _,  msd_fake_out, _ = self.model.msd(initial_wav, wav_fake.detach())

        mpd_disc_loss = self.criterion.discriminator_loss(mpd_gt_out, mpd_fake_out)
        msd_disc_loss = self.criterion.discriminator_loss(msd_gt_out, msd_fake_out)
        disc_loss = mpd_disc_loss + msd_disc_loss

        batch["mpd_disc_loss"] = mpd_disc_loss
        batch["msd_disc_loss"] = msd_disc_loss
        batch["disc_loss"] = disc_loss

        if self.is_train:
            self._clip_grad_norm(self.model.mpd)
            self._clip_grad_norm(self.model.msd)

        if self.is_train:
            disc_loss.backward()
            self.disc_optimizer.step()
            self.gen_optimizer.zero_grad()




        _, mpd_gt_feats, mpd_fake_out, mpd_fake_feats = self.model.mpd(initial_wav, wav_fake)

        _, msd_gt_features, msd_fake_out, msd_fake_feats = self.model.msd(initial_wav, wav_fake)     

        mpd_gen_loss = self.criterion.generator_loss(mpd_fake_out)
        msd_gen_loss = self.criterion.generator_loss(msd_fake_out)

        mel_spec_loss = self.criterion.melspec_loss(initial_melspec, mel_spec_fake)
        
        mpd_feats_gen_loss = self.criterion.fm_loss(mpd_gt_feats, mpd_fake_feats)
        msd_feats_gen_loss = self.criterion.fm_loss(msd_gt_features, msd_fake_feats)

        gen_loss = mpd_gen_loss + msd_gen_loss + mel_spec_loss + mpd_feats_gen_loss + msd_feats_gen_loss


        if self.is_train:
            self._clip_grad_norm(self.model.generator)

        if self.is_train:
            gen_loss.backward()
            self.gen_optimizer.step()

        batch["mpd_gen_loss"] = mpd_gen_loss
        batch["msd_gen_loss"] = msd_gen_loss
        batch["mel_spec_loss"] = mel_spec_loss
        batch["mpd_feats_gen_loss"] = mpd_feats_gen_loss
        batch["msd_feats_gen_loss"] = msd_feats_gen_loss
        batch["gen_loss"] = gen_loss
    


        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            self.log_spectrogram(**batch)
            self.log_audio(**batch)

        else:
            # Log Stuff
            self.log_spectrogram(**batch)
            self.log_audio(**batch)
            self.log_predictions(**batch)


    def log_audio(self, wav, generated_wav, **batch):
        self.writer.add_audio("initial_wav", wav[0], 22050)
        self.writer.add_audio("generated_wav", generated_wav[0], 22050)

    def log_spectrogram(self, melspec, **batch):
        spectrogram_for_plot = melspec[0].detach().cpu()
        image = plot_spectrogram(spectrogram_for_plot)
        self.writer.add_image("melspectrogram", image)

    def log_predictions(self, examples_to_log=10, **batch):
        # Note: by improving text encoder and metrics design
        # this logging can also be improved significantly
        all_wavs_generated = batch['generated_wav']
        paths = batch['path']
        rows = {}
        tuples = list(zip(all_wavs_generated, paths))
        for generated_wav, path in tuples[:examples_to_log]:
            mos = self.calc_mos.model.calculate_one(generated_wav)

            rows[Path(path).name] = {
                "MOS": mos,
                # "initial audio" :  self.writer.wandb.Audio(path, sample_rate),
                # "generated audio": self.writer.wandb.add_audio,
            }
            

        self.writer.add_table(
            "predictions", pd.DataFrame.from_dict(rows, orient="index")
        )