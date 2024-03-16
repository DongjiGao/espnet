# Learning from Flawed Data: Weakly Supervised Automatic Speech Recognition
# https://arxiv.org/pdf/2309.15796.pdf

# 2024 Dongji Gao

import logging  # k2 internal APIs
from typing import Union
import _k2
import k2
import torch


class OTC(torch.nn.Module):
    def __init__(self, otc_token="<star>", allow_bypass=False, allow_self_loop=False):
        super().__init__()
        self.otc_token = otc_token
        self.allow_bypass = allow_bypass
        self.allow_self_loop = allow_self_loop

    def forward(
        self,
        nnet_output,
        ys_pad,
        hlens,
        ylens,
        allow_bypass=None,
        allow_self_loop=None,
        bypass_weight=0.0,
        self_loop_weight=0.0,
    ):
        if allow_bypass is None:
            allow_bypass = self.allow_bypass
        if allow_self_loop is None:
            allow_self_loop = self.allow_self_loop

        # Reorder and filter out invalid examples:
        # A. K2 requires that hlens are in descending order;
        # B. remove all examples whose hlens is smaller than necessary.
        indices = torch.argsort(hlens, descending=True)
        ys, min_hlens = self.find_minimum_hlens(ys_pad[indices], ylens[indices])
        valid_sample_indices = (min_hlens <= hlens[indices]).nonzero(as_tuple=True)[0]

        if len(valid_sample_indices) == 0:
            logging.warning("All examples ars invalid in OTC. Skip this batch.")
            return torch.Tensor([0.0]).to(nnet_output.device)

        indices = indices[valid_sample_indices]
        nnet_output, hlens, ylens = nnet_output[indices], hlens[indices], ylens[indices]

    def forward_core(
        self,
        nnet_output,
        ys,
        hlens,
        allow_bypass,
        allow_self_loop,
        bypass_weight,
        self_loop_weight,
    ):
        #  (1) Set the probability of OTC token as the average of non-blank tokens
        #      under the assumption that blank is the first and OTC token is the last token in tokens.txt
        B, _, C = nnet_output.shape
        otc_token_log_prob = torch.logsumexp(
            nnet_output[:, :, 1:], dim=-1, keepdim=True
        ) - torch.log(torch.tensor([C - 1])).to(nnet_output.device)
        nnet_output = torch.cat([nnet_output, otc_token_log_prob], dim=-1)
        num_channels = C + 1

        # (2) Build DenseFsaVec and OTC graphs
        supervision_segments = torch.stack(
            [torch.arange(B), torch.zeros(B), hlens.cpu()], dim=1
        ).int()

        otc_training_graph = self.compile(
            text_ids=ys,
            num_channels=num_channels,
            allow_bypass=allow_bypass,
            allow_self_loop=allow_self_loop,
            bypass_weight=bypass_weight,
            self_loop_weight=self_loop_weight,
        )

    def compile(
        text_ids,
        allow_bypass,
        allow_self_loop,
        bypass_weight,
        self_loop_weight,
    ):
        pass

    def make_arc(
        self,
        from_state: int,
        to_state: int,
        symbol: Union[str, int],
        weight: float,
    ):
        return f"{from_state} {to_state} {symbol} {weight}"

    def find_minimum_hlens(self, ys_pad, ylens):
        device = ys_pad.device
        ys_pad, ylens = ys_pad.cpu().tolist(), ylens.cpu().tolist()
        ys, min_hlens = [], []

        for y_pad, ylen in zip(ys_pad, ylens):
            y, min_hlen = [], 0
            prev = None

            for i in range(ylen):
                y.append(y_pad[i])
                min_hlen += 1

                if y_pad[i] == prev:
                    min_hlen += 1

                prev = y_pad[i]

            ys.append(y)
            min_hlens.append(min_hlen)

        min_hlens = torch.Tensor(min_hlens).long().to(device)

        return ys, min_hlens
