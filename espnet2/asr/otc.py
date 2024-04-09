# Learning from Flawed Data: Weakly Supervised Automatic Speech Recognition
# https://arxiv.org/pdf/2309.15796.pdf

# 2024 Dongji Gao

import logging  # k2 internal APIs
from typing import Union
import _k2
import k2
import torch


class OTC(torch.nn.Module):
    def __init__(self, otc_token_id, allow_bypass=False, allow_self_loop=False,
                 initial_bypass_weight=0.0, bypass_weight_decay=0.0,
                 initial_self_loop_weight=0.0, self_loop_weight_decay=0.0):
        super().__init__()
        self.otc_token_id = otc_token_id
        self.allow_bypass = allow_bypass
        self.allow_self_loop = allow_self_loop
        self.initial_bypass_weight = initial_bypass_weight
        self.bypass_weight_decay = bypass_weight_decay
        self.initial_self_loop_weight = initial_self_loop_weight
        self.self_loop_weight_decay = self_loop_weight_decay
        self.bypass_weight = None
        self.self_loop_weight = None

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
        self.device = nnet_output.device
        if allow_bypass is None:
            allow_bypass = self.allow_bypass
        if allow_self_loop is None:
            allow_self_loop = self.allow_self_loop
        if allow_bypass and bypass_weight == 0.0:
            bypass_weight = self.bypass_weight
            assert(bypass_weight is not None), "bypass_weight should not be None"
        if allow_self_loop and self_loop_weight == 0.0:
            self_loop_weight = self.self_loop_weight
            assert(self_loop_weight is not None), "self_loop_weight should not be None"

        # Reorder and filter out invalid examples:
        # A. K2 requires that hlens are in descending order;
        # B. remove all examples whose hlens is smaller than necessary.
        indices = torch.argsort(hlens, descending=True).to(self.device)
        ys, min_hlens = self.find_minimum_hlens(ys_pad[indices], ylens[indices])
        hlens = hlens.to(self.device)
        valid_sample_indices = (min_hlens <= hlens[indices]).nonzero(as_tuple=True)[0]

        if len(valid_sample_indices) == 0:
            logging.warning("All examples ars invalid in OTC. Skip this batch.")
            return torch.Tensor([0.0]).to(self.device)

        indices = indices[valid_sample_indices]
        nnet_output, hlens, ylens = nnet_output[indices], hlens[indices], ylens[indices]
        ys = [ys[i.item()] for i in valid_sample_indices]

        otc_loss = self.forward_core(
            nnet_output,
            ys,
            hlens,
            allow_bypass,
            allow_self_loop,
            bypass_weight,
            self_loop_weight,
        )

        return otc_loss

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
        ) - torch.log(torch.tensor([C - 1])).to(self.device)
        nnet_output = torch.cat([nnet_output, otc_token_log_prob], dim=-1)
        num_channels = C + 1
        max_token_id = num_channels - 1

        # (2) Build DenseFsaVec and OTC graphs
        supervision_segments = torch.stack(
            [torch.arange(B), torch.zeros(B), hlens.cpu()], dim=1
        ).int()

        otc_training_graph = self.compile(
            text_ids=ys,
            max_token_id=max_token_id,
            allow_bypass=allow_bypass,
            allow_self_loop=allow_self_loop,
            bypass_weight=bypass_weight,
            self_loop_weight=self_loop_weight,
        )

        dense_fsa_vec = k2.DenseFsaVec(
            nnet_output,
            supervision_segments,
            allow_truncate=3,
        )

         # (3) Compute OTC loss
        otc_loss = k2.ctc_loss(
            decoding_graph=otc_training_graph,
            dense_fsa_vec=dense_fsa_vec,
            output_beam=10,
            reduction="sum",
            use_double_scores=True,
        )

        return otc_loss

    def compile(
        self,
        text_ids,
        max_token_id,
        allow_bypass,
        allow_self_loop,
        bypass_weight,
        self_loop_weight,
    ):
        ctc_topo = k2.ctc_topo(max_token_id, modified=False).to(self.device)

        transcript_fsa = self.convert_transcript_to_fsa(
            text_ids,
            self.otc_token_id,
            allow_bypass,
            allow_self_loop,
            bypass_weight,
            self_loop_weight,
        )
        transcript_fsa = transcript_fsa.to(self.device)
        fsa_with_self_loop = k2.remove_epsilon_and_add_self_loops(transcript_fsa)
        fsa_with_self_loop = k2.arc_sort(fsa_with_self_loop)

        graph = k2.compose(
            ctc_topo,
            fsa_with_self_loop,
            treat_epsilons_specially=False,
        )
        return graph

    def convert_transcript_to_fsa(
        self,
        texts_ids,
        otc_token_id,
        allow_bypass,
        allow_self_loop,
        bypass_weight,
        self_loop_weight,
    ):

        transcript_fsa_list = []
        # sentence
        for text_ids in texts_ids:
            text_piece_ids = []

            # word
            for piece_ids in text_ids:
                text_piece_ids.append([piece_ids])

            arcs = []
            start_state = 0
            cur_state = start_state
            next_state = 1

            # subword
            for piece_ids in text_piece_ids:
                bypass_cur_state = cur_state

                if allow_self_loop:
                    self_loop_arc = self.make_arc(
                        cur_state,
                        cur_state,
                        otc_token_id,
                        self_loop_weight,
                    )
                    arcs.append(self_loop_arc)

                for piece_id in piece_ids:
                    arc = self.make_arc(cur_state, next_state, piece_id, 0.0)
                    arcs.append(arc)

                    cur_state = next_state
                    next_state += 1

                bypass_next_state = cur_state
                if allow_bypass:
                    bypass_arc = self.make_arc(
                        bypass_cur_state,
                        bypass_next_state,
                        otc_token_id,
                        bypass_weight,
                    )
                    arcs.append(bypass_arc)
                bypass_cur_state = cur_state

            if allow_self_loop:
                self_loop_arc = self.make_arc(
                    cur_state,
                    cur_state,
                    otc_token_id,
                    self_loop_weight,
                )
                arcs.append(self_loop_arc)

            # Deal with final state
            final_state = next_state
            final_arc = self.make_arc(cur_state, final_state, -1, 0.0)
            arcs.append(final_arc)
            arcs.append(f"{final_state}")
            sorted_arcs = sorted(arcs, key=lambda a: int(a.split()[0]))

            transcript_fsa = k2.Fsa.from_str("\n".join(sorted_arcs))
            transcript_fsa = k2.arc_sort(transcript_fsa)
            transcript_fsa_list.append(transcript_fsa)

        transcript_fsa_vec = k2.create_fsa_vec(transcript_fsa_list)

        return transcript_fsa_vec

    def make_arc(
        self,
        from_state: int,
        to_state: int,
        symbol: Union[str, int],
        weight: float,
    ):
        return f"{from_state} {to_state} {symbol} {weight}"

    def find_minimum_hlens(self, ys_pad, ylens):
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

        min_hlens = torch.Tensor(min_hlens).long().to(self.device)

        return ys, min_hlens
