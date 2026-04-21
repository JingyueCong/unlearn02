import copy
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import LogitsProcessorList, StoppingCriteriaList
from transformers.modeling_outputs import CausalLMOutputWithPast

from .contrastllm import ContrastLLM
from .gen_util import LogitDecoderOnlyOutput


class DualContrastLLM(ContrastLLM):
    """Two-assistant ULD: final = base + w1 * A1 + w2 * A2.

    - A1 is trained to 'remember' forget-data AND R_sub (GD), and output uniform
      elsewhere. Used with negative weight to subtract.
    - A2 is trained to 'remember' R_sub only (GD) with uniform elsewhere. Used
      with positive weight so that on R_sub the two assistants cancel, leaving
      base unchanged.

    The top-logit filter is applied against the base logits (same as ULD); both
    assistants' logits are masked with the same mask to keep their shapes aligned.
    """

    def __init__(
        self,
        basellm: AutoModelForCausalLM,
        assist_llm_1: AutoModelForCausalLM,
        assist_llm_2: AutoModelForCausalLM,
        weight_a1: float = -1.0,
        weight_a2: float = 1.0,
        top_logit_filter: float = 0.0,
    ) -> None:
        # Intentionally bypass ContrastLLM.__init__ to avoid holding a single
        # assist_llm; we set fields manually.
        torch.nn.Module.__init__(self)
        self.basellm = basellm
        self.assist_llm_1 = assist_llm_1
        self.assist_llm_2 = assist_llm_2
        self.weight_a1 = weight_a1
        self.weight_a2 = weight_a2
        # Keep attributes that ContrastLLM's generation / loss code expects.
        self.weight = weight_a1  # unused, kept for parity
        self.device = self.basellm.device
        self.config = self.basellm.config
        self.generation_config = basellm.generation_config
        self.top_logit_filter = top_logit_filter
        self.tokenizer = AutoTokenizer.from_pretrained(self.config._name_or_path)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = False
        output_hidden_states = False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        fw_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        base_out = self.basellm(**fw_kwargs)
        a1_out = self.assist_llm_1(**fw_kwargs)
        a2_out = self.assist_llm_2(**fw_kwargs)

        base_logits = base_out.logits
        a1_logits = a1_out.logits
        a2_logits = a2_out.logits

        if self.top_logit_filter > 0.0:
            base_logits, mask, _ = self.relative_top_filter(base_logits, self.top_logit_filter)
            a1_logits[mask] = 0
            a2_logits[mask] = 0

        logits = base_logits + self.weight_a1 * a1_logits + self.weight_a2 * a2_logits

        loss = self.get_loss(logits, labels)
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

    # --------- generation overrides (both assistants with independent caches) ---------

    def prepare_inputs_for_generation_assist(self, *args, **kwargs):
        # Used for A1; A2 uses its own below. Same arch, same method.
        return self.assist_llm_1.prepare_inputs_for_generation(*args, **kwargs)

    def _prep_assist2(self, *args, **kwargs):
        return self.assist_llm_2.prepare_inputs_for_generation(*args, **kwargs)

    @torch.no_grad()
    def greedy_search(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        output_logits: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer=None,
        **model_kwargs,
    ):
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.generation_config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        raw_logits = () if (return_dict_in_generate and output_logits) else None
        assist_logits = () if (return_dict_in_generate and output_logits) else None
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        # Independent kv caches for A1 and A2.
        model_kwargs_a1 = copy.deepcopy(model_kwargs)
        model_kwargs_a2 = copy.deepcopy(model_kwargs)

        this_peer_finished = False
        while True:
            base_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            base_out = self.basellm(
                **base_inputs, return_dict=True,
                output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            )

            def _run_assist(assist_llm, prep_fn, m_kwargs):
                dev = assist_llm.device
                ids = input_ids if input_ids.device == dev else input_ids.to(dev)
                prepared = prep_fn(ids, **m_kwargs)
                for k, v in prepared.items():
                    if isinstance(v, torch.Tensor) and v.device != dev:
                        prepared[k] = v.to(dev)
                return assist_llm(
                    **prepared, return_dict=True,
                    output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                )

            a1_out = _run_assist(self.assist_llm_1, self.prepare_inputs_for_generation_assist, model_kwargs_a1)
            a2_out = _run_assist(self.assist_llm_2, self._prep_assist2, model_kwargs_a2)

            if synced_gpus and this_peer_finished:
                continue

            next_token_logits = base_out.logits[:, -1, :]
            a1_last = a1_out.logits[:, -1, :].to(next_token_logits.device)
            a2_last = a2_out.logits[:, -1, :].to(next_token_logits.device)

            if self.top_logit_filter > 0.0:
                next_token_logits, mask, _ = self.relative_top_filter(
                    next_token_logits, relative_top=self.top_logit_filter
                )
                a1_last = a1_last.clone(); a1_last[mask] = 0
                a2_last = a2_last.clone(); a2_last[mask] = 0
                next_token_logits = next_token_logits + self.weight_a1 * a1_last + self.weight_a2 * a2_last
                next_token_logits[mask] = -1e3
            else:
                next_token_logits = next_token_logits.log_softmax(dim=-1)
                a1_last = a1_last.log_softmax(dim=-1)
                a2_last = a2_last.log_softmax(dim=-1)
                next_token_logits = next_token_logits + self.weight_a1 * a1_last + self.weight_a2 * a2_last

            next_tokens_scores = logits_processor(input_ids, next_token_logits)

            if return_dict_in_generate:
                if output_scores:
                    scores += (next_tokens_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                    assist_logits += ((a1_last, a2_last),)
                if output_attentions:
                    decoder_attentions += (base_out.attentions,) if not self.config.is_encoder_decoder else (base_out.decoder_attentions,)
                if output_hidden_states:
                    decoder_hidden_states += (base_out.hidden_states,) if not self.config.is_encoder_decoder else (base_out.decoder_hidden_states,)

            if self.top_logit_filter > 0.0:
                next_tokens = torch.argmax(next_tokens_scores * (~mask), dim=-1)
            else:
                next_tokens = torch.argmax(next_tokens_scores, dim=-1)

            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())

            model_kwargs = self._update_model_kwargs_for_generation(
                base_out, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            model_kwargs_a1 = self._update_model_kwargs_for_generation(
                a1_out, model_kwargs_a1, is_encoder_decoder=self.config.is_encoder_decoder
            )
            model_kwargs_a2 = self._update_model_kwargs_for_generation(
                a2_out, model_kwargs_a2, is_encoder_decoder=self.config.is_encoder_decoder
            )

            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1)
                    .ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )
                if unfinished_sequences.max() == 0:
                    this_peer_finished = True

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            if this_peer_finished and not synced_gpus:
                break

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            return LogitDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
                assist_logits=assist_logits,
            )
        return input_ids
