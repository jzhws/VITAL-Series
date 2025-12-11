# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import warnings
from typing import List, Optional, Tuple, Union

import torch.distributed as dist
import torch.utils.checkpoint
import transformers
from internvl.conversation import get_conv_template
from internvl.model.internlm2.modeling_internlm2 import InternLM2ForCausalLM
from internvl.model.phi3.modeling_phi3 import Phi3ForCausalLM
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (AutoModel, GenerationConfig, LlamaForCausalLM,
                          LlamaTokenizer, Qwen2ForCausalLM)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging

from .configuration_internvl_chat import InternVLChatConfig
from .modeling_intern_vit import InternVisionModel, has_flash_attn
from pytorchvideo.models.hub import slowfast_r50
import torch.nn.functional as F
logger = logging.get_logger(__name__)


def version_cmp(v1, v2, op='eq'):
    import operator

    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))


# def init_weights(m):
#     if isinstance(m, nn.Linear):
#         # new_weight = torch.rand(1, 4096)
#         # m.weight.copy_(new_weight)
#         nn.init.uniform_(m.weight, a=0.0, b=1.0)
#         nn.init.zeros_(m.bias)
#         print('m.weight',m.weight)
#         print('m.bias',m.bias)
        
class MLP(nn.Module):
    def __init__(self, input_dim=4096):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 16)
        self.fc5 = nn.Linear(16, 1)

        # 初始化线性层权重在 [0, 1] 之间
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # print('m.weight1', m.weight)
                # with torch.no_grad():
                m.weight.data.uniform_(0.0, 1e-2)
                # print('m.weight2', m.weight)

                m.bias.data.zero_()
                # print('m.bias', m.bias)
                # if m.bias is not None:
                #     nn.init.uniform_(m.bias, a=0.0, b=1.0)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        return x


def pack_pathway_output(frames, device):
    """
    Prepare output as a list of tensors. Each tensor corresponding to a
    unique pathway.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `channel` x `num frames` x `height` x `width`.
    Returns:
        frame_list (list): list of tensors with the dimension of
            `channel` x `num frames` x `height` x `width`.
    """
    fast_pathway = frames  # Shape: [B, C, T, H, W]

    # Generate the index tensor on the same device as 'frames'
    index = torch.linspace(
        0, frames.shape[2] - 1, frames.shape[2] // 4
    ).long().to(frames.device)

    # Perform temporal sampling from the fast pathway
    slow_pathway = frames.index_select(2, index)
    
    # print("slow_pathway",slow_pathway.shape)
    # print("fast_pathway",fast_pathway.shape)

    frame_list = [slow_pathway.to(device), fast_pathway.to(device)]

    # fast_pathway = frames
    # # Perform temporal sampling from the fast pathway.
    # slow_pathway = torch.index_select(
    #     frames,
    #     2,
    #     torch.linspace(
    #         0, frames.shape[2] - 1, frames.shape[2] // 4
    #     ).long(),
    # )
    # frame_list = [slow_pathway.to(device), fast_pathway.to(device)]

    return frame_list

class slowfast(torch.nn.Module):
    def __init__(self):
        super(slowfast, self).__init__()
        slowfast_pretrained_features = nn.Sequential(*list(slowfast_r50(pretrained=True).children())[0])

        self.feature_extraction = torch.nn.Sequential()
        self.slow_avg_pool = torch.nn.Sequential()
        self.fast_avg_pool = torch.nn.Sequential()
        self.adp_avg_pool = torch.nn.Sequential()

        for x in range(0,5):
            self.feature_extraction.add_module(str(x), slowfast_pretrained_features[x])

        self.slow_avg_pool.add_module('slow_avg_pool', slowfast_pretrained_features[5].pool[0])
        self.fast_avg_pool.add_module('fast_avg_pool', nn.AdaptiveAvgPool2d(output_size=1))

        self.adp_avg_pool.add_module('adp_avg_pool', slowfast_pretrained_features[6].output_pool)
        

    def forward(self, x):
        # with torch.no_grad():
            
        x = self.feature_extraction(x)
        # x[0] = x[0].repeat_interleave(6, dim=2)
        # x[1] = x[1].repeat_interleave(6, dim=2)
        # slow_feature = self.slow_avg_pool(x[0])
        fast_feature = self.fast_avg_pool(x[1])

        # slow_feature = self.adp_avg_pool(slow_feature)
        # fast_feature = self.adp_avg_pool(fast_feature)
        # print(slow_feature.shape) #[2,2048,1,1,1]
        # print(fast_feature.shape) #[2,256,1,1,1]
        # feature_3D = torch.cat([slow_feature, fast_feature],dim=1) #[2,2304,1,1,1]
        feature_3D = fast_feature #[2,256,1,1,1]
        return feature_3D

class InternVLChatModel(PreTrainedModel):
    config_class = InternVLChatConfig
    main_input_name = 'pixel_values'
    base_model_prefix = 'language_model'
    _no_split_modules = ['InternVisionModel', 'LlamaDecoderLayer', 'InternLM2DecoderLayer',
                         'Phi3DecoderLayer', 'Qwen2DecoderLayer']
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True

    def __init__(self, config: InternVLChatConfig, vision_model=None, language_model=None, use_flash_attn=True):
        super().__init__(config)

        assert version_cmp(transformers.__version__, '4.37.0', 'ge')
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        self.llm_arch_name = config.llm_config.architectures[0]
        # Enable Flash Attention if supported, otherwise fall back to eager attention.
        use_flash_attn = use_flash_attn if has_flash_attn else False
        config.vision_config.use_flash_attn = True if use_flash_attn else False
        config.llm_config.attn_implementation = 'flash_attention_2' if use_flash_attn else 'eager'

        logger.info(f'num_image_token: {self.num_image_token}')
        logger.info(f'ps_version: {self.ps_version}')
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = InternVisionModel(config.vision_config)
        if language_model is not None:
            self.language_model = language_model
        else:
            if config.llm_config.architectures[0] == 'LlamaForCausalLM':
                self.language_model = LlamaForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'InternLM2ForCausalLM':
                self.language_model = InternLM2ForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Phi3ForCausalLM':
                self.language_model = Phi3ForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Qwen2ForCausalLM':
                self.language_model = Qwen2ForCausalLM(config.llm_config)
            else:
                raise NotImplementedError(f'{config.llm_config.architectures[0]} is not implemented.')

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )

        self.motion_mlp = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )
        
        for m in self.motion_mlp.modules():
            if isinstance(m, nn.Linear):
                # print('motion_mlp.weight1',m.weight)
                m.weight.data.uniform_(0.0, 1e-2)
                # print('motion_mlp.weight2',m.weight)
                m.bias.data.zero_()
                # print('motion_mlp.bias',m.bias)

        self.slowfast_model = slowfast()

        self.img_context_token_id = None
        self.conv_template = get_conv_template(self.template)
        if hasattr(config, 'system_message'):
            self.system_message = config.system_message
        else:
            self.system_message = self.conv_template.system_message
        self.num_samples = 0

        if config.use_backbone_lora:
            self.wrap_backbone_lora(r=config.use_backbone_lora, lora_alpha=2 * config.use_backbone_lora)

        if config.use_llm_lora:
            self.wrap_llm_lora(r=config.use_llm_lora, lora_alpha=2 * config.use_llm_lora)

    def wrap_backbone_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        lora_config = LoraConfig(
            r=r,
            target_modules=['attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.vision_model = get_peft_model(self.vision_model, lora_config)
        self.vision_model.print_trainable_parameters()

    def wrap_llm_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        # Determine the target modules based on the architecture of the language model
        if self.llm_arch_name == 'InternLM2ForCausalLM':
            target_modules = ['attention.wqkv', 'attention.wo', 'feed_forward.w1', 'feed_forward.w2', 'feed_forward.w3']
        elif self.llm_arch_name == 'Phi3ForCausalLM':
            target_modules = ['mlp.down_proj', 'mlp.gate_up_proj', 'self_attn.o_proj', 'self_attn.qkv_proj']
        elif self.llm_arch_name in ['Qwen2ForCausalLM', 'LlamaForCausalLM']:
            target_modules = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
                              'mlp.gate_proj', 'mlp.down_proj', 'mlp.up_proj']
        else:
            raise NotImplemented
        lora_config = LoraConfig(
            r=r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            task_type='CAUSAL_LM'
        )
        self.language_model = get_peft_model(self.language_model, lora_config)
        self.language_model.enable_input_require_grads()
        self.language_model.print_trainable_parameters()

    def forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            target_dist: Optional[torch.Tensor] = None,
            modality: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            pair_wise_label: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            statistics: Optional[torch.LongTensor] = None,
            loss_weight: Optional[List] = None,
            loss_reduction_all_gather: Optional[bool] = False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        image_flags = image_flags.squeeze(-1)
        # print("image_flags", image_flags.shape)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()
        # print("pixel_values", pixel_values.shape)
        vit_embeds = self.extract_feature(pixel_values[image_flags == 1])
        # vit_embeds = vit_embeds[image_flags == 1]
        # print("vit_embeds", vit_embeds.shape)
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        # print("input_embeds.shape", input_embeds.shape)
        input_embeds = input_embeds.reshape(B * N, C)
        frames = pixel_values.view(1, -1, 3, 448, 448)
        # print(frames.shape)
        if modality[0]==0:
            frames=frames.repeat(1,4,1,1,1)
        # if frames.shape[1]%4!=0:
        #     frames=torch.cat((frames,frames[:,frames.shape[1]:max(frames.shape[1]//4*4,4)]),dim=1)
        # print(frames.shape)
        frames = frames.permute(0, 2, 1, 3, 4)
        device = pixel_values.device

        # # Prepare inputs for slow_fast model
        inputs = pack_pathway_output(frames, device)  # Returns [slow_pathway, fast_pathway]
        motion_feature = self.slowfast_model(inputs)
        # print("motion_feature", motion_feature.shape)
        motion_feature = motion_feature.view(frames.shape[2], -1)
        motion_embeds = self.motion_mlp(motion_feature)
        # print("motion_embeds", motion_embeds.shape)
     

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')
            if statistics is not None:
                num_samples, num_padding_tokens, num_padding_images = statistics.tolist()
                self.num_samples += num_samples
                print(f'total_samples={self.num_samples}, {num_samples=}, {num_padding_tokens=}, {num_padding_images=}')

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
       

        input_ids = input_ids.reshape(B * N)

        # try:
        # print(torch.cat((vit_embeds.reshape(-1, C),motion_embeds),0).shape)
        # print(input_embeds[selected].shape)
        if modality[0]==1:
            input_embeds[selected] = input_embeds[selected] * 0.0 + torch.cat((vit_embeds.reshape(-1, C),motion_embeds),0)
        else:
            # print("image",vit_embeds.shape)
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)+0.0*torch.cat((motion_embeds,vit_embeds.reshape(-1, C)[:vit_embeds.reshape(-1, C).shape[0]-motion_embeds.shape[0]]),0)
        
        ignore_flag = False
       
        input_embeds = input_embeds.reshape(B, N, C)
        input_ids=input_ids.reshape(B , N)
        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None and loss_weight is not None:
            # print("loss_weight is not None")
            loss_weight = torch.tensor(loss_weight, dtype=torch.float32, device=labels.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_weights = loss_weight[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction='none')
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_weights = shift_weights.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            shift_weights = shift_weights.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

            shift_weights_sum = shift_weights.sum()
            if loss_reduction_all_gather:
                dist.all_reduce(shift_weights_sum, op=dist.ReduceOp.AVG)

            loss = loss * shift_weights
            loss = loss.sum() / shift_weights_sum
            if ignore_flag:
                loss = loss * 0.0
        elif labels is not None:
            def rating_loss(pred_score_A, pred_score_B,pred_std_A,pred_std_B, pair_wise_label):
                # print("pred_score_A",pred_score_A.requires_grad)
                # print("pred_score_B",pred_score_B.requires_grad)
                # print("pred_std_B",pred_std_B.requires_grad)
                # print("pred_std_A",pred_std_A.requires_grad)
                eps=1e-8#CRUCIAL!!!
                # pred = 0.5 * (1 + torch.erf((pred_score_A - pred_score_B) / (torch.sqrt(2*(pred_std_A**2+pred_std_B**2))+eps)))  # 2 -> sqrt(2 * (1**2 + 1**2))
                # pred = 0.5 * (1 + torch.erf((pred_score_A - pred_score_B) / 2)) # 2 -> sqrt(2 * (1**2 + 1**2))
                pred_std_A = pred_std_A.detach()
                pred_std_B = pred_std_B.detach()
                pred = 0.5 * (1 + torch.erf((pred_score_A - pred_score_B) / (2*(pred_std_A**2+pred_std_B**2) + eps).sqrt())) # 2 -> sqrt(2 * (1**2 + 1**2))
                pred1=pred
                # pred=torch.stack([pred,1-pred]).to(pred.dtype).to(pred.device)
                pred=pred.unsqueeze(0).to(pred.dtype).to(pred.device)
                # print("pred",pred.requires_grad)
                # print("pred",pred.requires_grad)
                # print("rating_loss",pred)
                # pred=torch.stack([pred,1-pred]).to(pred.dtype).to(pred.device)
                # print("pred",pred.requires_grad)
                if pair_wise_label[0]==1:
                    gt = torch.tensor([1]).unsqueeze(0).to(pred.dtype).to(pred.device)
                elif pair_wise_label[0]==0:
                    gt = torch.tensor([0]).unsqueeze(0).to(pred.dtype).to(pred.device)
                elif pair_wise_label[0]==2:
                    gt = torch.tensor([0.5]).unsqueeze(0).to(pred.dtype).to(pred.device)
                else:
                    gt = torch.tensor([pred1]).unsqueeze(0).to(pred.dtype).to(pred.device)
                loss = (1 - (pred * gt + eps).sqrt() - ((1 - pred) * (1 - gt) + eps).sqrt()).mean()

                # if pair_wise_label[0]==1:
                #     gt = torch.tensor(1).to(pred.dtype).to(pred.device)
                # if pair_wise_label[0]==0:
                #     gt = torch.tensor(0).to(pred.dtype).to(pred.device)
                # if pair_wise_label[0]==2:
                #     gt = torch.tensor(0.5).to(pred.dtype).to(pred.device)
                # gt = gt.detach()
                # loss = (1 - (pred * gt + eps).sqrt() - ((1 - pred) * (1 - gt) + eps).sqrt())
                # loss=F.kl_div(torch.log(pred), gt, reduction="batchmean")
                # print(loss)
                return loss
            def cal_dist(logits, input_ids):
                idx_level_label = find_last_element([input_ids[i] for i in range(input_ids.shape[0])],[1550,1661,6624,7852,3347])
                idx_level_logit = idx_level_label
                logits_level_ids = logits[
                    idx_level_logit-1
                ].contiguous()  # [B, V]
                preds = torch.softmax(logits_level_ids, dim=0).index_select(0, torch.tensor([1550,1661,6624,7852,3347]).to(logits.device))
                return preds
            def softkl_loss(logits, input_ids, target_dist):
                idx_level_label = find_last_element([input_ids[i] for i in range(input_ids.shape[0])],[1550,1661,6624,7852,3347])
                idx_level_logit = idx_level_label
                logits_level_ids = logits[
                    idx_level_logit-1
                ].contiguous()  # [B, V]
                preds = torch.softmax(logits_level_ids, dim=0)
                # preds = torch.softmax(logits_level_ids, dim=1).index_select(1, torch.tensor([1550,1661,6624,7852,3347]).to(logits.device))
                target = torch.zeros_like(preds).to(torch.bfloat16)  # [B, V]
                target[[1550,1661,6624,7852,3347]] = target_dist.to(torch.bfloat16)
                # target = target_dist.to(torch.bfloat16)
                target = target.detach()

                pred_log = torch.log(preds)
                loss_kl = F.kl_div(pred_log.unsqueeze(0), target.unsqueeze(0), reduction="batchmean")
                return loss_kl,idx_level_label
            def find_last_element(lst,target):
                for num,i in enumerate(range(len(lst)-1, -1, -1)):
                    if lst[i] in target:
                        return len(lst)-1-num
                return -1  # 如果没有找到目标元素
            dists=[]
            for i in range(logits.shape[0]):
                dists.append(cal_dist(logits[i],input_ids[i]))
                # print("dists",cal_dist(logits[i],input_ids[i]).requires_grad)
            dists=torch.stack(dists)
            pred_moss=[]
            pred_stds=[]
            loss_fct = CrossEntropyLoss()
            # if pair_wise_label is not None:
            for i in range(logits.shape[0]):
                weight = torch.tensor([4.5, 3.5, 2.5, 1.5, 0.5]).to(dists[i])
                score = torch.matmul(dists[i], weight)
                pred_moss.append(score)
                # print("score",score.requires_grad)
                variance = (weight - score.unsqueeze(0)) ** 2
                std = torch.sqrt(torch.sum(dists[i] * variance, dim=0))
                # print("std",std.requires_grad)
                pred_stds.append(std)
            if pair_wise_label is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_kl1,idx_level_label1=softkl_loss(logits[0],input_ids[0],target_dist[0])
                loss_kl2,idx_level_label2=softkl_loss(logits[1],input_ids[1],target_dist[1])
                mask = torch.ones([*shift_logits.shape[:2]], dtype=torch.bool)
                mask[0,idx_level_label1-1]=False
                mask[1,idx_level_label2-1]=False
                shift_logits=shift_logits[mask]
                shift_labels=shift_labels[mask]
                ce=loss_fct(shift_logits,shift_labels)
                # print("kl",loss_kl1)
                # print("ce",ce)
                # print('rating_LOSS',rating_loss(pred_moss[0], pred_moss[1], pred_stds[0], pred_stds[1], pair_wise_label))
                loss = rating_loss(pred_moss[0], pred_moss[1], pred_stds[0], pred_stds[1], pair_wise_label)+0.5*(loss_kl1+loss_kl2)+0.05*ce
                print("pairwise",loss)
            # # shift_logits = logits[..., :-1, :].contiguous()
            # # shift_labels = labels[..., 1:].contiguous()
            # # loss_kl1,idx_level_label1=softkl_loss(logits[0],input_ids[0],target_dist[0])
            # # loss_kl2,idx_level_label2=softkl_loss(logits[1],input_ids[1],target_dist[1])
            # # mask = torch.ones([*shift_logits.shape[:2]], dtype=torch.bool)
            # # mask[0,idx_level_label1-1]=False
            # # mask[1,idx_level_label2-1]=False
            # # shift_logits=shift_logits[mask]
            # # shift_labels=shift_labels[mask]
            # # ce=loss_fct(shift_logits,shift_labels)
            # # print(loss_kl1)
            # # loss = rating_loss(pred_moss[0], pred_moss[1], pred_stds[0], pred_stds[1], pair_wise_label)+0.1*(loss_kl1+loss_kl2)+0.005*ce
            # # print('LOSS',loss)
            elif target_dist is not None:
               
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_kl1,idx_level_label1=softkl_loss(logits[0],input_ids[0],target_dist[0])
                loss_kl2,idx_level_label2=softkl_loss(logits[1],input_ids[1],target_dist[1])
                mask = torch.ones([*shift_logits.shape[:2]], dtype=torch.bool)
                mask[0,idx_level_label1-1]=False
                mask[1,idx_level_label2-1]=False
                shift_logits=shift_logits[mask]
                shift_labels=shift_labels[mask]
                # print(shift_logits.shape)
                # print(shift_labels.shape)
                ce=loss_fct(shift_logits,shift_labels)
                
                # print("kl",loss_kl1)
                # print("ce",ce)
                # print('rating_LOSS',rating_loss(pred_moss[0], pred_moss[1], pred_stds[0], pred_stds[1], pair_wise_label))
                loss = (loss_kl1+loss_kl2)+0.05*ce
                print("dist",loss)

            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = CrossEntropyLoss()
                # loss_fct = CrossEntropyLoss(reduction='none')
                # reduction='mean'
                # print(shift_logits.shape)
                # print(shift_labels.shape)
                shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                assert shift_labels.max() < self.language_model.config.vocab_size
    # 对输入logits进行softmax操作，得到概率分布
                # softmax_output = torch.softmax(shift_logits, dim=1)
                # print(softmax_output.shape)
                # print(shift_labels.shape)
                # print(softmax_output.gather(1, shift_labels.view(-1, 1)).shape)
                # 计算每个样本的交叉熵损失（不进行reduction）
                # loss = -torch.log(softmax_output.gather(1, shift_labels.view(-1, 1))+1e-8)
                
                # 返回每个样本的损失值
                # return loss.view(-1)
                loss = loss_fct(shift_logits, shift_labels)
                # print(torch.sum(loss))
                # print(loss.shape)
                # #DFT
                # loss = loss * torch.softmax(shift_logits, dim=-1).gather(1, shift_labels.unsqueeze(-1)).squeeze(-1).detach()
                # loss = loss * torch.exp(-1*loss)
                # loss = torch.sum(loss) / shift_logits.shape[0]
                #focal
                # loss = loss *torch.exp(1-torch.softmax(shift_logits, dim=-1).gather(1, shift_labels.unsqueeze(-1)).squeeze(-1).detach(),2)
                # loss = loss *torch.pow(1-torch.exp(-1*loss),2)
                # loss = torch.sum(loss) / shift_logits.shape[0]
                # loss = loss / shift_logits.shape[0]
                # shift_logits = logits[..., :-1, :].contiguous()
                # shift_labels = labels[..., 1:].contiguous()
                # ce=loss_fct(shift_logits,shift_labels)
                # loss = ce
            # print('LOSS',loss)
            # loss = (loss_kl1+loss_kl2)+0.005*ce
                # loss = rating_loss(pred_moss[0], pred_moss[1], pred_stds[0], pred_stds[1], pair_wise_label)
            # print(loss)
            # if ignore_flag:
            #     loss = loss * 0.0

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    def batch_chat(self, tokenizer, pixel_values, questions, generation_config, num_patches_list=None,
                   history=None, return_history=False, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>',
                   IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', verbose=False, image_counts=None):
        if history is not None or return_history:
            print('Now multi-turn chat is not supported in batch_chat.')
            raise NotImplementedError

        if image_counts is not None:
            num_patches_list = image_counts
            print('Warning: `image_counts` is deprecated. Please use `num_patches_list` instead.')

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question
            template = get_conv_template(self.template)
            template.system_message = self.system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
            queries.append(query)

        tokenizer.padding_side = 'left'
        model_inputs = tokenizer(queries, return_tensors='pt', padding=True)
        device = torch.device(self.language_model.device if torch.cuda.is_available() else 'cpu')
        input_ids = model_inputs['input_ids'].to(device)
        attention_mask = model_inputs['attention_mask'].to(device)
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        responses = [response.split(template.sep.strip())[0].strip() for response in responses]
        return responses

    def chat(self, tokenizer, pixel_values, question, generation_config, history=None, return_history=False,
             num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
             verbose=False):

        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        # assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors='pt')
        device = torch.device(self.language_model.device if torch.cuda.is_available() else 'cpu')
        input_ids = model_inputs['input_ids'].to(device)
        attention_mask = model_inputs['attention_mask'].to(device)
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep.strip())[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            if verbose:
                print(query_to_print, response)
            return response

    def chat2(self, tokenizer, pixel_values, input_ids, generation_config, attention_mask, history=None, return_history=False,
             image_flags=None,modality=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
             verbose=False):

        image_flags = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()
        # input_embeds2 = self.language_model.get_input_embeddings()(input_ids).clone()

        vit_embeds = self.extract_feature(pixel_values)
        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]
        
        # vit_embeds2 = self.extract_feature(pixel_values2)
        # vit_embeds2 = vit_embeds2[image_flags == 1]

        B, N, C = input_embeds.shape
        # print("input_embeds.shape", input_embeds.shape)
        input_embeds = input_embeds.reshape(B * N, C)
        
        frames = pixel_values.view(1, -1, 3, 448, 448)
        if modality==0:
            frames=frames.repeat(1,4,1,1,1)
        frames = frames.permute(0, 2, 1, 3, 4)
        device = pixel_values.device
       
        # # Prepare inputs for slow_fast model
        inputs = pack_pathway_output(frames, device)  # Returns [slow_pathway, fast_pathway]
        motion_feature = self.slowfast_model(inputs)
        # print("motion_feature", motion_feature.shape)
        motion_feature = motion_feature.view(frames.shape[2], -1)
        # motion_feature = motion_feature.view(pixel_values.shape[0], -1)
        motion_embeds = self.motion_mlp(motion_feature)
     
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')
            # if statistics is not None:
            #     num_samples, num_padding_tokens, num_padding_images = statistics.tolist()
            #     self.num_samples += num_samples
            #     print(f'total_samples={self.num_samples}, {num_samples=}, {num_padding_tokens=}, {num_padding_images=}')

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
       

        # input_ids = input_ids.reshape(B * N)
        if modality==0:
            
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)+0.0*torch.cat((motion_embeds,vit_embeds.reshape(-1, C)[:vit_embeds.reshape(-1, C).shape[0]-motion_embeds.shape[0]]),0)
        else:
            # print("image",vit_embeds.shape)
            input_embeds[selected] = input_embeds[selected] * 0.0 + torch.cat((vit_embeds.reshape(-1, C),motion_embeds),0)
        
        # try:
        # print(torch.cat((vit_embeds.reshape(-1, C),motion_embeds),0).shape)
        # print(input_embeds[selected].shape)
        # input_embeds[selected] = input_embeds[selected] * 0.0 + torch.cat((vit_embeds.reshape(-1, C),motion_embeds),0)
        # input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
      
            
        input_embeds = input_embeds.reshape(B, N, C)
        
        # if history is None and pixel_values is not None and '<image>' not in question:
        #     question = '<image>\n' + question

        # if num_patches_list is None:
        #     num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        # assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)

        # history = [] if history is None else history
        # for (old_question, old_answer) in history:
        #     template.append_message(template.roles[0], old_question)
        #     template.append_message(template.roles[1], old_answer)
        # template.append_message(template.roles[0], question)
        # template.append_message(template.roles[1], None)
        # query = template.get_prompt()

        # if verbose and pixel_values is not None:
        #     image_bs = pixel_values.shape[0]
        #     print(f'dynamic ViT batch size: {image_bs}')

        # for num_patches in num_patches_list:
        #     image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
        #     query = query.replace('<image>', image_tokens, 1)

        # model_inputs = tokenizer(query, return_tensors='pt')
        # input_ids = model_inputs['input_ids'].to(self.device)
        # attention_mask = model_inputs['attention_mask'].to(self.device)
        # generation_config['eos_token_id'] = eos_token_id
        # generation_config['pad_token_id'] = tokenizer.pad_token_id
        generation_config = GenerationConfig(
            max_new_tokens=2048,
            do_sample=True,
            temperature=1,
            eos_token_id=tokenizer.eos_token_id,   # ✅ 正确设置
            pad_token_id=tokenizer.pad_token_id,    # ✅ 必须加上，避免 generate 报错
            bos_token_id=151644  # ✅ 必须有
        )

        # generation_output = self.language_model.generate(
        #     inputs_embeds=input_embeds,
        #     attention_mask=attention_mask,
        #     generation_config=generation_config,
        #     use_cache=True,
        # )
        # print("attention_mask", attention_mask)
        generation_output = self.generate2(
            input_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        # print('response',response)
        # response = response.split(template.sep)[0].strip()
        # print(response)
        # history.append((question, response))
        if return_history:
            return response, history
        else:
            # query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            # query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            # if verbose:
            #     print(query_to_print, response)
            return response
    
    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:

        assert self.img_context_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.extract_feature(pixel_values)
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)
            

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs
    
    @torch.no_grad()
    def generate2(
            self,
            input_embeds,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:

        # assert self.img_context_token_id is not None
        # if pixel_values is not None:
        #     if visual_features is not None:
        #         vit_embeds = visual_features
        #     else:
        #         vit_embeds = self.extract_feature(pixel_values)
        #     input_embeds = self.language_model.get_input_embeddings()(input_ids)
        #     B, N, C = input_embeds.shape
        #     input_embeds = input_embeds.reshape(B * N, C)

        #     input_ids = input_ids.reshape(B * N)
        #     selected = (input_ids == self.img_context_token_id)
        #     assert selected.sum() != 0
        #     input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

        #     input_embeds = input_embeds.reshape(B, N, C)
        # else:
        #     input_embeds = self.language_model.get_input_embeddings()(input_ids)

        # print("generation_config", generation_config)
        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs


    @property
    def lm_head(self):
        return self.language_model.get_output_embeddings()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    @property
    def lm_head(self):
        return self.language_model.get_output_embeddings()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()
