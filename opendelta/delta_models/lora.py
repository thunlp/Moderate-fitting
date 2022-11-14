from turtle import forward
from typing import Optional, Union

from opendelta.utils.signature import get_arg_names, get_arg_names_inside_func
from opendelta.utils.name_based_addressing import *
from opendelta.basemodel import DeltaBase
from transformers.models.t5 import T5ForConditionalGeneration
import torch
import torch.nn as nn
from opendelta import BaseDeltaConfig
import math

class ReparameterizeFunction(nn.Module):
    """
    """
    def __init__(self, in_dim, mid_dim=512):
        super().__init__()
        self.in_dim = in_dim
        self.mid_dim = mid_dim

    def instantiate(self, module_list=[]):
        self.module_list = module_list
        self.record_parameters()
        self.define_reparameterization_network()

    def goto(self, module, path):
        n = module
        for p in path:
            n = getattr(n, p)
        return n
        
    def record_parameters(self):
        r""" Enumerate the parameters that need to be reparameterized.
        Then, delete the original parameters. 
        """
        tot1 = 0
        tot2 = 0
        ith = 0

        self.path_list = []
        for module_idx, module in enumerate(self.module_list):
            self.path_list.append([])
            for n, parameters in module.named_parameters():
                ith ^= 1
                numel = parameters.numel()
                shape = parameters.size()

                names = n.split('.')
                path = names[:-1]
                name = names[-1]
                if(name=="bias" or name=="weight" or name=="lora_A"):
                    tot1 +=numel
                elif(name=="lora_B"):
                    tot2 +=numel
                self.path_list[-1].append((path, name, numel, shape))
            for path, name, _, _ in self.path_list[-1]:
                delattr(self.goto(module, path), name)
        self.total_parameters_num1 = tot1
        self.total_parameters_num2 = tot2

    def define_reparameterization_network(self) -> None:
        r""" Build the reparameterize module 
        """
        in_embed1 = torch.empty(self.in_dim)
        nn.init.uniform_(in_embed1)
        self.in_embed1 = nn.Parameter(in_embed1)
        self.transform1 = nn.Sequential(
            nn.Linear(self.in_dim, self.mid_dim, bias=False),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.total_parameters_num1, bias=False)
        )

        in_embed2 = torch.empty(self.in_dim)
        nn.init.zeros_(in_embed2)
        self.in_embed2 = nn.Parameter(in_embed2)
        self.transform2 = nn.Sequential(
            nn.Linear(self.in_dim, self.mid_dim, bias=False),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.total_parameters_num2, bias=False)
        )
    
    def allocate_parameter(self):
        r""" At the beginning of each forward pass through the whole network(PLM), 
        cacalulate the reparameterized params for later use in each layer.
        """
        params1 = self.transform1(self.in_embed1)
        params2 = self.transform2(self.in_embed2)

        cur1 = 0
        cur2 = 0
        ith = 0

        for module_idx, module in enumerate(self.module_list):
            for path, name, numel, shape in self.path_list[module_idx]:
                ith ^= 1
                if(name=="bias" or name=="weight" or name=="lora_A"):
                    setattr(self.goto(module, path), name, params1[cur1: cur1 + numel].view(shape))
                    cur1 +=numel
                elif(name=="lora_B"):
                    setattr(self.goto(module, path), name, params2[cur2: cur2 + numel].view(shape))
                    cur2 += numel

    def pre_forward(self, *args, **kwargs):
        r""" Firstly forward through the reparameterized network, and then go through normal forward pass of the PLM.
        """
        self.allocate_parameter()
        return args, kwargs


class LowRankLinear(nn.Module):
    #  ------------------------------------------------------------------------------------------
    #  Copyright (c) Microsoft Corporation. All rights reserved.
    #  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
    #  ------------------------------------------------------------------------------------------
    #  copy from loralib and do some refactor
    def __init__(self,
        in_features,
        out_features,
        weight,
        r=8, 
        lora_alpha=16,
        lora_dropout=0.0,
    ):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        # self.lin = nn.Linear(in_features, out_features) #
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        if r > 0:
            self.lora_A = nn.Parameter(weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # self.lin.reset_parameters() #
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling


class LoraConfig(BaseDeltaConfig):
    r"""
    This is the configuration class to store the configuration of a :py:class:`~LoraModel`

    """
    def __init__(
        self, 
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        in_dim =4,
        mid_dim=4,
        reparameterize=False,
        **kwargs
    ): 
        super().__init__(**kwargs)
        arg_names = get_arg_names_inside_func(self.__init__)
        for arg_name in arg_names:
            if not hasattr(self, arg_name): # the arg has not been registered in parent config
                setattr(self, arg_name, locals()[arg_name])


class LoraModel(DeltaBase):

    config_class = LoraConfig
    delta_type = "lora"
    default_modified_modules = ['attn.q', 'attn.v']
    def __init__(self,
                 backbone_model: nn.Module, 
                 lora_r=8,
                 lora_alpha=16,
                 lora_dropout=0.0,
                 in_dim=4,
                 mid_dim=4,
                 modified_modules: Optional[bool] = None,
                 unfrozen_modules: Optional[bool] = None,
                 common_structure: Optional[bool] = None,
                 interactive_modify: Optional[Union[bool, int]] = False,
                 reparameterize: Optional[bool] = False,
                 ):
        DeltaBase.__init__(self, 
                           backbone_model, 
                           modified_modules=modified_modules,
                           unfrozen_modules=unfrozen_modules,
                           common_structure=common_structure,
                           interactive_modify=interactive_modify,
                           )
        arg_names = get_arg_names_inside_func(self.__init__)
        for arg_name in arg_names:
            if not hasattr(self, arg_name): # not registered in parent class
                setattr(self, arg_name, locals()[arg_name])
        self.delta_modules = nn.ModuleList()

        if reparameterize:
            reparams = ReparameterizeFunction(
                in_dim = self.in_dim,
                mid_dim = self.mid_dim,
            )
        else:
            reparams = None
        self.delta_modules.append(self.backbone_model.classifier)
        self.add_all_delta_to_backbone(self.backbone_model,
                                   self.modified_modules,
                                   reparams = reparams,
                                   )
    
    
    def update_module(self, module: nn.Module, key: str):
        parent_ref, child_name, child_ref = self.find_module(module, key)
        parallel_module = self.new_module_like(child_module=child_ref)
        self.insert_parallel_module(child_ref, delta_module=parallel_module, delta_name="lora")
        
    def _pseudo_data_to_instantiate(self, module):
        # no need to pass pseudo input, so overwrite it
        pass

    def new_module_like(self, child_module):
        if isinstance(child_module, nn.Linear):
            in_features, out_features = child_module.in_features, child_module.out_features
            new_module = LowRankLinear(in_features = in_features, 
                                     out_features = out_features, 
                                     weight = child_module.weight,
                                     r = self.lora_r, 
                                     lora_alpha = self.lora_alpha,
                                     lora_dropout = self.lora_dropout)
            self.delta_modules.append(new_module)  
        else:
            raise NotImplementedError
        return new_module
