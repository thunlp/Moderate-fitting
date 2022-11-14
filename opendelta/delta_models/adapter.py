from functools import partial
from random import random
from typing import Optional, Union
from opendelta.utils.signature import get_arg_names_inside_func
from opendelta.utils.name_based_addressing import *
from opendelta.utils.cuda import get_device
from opendelta.basemodel import DeltaBase
import loralib as lora
import torch.nn as nn
import torch
import math
from opendelta.delta_models.layers.activations import Activations
import inspect
from opendelta import BaseDeltaConfig
import opendelta.utils.logging as logging
logger = logging.get_logger(__name__)

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
        tot = 0 
        tot1 = 0
        tot2 = 0

        self.path_list = []
        for module_idx, module in enumerate(self.module_list):
            self.path_list.append([])
            for n, parameters in module.named_parameters():
                numel = parameters.numel()
                shape = parameters.size()
                tot += numel

                names = n.split('.')
                path = names[:-1]
                name = names[-1]
                if(len(path)>1 and path[1]=="up_proj"):
                    tot1+=numel
                self.path_list[-1].append((path, name, numel, shape))
            for path, name, _, _ in self.path_list[-1]:
                delattr(self.goto(module, path), name)
        self.total_parameters_num = tot

    def define_reparameterization_network(self) -> None:
        r""" Build the reparameterize module
        """
        in_embed1 = torch.empty(self.in_dim)
        in_embed1.normal_(mean=0.0, std=0.01)
        self.in_embed1 = nn.Parameter(in_embed1)
        self.transform1 = nn.Sequential(
            nn.Linear(self.in_dim, self.mid_dim, bias=False),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.total_parameters_num, bias=False)
        )
        self.transform1[0].weight.data.normal_(mean=0.0,std=0.01)
        self.transform1[2].weight.data.normal_(mean=0.0,std=0.01)


    def allocate_parameter(self):
        r""" At the beginning of each forward pass through the whole network(PLM),
        cacalulate the reparameterized params for later use in each layer.
        """
        params1 = self.transform1(self.in_embed1)

        cur1 = 0
        cur2 = 0
        for module_idx, module in enumerate(self.module_list):
            for path, name, numel, shape in self.path_list[module_idx]:
                setattr(self.goto(module, path), name, params1[cur1: cur1 + numel].view(shape))
                cur1 += numel

    def pre_forward(self, *args, **kwargs):
        r""" Firstly forward through the reparameterized network, and then go through normal forward pass of the PLM.
        """
        self.allocate_parameter()
        return args, kwargs

class AdapterLayer(nn.Module):
    r"""A layer of adapter tuning module. 
    """
    layer_count = 0

    @classmethod
    def count_layer(cls):
        cls.layer_count += 1
    
    @classmethod
    def get_layer_count(cls):
        return cls.layer_count

    def __init__(self, bottleneck_dim=24, non_linearity='gelu_new', device=None):
        super().__init__()
        self.bottleneck_dim = bottleneck_dim
        self.device = device
        self.instantiated = False
        self.non_linearity = non_linearity
        
        self.layer_id = AdapterLayer.get_layer_count()
        AdapterLayer.count_layer()
        
    
    def instantiate(self, hidden_dim):
        self.modulelist = nn.Sequential()
        self.modulelist.add_module("down_proj", nn.Linear(hidden_dim, self.bottleneck_dim, device=self.device))

        # select non-linearity
        self.modulelist.add_module("non_linear", Activations(self.non_linearity.lower()))

        self.modulelist.add_module("up_proj", nn.Linear(self.bottleneck_dim, self.hidden_dim,  device=self.device))

        # TODO:
        # If we want to have a layer norm on output, we apply it later after a separate residual connection
        # This means that we learn a new output layer norm, which replaces another layer norm learned in the bert layer
        # if self.add_layer_norm_after:
        #     self.adapter_norm_after = nn.LayerNorm(self.input_size)

        self.instantiated = True
        # initialize the weight, which is important for fast convergence and better performance. 
        self.apply(self._init_weight)
    
    def _init_weight(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.01) 
            if module.bias is not None:
                module.bias.data.zero_()
        
    
    def post_forward(self, output):
        r""" Get the hidden_states from the PLM's layer output, pass it into the adapter, 
        then combined with the main hidden_states. Finally pass it into the subsequent layer.

        normal_(mean=0.0, std=0.01)"""
        if isinstance(output, tuple):
            hiddens = output[0]
        elif isinstance(output, torch.Tensor):
            hiddens = output
        else:
            raise TypeError


        if not self.instantiated:
            self.hidden_dim = hiddens.shape[-1]
            logger.debug(f"Got hidden dim hidden_dim {self.hidden_dim}")
            self.instantiate(hidden_dim=self.hidden_dim)
                

        adapter_output = self.modulelist(hiddens)
        modified_output = adapter_output + hiddens # TODO option: disable residual_connection
        if isinstance(output, tuple):
            output = (modified_output,) + output[1:]
        elif isinstance(output, torch.Tensor):
            output = modified_output
        else:
            raise TypeError
        return output
    
  

class AdapterConfig(BaseDeltaConfig):
    r"""
    This is the configuration class to store the configuration of a :py:class:`~AdapterModel`

    """
    def __init__(
        self, 
        bottleneck_dim: Optional[int]=24, 
        non_linearity: Optional[str]='gelu_new',
        sequential: Optional[str] = True,
        in_dim: Optional[int] = 4,
        mid_dim: Optional[int] = 4,
        reparameterize=False,
        **kwargs
    ): 
        super().__init__(**kwargs)
        arg_names = get_arg_names_inside_func(self.__init__)
        for arg_name in arg_names:
            if not hasattr(self, arg_name): # the arg has not been registered in parent config
                setattr(self, arg_name, locals()[arg_name])



class AdapterModel(DeltaBase):
    config_class = AdapterConfig
    delta_type = "adapter"
    default_modified_modules = ["attn", "ff"]
    def __init__(self,
                 backbone_model: nn.Module, 
                 bottleneck_dim: Optional[int]=24, 
                 non_linearity: Optional[str]='gelu_new',
                 sequential: Optional[str] = True,
                 modified_modules: Optional[bool] = None,
                 unfrozen_modules: Optional[bool] = None,
                 common_structure: Optional[bool] = None,
                 interactive_modify: Optional[Union[bool, int]] = False,
                 reparameterize: Optional[bool] = False,
                 in_dim: Optional[int] = 4,
                 mid_dim: Optional[int] = 4,

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
        _, _, ref = self.find_module(module, key)
        adapterlayer = self.new_module_like(ref)
        self.insert_sequential_module(ref, delta_module=adapterlayer, delta_name="adapter")
    
    def new_module_like(self, module):
        module_device = get_device(module)
        adapterlayer = AdapterLayer(bottleneck_dim=self.bottleneck_dim, non_linearity=self.non_linearity, device=module_device)
        self.delta_modules.append(adapterlayer)  
        return adapterlayer
