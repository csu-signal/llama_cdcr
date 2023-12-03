import torch.nn as nn
import torch
from transformers import AutoModel, AutoTokenizer
from transformers import LongformerForMaskedLM, LongformerTokenizer, LlamaForCausalLM, LlamaForSequenceClassification, LlamaModel, LlamaConfig
import numpy as np
import pyhocon
import os
from inspect import getfullargspec
from collections import defaultdict
import sklearn.linear_model
import copy


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.uniform_(m.bias)


class Student_teacher_model(nn.Module):

    """
    Initializes the teacher model (llama 7b-chat) and student model (Longformer)
    with special tokens and outputs pairwise scores and teacher/student represenations for training. 
    Parameters

    """

    def __init__(self, is_training=True, long=True, student_model_name='allenai/longformer-base-4096',
    teacher_model_name = None, 
                 linear_weights=None, teacher_to_student_weights = None, e1_linear = None, e_2_linear = None):
        super(Student_teacher_model, self).__init__()

        PATH_TO_CONVERTED_WEIGHTS =  " " # model path to your teacher model 
        PATH_TO_CONVERTED_TOKENIZER = " " # model path to your teacher tokenizer 
        teacher_config = LlamaConfig.from_json_file('') " " # model path to your teacher config files 
        teacher_config.num_labels = 2
        

        self.student_tokenizer = AutoTokenizer.from_pretrained(student_model_name) #longformer as the student model 
        self.teacher_tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)
        self.teacher_tokenizer.pad_token = "[PAD]"
        self.teacher_tokenizer.padding_side = "right"
        #self.teacher_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        #teacher_config.pad_token_id = 0
        self.long = long 
        self.teacher_model = LlamaForSequenceClassification.from_pretrained(PATH_TO_CONVERTED_WEIGHTS, load_in_8bit=False, config =teacher_config)
         #use the finetuned teacher model (frozen) with a sequence classification head 
        self.teacher_model.config.pad_token_id = self.teacher_model.config.eos_token_id
        if is_training:
            self.student_tokenizer.add_tokens(['<m>', '</m>'], special_tokens=True)
            self.student_tokenizer.add_tokens(['<doc-s>', '</doc-s>'], special_tokens=True)
            self.student_tokenizer.add_tokens(['<g>'], special_tokens=True)
            self.student_model = AutoModel.from_pretrained(student_model_name)

            #self.model = LongformerForMaskedLM.from_pretrained('allenai/longformer-base-4096', return_dict=True)
            self.student_model.resize_token_embeddings(len(self.student_tokenizer))
        else:
            self.student_model = AutoModel.from_pretrained(student_model_name)

        self.start_id = self.student_tokenizer.encode('<m>', add_special_tokens=False)[0]
        self.end_id = self.student_tokenizer.encode('</m>', add_special_tokens=False)[0]

        self.hidden_size_student = self.student_model.config.hidden_size
        self.hidden_size_teacher = self.teacher_model.config.hidden_size
        



        self.scorer = nn.Sequential(
            nn.Linear(self.hidden_size_student* 4, self.hidden_size_student),
            nn.Tanh(),
            nn.Linear(self.hidden_size_student, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        self.logit_scorer = nn.Sequential(
            nn.Linear(self.hidden_size_student* 4, self.hidden_size_student),
            nn.Tanh(),
            nn.Linear(self.hidden_size_student, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
         
        )

        self.e1_im_scorer = nn.Sequential(
            nn.Linear(self.hidden_size_student* 2, self.hidden_size_student),
            nn.Tanh(),
            nn.Linear(self.hidden_size_student, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        self.e2_im_scorer = nn.Sequential(
            nn.Linear(self.hidden_size_student*2 , self.hidden_size_student),
            nn.Tanh(),
            nn.Linear(self.hidden_size_student, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        self.teacher_to_student_layer = nn.Sequential(
            nn.Linear(self.hidden_size_teacher, self.hidden_size_student)
        )

        if linear_weights is None:
            self.scorer.apply(init_weights)
            self.logit_scorer.apply(init_weights)
        else:
            self.scorer.load_state_dict(linear_weights)

        if teacher_to_student_weights is None:
            self.teacher_to_student_layer.apply(init_weights)
        else:
            self.teacher_to_student_layer.load_state_dict(teacher_to_student_weights)
        if e1_linear is None:
            self.e1_im_scorer.apply(init_weights)
        else:
            self.e1_im_scorer.load_state_dict(e1_linear)
        if e_2_linear is None:
            self.e2_im_scorer.apply(init_weights)
        else:
            self.e2_im_scorer.load_state_dict(e_2_linear)
        


    def generate_cls_arg_vectors(self, input_ids, attention_mask, position_ids,
                                 global_attention_mask, arg1, arg2):
        arg_names = set(getfullargspec(self.student_model).args)

        if self.long:
            output = self.student_model(input_ids,
                                position_ids=position_ids,
                                attention_mask=attention_mask,
                                global_attention_mask=None,  output_hidden_states =True, output_attentions = True)
        else:
            output = self.student_model(input_ids,
                                attention_mask=attention_mask,  output_hidden_states =True, output_attentions = True)

       

        last_hidden_states = output.last_hidden_state
        cls_vector = output.pooler_output
        #print("output keys",output.keys())

        arg1_vec = None
        if arg1 is not None:
            arg1_vec = (last_hidden_states * arg1.unsqueeze(-1)).sum(1)
        arg2_vec = None
        if arg2 is not None:
            arg2_vec = (last_hidden_states * arg2.unsqueeze(-1)).sum(1)


        return cls_vector, arg1_vec, arg2_vec, output
    def generate_model_output(self, input_ids, attention_mask, position_ids,
                              global_attention_mask, arg1, arg2):
        cls_vector, arg1_vec, arg2_vec, output = self.generate_cls_arg_vectors(input_ids, attention_mask, position_ids,
                                                                       global_attention_mask, arg1, arg2)

        return torch.cat([cls_vector, arg1_vec, arg2_vec, arg1_vec * arg2_vec], dim=1), output

    def generate_teacher_student_output(self, input_ids, attention_mask, position_ids,
                              global_attention_mask, arg1, arg2,
                              im_student_input_ids,
                              im_student_am, 
                              im_teacher_input_ids,
                              im_teacher_am
                              ):

        cls_vector, arg1_vec, arg2_vec, no_im_output_full = self.generate_cls_arg_vectors(input_ids, attention_mask, position_ids,
                                                                       global_attention_mask, arg1, arg2)
         #get the IM sample represenatations for teacher and student models 

        self.teacher_tokenizer.pad_token = "[PAD]"
        self.teacher_tokenizer.padding_side = "right"
        student_output_im = self.student_model(im_student_input_ids,
                                attention_mask=im_student_am,  output_hidden_states =True, output_attentions = True)
        #print("student attn size at incidence", student_output_im['attentions'][-1][:, :, :,:].size())
        with torch.no_grad():                     
            teacher_output_im = self.teacher_model(im_teacher_input_ids,
                                    attention_mask=im_teacher_am,  output_hidden_states =True, output_attentions = True)

            #teacher_last_attn = teacher_output_im['attentions'][-1][:, -1, :,:]
            teacher_last_attn = teacher_output_im['attentions'][-1] 
            #teacher_last_hidden_reps = teacher_output_im['hidden_states'][-1][:,:,:]
            teacher_last_hidden_reps = teacher_output_im['hidden_states'][-1][:,0,:]
            print("last hidden teacher size", teacher_last_hidden_reps.size())
            teacher_logits = teacher_output_im['logits']



        no_im_output  = torch.cat([cls_vector, arg1_vec, arg2_vec, arg1_vec * arg2_vec], dim=1) 
        
        return no_im_output, no_im_output_full, student_output_im,teacher_last_attn, teacher_last_hidden_reps, teacher_logits
  

    def frozen_forward(self, input_):
        return self.linear(input_)

    def forward(self, input_ids, attention_mask=None, position_ids=None,
                global_attention_mask=None, arg1=None, arg2=None, concat_reps_teacher_student = None, lm_only=False, pre_lm_out=False, ann_attn_hidden_logits=False):

        if pre_lm_out:
            return self.linear(input_ids)
        if ann_attn_hidden_logits:
            #print("getting teacher and student for IM samples")
         
            im_student_input_ids, im_student_am, im_teacher_input_ids,im_teacher_am = concat_reps_teacher_student
            no_im_output, no_im_output_full, student_output_im,teacher_last_attn, teacher_last_hidden_reps, teacher_logits= self.generate_teacher_student_output(input_ids, attention_mask=attention_mask,
                                                global_attention_mask=global_attention_mask,
                                                position_ids=position_ids,
                                                arg1=arg1, arg2=arg2,
                                                im_student_input_ids = im_student_input_ids,
                                                im_student_am = im_student_am,
                                                im_teacher_input_ids = im_teacher_input_ids,
                                                im_teacher_am = im_teacher_am
                                                
                                                
                                                )
            #teacher_last_hidden_reps = torch.squeeze(teacher_last_hidden_reps)       
            #print("last hidden teacher size after squeezing", teacher_last_hidden_reps.size())                            
            teacher_last_hidden_reps = self.teacher_to_student_layer(teacher_last_hidden_reps)

            sd = copy.deepcopy(self.scorer.state_dict())
            self.logit_scorer.load_state_dict(sd) #loading the state dict of scorer module to get the logits 
            student_logits = self.logit_scorer(no_im_output)
            return self.scorer(no_im_output), no_im_output_full,student_output_im, teacher_last_attn, teacher_last_hidden_reps, teacher_logits, student_logits

        else:
            #print("no IM samples/hypothesis ")
            lm_output, output = self.generate_model_output(input_ids, attention_mask=attention_mask,
                                                global_attention_mask=global_attention_mask,
                                                position_ids=position_ids,
                                                arg1=arg1, arg2=arg2)
            return self.scorer(lm_output)
       
class CrossEncoder(nn.Module):
    """
    Initializes the student model for cross encoding event pairs
    Parameters

    """
    def __init__(self, is_training=True, long=True, model_name='allenai/longformer-base-4096',
                 linear_weights=None):
        super(CrossEncoder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.long = long

        if is_training:
            self.tokenizer.add_tokens(['<m>', '</m>'], special_tokens=True)
            self.tokenizer.add_tokens(['<doc-s>', '</doc-s>'], special_tokens=True)
            self.tokenizer.add_tokens(['<g>'], special_tokens=True)
            self.model = AutoModel.from_pretrained(model_name)
            #self.model = LongformerForMaskedLM.from_pretrained('allenai/longformer-base-4096', return_dict=True)
            self.model.resize_token_embeddings(len(self.tokenizer))
        else:
            self.model = AutoModel.from_pretrained(model_name)

        self.start_id = self.tokenizer.encode('<m>', add_special_tokens=False)[0]
        self.end_id = self.tokenizer.encode('</m>', add_special_tokens=False)[0]

        self.hidden_size = self.model.config.hidden_size

        self.linear = nn.Sequential(
            nn.Linear(self.hidden_size * 4, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        if linear_weights is None:
            self.linear.apply(init_weights)
        else:
            self.linear.load_state_dict(linear_weights)

    def generate_cls_arg_vectors(self, input_ids, attention_mask, position_ids,
                                 global_attention_mask, arg1, arg2):
        arg_names = set(getfullargspec(self.model).args)

        if self.long:
            output = self.model(input_ids,
                                position_ids=position_ids,
                                attention_mask=attention_mask,
                                global_attention_mask=None,  output_hidden_states =True, output_attentions = True)
        else:
            output = self.model(input_ids,
                                attention_mask=attention_mask,  output_hidden_states =True, output_attentions = True)

        last_hidden_states = output.last_hidden_state
        cls_vector = output.pooler_output
       # print("output keys",output.keys())

        arg1_vec = None
        if arg1 is not None:
            arg1_vec = (last_hidden_states * arg1.unsqueeze(-1)).sum(1)
        arg2_vec = None
        if arg2 is not None:
            arg2_vec = (last_hidden_states * arg2.unsqueeze(-1)).sum(1)


        return cls_vector, arg1_vec, arg2_vec, output
    def generate_model_output(self, input_ids, attention_mask, position_ids,
                              global_attention_mask, arg1, arg2):
        cls_vector, arg1_vec, arg2_vec, output = self.generate_cls_arg_vectors(input_ids, attention_mask, position_ids,
                                                                       global_attention_mask, arg1, arg2)

        return torch.cat([cls_vector, arg1_vec, arg2_vec, arg1_vec * arg2_vec], dim=1), output

    def frozen_forward(self, input_):
        return self.linear(input_)

    def forward(self, input_ids, attention_mask=None, position_ids=None,
                global_attention_mask=None, arg1=None, arg2=None, lm_only=False, pre_lm_out=False, ann_attn_hidden_logits=False):

        # attention_mask[global_attention_mask == 1] = 2

        if pre_lm_out:
            return self.linear(input_ids)

        lm_output, output = self.generate_model_output(input_ids, attention_mask=attention_mask,
                                               global_attention_mask=global_attention_mask,
                                               position_ids=position_ids,
                                               arg1=arg1, arg2=arg2)
        if lm_only:
            return lm_output
        if ann_attn_hidden_logits:
            # pairwise_scores = self.linear(lm_output)
            return self.linear(lm_output), output

        return self.linear(lm_output)
