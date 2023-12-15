import pickle
import torch
#from utils import tokenize,teacher_student_tokenize, forward_ab, f1_score, accuracy, precision, recall
from utils import *
from prediction import predict_long,predict_with_inner_monologue 
import random
from tqdm import tqdm
import os
from teacher_student_modelling import CrossEncoder, Student_teacher_model
import matplotlib.pyplot as plt
import numpy as np

from coval.coval.conll.reader import get_coref_infos
from coval.coval.eval.evaluator import evaluate_documents as evaluate
from coval.coval.eval.evaluator import muc, b_cubed, ceafe, lea
from heuristic import lh_split
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
import pandas as pd


def predict_with_inner_monologue(parallel_model, dev_ab,  device, batch_size):
    n = dev_ab['input_ids'].shape[0]
    indices = list(range(n))
    all_scores_ab = []
    with torch.no_grad():
        for i in tqdm(range(0, n, batch_size), desc='Predicting'):
            batch_indices = indices[i: i + batch_size]
            scores_ab = forward_ab(parallel_model, dev_ab, device, batch_indices, ann_attn_hidden_logits = False)
            all_scores_ab.append(scores_ab.detach().cpu())
    return torch.cat(all_scores_ab) 


def predict_with_student_model(mention_map, model_name, linear_weights_path,linear_layer_path, test_pairs, text_key='bert_doc', max_sentence_len=512, long=True):
    device = torch.device('cuda:0')
    
    device_ids = list(range(1))
    #device_ids= [1]

    student_model_path = # Path for the trained student model. If not trained, use the Longformer base model 
 
    # At inference, load trained state dictionaries and linear layer weights for initialization. 
    
    print("model path", bert_path)
    linear_weights = torch.load(linear_weights_path,  map_location=torch.device('cpu'))
    layer_weights = torch.load(layer_path,  map_location=torch.device('cpu'))
    scorer_module = Student_teacher_model(is_training=False,long=True, student_model_name=student_model_name, linear_weights=linear_weights, teacher_to_student_weights = layer_weights).to(device)
    parallel_model = torch.nn.DataParallel(scorer_module, device_ids=device_ids)
    parallel_model.module.to(device)
    student_tokenizer = parallel_model.module.student_tokenizer
    teacher_tokenizer = parallel_model.module.teacher_tokenizer
    print(student_tokenizer, teacher_tokenizer)
    m_end = parallel_model.module.end_id

    tokenizer = parallel_model.module.tokenizer
    # prepare data

    test_scores = tokenize(student_tokenizer, test_pairs, mention_map, parallel_model.module.end_id, text_key=text_key, max_sentence_len=max_sentence_len)

    scores= predict_with_inner_monologue(parallel_model, test_ab, test_ba, device, batch_size=128)

    return scores, test_pairs

def get_llm_scores(dataset, heu, split,model = None):
    dataset_folder = f'./datasets/{dataset}/'
    if dataset == 'ldc':
        mps, mps_trans =  pickle.load(open(dataset_folder + f'/{heu}/mp_mp_t_test_0.03_new.pkl', 'rb'))
        tps, fps, tns, fns = mps

        tps = tps
        fps = fps
        test_pairs = tps + fps
        test_labels = [1]*len(tps) + [0]*len(fps)  
        pairs = test_pairs
        #scores_ab =  get_gpt_scores_aida()
        bad_idx = pickle.load(open(dataset_folder + f"/bad_test_indices_{dataset}.pkl", 'rb')) 
        pairs = [y for x, y in enumerate(pairs) if x not in bad_idx]
    else:
        test_mp_mpt, _ = pickle.load(open(dataset_folder + f'/{heu}/mp_mp_t_test.pkl', 'rb'))
        tps_test, fps_test, _, _ = test_mp_mpt
        test_pairs = list(tps_test + fps_test)
        test_labels = [1] * len(tps_test) + [0] * len(fps_test)
        pairs = test_pairs
    scores1 = pickle.load(open(dataset_folder + f"/llama_full_scores_kd/{split}_{heu}_scores.pkl", 'rb'))
  
    if len(scores1)==1:
        scores1 = np.array(scores1[0])
    scores2 = scores1
    score_map = {}
    for b, ab, ba in zip(pairs, scores1, scores2):
        score_map[tuple(b)] = (float(ab), float(ba))
    #print("model score map", len(dpos_map))
    return score_map 
    
def get_cluster_scores(dataset_folder, evt_mention_map, all_mention_pairs, dataset, split, heu, similarities, long_score_map, out_name, threshold):
    curr_mentions = sorted(evt_mention_map.keys())

    # generate gold clusters key file
    curr_gold_cluster_map = [(men, evt_mention_map[men]['gold_cluster']) for men in curr_mentions]
    gold_key_file = dataset_folder + f'/evt_gold_{split}.keyfile'
    generate_key_file(curr_gold_cluster_map, 'evt', dataset_folder, gold_key_file)

    w_long_sims = []
    for p, sim in zip(all_mention_pairs, similarities):
        if tuple(p) in long_score_map:
            w_long_sims.append(long_score_map[p][0])

        elif (p[1], p[0]) in long_score_map:
            #w_long_sims.append(np.mean(long_score_map[p[1], p[0]]))
            w_long_sims.append(np.mean(long_score_map[p[0]]))
            print("coref scores", long_score_map[p[0]])
        else:
            w_long_sims.append(sim)
    print(len(curr_mentions), len(all_mention_pairs), len(w_long_sims))

    mid2cluster = cluster(curr_mentions, all_mention_pairs, w_long_sims, threshold)
    system_key_file = dataset_folder + f'/evt_gold_long_{out_name}.keyfile'
    generate_key_file(mid2cluster.items(), 'evt', dataset_folder, system_key_file)
    doc = read(gold_key_file, system_key_file)

    mr, mp, mf = np.round(np.round(evaluate(doc, muc), 3) * 100, 1)
    br, bp, bf = np.round(np.round(evaluate(doc, b_cubed), 3) * 100, 1)
    cr, cp, cf = np.round(np.round(evaluate(doc, ceafe), 3) * 100, 1)
    lr, lp, lf = np.round(np.round(evaluate(doc, lea), 3) * 100, 1)
    

    conf = np.round((mf + bf + cf) / 3, 1)
    print(dataset, split)
    final_frame = [mr, mp, mf,br, bp, bf,cr, cp, cf,  lr, lp, lf,conf ]
    result_string = f'& {heu} && {mr}  & {mp} & {mf} && {br} & {bp} & {bf} && {cr} & {cp} & {cf} && {lr} & {lp} & {lf} && {conf} \\'

    print(result_string)
    return conf, result_string, final_frame



def get_coref_scores_with_student(dataset, split, score_map, heu='lh_llama', threshold=0.5):
    dataset_folder = f'./datasets/{dataset}/'
    mention_map = pickle.load(open(dataset_folder + "/mention_map.pkl", 'rb'))
    evt_mention_map = {m_id: m for m_id, m in mention_map.items() if m['men_type'] == 'evt' and m['split'] == split}
    print("heu", heu)
    mps, mps_trans = pickle.load(open(f'./datasets/{dataset}/{heu}/mp_mp_t_{split}.pkl', 'rb'))
    print("heu all mention pairs", heu)
    _, _, _, fns = mps_trans
    tps, fps, tns, fns_nt = mps
    print(len(tps), len(fps), len(fns))
    all_mention_pairs = tps + fps
    heu_predictions = np.array([1] * len(tps) + [0] * len(fps))
    # print(len(fps,))
    conf, final_scores, final_frame = get_cluster_scores(dataset_folder, evt_mention_map, all_mention_pairs, dataset, split, heu, heu_predictions, long_score_map, out_name=heu, threshold=threshold)
    return conf,final_scores, final_frame


def save_student_scores(dataset, split, student_folder, heu='lh', threshold=0.999, text_key='bert_doc', max_sentence_len=512, long=True):
    threshold = 0.5
    dataset_folder = f'./datasets/{dataset}/'
    mention_map = pickle.load(open(dataset_folder + "/mention_map.pkl", 'rb'))
    evt_mention_map = {m_id: m for m_id, m in mention_map.items() if m['men_type'] == 'evt' and m['split'] == split}
    curr_mentions = list(evt_mention_map.keys())
    mps, mps_trans = pickle.load(open(f'./datasets/{dataset}/{heu}/mp_mp_t_{split}.pkl', 'rb'))
    tps, fps, tns, fns = mps
    tps = tps
    fps = fps
    test_pairs = tps + fps + fns
    test_labels = [1]*len(tps) + [0]*len(fps) + [1]*len(fns) 
    linear_weights_path = student_folder + "/linear.chkpt"  
    linear_layer_path = student_folder + "/teacher_to_student_layer.chkpt" #linear weights from the hidden state loss function that projects llama 4096 dim to Longformer 768
    bert_path = student_folder + '/bert'
    scores, pairs = predict_with_student_model(evt_mention_map, bert_path, linear_weights_path,linear_layer_path, test_pairs, text_key, max_sentence_len, long=False)
    predictions = scores 

    predictions = torch.squeeze(predictions) > threshold

    test_labels = torch.LongTensor(test_labels)

    print("Test accuracy:", accuracy(predictions, test_labels))
    print("Test precision:", precision(predictions, test_labels))
    print("Test recall:", recall(predictions, test_labels))
    print("Test f1:", f1_score(predictions, test_labels))

    pickle.dump(test_pairs, open(dataset_folder + f'/llama_full_scores_kd/{split}_{heu}_pairs.pkl', 'wb'))
    pickle.dump(scores open(dataset_folder + f'/llama_full_scores_kd/{split}_{heu}_scores.pkl', 'wb'))
  

if __name__ == '__main__':
    
    dataset = 'gvc'
    split = 'test'
    heu = 'lh_llama'

    student_model_path = # replace with the trianed student model path after Phase2 (CKD)
    save_student_scores(dataset, split, student_folder, heu='lh', threshold=0.5, text_key='bert_doc', max_sentence_len=512, long=True)
    #get the saved score dictionary/map 
    score_map = get_llm_scores(dataset, heu, split,model = None)
    conf,final_scores, final_frame = get_coref_scores_with_student(dataset, split, score_map, heu='lh_llama', threshold=0.5)
 
        