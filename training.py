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


def soft_cross_entropy(predicts, targets):
                student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
                targets_prob = torch.nn.functional.softmax(targets, dim=-1)
                return (- targets_prob * student_likelihood).mean()
            
def read(key, response):
    return get_coref_infos('%s' % key, '%s' % response,
            False, False, True)            
            
def train_student_teacher(dataset, student_model_name=None):
    '''
    This function traines the student model (Longformer-base)
    using soft-supervision from the Teacher (Llama2-7b-chat) in an offline manner.
    The teacher is not trained, only the rationale representations 
    from the hidden state and the attentions headsthe teacher are used during training. 
    '''
    final_coref = []
    final_f = []
    final_conf = []
    dataset_folder = f'./datasets/{dataset}/'
    print("dataset",dataset_folder)
    mention_map = pickle.load(open(dataset_folder + "/mention_map.pkl", 'rb'))
    evt_mention_map = {m_id: m for m_id, m in mention_map.items() if m['men_type'] == 'evt'}
    inner_monologue_map = pickle.load(open(dataset_folder + f'/im_map_{dataset}.pkl', 'rb'))

    device = torch.device('cuda:0')
    device_ids = list(range(1))
    #device_ids = [1]
    train_mp_mpt, _ = pickle.load(open(dataset_folder + '/lh_llama/mp_mp_t_train.pkl', 'rb'))
    dev_mp_mpt, _ = pickle.load(open(dataset_folder + '/lh_llama/mp_mp_t_dev.pkl', 'rb'))

    tps_train, fps_train, _, _ = train_mp_mpt
    tps_dev, fps_dev, _, _ = dev_mp_mpt
    train_pairs = list(tps_train + fps_train)
    train_labels = [1]*len(tps_train) + [0]*len(fps_train)
    dev_pairs = list(tps_dev + fps_dev)
    dev_labels = [1] * len(tps_dev) + [0] * len(fps_dev)
  
    student_model_path = # Path for the trained student model from the ROEC Phase
 
    # At inference, load trained state dictionaries and linear layer weights for initialization. 
    # linear_weights_path = student_model_path + "/linear.chkpt"  
    # bert_path = student_model_path + '/bert'
    # print("model path", bert_path)
    #linear_weights = torch.load(linear_weights_path,  map_location=torch.device('cpu'))
    scorer_module = Student_teacher_model(is_training=True,long=True, student_model_name=student_model_name, linear_weights=None, teacher_to_student_weights = None).to(device)
    print(scorer_module)
    parallel_model = torch.nn.DataParallel(scorer_module, device_ids=device_ids)
    parallel_model.module.to(device)
    conf, final_coref_results, final_coref_frame = train(train_pairs, train_labels,dev_pairs, dev_labels, parallel_model, \
                evt_mention_map, inner_monologue_map,dataset_folder, device,
              batch_size=16, n_iters=10, max_seq_length =512, lr_lm=0.00001, lr_class=0.0001,  lr_e1_scorer=0.0001,
              lr_e2_scorer=0.0001)
        final_coref.append(final_coref_results)
        final_f.append(final_coref_frame)
        final_conf.append(conf)
        
    print("final results,",final_coref )
    results_folder = dataset_folder + f'/llama_full_scores_kd/results/'
    final_coref_df = pd.DataFrame(final_coref)

    final_frame_df = pd.DataFrame(final_f)
   
    final_conf_df =  pd.DataFrame(final_conf)
    
    
    final_coref_df.to_csv(results_folder + f"/{dataset}results.csv")
    final_frame_df.to_csv(results_folder + f"/{dataset}results.csv")
       
    return final_coref

def get_llm_scores(pairs,scores_ab):

    scores_ab = np.array(scores_ab[0] )
    print("scores size", scores_ab.shape )

    scores_ba = scores_ab
    long_map = {}
    
    
    for b, ab, ba in zip(pairs, scores_ab, scores_ba):
        long_map[tuple(b)] = (float(ab), float(ba))
    print("d pos map", len(long_map))
    return long_map


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



def predict_with_student(dataset, split, long_score_map, heu='lh_llama', threshold=0.5):
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
    
    
def train(train_pairs,
          train_labels,
          dev_pairs,
          dev_labels,
          parallel_model,
          mention_map,
          inner_monologue_map,
          working_folder,
          device,
          batch_size=16,
          n_iters=10,
          max_seq_length = 512,
          lr_lm=0.00001,
          lr_class=0.001,
          lr_e1_scorer=0.0001,
          lr_e2_scorer=0.0001,
        ):
    bce_loss = torch.nn.BCELoss()
    mse_loss = torch.nn.MSELoss()
    
    dataset = 'gvc'
    split = 'dev'
    heu = 'lh_llama'
    
    dataset_folder = f'./datasets/{dataset}/'
    print("dataset",dataset_folder)
    
    train_loss = []
    validation_loss = []
    attention_loss= []
    last_hidden_loss = []
    logits_loss = []
    train_scores = []
#     val_scores = []
    train_logits_teacher = []
    train_logits_student = []
    loss_dict = {}
    final_coref_results = []
    final_coref_frame = []
    final_conf = []
    

    optimizer = torch.optim.AdamW([
        {'params': parallel_model.module.student_model.parameters(), 'lr': lr_lm},
        {'params': parallel_model.module.scorer.parameters(), 'lr': lr_class},
        {'params': parallel_model.module.e1_im_scorer.parameters(), 'lr': lr_e1_scorer},
        {'params': parallel_model.module.e2_im_scorer.parameters(), 'lr': lr_e2_scorer},
        
        
    ])

    student_tokenizer = parallel_model.module.student_tokenizer
    teacher_tokenizer = parallel_model.module.teacher_tokenizer
    print(student_tokenizer, teacher_tokenizer)
    m_end = parallel_model.module.end_id

    # tokenize data
    train_ab = teacher_student_tokenize(student_tokenizer, teacher_tokenizer, train_pairs,inner_monologue_map, \
    mention_map, m_end, max_sentence_len=max_seq_length, text_key='bert_doc', truncate=True, experiment_type = "with_hypothesis")
    dev_ab = teacher_student_tokenize(student_tokenizer, teacher_tokenizer, dev_pairs,inner_monologue_map,\
    mention_map, m_end, max_sentence_len=max_seq_length, text_key='bert_doc', truncate=True, experiment_type = "with_hypothesis")

    train_labels = torch.FloatTensor(train_labels)
    dev_float_labels = torch.FloatTensor(dev_labels)
    dev_labels = torch.LongTensor(dev_labels)
    
    pairwise_loss = 0.
    iteration_loss_attn = 0.
    iteration_loss_hidden = 0.
    iteration_loss_logits = 0.
    val_loss_ = 0.
    
    
    for n in range(n_iters):
        val_scores = []
        train_indices = list(range(len(train_pairs)))
        random.shuffle(train_indices)
        iteration_loss = 0.
        new_batch_size = batch_size
        for i in tqdm(range(0, len(train_indices), new_batch_size), desc='Training'):
            optimizer.zero_grad()
            batch_indices = train_indices[i: i + new_batch_size]

            scores_ab, no_im_output_full,student_output_im, teacher_last_attn, teacher_last_hidden_reps, teacher_logits, student_logits = \
            forward_ab(parallel_model, train_ab, device, batch_indices, ann_attn_hidden_logits = True) 
            train_scores.append(scores_ab)
            train_logits_teacher.append(teacher_logits)
            train_logits_student.append(student_logits)
            batch_labels = train_labels[batch_indices].reshape((-1, 1)).to(device)
            pairwise_loss = bce_loss(scores_ab, batch_labels)
            student_att = student_output_im['attentions'][-1][:, :, :,:max_seq_length] # all layes for student: use the global seq length
            teacher_att = teacher_last_attn[:, 20:32, :, :] # last n layers for teacher where n is max of student layers. Check mapping formula(f (i) = i+H −h; 0 < i ≤ h) 
            student_att = torch.where(student_att<= -1e2, torch.zeros_like(student_att).to(device),
                                      student_att)
            teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(device),
                                      teacher_att)
            attention_loss_iter = mse_loss(student_att, teacher_att )
            student_output_im['last_hidden_state'].size(),teacher_last_hidden_reps.size()
            student_rep = student_output_im['last_hidden_state'][:,0, :]
            teacher_rep = teacher_last_hidden_reps 
            student_rep.size(), teacher_last_hidden_reps.size()
            last_hidden_loss_iter = mse_loss(student_rep, teacher_rep )
            temp_label = torch.zeros(len(batch_indices),1).to(device)
            student_log = torch.cat((temp_label, student_logits), 1).to(device)
            temperature = 0.7
            logits_loss_iter = soft_cross_entropy(student_log / temperature,
                                                                  teacher_logits / temperature)
      
            
            pairwise_loss = pairwise_loss + attention_loss_iter + 0.01*last_hidden_loss_iter  #final results do NOT include logits loss since the model did not learn much with it when analyzed individually

            pairwise_loss.backward()

            optimizer.step()

            iteration_loss += pairwise_loss.item()
            iteration_loss_attn += attention_loss_iter.item()
            iteration_loss_hidden += last_hidden_loss_iter.item()
            iteration_loss_logits += logits_loss_iter.item()
            

      
        with torch.no_grad():
            print(f'Epoch {n} Loss:', iteration_loss / len(train_pairs))
            train_loss.append(iteration_loss / len(train_pairs))
           
            attention_loss.append(iteration_loss_attn/len(train_pairs))
            last_hidden_loss.append(0.01*iteration_loss_hidden/len(train_pairs))
            logits_loss.append(iteration_loss_logits/len(train_pairs))
            dev_scores = predict_with_inner_monologue(parallel_model, dev_ab, device, batch_size)
       
            dev_predictions = dev_scores  
            print(dev_predictions.size(), dev_float_labels.size())

            val_loss = bce_loss(torch.squeeze(dev_predictions), dev_float_labels)
            val_loss_ += val_loss.item()
            val_loss_ = val_loss_/len(dev_float_labels)
            
            validation_loss.append(val_loss_)
            print("validation loss", val_loss)
            val_scores.append(dev_predictions)
            dev_predictions = dev_predictions > 0.5
            dev_predictions = torch.squeeze(dev_predictions)
            

            print("dev accuracy:", accuracy(dev_predictions, dev_labels))
            print("dev precision:", precision(dev_predictions, dev_labels))
            print("dev recall:", recall(dev_predictions, dev_labels))
            print("dev f1:", f1_score(dev_predictions, dev_labels))
        if n % 1 == 0:
                loss_dict['train_loss'] = train_loss
                loss_dict['validation_loss'] = validation_loss
                loss_dict['logits_loss'] = logits_loss
                loss_dict['attention_loss'] = attention_loss
                loss_dict['last_hidden_loss'] = last_hidden_loss
                
                scorer_folder = working_folder + f'/s_t_fullloss_scorer_final/chk_{n}'
                if not os.path.exists(scorer_folder):
                    os.makedirs(scorer_folder)
                pickle.dump(loss_dict, open(scorer_folder + f'/loss_dict_student_teacher_full_longformer_{n}.pkl', 'wb'))
                model_path = scorer_folder + '/linear.chkpt'
                st_layer_path = scorer_folder + '/teacher_to_student_layer.chkpt'
                pickle.dump(dev_pairs, open(dataset_folder + f'/llama_full_scores_kd/results/{split}_{heu}_pairs.pkl', 'wb'))
                #call the final scoring function here
                long = get_llm_scores(dev_pairs,val_scores) #get the scores while training 
                conf, final_scores, final_frame = predict_with_student(dataset, split, long, heu=heu)
                print("final scores", final_scores)
                final_coref_results.append([n, n,final_scores ])
                final_coref_frame.append([samnpling, n,conf ])
                final_conf.append([n, n,final_frame ])
                pickle.dump(val_scores, open(dataset_folder + f'/llama_full_scores_kd/results/{split}_{heu}_scores_ab_{n}.pkl', 'wb'))
                
                torch.save(parallel_model.module.scorer.state_dict(), model_path)
                torch.save(parallel_model.module.teacher_to_student_layer.state_dict(), st_layer_path)
                parallel_model.module.student_model.save_pretrained(scorer_folder + '/bert')
                parallel_model.module.student_tokenizer.save_pretrained(scorer_folder + '/bert')
                print(f'saved model at {n}')
                
                
                 
 

    scorer_folder = working_folder + '/s_t_fullloss_scorer_final/'
    if not os.path.exists(scorer_folder):
        os.makedirs(scorer_folder)
    model_path = scorer_folder + '/linear.chkpt'
    st_layer_path = scorer_folder + '/teacher_to_student_layer.chkpt'
    torch.save(parallel_model.module.scorer.state_dict(), model_path)
    torch.save(parallel_model.module.teacher_to_student_layer.state_dict(), st_layer_path)
    parallel_model.module.student_model.save_pretrained(scorer_folder + '/bert')
    parallel_model.module.student_tokenizer.save_pretrained(scorer_folder + '/bert')
    
    pickle.dump(dev_pairs, open(dataset_folder + f'/llama_full_scores_kd/results/{split}_{heu}_pairs.pkl', 'wb'))
    pickle.dump(val_scores, open(dataset_folder + f'/llama_full_scores_kd/results/{split}_{heu}_scores_ab_{n}.pkl', 'wb'))
    
    plt.plot(range(n_iters), train_loss)
    plt.plot(range(n_iters), validation_loss)
    plt.plot(range(n_iters), logits_loss)
    plt.plot(range(n_iters), attention_loss)
    
    plt.plot(range(n_iters), last_hidden_loss)
  
    
    
    
    

    return final_conf, final_coref_results, final_coref_frame

if __name__ == '__main__':
    
    dataset = 'gvc'
    split = 'train'
    heu = 'lh_llama'
    
        
    final_coref_results = train_student_teacher('gvc', student_model_name='allenai/longformer-base-4096')
        
       
     
    
    



