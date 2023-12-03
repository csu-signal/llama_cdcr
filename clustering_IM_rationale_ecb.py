import os
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification,AutoTokenizer, LongformerForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from transformers import AdamW
import matplotlib.pyplot as plt
import pickle
import collections

def main(dataset):
    
    dataset_folder = f'./datasets/{dataset}/'
    mention_map = pickle.load(open(dataset_folder + "/mention_map.pkl", 'rb'))
    evt_mention_map = {m_id: m for m_id, m in mention_map.items() if m['men_type'] == 'evt'}
    evt_mention_map_train = {m_id: m for m_id, m in mention_map.items() if m['men_type'] == 'evt' and m['split']=='train'}
    evt_mention_map_dev = {m_id: m for m_id, m in mention_map.items() if m['men_type'] == 'evt' and m['split']=='dev'}
    inner_monologue_map = pickle.load(open(dataset_folder + f'/im_map_{dataset}.pkl', 'rb'))
    inner_monologue_map_train =  {im_id: m for im_id, m in inner_monologue_map.items() if m['split']=='train'}
    inner_monologue_map_dev = {im_id: m for im_id, m in inner_monologue_map.items() if m['split']=='dev'}

    evt_mention_map_train.update(evt_mention_map_dev)

    #get the clusters

    cluster_set = []
    for x, y in evt_mention_map_train.items():
        cluster_set.append(y['gold_cluster'])

    len(set(cluster_set))

    counter = collections.Counter(cluster_set)
    #cluster_dict = counter.items()

    cluster_dict = dict(counter)
    #cluster_dict = dict(sorted(cluster_dict.items(), key=lambda item: item[1]))
    clus = [x for x, y in cluster_dict.items() if y >1] # without considering the singletons # appply kenyon dean's techique
    clus.append("dummy")

    #make a cluster to id map
    cluster_to_label_map = {x:y for x, y in enumerate(clus) } # if pairs is not in the same cluster--> add dummy class 
    label_to_cluster_map = {y:x for x, y in cluster_to_label_map.items()}


    #get the clusters for all the train and dev pairs 
   
    dataset_folder = f'./datasets/{dataset}/'
    mention_map = pickle.load(open(dataset_folder + "/mention_map.pkl", 'rb'))
    evt_mention_map = {m_id: m for m_id, m in mention_map.items() if m['men_type'] == 'evt'}
    inner_monologue_map = pickle.load(open(dataset_folder + f'/im_map_{dataset}.pkl', 'rb'))

    device = torch.device('cuda:0')
    device_ids = list(range(1))
    device_ids = [1]
    train_mp_mpt, _ = pickle.load(open(dataset_folder + '/lh_llama/mp_mp_t_train.pkl', 'rb'))
    dev_mp_mpt, _ = pickle.load(open(dataset_folder + '/lh_llama/mp_mp_t_dev.pkl', 'rb'))
    #test_mp_mpt, _ = pickle.load(open(dataset_folder + '/lh_llama/mp_mp_t_test.pkl', 'rb')) # not needed for clustering since we're validating on dev set
    test_mp_mpt, _ = pickle.load(open(dataset_folder + '/lh_llama/mp_mp_t_test.pkl', 'rb'))



    tps_test, fps_test, _, _ = test_mp_mpt
    test_pairs = list(tps_test + fps_test)
    test_labels = [1] * len(tps_test) + [0] * len(fps_test)

    tps_train, fps_train, _, _ = train_mp_mpt
    tps_dev, fps_dev, _, _ = dev_mp_mpt

    train_pairs = list(tps_train + fps_train)
    train_labels = [1]*len(tps_train) + [0]*len(fps_train)

    dev_pairs = list(tps_dev + fps_dev)
    dev_labels = [1] * len(tps_dev) + [0] * len(fps_dev)
    len(train_pairs), len(dev_pairs)


    #create cluster to label maps 
    cluster_labels_train = []
    cluster_labels_dev = []

    for p,l in zip(train_pairs, train_labels):
        if l ==1:
            cluster = evt_mention_map_train[p[0]]['gold_cluster']
            label = label_to_cluster_map[cluster]
            cluster_labels_train.append(label)
        else:
            cluster_labels_train.append(label_to_cluster_map['dummy'])

    for p,l in zip(dev_pairs, dev_labels):
        if l ==1:
            cluster = evt_mention_map_train[p[0]]['gold_cluster']
            label = label_to_cluster_map[cluster]
            cluster_labels_dev.append(label)
        else:
            cluster_labels_dev.append(label_to_cluster_map['dummy'])

    pairwise_bert_instances_train = []
    pairwise_bert_instances_dev = []
    event_pair_sep_train = []
    hypo_sep_train = []
    event_pair_sep_dev = []
    hypo_sep_dev = []

    text_key = 'bert_sentence'

    doc_start = '<doc-s>'
    doc_end = '</doc-s>'
    # print("mention pair length" ,len(mention_pairs))

    def make_instance_full(sent_a, sent_b, sent_c=None):
        return ' '.join(['<g>', doc_start, sent_a, doc_end, doc_start, sent_b,\
                       doc_end,  doc_start, sent_c, doc_end ]) 

    for (m1, m2) in train_pairs:
        #print("mentions", m1,m2)

        sentence_a = evt_mention_map[m1][text_key]
        sentence_b = evt_mention_map[m2][text_key]
        #print("sentence", sentence_a, sentence_b)

        im_sample = inner_monologue_map[(m1, m2)]['im_generated_sample'].replace("\n", "")
        full_doc = make_instance_full(sentence_a, sentence_b,im_sample)
        event = make_instance_full(sentence_a, sentence_b,'')
        hypo = doc_start + im_sample + doc_end

        pairwise_bert_instances_train.append(full_doc)
        event_pair_sep_train.append(event)
        hypo_sep_train.append(hypo)
    for (m1, m2) in dev_pairs:
        #print("mentions", m1,m2)

        sentence_a = evt_mention_map[m1][text_key]
        sentence_b = evt_mention_map[m2][text_key]
        #print("sentence", sentence_a, sentence_b)

        im_sample = inner_monologue_map[(m1, m2)]['im_generated_sample'].replace("\n", "")
        full_doc = make_instance_full(sentence_a, sentence_b,im_sample)
        event = make_instance_full(sentence_a, sentence_b,'')
        hypo = doc_start + im_sample + doc_end

        pairwise_bert_instances_dev.append(full_doc)
        event_pair_sep_dev.append(event)
        hypo_sep_dev.append(hypo)

    # pairwise_bert_instances_train[1], cluster_labels_train[1], pairwise_bert_instances_dev[0], cluster_labels_dev[0]
    len(pairwise_bert_instances_train), len(cluster_labels_train),\
    len(pairwise_bert_instances_dev), len(cluster_labels_dev),\
    len(event_pair_sep_train), len(event_pair_sep_dev), \
    len(hypo_sep_train), len(hypo_sep_dev)

    train_texts = pairwise_bert_instances_train
    train_labels = cluster_labels_train 
    test_texts = pairwise_bert_instances_dev 
    test_labels =  cluster_labels_dev 
    train_hypo = hypo_sep_train 
    test_hypo = hypo_sep_dev
    print(len(train_texts)), print(len(test_texts)), print(len(train_labels)), print(len(test_labels))

    num_labels = len(label_to_cluster_map)

    tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
    model = LongformerForSequenceClassification.from_pretrained("allenai/longformer-base-4096", num_labels=num_labels)
    special_tokens_dict = {'additional_special_tokens': ['<g>','<m>', '</m>', '<doc-s>', '</doc-s>','SEP']} #special tokens for global attention and document boundaries 
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    start_id = tokenizer.encode('<doc-s>', add_special_tokens=False)[0]
    end_id = tokenizer.encode('</doc-s>', add_special_tokens=False)[0]
    sep_id = tokenizer.encode('[SEP]', add_special_tokens=False)[0]
    start_id,end_id,sep_id


    # Tokenize the input texts
    train_encodings_event = tokenizer(train_texts, truncation=True, padding=True, max_length=512, return_tensors='pt')
    train_encodings_hypo = tokenizer(train_hypo, truncation=True, padding=True, max_length=512, return_tensors='pt')

    test_encodings_event = tokenizer(test_texts, truncation=True, padding=True, max_length=512, return_tensors='pt')
    test_encodings_hypo = tokenizer(test_hypo, truncation=True, padding=True, max_length=512, return_tensors='pt')


    print(train_encodings_event['input_ids'].size(), train_encodings_event['attention_mask'].size(),\
    train_encodings_hypo['input_ids'].size(), train_encodings_hypo['attention_mask'].size())
    print(test_encodings_event['input_ids'].size(), test_encodings_event['attention_mask'].size(),\
    test_encodings_hypo['input_ids'].size(), test_encodings_hypo['attention_mask'].size())

    # Convert labels to PyTorch tensors
    train_labels = torch.tensor(train_labels)
    test_labels = torch.tensor(test_labels)

    # Create DataLoader for training and testing sets

    #Tokenizer events and the IM hypothesis separately and pad to max-length 

    train_dataset = TensorDataset(train_encodings_event['input_ids'], train_encodings_event['attention_mask'],\
                                  train_encodings_hypo['input_ids'], train_encodings_hypo['attention_mask'], train_labels)
    test_dataset = TensorDataset(test_encodings_event['input_ids'], test_encodings_event['attention_mask'],\
                                 test_encodings_hypo['input_ids'], test_encodings_hypo['attention_mask'],test_labels)

    train_loader = DataLoader(train_dataset, batch_size=40, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=40, shuffle=False)

    # Set up optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=1e-5)
    cross_entropy_loss_fn = torch.nn.CrossEntropyLoss()

    # Custom loss function with CrossEntropyLoss and cosine regularization of IM rationales with event pairs 
    def custom_loss(outputs, labels, event_embedding, rationale_embedding):
        # CrossEntropyLoss
        alpha = 0. # cosine distance regularization parameter after tuning 
        ce_loss = cross_entropy_loss_fn(outputs.logits, labels)

        cosine_similarity = torch.nn.functional.cosine_similarity(event_embedding, rationale_embedding)
        cosine_distance = 1 - cosine_similarity
        cosine_loss = torch.mean(cosine_distance)
        loss = ce_loss+ alpha*cosine_loss
        return ce_loss

    # Training loop
    num_epochs = 30 # try with various epochs since there are ~500 clusters 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    train_losses = []  # Track training losses
    val_losses = []    # Track validation losses
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            input_ids, attention_mask,input_ids_hypo, attention_mask_hypo, labels = batch
            input_ids, attention_mask,input_ids_hypo, attention_mask_hypo, labels = input_ids.to(device), \
            attention_mask.to(device), input_ids_hypo.to(device), attention_mask_hypo.to(device),labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states =True)
            outputs_hypo = model(input_ids_hypo, attention_mask=attention_mask_hypo, labels=labels, output_hidden_states =True)
            cls_token = outputs['hidden_states'][-1][:,0,:]  # CLS token
            cls_token_hypo = outputs_hypo['hidden_states'][-1][:,0,:]  # CLS token for IM rationale 
            #print(cls_token.size())
            #last_hidden_state = outputs.last_hidden_state  # Last hidden state

            loss = outputs.loss
            loss = custom_loss(outputs, labels, cls_token, cls_token_hypo)

            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss}')
        train_losses.append(avg_loss)
        
        # Evaluation
        # Evaluation on the validation set
        model.eval()
        all_preds = []
        all_labels = []
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Evaluating'):
                input_ids, attention_mask,input_ids_hypo, attention_mask_hypo, labels= batch
                input_ids, attention_mask,input_ids_hypo, attention_mask_hypo, labels = input_ids.to(device), \
            attention_mask.to(device), input_ids_hypo.to(device), attention_mask_hypo.to(device),labels.to(device)

                outputs = model(input_ids, attention_mask=attention_mask,output_hidden_states =True)
                outputs_hypo = model(input_ids_hypo, attention_mask=attention_mask_hypo, labels=labels, output_hidden_states =True)

                # Retrieve the CLS token and the last hidden state
                cls_token = outputs.hidden_states[-1][:, 0, :]  # CLS token
                cls_token_hypo = outputs_hypo['hidden_states'][-1][:,0,:]  # CLS token for IM rationale 
                #last_hidden_state = outputs.hidden_states[-1]  # Last hidden state

                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                #loss = outputs.loss
                loss = custom_loss(outputs, labels, cls_token, cls_token_hypo)
                total_val_loss += loss.item()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                print("prediction",preds )
                print("labels",labels )
        avg_val_loss = total_val_loss / len(test_loader)
        # Calculate accuracy
        accuracy = accuracy_score(all_labels, all_preds)
        print(f'Epoch {epoch + 1}/{num_epochs}, Average Validation Loss: {avg_val_loss}')
        print(f'Accuracy on the validation set: {accuracy}')
        val_losses.append(avg_val_loss)
        if epoch % 5 == 0:

        scorer_folder = dataset_folder + f'/rational_cluster_scorer/chk_{epoch}'
        if not os.path.exists(scorer_folder):
            os.makedirs(scorer_folder)
    
        model.save_pretrained(scorer_folder + '/bert')
        tokenizer.save_pretrained(scorer_folder + '/bert')
        print(f'saved model at {epoch}')

        
        
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.savefig(scorer_folder + 'loss_plot.png')
    scorer_folder = dataset_folder + f'/rational_cluster_scorer/chk_{epoch}'
    if not os.path.exists(scorer_folder):
        os.makedirs(scorer_folder)

    model.save_pretrained(scorer_folder + '/bert')
    tokenizer.save_pretrained(scorer_folder + '/bert')
    print(f'saved model at {epoch}')

   


if __name__ == '__main__':
    main(dataset = 'ecb')






