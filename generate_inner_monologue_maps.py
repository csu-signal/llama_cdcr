#load and create IM maps for ECB and GVC and then pickle them

import pickle
from collections import defaultdict

def create_system_user_prompts(dataset, mention_map, split, men_type = "evt", heu = "lh"):
    """
    Generates the Inner-Monologue system and user prompts using trigger words and sentences of paired events
    from LH or LH oracle heuristic.
    Returns: list of system and user prompts for each event mention pair (Str)
    """
    dataset_folder = f'/s/chopin/d/proj/ramfis-aida/multimodal_NLI/Multimodal_CDCR/acl_submission_2023-main/datasets/{dataset}/'
    mention_map = pickle.load(open(dataset_folder + "/mention_map.pkl", 'rb'))
    curr_mention_map = {m_id: m for m_id, m in mention_map.items() if m['men_type'] == men_type and m['split'] == split}
    split_mp_mpt, _ = pickle.load(open(dataset_folder + f'/{heu}/mp_mp_t_{split}.pkl', 'rb'))
    tps, fps, _, _ = split_mp_mpt
    pairs_lh = list(tps + fps)
    labels = [1]*len(tps) + [0]*len(fps)

    print(f"Generating Inner Monologue Samples for {len(pairs_lh), len(pairs_lh)} samples for {split} split")

    prompt_list = []
    dialog_list = []
    for i, (x, y) in enumerate(zip(pairs_lh,labels )):

        sentence_a = curr_mention_map[x[0]]['bert_sentence']
        sentence_b = curr_mention_map[x[1]]['bert_sentence']
        inner_mon_prompt_1 = f'''Create an inner monologue based on chain-of-thought
    reasoning to answer a question about documents A and B.
    Then, answer whether they are talking about the same event or not. '''
        inner_mon_prompt_2 = f'''

    The event trigger-words are <m> {curr_mention_map[x[0]]['mention_text']} </m> in
    document A and <m> {curr_mention_map[x[1]]['mention_text']} </m> in document B.
    The inner monologue must make this decision after identifying common
    context, actions, actors, objects and locations.
    For instance, are the entities related to the event-trigger words
    <m> {curr_mention_map[x[0]]['mention_text']} </m>
    and <m> {curr_mention_map[x[1]]['mention_text']} </m> the same?
    Resolve who is being referred to by the pronouns used to make your decision if needed.
    The language in the two documents can be ambiguous so you must reason carefully.
    Do not mix the entities talked about in the documents and answer based on facts.
    Use your inner monologue to decide whether or not the two events are the same event.

    '''

        if y==0 :
            user_prompt = f'''Document A is:{sentence_a}. Document B is: {sentence_b}'''
            hint_promt = f'''Hint: They are NOT referring to the same event, so reverse-engineer your reasoning!!'''
            inner_mon_prompt_1 = "".join(inner_mon_prompt_1)
            inner_mon_prompt_2 = "".join(inner_mon_prompt_2)
            final_sytem_prompt = inner_mon_prompt_1 + hint_promt + inner_mon_prompt_2
            final_sytem_prompt = final_sytem_prompt.replace("\n", "")
            final_sytem_prompt = "".join(final_sytem_prompt)
            prompt_list.append(final_sytem_prompt)
            system_dict = {
                    "role": "system",
                    "content": final_sytem_prompt,
                }
            user_dict = {"role": "user", "content": user_prompt}
            final_dialog = [system_dict, user_dict]
            dialog_list.append(final_dialog)
        else:
            user_prompt = f'''Document A is:{sentence_a}. Document B is: {sentence_b}'''
            hint_promt = f'''Hint: They ARE referring to the same event, so reverse-engineer your reasoning!!'''
            inner_mon_prompt_1 = "".join(inner_mon_prompt_1)
            inner_mon_prompt_2 = "".join(inner_mon_prompt_2)
            final_sytem_prompt = inner_mon_prompt_1 + hint_promt + inner_mon_prompt_2
            final_sytem_prompt = final_sytem_prompt.replace("\n", "")
            final_sytem_prompt = "".join(final_sytem_prompt)
            prompt_list.append(final_sytem_prompt)
            system_dict = {
                    "role": "system",
                    "content": final_sytem_prompt,
                }
            user_dict = {"role": "user", "content": user_prompt}
            final_dialog = [system_dict, user_dict]
            dialog_list.append(final_dialog)

    return dialog_list

def test_length_a_equals_length_b(IM_samples,LH_samples):
    assert len(IM_samples) ==  len(LH_samples), f"IM samples != LH samples"


def get_inner_monologue_samples(dataset):
    """
    Loads the LLM generated Inner-Monologue samples.
    Returns: List of generated IM samples for each split (train, dev and test)
    """

    if dataset == 'ecb':
        dataset_folder = f'./datasets/{dataset}'
        mention_map = pickle.load(open(dataset_folder + "/mention_map.pkl", 'rb'))
        evt_mention_map_train = {m_id: m for m_id, m in mention_map.items() if m['men_type'] == 'evt' and m['split'] == 'train'}
        evt_mention_map_dev = {m_id: m for m_id, m in mention_map.items() if m['men_type'] == 'evt' and m['split'] == 'dev'}
        evt_mention_map_test = {m_id: m for m_id, m in mention_map.items() if m['men_type'] == 'evt' and m['split'] == 'test'}

        mention_map = pickle.load(open(dataset_folder + "/mention_map.pkl", 'rb'))
        evt_mention_map = {m_id: m for m_id, m in mention_map.items() if m['men_type'] == 'evt'}

        train_mp_mpt, _ = pickle.load(open(dataset_folder + '/lh/mp_mp_t_train.pkl', 'rb'))
        dev_mp_mpt, _ = pickle.load(open(dataset_folder + '/lh/mp_mp_t_dev.pkl', 'rb'))
        test_mp_mpt, _ = pickle.load(open(dataset_folder + '/lh/mp_mp_t_test.pkl', 'rb'))
        tps_train, fps_train, _, _ = train_mp_mpt
        tps_dev, fps_dev, _, _ = dev_mp_mpt
        tps_test, fps_test, _, _ = test_mp_mpt
        train_pairs_lh = list(tps_train + fps_train)
        train_labels = [1]*len(tps_train) + [0]*len(fps_train)
        dev_pairs_lh = list(tps_dev + fps_dev)
        dev_labels = [1] * len(tps_dev) + [0] * len(fps_dev)
        test_pairs_lh = list(tps_test + fps_test)
        test_labels = [1] * len(tps_test) + [0] * len(fps_test)
        ecb_train_dict = pickle.load(open(f"result_dict_train_falcon.pkl", 'rb'))
        ecb_train_dict_1 = pickle.load(open(f"result_dict_train_2.pkl", 'rb'))
        ecb_dev_dict =  pickle.load(open(f"result_dict_dev_falcon.pkl", 'rb'))
        ecb_test_dict =  pickle.load(open(f"result_dict_test_falcon.pkl", 'rb'))

        ecb_train_dict = [item for sublist in ecb_train_dict for item in sublist]
        ecb_train_dict_1 = [item for sublist in ecb_train_dict_1 for item in sublist]
        im_samples_dev_full = [item for sublist in ecb_dev_dict for item in sublist]
        im_samples_test_full = [item for sublist in ecb_test_dict for item in sublist]
        im_samples_train_full = ecb_train_dict + ecb_train_dict_1
        print(len(ecb_train_dict), len(ecb_dev_dict), len(ecb_test_dict))
        print(f"{dataset} IM samples splits: train, dev and test", len(im_samples_train_full),len(im_samples_dev_full),len(im_samples_test_full) )
        print(f"{dataset} LH heuristic sample splits: train, dev and test", len(train_pairs_lh), len(dev_pairs_lh), len(test_pairs_lh))
        print(f"{dataset} LH label splits: train, dev and test", len(train_labels), len(dev_labels), len(test_labels))

    else:

        dataset_folder = f'./datasets/{dataset}'
        mention_map = pickle.load(open(dataset_folder + "/mention_map.pkl", 'rb'))
        evt_mention_map_train = {m_id: m for m_id, m in mention_map.items() if m['men_type'] == 'evt' and m['split'] == 'train'}
        evt_mention_map_dev = {m_id: m for m_id, m in mention_map.items() if m['men_type'] == 'evt' and m['split'] == 'dev'}
        evt_mention_map_test = {m_id: m for m_id, m in mention_map.items() if m['men_type'] == 'evt' and m['split'] == 'test'}

        mention_map = pickle.load(open(dataset_folder + "/mention_map.pkl", 'rb'))
        evt_mention_map = {m_id: m for m_id, m in mention_map.items() if m['men_type'] == 'evt'}

        train_mp_mpt, _ = pickle.load(open(dataset_folder + '/lh/mp_mp_t_train.pkl', 'rb'))
        dev_mp_mpt, _ = pickle.load(open(dataset_folder + '/lh/mp_mp_t_dev.pkl', 'rb'))

        test_mp_mpt, _ = pickle.load(open(dataset_folder + '/lh/mp_mp_t_test.pkl', 'rb'))

        tps_train, fps_train, _, _ = train_mp_mpt
        tps_dev, fps_dev, _, _ = dev_mp_mpt
        tps_test, fps_test, _, _ = test_mp_mpt


        train_pairs_lh = list(tps_train + fps_train)
        train_labels = [1]*len(tps_train) + [0]*len(fps_train)



        dev_pairs_lh = list(tps_dev + fps_dev)
        dev_labels = [1] * len(tps_dev) + [0] * len(fps_dev)
        test_pairs_lh = list(tps_test + fps_test)
        test_labels = [1] * len(tps_test) + [0] * len(fps_test)
        #GVC train sets

        im_samples_train_gvc1 = pickle.load(open(f"gvc/result_dict_train_gvc_15000.pkl", 'rb'))
        im_samples_train_gvc2 = pickle.load(open(f"gvc/result_dict_train_gvc_full_part2.pkl", 'rb'))
        im_samples_train_gvc3 = pickle.load(open(f"gvc/result_dict_train_gvc_full_part3.pkl", 'rb'))


        im_samples_train_gvc1 = [item for sublist in im_samples_train_gvc1 for item in sublist]
        im_samples_train_gvc2 = [item for sublist in im_samples_train_gvc2 for item in sublist]
        im_samples_train_gvc3 = [item for sublist in im_samples_train_gvc3 for item in sublist]
        print(len(im_samples_train_gvc1), len(im_samples_train_gvc2), len(im_samples_train_gvc3))

        print(len(im_samples_train_gvc1[0:15000])+ len(im_samples_train_gvc2) + len(im_samples_train_gvc3))

        #GVC dev sets:
        im_samples_dev_gvc1 = pickle.load(open(f"gvc/result_dict_dev_gvc_full.pkl", 'rb'))
        im_samples_dev_gvc2 = pickle.load(open(f"gvc/result_dict_dev_gvc_part2.pkl", 'rb'))

        im_samples_dev_gvc1 =  [item for sublist in im_samples_dev_gvc1 for item in sublist]
        im_samples_dev_gvc2 =  [item for sublist in im_samples_dev_gvc2 for item in sublist]

        #GVC test sets:

        im_samples_test_gvc1 = pickle.load(open(f"gvc/result_dict_test_gvc_full.pkl", 'rb'))
        im_samples_test_gvc1 =  [item for sublist in im_samples_test_gvc1 for item in sublist]


        im_samples_test_gvc2 = pickle.load(open(f"gvc/result_dict_test_gvc_full_part2.pkl", 'rb'))
        im_samples_test_gvc2 =  [item for sublist in im_samples_test_gvc2 for item in sublist]

        im_samples_train_full = im_samples_train_gvc1[0:15000]+ im_samples_train_gvc2 + im_samples_train_gvc3
        im_samples_dev_full = im_samples_dev_gvc1 + im_samples_dev_gvc2
        im_samples_test_full = im_samples_test_gvc1 + im_samples_test_gvc2

        test_length_a_equals_length_b(im_samples_train_full, train_pairs_lh)
        test_length_a_equals_length_b(im_samples_dev_full, dev_pairs_lh)
        test_length_a_equals_length_b(im_samples_test_full, test_pairs_lh)

        print(len(im_samples_train_gvc1[0:15000])+ len(im_samples_train_gvc2) + len(im_samples_train_gvc3), len(im_samples_dev_gvc1) + len(im_samples_dev_gvc2),len(im_samples_test_gvc1)+len(im_samples_test_gvc2))
        #find out some important samples of IM generation
        print(f"{dataset}  IM samples splits: train, dev and test", len(im_samples_train_full),len(im_samples_dev_full),len(im_samples_test_full) )
        print(f"{dataset}  LH sample splits: train, dev and test", len(train_pairs_lh), len(dev_pairs_lh), len(test_pairs_lh))
        print(f"{dataset}  LH label splits: train, dev and test", len(train_labels), len(dev_labels), len(test_labels))

    return im_samples_train_full, im_samples_dev_full, im_samples_test_full, train_pairs_lh, dev_pairs_lh, test_pairs_lh, train_labels, dev_labels, test_labels


def create_im_generated_maps(pairs,mention_map, im_samples, dialog_list,labels, split ):
    """
    Generates maps between mention pairs and IM-generated samples
    """
    im_dict = defaultdict(dict)

    for x, (i,j, k, l) in enumerate(zip(pairs,im_samples, dialog_list,labels)):
        im_dict[i]['im_generated_sample'] = j['generation']['content']
        im_dict[i]['mention_text_1'] = mention_map[i[0]]['mention_text']
        im_dict[i]['mention_text_2'] = mention_map[i[1]]['mention_text']
        im_dict[i]['im_abductive_system_prompt'] = k[0]['content']
        im_dict[i]['im_abductive_user_prompt'] = k[1]['content']
        im_dict[i]['sentence_1'] = mention_map[i[0]]['bert_sentence']
        im_dict[i]['sentence_2'] = mention_map[i[1]]['bert_sentence']
        im_dict[i]['label'] =l
        im_dict[i]['split'] = split

    return im_dict

if __name__ == '__main__':

    dataset = 'ecb'

    dialog_ecb_train = create_system_user_prompts(dataset, mention_map, split ="train", men_type = "evt", heu = "lh")
    dialog_ecb_test = create_system_user_prompts(dataset, mention_map, split ="test", men_type = "evt", heu = "lh")
    dialog_ecb_dev = create_system_user_prompts(dataset, mention_map, split ="dev", men_type = "evt", heu = "lh")
    im_samples_train_full, im_samples_dev_full, im_samples_test_full, train_pairs_lh, dev_pairs_lh, test_pairs_lh, train_labels, dev_labels, test_labels = get_inner_monologue_samples(dataset)

    dataset_folder = f'./datasets/{dataset}'
    mention_map = pickle.load(open(dataset_folder + "/mention_map.pkl", 'rb'))
    evt_mention_map_train = {m_id: m for m_id, m in mention_map.items() if m['men_type'] == 'evt' and m['split'] == 'train'}
    evt_mention_map_dev = {m_id: m for m_id, m in mention_map.items() if m['men_type'] == 'evt' and m['split'] == 'dev'}
    evt_mention_map_test = {m_id: m for m_id, m in mention_map.items() if m['men_type'] == 'evt' and m['split'] == 'test'}


    im_map_ecb_train  = create_im_generated_maps(train_pairs_lh, mention_map, im_samples_train_full, dialog_ecb_train,train_labels, 'train' )
    im_map_ecb_dev  = create_im_generated_maps(dev_pairs_lh, mention_map,im_samples_dev_full, dialog_ecb_dev,dev_labels, 'dev' )
    im_map_ecb_test = create_im_generated_maps(test_pairs_lh,mention_map, im_samples_test_full, dialog_ecb_test,test_labels, 'test' )
    print(len(im_map_ecb_train), len(im_map_ecb_dev), len(im_map_ecb_test))
    im_map_ecb_train.update(im_map_ecb_dev)
    im_map_ecb_train.update(im_map_ecb_test)
    print(len(im_map_ecb_train), len(im_map_ecb_dev), len(im_map_ecb_test))

    pickle.dump(im_map_ecb_train, open(dataset_folder + f'/im_map_{dataset}.pkl', 'wb'))

    dataset = 'gvc'
    dialog_gvc_train = create_system_user_prompts('gvc', mention_map, split ="train", men_type = "evt", heu = "lh")
    dialog_gvc_test = create_system_user_prompts('gvc', mention_map, split ="test", men_type = "evt", heu = "lh")
    dialog_gvc_dev = create_system_user_prompts('gvc', mention_map, split ="dev", men_type = "evt", heu = "lh")
    im_samples_train_full, im_samples_dev_full, im_samples_test_full, train_pairs_lh, dev_pairs_lh, test_pairs_lh, train_labels, dev_labels, test_labels = get_inner_monologue_samples('gvc')

    dataset_folder = f'./datasets/{dataset}'
    mention_map = pickle.load(open(dataset_folder + "/mention_map.pkl", 'rb'))
    evt_mention_map_train = {m_id: m for m_id, m in mention_map.items() if m['men_type'] == 'evt' and m['split'] == 'train'}
    evt_mention_map_dev = {m_id: m for m_id, m in mention_map.items() if m['men_type'] == 'evt' and m['split'] == 'dev'}
    evt_mention_map_test = {m_id: m for m_id, m in mention_map.items() if m['men_type'] == 'evt' and m['split'] == 'test'}


    im_map_gvc_train  = create_im_generated_maps(train_pairs_lh, mention_map, im_samples_train_full, dialog_gvc_train,train_labels, 'train' )
    im_map_gvc_dev  = create_im_generated_maps(dev_pairs_lh, mention_map,im_samples_dev_full, dialog_gvc_dev,dev_labels, 'dev' )
    im_map_gvc_test = create_im_generated_maps(test_pairs_lh,mention_map, im_samples_test_full, dialog_gvc_test,test_labels, 'test' )
    print(len(im_map_gvc_train), len(im_map_gvc_dev), len(im_map_gvc_test))
    im_map_gvc_train.update(im_map_gvc_dev)
    im_map_gvc_train.update(im_map_gvc_test)
    print(len(im_map_gvc_train), len(im_map_gvc_dev), len(im_map_gvc_test))

    pickle.dump(im_map_gvc_train, open(dataset_folder + f'/im_map_{dataset}.pkl', 'wb'))
