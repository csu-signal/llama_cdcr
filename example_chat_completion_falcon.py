# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional
import os
import fire
from tqdm import tqdm
from llama import Llama
import spacy
import pickle
from collections import defaultdict
output_list = []
dataset = 'gvc'
generated_IM_samples = defaultdict(list)

output_folder = f'/s/chopin/d/proj/ramfis-aida/llama2/llama/{dataset}/'
def create_inner_monologue_prompts(dataset_folder =None):

    dataset_folder = f'/s/chopin/d/proj/ramfis-aida/multimodal_NLI/Multimodal_CDCR/acl_submission_2023-main/datasets/{dataset}/'
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
    len(train_pairs_lh), len(dev_pairs_lh), len(test_pairs_lh)

    train_pairs_lh = train_pairs_lh[15000:35000] #get the first 20000, then get the rest.
    train_labels = train_labels[15000:35000]
    print(f"Generating Inner Monologue Samples for {len(train_pairs_lh), len(train_labels)} samples")

    prompt_list = []
    dialog_list = []

    def trim_mention_sentences(mention_map):
        list_keys = []
        nlp = spacy.load('en_core_web_sm')
        for x, y in mention_map.items():
            if len(mention_map[x]['bert_sentence'].split())>=300:
                list_keys.append(x)
                sentence = mention_map[x]['bert_sentence']
                doc = nlp(sentence)
                start_token_index = sentence.find('<m>')
                end_token_index =  sentence.find('</m>')
                word = doc.char_span(start_token_index+1,end_token_index-1)
                # word will be None if the char span isn't valid
                if word is not None:
                    sent = word.sent
                    new_bert_sentence = str(sent)
                    mention_map[x]['bert_sentence'] = new_bert_sentence
                    #print(x, new_bert_sentence)
        return mention_map

    evt_mention_map_train = trim_mention_sentences(evt_mention_map_train)
    for i, (x, y) in enumerate(zip(train_pairs_lh,train_labels )):

        sentence_a = evt_mention_map_train[x[0]]['bert_sentence']
        sentence_b = evt_mention_map_train[x[1]]['bert_sentence']
        inner_mon_prompt_1 = f'''Create an inner monologue based on chain-of-thought
    reasoning to answer a question about documents A and B.
    Then, answer whether they are talking about the same event or not. '''
        inner_mon_prompt_2 = f'''

    The event trigger-words are <m> {evt_mention_map_train[x[0]]['mention_text']} </m> in
    document A and <m> {evt_mention_map_train[x[1]]['mention_text']} </m> in document B.
    The inner monologue must make this decision after identifying common
    context, actions, actors, objects and locations.
    For instance, are the entities related to the event-trigger words
    <m> {evt_mention_map_train[x[0]]['mention_text']} </m>
    and <m> {evt_mention_map_train[x[1]]['mention_text']} </m> the same?
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

    return dialog_list,train_pairs_lh, train_labels
def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_seq_len: int = 4096,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    dialogs, train_pairs, train_labels = create_inner_monologue_prompts()
    generated_IM_samples = {}
#     dialogs = [


#         [
#             {
#                 "role": "system",
#                 "content": "Are sentences A and B talking about the same event? The event in each sentence is surrounded by the tokens, <m> and </m>. Please explain.",
#             },
#             {"role": "user", "content": "Sentence A: John <m> acquired <m> a huge \
#             \
#             amount property through inheritance. Sentence B: As part of its efforts to support energy - efficient computing , HP Monday announced it has signed an agreement to\
#             <m> acquire </m> facilities consulting firm EYP Mission Critical Facilities for an undisclosed sum."},
#         ],

#         [
#             {
#                 "role": "system",
#                 "content": "What are the important entities being talked about in document A and B? Are there any overlaps?",
#             },
#             {"role": "user", "content": "Sentence A: John <m> acquired <m> a huge \
#             \
#             amount property through inheritance. Sentence B: As part of its efforts to support energy - efficient computing , HP Monday announced it has signed an agreement to\
#             <m> acquire </m> facilities consulting firm EYP Mission Critical Facilities for an undisclosed sum."},
#         ],
#         [
#             {
#                 "role": "system",
#                 "content": "Create an inner monologue based on chain of thought reasoning for document A and B. Then, answer whether they are talking about the same event or not. The inner monologue must make this decision after identifying common context, actions, actors, objects and locations. Sometimes the language in the two documents can be ambiguous so you must be fact-check and reason carefully. Do not mix the entities talked about in the documents and answer based on facts. Use your inner monologue to decide whether or not the two events are the same event. The event trigger-words are <m> and </m> tokens in each document. ",
#             },
#             {"role": "user", "content": "Sentence A: John <m> acquired <m> a huge \
#             \
#             amount property through inheritance. \
#             Sentence B: As part of its efforts to support energy - efficient computing , HP Monday announced it has signed an agreement to\
#             <m> acquire </m> facilities consulting firm EYP Mission Critical Facilities for an undisclosed sum."},
#         ],
#         [
#             {
#                 "role": "system",
#                 "content": "Create an inner monologue based on chain of thought reasoning for a question about document A and B. Then, answer whether they are talking about the same event or not. Hint: They are not the same, so reverse-engineer your reasoning!! The event trigger-words are <m> walking </m> in document A and <m> fled </m> in document B. The inner monologue must make this decision after identifying common context, actions, actors, objects and locations. For instance, are the entities related to the event-trigger words <m> walking </m> and <m> fled </m> the same? Resolve who is being referred to by the pronouns used to make your decision if needed.  Sometimes the language in the two documents can be ambiguous so you must reason carefully. Do not mix the entities talked about in the documents and answer based on facts. Use your inner monologue to decide whether or not the two events are the same event. ",
#             },
#             {"role": "user", "content": "Document A is: An initial investigation of the scene revealed that Chohan had been <m> walking </m> alongside the parked cars on 117th Street when she was struck by a vehicle heading northbound , police said ., Document B is : Raj Chohan , 59 , of Queens , had just parked her car around 7 p . m . Friday at 97th Avenue and 117th Street in Richmond Hill when she was hit by a 2013 Toyota Camry , which then <m> fled </m> , police told 1010 WINS ’ Glenn Schuck ."},
#         ],
#     ]
    batch_size = 40
    if not os.path.exists(output_folder):
            os.makedirs(output_folder)
    for i in tqdm(range(0, len(dialogs), batch_size)):
        batched_prompts =  dialogs[i:i + batch_size]
        #batched_pairs = train_pairs[i:i + batch_size]

        results = generator.chat_completion(
            batched_prompts,  # type: ignore
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        output_list.append(results)
        # generated_IM_samples[i].append(results)
        # generated_IM_samples = {x:y for x, y in zip(batched_pairs,results)}

        if i%5000 == 0:

            with open(f'{output_folder}/result_dict_train_gvc_{i}_part2.pkl', 'wb') as fp:
                pickle.dump(output_list, fp)
    with open(f'{output_folder}/result_dict_train_gvc_full_part2.pkl', 'wb') as fp:
                pickle.dump(output_list, fp)

#     for dialog, result, pairs in zip(dialogs, results, train_labels):
#         for msg in dialog:

#             print(f"{msg['role'].capitalize()}: {msg['content']}\n")
#         print(
#             f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
#         )
#         print("\n==================================\n")
    return results

if __name__ == "__main__":
    results= fire.Fire(main)
