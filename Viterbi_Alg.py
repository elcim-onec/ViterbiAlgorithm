import nltk
from nltk.corpus import brown

sent_tag = brown.tagged_sents()
sent_tag = []
for s in sent_tag:
    s.insert(0, ('##', '##'))
    s.append(('&&', '&&'))
    sent_tag.append(s)

split_num = int(len(sent_tag) * 0.8)
train_data = sent_tag[0:split_num]
test_data = sent_tag[split_num:]

word_tag_dict = {}
for s in train_data:
    for (w, t) in s:
        w = w.lower()
        try:
            try:
                word_tag_dict[t][w] += 1
            except:
                word_tag_dict[t][w] = 1
        except:
            word_tag_dict[t] = {w: 1}

unique_words = len(set(word_tag_dict.keys()))
print("unique word amount: ", unique_words)

count_words = len(word_tag_dict.keys())
print("word amount: ", count_words)

ems_prob = {}
for k in word_tag_dict.keys():
    ems_prob[k] = {}
    count = sum(word_tag_dict[k].values())
    for k2 in word_tag_dict[k].keys():
        ems_prob[k][k2] = word_tag_dict[k][k2] / count

bigram_tags = {}
for s in train_data:
    bi = list(nltk.bigrams(s))
    for b1, b2 in bi:
        try:
            try:
                bigram_tags[b1[1]][b2[1]] += 1

            except:
                bigram_tags[b1[1]][b2[1]] = 1
        except:
            bigram_tags[b1[1]] = {b2[1]: 1}

bigram_tag_prob = {}
for k in bigram_tags.keys():
    bigram_tag_prob[k] = {}
    count = sum(bigram_tags[k].values())
    for k2 in bigram_tags[k].keys():
        bigram_tag_prob[k][k2] = bigram_tags[k][k2] / count

train_tokens = {}
count = 0
for s in train_data:
    for (w, t) in s:
        w = w.lower()
        try:
            if t not in train_tokens[w]:
                train_tokens[w].append(t)
        except:
            z = [t]
            train_tokens[w] = z

for s in test_data:
    for (w, t) in s:
        w = w.lower()
        try:
            if t not in train_tokens[w]:
                train_tokens[w].append(t)
        except:
            z = []
            z.append(t)
            train_tokens[w] = z

test_words = []
test_tags = []
for s in test_data:
    temp_word = []
    temp_tag = []
    for (w, t) in s:
        temp_word.append(w.lower())
        temp_tag.append(t)
    test_words.append(temp_word)
    test_tags.append(temp_tag)

# Viterbi Algorithm
predicted_tags = []
for x in range(len(test_words)):
    s = test_words[x]

    storing_values = {}
    for q in range(len(s)):
        step = s[q]

        if q == 1:
            storing_values[q] = {}
            tags = train_tokens[step]
            for t in tags:

                try:
                    storing_values[q][t] = ['##', bigram_tag_prob['##'][t] * ems_prob[t][step]]

                except:
                    storing_values[q][t] = ['##', 0.0001]  # *train_emission_prob[t][step]]
                    nltk.pprint(storing_values)

        if q > 1:
            storing_values[q] = {}
            previous_states = list(storing_values[q - 1].keys())
            current_states = train_tokens[step]
            for t in current_states:
                temp = []
                for pt in previous_states:
                    try:
                        temp.append(
                            storing_values[q - 1][pt][1] * bigram_tag_prob[pt][t] * ems_prob[t][step])
                    except:
                        temp.append(storing_values[q - 1][pt][1] * 0.0001)
                max_temp_index = temp.index(max(temp))
                best_pt = previous_states[max_temp_index]
                storing_values[q][t] = [best_pt, max(temp)]

    pred_tags = []
    total_steps_num = storing_values.keys()
    last_step_num = max(total_steps_num)
    for bs in range(len(total_steps_num)):
        step_num = last_step_num - bs
        if step_num == last_step_num:
            pred_tags.append('&&')
            pred_tags.append(storing_values[step_num]['&&'][0])
        if step_num < last_step_num and step_num > 0:
            pred_tags.append(storing_values[step_num][pred_tags[len(pred_tags) - 1]][0])
    predicted_tags.append(list(reversed(pred_tags)))
    print(pred_tags)

right = 0
wrong = 0
for i in range(len(test_tags)):
    gt = test_tags[i]
    pred = predicted_tags[i]
    for h in range(len(gt)):
        if gt[h] == pred[h]:
            right = right + 1
        else:
            wrong = wrong + 1

print('Accuracy on the test data is: ', right / (right + wrong))
print('Loss on the test data is: ', wrong / (right + wrong))

