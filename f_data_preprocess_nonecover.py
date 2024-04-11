#%%
from utils import *
import os
import pickle
import random
import numpy as np
from collections import Counter
import sys
from tqdm import tqdm

dataset_path = '../dataset/YAGO1830'
lower_bound_to_select_task_entity = 10
upper_bound_to_select_task_entity = 30
meta_train_ratio = 0.7
meta_valid_ratio = 0.15
meta_test_ratio = 0.15
meta_task_ratio = 0.5

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)

ent2id, rel2id, time2id, train_quads, valid_quads, test_quads, all_quads = load_data_quadruples(dataset_path)
'''train_quads [[ 920  207  733    0]
 [ 467  224 1176    0]
 [2274  157 3262    0]
 ...
 [1161  156 2442 7104]
 [2315  160 3262 7104]
 [1121  224 5423 7104]]'''
# np.random.shuffle(all_quads)

head, relation, tail, timestamp = all_quads.transpose()

entity = np.concatenate([head, tail])
entity_counter = Counter(entity) # 对字符串\列表\元祖\字典进行计数,返回一个字典类型的数据,键是元素,值是元素出现的次数
entity_counter_values_counter = Counter(entity_counter.values())

low_frequency_entity_list = []

for entity_id in entity_counter:
    entity_frequency = entity_counter[entity_id]
    if (entity_frequency >= lower_bound_to_select_task_entity) and (entity_frequency <= upper_bound_to_select_task_entity):
        low_frequency_entity_list.append(entity_id)


n_low_freq_entity = len(low_frequency_entity_list)
n_sampled_entity = int(n_low_freq_entity * meta_task_ratio)  # 0.5
meta_train_number = int(meta_train_ratio * n_sampled_entity)  # 0.8
meta_valid_number = int(meta_valid_ratio * n_sampled_entity)  # 0.1
meta_test_number = int(meta_test_ratio * n_sampled_entity)
print("num_low_frequency_entity: {}".format(n_low_freq_entity))
print("sampled_num_low_frequency_entity: {}".format(n_sampled_entity))
print(meta_train_number, meta_valid_number, meta_test_number)

# sampled_low_frequency_entity_list = np.random.choice(low_frequency_entity_list, n_sampled_entity, replace=False)
f_sampled_low_frequency_entity_list = low_frequency_entity_list[1::2] #隔一个取一个，按顺序取 481

#-------------------------------------确定哪些实体在测试集和验证集出现过----------------------------
test_head, test_relation, test_tail, test_timestamp = test_quads.transpose()
valid_head, valid_relation, valid_tail, valid_timestamp = valid_quads.transpose()
train_head, train_relation, train_tail, train_timestamp = train_quads.transpose()

test_entity = np.concatenate([test_head, test_tail])
valid_entity = np.concatenate([valid_head, valid_tail])
train_entity = np.concatenate([train_head, train_tail])

test_entity = np.unique(test_entity)
valid_entity = np.unique(valid_entity)
train_entity = np.unique(train_entity)

test_sample = []
valid_sample = []
train_sample = []
for id in tqdm(range(len(f_sampled_low_frequency_entity_list)-1, -1, -1)):
    entity_id = f_sampled_low_frequency_entity_list[id]
    appear = (test_entity == entity_id)
    appear_sum = sum(appear)
    if appear_sum >= 1:
        test_sample.append(entity_id)
        f_sampled_low_frequency_entity_list.pop(id)
    if len(test_sample) == meta_test_number:
        break

for id in tqdm(range(len(f_sampled_low_frequency_entity_list)-1, -1, -1)):
    entity_id = f_sampled_low_frequency_entity_list[id]
    appear = (valid_entity == entity_id)
    appear_sum = sum(appear)
    if appear_sum >= 1:
        valid_sample.append(entity_id)
        f_sampled_low_frequency_entity_list.pop(id)
    if len(valid_sample) == meta_valid_number:
        break

for id in tqdm(range(len(f_sampled_low_frequency_entity_list)-1, -1, -1)):
    entity_id = f_sampled_low_frequency_entity_list[id]
    appear = (train_entity == entity_id)
    appear_sum = sum(appear)
    if appear_sum >= 1:
        train_sample.append(entity_id)
        f_sampled_low_frequency_entity_list.pop(id)


if len(train_sample) != (n_sampled_entity - meta_valid_number - meta_test_number):
    print('less number')
    # sys.exit()

print('train_sample num:', len(train_sample))
print('valid_sample num:', len(valid_sample))
print('test_sample num:', len(test_sample))
f_sampled_low_frequency_entity_list = train_sample + valid_sample + test_sample
f_sampled_low_frequency_entity_list = np.array(f_sampled_low_frequency_entity_list)



# #--------------------------------------------old-------------------------------------
# filtered_index = []  
# for index, quadruple in enumerate(all_quads):
#     if quadruple[0] in sampled_low_frequency_entity_list:
#         continue
#     elif quadruple[2] in sampled_low_frequency_entity_list:
#         continue
#     filtered_index.append(index)  # index for quadruples that don't have unseen_entity   83421

#----------------------------------------------new-----------------------------------
f_filtered_index = []
for index, quadruple in enumerate(all_quads):
    if quadruple[0] in f_sampled_low_frequency_entity_list:
        continue
    elif quadruple[2] in f_sampled_low_frequency_entity_list:
        continue
    f_filtered_index.append(index)  # index for f_quadruples that don't have unseen_entity  83356

# filtered_index = np.array(filtered_index)
# filtered_quadruples = all_quads[filtered_index] #(83421, 4)
f_filtered_index = np.array(f_filtered_index)
f_filtered_quadruples = all_quads[f_filtered_index] #(83356, 4)




# meta_train_entity_list = sampled_low_frequency_entity_list[:meta_train_number]
# meta_valid_entity_list = sampled_low_frequency_entity_list[meta_train_number:meta_train_number + meta_valid_number]
# meta_test_entity_list = sampled_low_frequency_entity_list[meta_train_number + meta_valid_number:]

f_meta_train_entity_list = np.array(train_sample)
f_meta_valid_entity_list = np.array(valid_sample)
f_meta_test_entity_list = np.array(test_sample)


def process_meta_set(index, quadruple, self_list, other_list,
                     idx_list, cross_idx_list, self_dict, self_cross_dict): 
    '''
    params:
        self_list: meta_train_entity_list
        other_list: (meta_valid_entity_list, meta_test_entity_list)
        self_dict: meta_train_task_entity_to_quadruples
        self_cross_dict: meta_train_task_entity_to_unseen_quadruples
        idx_list: meta_train_task_index
        cross_idx_list: meta_train_unseen_task_index
    global:
        meta_train_task_index
        meta_train_unseen_task_index
    '''
    if (quadruple[0] in self_list) or (quadruple[2] in self_list):
        if (quadruple[0] not in other_list[0]) and (quadruple[2] not in other_list[0]) \
                and (quadruple[0] not in other_list[1]) and (quadruple[2] not in other_list[1]):

            idx_list.append(index)

            # quadruples between two meta_train entity
            if (quadruple[0] in self_list) and (quadruple[2] in self_list):
                cross_idx_list.append(index)
                i = random.randrange(0, 4, 2)
                # randomly add to one of the unseen entity in the dictionary
                if quadruple[i] in self_cross_dict:
                    self_cross_dict[quadruple[i]].append(quadruple)
                else:
                    self_cross_dict[quadruple[i]] = [quadruple]
            # quadruples between meta_train entity to in-graph entity
            elif (quadruple[0] in self_list):
                i = 0
            elif (quadruple[2] in self_list):
                i = 2

            if quadruple[i] in self_dict:
                self_dict[quadruple[i]].append(quadruple)
            else:
                self_dict[quadruple[i]] = [quadruple]

#--------------------------------------------old-------------------------------------
# print('#--------------------------------------------old-------------------------------------')
# meta_train_task_index = []  # index for quadruples: meta_train entity to in-graph & other meta_train
# meta_valid_task_index = []
# meta_test_task_index = []

# meta_train_task_entity_to_quadruples = {}
# meta_valid_task_entity_to_quadruples = {}
# meta_test_task_entity_to_quadruples = {}

# meta_train_unseen_task_index = []  # quadruples between meta_train entity and other meta_train entity
# meta_valid_unseen_task_index = []
# meta_test_unseen_task_index = []

# meta_train_task_entity_to_unseen_quadruples = {}
# meta_valid_task_entity_to_unseen_quadruples = {}
# meta_test_task_entity_to_unseen_quadruples = {}

# for index, quadruple in tqdm(enumerate(all_quads)):
#     # Meta-Train Quadruples
#     # quadruple include meta_train_entity
#     process_meta_set(index, quadruple, meta_train_entity_list, (meta_valid_entity_list, meta_test_entity_list),
#                      meta_train_task_index, meta_train_unseen_task_index,
#                      meta_train_task_entity_to_quadruples, meta_train_task_entity_to_unseen_quadruples)

#     # Meta-Valid quadruple
#     process_meta_set(index, quadruple, meta_valid_entity_list, (meta_train_entity_list, meta_test_entity_list),
#                      meta_valid_task_index, meta_valid_unseen_task_index,
#                      meta_valid_task_entity_to_quadruples, meta_valid_task_entity_to_unseen_quadruples)

#     process_meta_set(index, quadruple, meta_test_entity_list, (meta_train_entity_list, meta_valid_entity_list),
#                      meta_test_task_index, meta_test_unseen_task_index,
#                      meta_test_task_entity_to_quadruples, meta_test_task_entity_to_unseen_quadruples)

# meta_train_task_quadruples = all_quads[meta_train_task_index]
# meta_valid_task_quadruples = all_quads[meta_valid_task_index]
# meta_test_task_quadruples = all_quads[meta_test_task_index]

# meta_train_unseen_task_quadruples = all_quads[meta_train_unseen_task_index]
# meta_valid_unseen_task_quadruples = all_quads[meta_valid_unseen_task_index]
# meta_test_unseen_task_quadruples = all_quads[meta_test_unseen_task_index]

# print('num_meta_train_task_quadruples: {}'.format(len(meta_train_task_quadruples)))
# print('num_meta_valid_task_quadruples: {}'.format(len(meta_valid_task_quadruples)))
# print('num_meta_test_task_quadruples: {}'.format(len(meta_test_task_quadruples)))

# print('num_meta_train_entity: {}'.format(len(meta_train_task_entity_to_quadruples.keys())))
# print('num_meta_valid_entity: {}'.format(len(meta_valid_task_entity_to_quadruples.keys())))
# print('num_meta_test_entity: {}'.format(len(meta_test_task_entity_to_quadruples.keys())))

# print('num_meta_train_unseen_task_quadruples: {}'.format(len(meta_train_unseen_task_quadruples)))
# print('num_meta_valid_unseen_task_quadruples: {}'.format(len(meta_valid_unseen_task_quadruples)))
# print('num_meta_test_unseen_task_quadruples: {}'.format(len(meta_test_unseen_task_quadruples)))

# print('num_meta_train_unseen_entity_pair: {}'.format(len(meta_train_task_entity_to_unseen_quadruples.keys())))
# print('num_meta_valid_unseen_entity_pair: {}'.format(len(meta_valid_task_entity_to_unseen_quadruples.keys())))
# print('num_meta_test_unseen_entity_pair: {}'.format(len(meta_test_task_entity_to_unseen_quadruples.keys())))

#----------------------------------------------new-----------------------------------
print('#----------------------------------------------new-----------------------------------')
f_meta_train_task_index = []  # index for quadruples: meta_train entity to in-graph & other meta_train
f_meta_valid_task_index = []
f_meta_test_task_index = []

f_meta_train_task_entity_to_quadruples = {}
f_meta_valid_task_entity_to_quadruples = {}
f_meta_test_task_entity_to_quadruples = {}

f_meta_train_unseen_task_index = []  # quadruples between meta_train entity and other meta_train entity
f_meta_valid_unseen_task_index = []
f_meta_test_unseen_task_index = []

f_meta_train_task_entity_to_unseen_quadruples = {}
f_meta_valid_task_entity_to_unseen_quadruples = {}
f_meta_test_task_entity_to_unseen_quadruples = {}

for index, quadruple in tqdm(enumerate(all_quads)):
    # Meta-Train Quadruples
    # quadruple include meta_train_entity
    process_meta_set(index, quadruple, f_meta_train_entity_list, (f_meta_valid_entity_list, f_meta_test_entity_list),
                     f_meta_train_task_index, f_meta_train_unseen_task_index,
                     f_meta_train_task_entity_to_quadruples, f_meta_train_task_entity_to_unseen_quadruples)

    # Meta-Valid quadruple
    process_meta_set(index, quadruple, f_meta_valid_entity_list, (f_meta_train_entity_list, f_meta_test_entity_list),
                     f_meta_valid_task_index, f_meta_valid_unseen_task_index,
                     f_meta_valid_task_entity_to_quadruples, f_meta_valid_task_entity_to_unseen_quadruples)

    process_meta_set(index, quadruple, f_meta_test_entity_list, (f_meta_train_entity_list, f_meta_valid_entity_list),
                     f_meta_test_task_index, f_meta_test_unseen_task_index,
                     f_meta_test_task_entity_to_quadruples, f_meta_test_task_entity_to_unseen_quadruples)



f_meta_train_task_quadruples = all_quads[f_meta_train_task_index]
f_meta_valid_task_quadruples = all_quads[f_meta_valid_task_index]
f_meta_test_task_quadruples = all_quads[f_meta_test_task_index]

f_meta_train_unseen_task_quadruples = all_quads[f_meta_train_unseen_task_index]
f_meta_valid_unseen_task_quadruples = all_quads[f_meta_valid_unseen_task_index]
f_meta_test_unseen_task_quadruples = all_quads[f_meta_test_unseen_task_index]

all_mata_set = np.concatenate([f_meta_train_task_quadruples, f_meta_valid_task_quadruples, f_meta_test_task_quadruples])
all_mata_set = all_mata_set[np.argsort(all_mata_set[:,3])]



print('f_num_meta_train_task_quadruples: {}'.format(len(f_meta_train_task_quadruples)))
print('f_num_meta_valid_task_quadruples: {}'.format(len(f_meta_valid_task_quadruples)))
print('f_num_meta_test_task_quadruples: {}'.format(len(f_meta_test_task_quadruples)))

print('f_num_meta_train_entity: {}'.format(len(f_meta_train_task_entity_to_quadruples.keys())))
print('f_num_meta_valid_entity: {}'.format(len(f_meta_valid_task_entity_to_quadruples.keys())))
print('f_num_meta_test_entity: {}'.format(len(f_meta_test_task_entity_to_quadruples.keys())))

print('f_num_meta_train_unseen_task_quadruples: {}'.format(len(f_meta_train_unseen_task_quadruples)))
print('f_num_meta_valid_unseen_task_quadruples: {}'.format(len(f_meta_valid_unseen_task_quadruples)))
print('f_num_meta_test_unseen_task_quadruples: {}'.format(len(f_meta_test_unseen_task_quadruples)))

print('f_num_meta_train_unseen_entity_pair: {}'.format(len(f_meta_train_task_entity_to_unseen_quadruples.keys())))
print('f_num_meta_valid_unseen_entity_pair: {}'.format(len(f_meta_valid_task_entity_to_unseen_quadruples.keys())))
print('f_num_meta_test_unseen_entity_pair: {}'.format(len(f_meta_test_task_entity_to_unseen_quadruples.keys())))

#-------------------------------------------------------------------

# print('f_num_meta_train_task_quadruples: ', f_meta_train_task_quadruples)
# print('f_num_meta_valid_task_quadruples: ', f_meta_valid_task_quadruples)
# print('f_num_meta_test_task_quadruples: ', f_meta_test_task_quadruples)

# print('#--------------------------------------------old-------------------------------------')
# count_lowfreq = 0
# count_len = 0
# for task_entity, quadruples in meta_valid_task_entity_to_quadruples.items():
#     count_len += len(quadruples)
#     if len(quadruples) < 2:
#         count_lowfreq += 1
# print("number of quadruples in entity_to_quadruples dictionary: {}".format(count_len))
# print("number of entity with less than 2 quadruples in meta_valid: {}".format(count_lowfreq))

# count_lowfreq = 0
# count_len = 0
# for task_entity, quadruples in meta_test_task_entity_to_quadruples.items():
#     count_len += len(quadruples)
#     if len(quadruples) < 2:
#         count_lowfreq += 1
# print("number of quadruples in entity_to_quadruples dictionary: {}".format(count_len))
# print("number of entity with less than 2 quadruples in meta_test: {}".format(count_lowfreq))

count_lowfreq = 0
count_len = 0
for task_entity, quadruples in f_meta_valid_task_entity_to_quadruples.items():
    count_len += len(quadruples)
    if len(quadruples) < 2:
        count_lowfreq += 1
print("number of quadruples in entity_to_quadruples dictionary: {}".format(count_len))
print("number of entity with less than 2 quadruples in meta_valid: {}".format(count_lowfreq))

count_lowfreq = 0
count_len = 0
for task_entity, quadruples in f_meta_test_task_entity_to_quadruples.items():
    count_len += len(quadruples)
    if len(quadruples) < 2:
        count_lowfreq += 1
print("number of quadruples in entity_to_quadruples dictionary: {}".format(count_len))
print("number of entity with less than 2 quadruples in meta_test: {}".format(count_lowfreq))

# print('new_f_meta_train_task_entity_to_quadruples', f_meta_train_task_entity_to_quadruples)
# sys.exit()

#调整数据集要调整这里
#----------------------------------------------构造训练集--------------------------------------------------------------
new_f_meta_train_task_quadruples = np.array([[0, 0, 0, 0]]) #(135, 4)
for id in tqdm(range(f_meta_train_task_quadruples.shape[0])):
    if (f_meta_train_task_quadruples[id][3] <= 166):
        new_f_meta_train_task_quadruples = np.concatenate([new_f_meta_train_task_quadruples, f_meta_train_task_quadruples[id][np.newaxis,:]], axis=0) 
new_f_meta_train_task_quadruples =np.delete(new_f_meta_train_task_quadruples, 0, axis=0)

#----------------------------------------------构造验证集--------------------------------------------------------------(478, 4)
new_f_meta_valid_task_quadruples = np.array([[0, 0, 0, 0]]) #(135, 4)
for id in tqdm(range(f_meta_valid_task_quadruples.shape[0])):
    if (f_meta_valid_task_quadruples[id][3] >= 167) and (f_meta_valid_task_quadruples[id][3] < 176):
        new_f_meta_valid_task_quadruples = np.concatenate([new_f_meta_valid_task_quadruples, f_meta_valid_task_quadruples[id][np.newaxis,:]], axis=0) 
new_f_meta_valid_task_quadruples =np.delete(new_f_meta_valid_task_quadruples, 0, axis=0)

#----------------------------------------------构造测试集--------------------------------------------------------------(462, 4)
new_f_meta_test_task_quadruples = np.array([[0, 0, 0, 0]]) #(135, 4)
for id in tqdm(range(f_meta_test_task_quadruples.shape[0])):
    if (f_meta_test_task_quadruples[id][3] >= 176):
        new_f_meta_test_task_quadruples = np.concatenate([new_f_meta_test_task_quadruples, f_meta_test_task_quadruples[id][np.newaxis,:]], axis=0) 
new_f_meta_test_task_quadruples =np.delete(new_f_meta_test_task_quadruples, 0, axis=0)


print('mata_train_quadruples:', new_f_meta_train_task_quadruples.shape[0])
print('mata_valid_quadruples:', new_f_meta_valid_task_quadruples.shape[0])
print('mata_test_quadruples:', new_f_meta_test_task_quadruples.shape[0])


#----------------------------------------------构造小样本entity_to_quadruples--------------------------------------------------------------
def entity_to_quadruples(quadruple, self_list, other_list, self_dict):
    if (quadruple[0] in self_list) or (quadruple[2] in self_list):
        if (quadruple[0] not in other_list[0]) and (quadruple[2] not in other_list[0]) \
                and (quadruple[0] not in other_list[1]) and (quadruple[2] not in other_list[1]):
            # quadruples between two meta_train entity
            if (quadruple[0] in self_list) and (quadruple[2] in self_list):
                i = random.randrange(0, 4, 2)
            elif (quadruple[0] in self_list):
                i = 0
            elif (quadruple[2] in self_list):
                i = 2

            if quadruple[i] in self_dict:
                self_dict[quadruple[i]].append(quadruple)
            else:
                self_dict[quadruple[i]] = [quadruple]
    else:
        print('error')

train_entity_to_quadruples = {}
valid_entity_to_quadruples = {}
test_entity_to_quadruples = {}
for index, quadruple in tqdm(enumerate(new_f_meta_train_task_quadruples)):
    entity_to_quadruples(quadruple, f_meta_train_entity_list, (f_meta_valid_entity_list, f_meta_test_entity_list),
                        train_entity_to_quadruples)
for index, quadruple in tqdm(enumerate(new_f_meta_valid_task_quadruples)):
    entity_to_quadruples(quadruple, f_meta_valid_entity_list, (f_meta_train_entity_list, f_meta_test_entity_list),
                        valid_entity_to_quadruples)
for index, quadruple in tqdm(enumerate(new_f_meta_test_task_quadruples)):
    entity_to_quadruples(quadruple, f_meta_test_entity_list, (f_meta_train_entity_list, f_meta_valid_entity_list), 
                        test_entity_to_quadruples)
print('train_entity_to_quadruples: {}'.format(len(train_entity_to_quadruples.keys())))
print('valid_entity_to_quadruples: {}'.format(len(valid_entity_to_quadruples.keys())))
print('test_entity_to_quadruples: {}'.format(len(test_entity_to_quadruples.keys())))

all_quadruples_to_use = np.concatenate(
    (f_filtered_quadruples, new_f_meta_train_task_quadruples, new_f_meta_valid_task_quadruples, new_f_meta_test_task_quadruples))
# print('num_all_quadruples: {}'.format(len(all_quadruples_to_use)))

save_folder = '../dataset/YAGO1830/none_cover_processed_data/' ####update
os.makedirs(save_folder, exist_ok=True)

#------------------------------------保存原始数据集------------------------------------
with open(save_folder + 'f_filtered_quadruples.pickle', 'wb') as f:
    pickle.dump(f_filtered_quadruples, f)
with open(save_folder + 'all_mata_set.pickle', 'wb') as f:
    pickle.dump(all_mata_set, f)
with open(save_folder + 'f_meta_train_task_entity_to_quadruples.pickle', 'wb') as f:
    pickle.dump(f_meta_train_task_entity_to_quadruples, f)
with open(save_folder + 'f_meta_valid_task_entity_to_quadruples.pickle', 'wb') as f:
    pickle.dump(f_meta_valid_task_entity_to_quadruples, f)
with open(save_folder + 'f_meta_test_task_entity_to_quadruples.pickle', 'wb') as f:
    pickle.dump(f_meta_test_task_entity_to_quadruples, f)

#------------------------------------保存小样本数据集------------------------------------
with open(save_folder + 'mata_train_quadruples.pickle', 'wb') as f:
    pickle.dump(new_f_meta_train_task_quadruples, f)
with open(save_folder + 'mata_valid_quadruples.pickle', 'wb') as f:
    pickle.dump(new_f_meta_valid_task_quadruples, f)
with open(save_folder + 'mata_test_quadruples.pickle', 'wb') as f:
    pickle.dump(new_f_meta_test_task_quadruples, f)


#------------------------------------保存小样本entity_to_quadruples------------------------------------
with open(save_folder + 'train_entity_to_quadruples.pickle', 'wb') as f:
    pickle.dump(train_entity_to_quadruples, f)
with open(save_folder + 'valid_entity_to_quadruples.pickle', 'wb') as f:
    pickle.dump(valid_entity_to_quadruples, f)
with open(save_folder + 'test_entity_to_quadruples.pickle', 'wb') as f:
    pickle.dump(test_entity_to_quadruples, f)

print(f"save processed data in folder {save_folder}")


# %%
