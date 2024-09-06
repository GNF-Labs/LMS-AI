import argparse
import os
import sys
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from tqdm import trange
from library_data import *
from library_models import *

# INITIALIZE PARAMETERS
parser = argparse.ArgumentParser()
parser.add_argument('--network', required=True, help='Network name')
parser.add_argument('--model', default='jodie', help="Model name")
parser.add_argument('--gpu', default=-1, type=int, help='ID of the gpu to run on. If set to -1 (default), the GPU with most free memory will be chosen.')
parser.add_argument('--epoch', default=50, type=int, help='Epoch id to load')
parser.add_argument('--embedding_dim', default=128, type=int, help='Number of dimensions')
parser.add_argument('--train_proportion', default=0.8, type=float, help='Proportion of training interactions')
parser.add_argument('--state_change', default=True, type=bool, help='True if training with state change of users in addition to the next interaction prediction. False otherwise. By default, set to True. MUST BE THE SAME AS THE ONE USED IN TRAINING.') 
args = parser.parse_args()
args.datapath = "data/%s.csv" % args.network
if args.train_proportion > 0.8:
    sys.exit('Training sequence proportion cannot be greater than 0.8.')
if args.network == "mooc":
    print("No interaction prediction for %s" % args.network)
    sys.exit(0)
    
# SET GPU
args.gpu = select_free_gpu()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# CHECK IF THE OUTPUT OF THE EPOCH IS ALREADY PROCESSED. IF SO, MOVE ON.
output_fname = "results/interaction_prediction_%s.txt" % args.network
if os.path.exists(output_fname):
    f = open(output_fname, "r")
    search_string = 'Test performance of epoch %d' % args.epoch
    for l in f:
        l = l.strip()
        if search_string in l:
            print("Output file already has results of epoch %d" % args.epoch)
            sys.exit(0)
    f.close()

# LOAD NETWORK
print("args :", args)
[user2id, user_sequence_id, user_timediffs_sequence, user_previous_itemid_sequence, \
 item2id, item_sequence_id, item_timediffs_sequence, \
 timestamp_sequence, \
 feature_sequence, \
 y_true] = load_network(args)
num_interactions = len(user_sequence_id)
num_features = len(feature_sequence[0])
num_users = len(user2id)
num_items = len(item2id) + 1
true_labels_ratio = len(y_true)/(sum(y_true)+1)
print("*** Network statistics:\n  %d users\n  %d items\n  %d interactions\n  %d/%d true labels ***\n\n" % (num_users, num_items, num_interactions, sum(y_true), len(y_true)))

# SET BATCHING TIMESPAN
timespan = timestamp_sequence[-1] - timestamp_sequence[0]
tbatch_timespan = timespan / 500 

# INITIALIZE MODEL PARAMETERS
model = JODIE(args, num_features, num_users, num_items).cuda()
weight = torch.Tensor([1,true_labels_ratio]).cuda()
crossEntropyLoss = nn.CrossEntropyLoss(weight=weight)
MSELoss = nn.MSELoss()

# INITIALIZE MODEL
learning_rate = 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# LOAD THE MODEL
model, optimizer, user_embeddings_dystat, item_embeddings_dystat, user_embeddings_timeseries, item_embeddings_timeseries, train_end_idx_training = load_model(model, optimizer, args, args.epoch)

# SET THE USER AND ITEM EMBEDDINGS TO THEIR STATE AT THE END OF THE TRAINING PERIOD
embeddings_training_end(user_embeddings_dystat, item_embeddings_dystat, user_embeddings_timeseries, item_embeddings_timeseries, user_sequence_id, item_sequence_id, train_end_idx_training) 

# LOAD THE EMBEDDINGS: DYNAMIC AND STATIC
item_embeddings = item_embeddings_dystat[:, :args.embedding_dim]
item_embeddings = item_embeddings.clone()
item_embeddings_static = item_embeddings_dystat[:, args.embedding_dim:]
item_embeddings_static = item_embeddings_static.clone()

user_embeddings = user_embeddings_dystat[:, :args.embedding_dim]
user_embeddings = user_embeddings.clone()
user_embeddings_static = user_embeddings_dystat[:, args.embedding_dim:]
user_embeddings_static = user_embeddings_static.clone()

# PERFORMANCE METRICS
validation_ranks = []
test_ranks = []

tbatch_start_time = None
loss = 0

# PREDICT INTERACTIONS FOR ALL USERS
print("*** Making interaction predictions for all users ***")
for userid in range(num_users):
    # Initialize user interaction predictions for this user
    user_item_predictions = []
    
    # Iterate through all items to predict interaction
    for itemid in range(num_items):
        # LOAD USER AND ITEM EMBEDDING
        user_embedding_input = user_embeddings[torch.cuda.LongTensor([userid])]
        user_embedding_static_input = user_embeddings_static[torch.cuda.LongTensor([userid])]
        item_embedding_input = item_embeddings[torch.cuda.LongTensor([itemid])]
        item_embedding_static_input = item_embeddings_static[torch.cuda.LongTensor([itemid])]
        
        # Generate fake feature tensor and timediffs for prediction
        feature_tensor = Variable(torch.zeros(1, num_features).cuda())
        user_timediffs_tensor = Variable(torch.zeros(1).cuda())
        item_timediffs_tensor = Variable(torch.zeros(1).cuda())
        item_embedding_previous = item_embeddings[torch.cuda.LongTensor([itemid])]

        # PROJECT USER EMBEDDING
        user_projected_embedding = model.forward(user_embedding_input, item_embedding_previous, timediffs=user_timediffs_tensor, features=feature_tensor, select='project')
        user_item_embedding = torch.cat([user_projected_embedding, item_embedding_previous, item_embeddings_static[torch.cuda.LongTensor([itemid])], user_embedding_static_input], dim=1)

        # PREDICT ITEM EMBEDDING
        predicted_item_embedding = model.predict_item_embedding(user_item_embedding)

        # CALCULATE DISTANCE OF PREDICTED ITEM EMBEDDING TO ALL ITEMS 
        euclidean_distances = nn.PairwiseDistance()(predicted_item_embedding.repeat(num_items, 1), torch.cat([item_embeddings, item_embeddings_static], dim=1)).squeeze(-1) 
        
        # FIND THE TOP 10 NEXT PREDICTED ITEMS
        top_10_items = torch.topk(-euclidean_distances, 10).indices.cpu().numpy()  # Taking negative to get smallest distances
        
        user_item_predictions.append((itemid, top_10_items))

    # Save the user_id and top 10 items to a file
    with open('user_recommendations.txt', 'a') as f:
        for itemid, top_10_items in user_item_predictions:
            f.write(f"User {userid} -> Top 10 items for item {itemid}: {top_10_items}\n")


# CALCULATE THE PERFORMANCE METRICS
performance_dict = dict()

def calculate_metrics(ranks):
    if len(ranks) == 0:
        return [float('nan'), float('nan')]  # Return NaN if no data to avoid division by zero
    mrr = np.mean([1.0 / r for r in ranks])
    rec10 = sum(np.array(ranks) <= 10) * 1.0 / len(ranks)
    return [mrr, rec10]

ranks = validation_ranks
performance_dict['validation'] = calculate_metrics(ranks)

ranks = test_ranks
performance_dict['test'] = calculate_metrics(ranks)

# PRINT AND SAVE THE PERFORMANCE METRICS
fw = open(output_fname, "a")
metrics = ['Mean Reciprocal Rank', 'Recall@10']

print('\n\n*** Validation performance of epoch %d ***' % args.epoch)
fw.write('\n\n*** Validation performance of epoch %d ***\n' % args.epoch)
for i in trange(len(metrics)):
    print(metrics[i] + ': ' + str(performance_dict['validation'][i]))
    fw.write("Validation: " + metrics[i] + ': ' + str(performance_dict['validation'][i]) + "\n")
    
print('\n\n*** Test performance of epoch %d ***' % args.epoch)
fw.write('\n\n*** Test performance of epoch %d ***\n' % args.epoch)
for i in trange(len(metrics)):
    print(metrics[i] + ': ' + str(performance_dict['test'][i]))
    fw.write("Test: " + metrics[i] + ': ' + str(performance_dict['test'][i]) + "\n")

fw.flush()
fw.close()
