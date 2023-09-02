import torch
import torch.nn as nn
import transformers
import numpy as np
from tqdm import trange
import copy
import random
# import pdb

class ReplayMemory(object):
    """
        Create the empty memory buffer
    """

    def __init__(self, buffer=None):

        if buffer is None:
            self.memory = {}
        else:
            self.memory = buffer
            total_keys = len(buffer.keys())
            # convert the keys from np.bytes to np.float32
            self.all_keys = np.frombuffer(
                np.asarray(list(self.memory.keys())), dtype=np.float32).reshape(total_keys, 768)

    def push(self, keys, examples):
        """
        Add the examples as key-value pairs to the memory dictionary with content,attention_mask,label tuple as value
        and key determined by key network
        """
        contents, attn_masks, labels = examples
        # update the memory dictionary
        for i, key in enumerate(keys):
            # numpy array cannot be used as key since it is non-hashable, hence convert it to bytes to use as key
            self.memory.update(
                {key.tobytes(): (contents[i], attn_masks[i], labels[i])})

    def _prepare_batch(self, sample):
        """
        Parameter:
        sample -> list of tuple of experiences
               -> i.e, [(content_1,attn_mask_1,label_1),.....,(content_k,attn_mask_k,label_k)]
        Returns:
        batch -> tuple of list of content,attn_mask,label
              -> i.e, ([content_1,...,content_k],[attn_mask_1,...,attn_mask_k],[label_1,...,label_k])
        """
        contents = []
        attn_masks = []
        labels = []
        # Iterate over experiences
        for content, attn_mask, label in sample:
            # convert the batch elements into torch.LongTensor
            contents.append(content)
            attn_masks.append(attn_mask)
            labels.append(label)

        return (torch.LongTensor(contents), torch.LongTensor(attn_masks), torch.LongTensor(labels))

    def get_neighbours(self, keys, k=32):
        """
        Returns samples from buffer using nearest neighbour approach
        """
        samples = []
        # Iterate over all the input keys
        # to find neigbours for each of them
        for key in keys:
            # compute similarity scores based on Euclidean distance metric
            similarity_scores = np.dot(self.all_keys, key.T)
            K_neighbour_keys = self.all_keys[np.argpartition(
                similarity_scores, -k)[-k:]]
            neighbours = [self.memory[nkey.tobytes()]
                          for nkey in K_neighbour_keys]
            # converts experiences into batch
            batch = self._prepare_batch(neighbours)
            samples.append(batch)

        return samples
    
    def sample(self, sample_size):
        keys = random.sample(list(self.memory),sample_size)
        contents = np.array([self.memory[k][0] for k in keys])
        attn_masks = np.array([self.memory[k][1] for k in keys])
        labels = np.array([self.memory[k][2] for k in keys])
        return (torch.LongTensor(contents), torch.LongTensor(attn_masks), torch.LongTensor(labels))
        


class MbPAplusplus(nn.Module):
    """
    Implements Memory based Parameter Adaptation model
    """

    def __init__(self, L=30, model_state=None):
        super(MbPAplusplus, self).__init__()

        if model_state is None:
            # Key network to find key representation of content
            self.key_encoder = transformers.BertModel.from_pretrained(
                'bert-base-uncased')
            # Bert model for text classification
            self.classifier = transformers.BertForSequenceClassification.from_pretrained(
                'bert-base-uncased', num_labels=33)

        else:

            cls_config = transformers.BertConfig.from_pretrained(
                'bert-base-uncased', num_labels=33)
            self.classifier = transformers.BertForSequenceClassification(
                cls_config)
            self.classifier.load_state_dict(model_state['classifier'])
            key_config = transformers.BertConfig.from_pretrained(
                'bert-base-uncased')
            self.key_encoder = transformers.BertModel(key_config)
            self.key_encoder.load_state_dict(model_state['key_encoder'])
            # load base model weights
            # we need to detach since parameters() method returns reference to the original parameters
            self.base_weights = self.classifier.parameters(
            ).clone().detach().to("cuda" if torch.cuda.is_available() else "cpu")
        # local adaptation learning rate - 1e-3 or 5e-3
        self.loc_adapt_lr = 1e-3
        # Number of local adaptation steps
        self.L = L

    def classify(self, content, attention_mask, labels):
        """
        Bert classification model
        """
        loss, logits = self.classifier(
            content, attention_mask=attention_mask, labels=labels)
        return loss, logits

    def get_keys(self, contents, attn_masks):
        """
        Return key representation of the documents
        """
        # Freeze the weights of the key network to prevent key
        # representations from drifting as data distribution changes
        with torch.no_grad():
            last_hidden_states, _ = self.key_encoder(
                contents, attention_mask=attn_masks)
        # Obtain key representation of every text content by selecting the its [CLS] hidden representation
        keys = last_hidden_states[:, 0, :]

        return keys

    def infer(self, content, attn_mask, K_contents, K_attn_masks, K_labels):
        """
        Function that performs inference based on memory based local adaptation
        Parameters:
        content   -> document that needs to be classified
        attn_mask -> attention mask over document
        rt_batch  -> the batch of samples retrieved from the memory using nearest neighbour approach

        Returns:
        logit -> label corresponding to the single document provided,i.e, content
        """

        # create a local copy of the classifier network
        adaptive_classifier = copy.deepcopy(self.classifier)
        optimizer = transformers.AdamW(
            adaptive_classifier.parameters(), lr=self.loc_adapt_lr)

        # Current model weights
        curr_weights = list(adaptive_classifier.parameters())
        # Train the adaptive classifier for L epochs with the rt_batch
        for _ in range(self.L):

            # zero out the gradients
            optimizer.zero_grad()
            likelihood_loss, _ = adaptive_classifier(
                K_contents, attention_mask=K_attn_masks, labels=K_labels)
            # Initialize diff_loss to zero and place it on the appropriate device
            diff_loss = torch.Tensor([0]).to(
                "cuda" if torch.cuda.is_available() else "cpu")
            # Iterate over base_weights and curr_weights and accumulate the euclidean norm
            # of their differences
            for base_param, curr_param in zip(self.base_weights, curr_weights):
                diff_loss += (curr_param-base_param).pow(2).sum()

            # Total loss due to log likelihood and weight restraint
            total_loss = 0.001*diff_loss + likelihood_loss
            total_loss.backward()
            optimizer.step()

        logits, = adaptive_classifier(content.unsqueeze(
            0), attention_mask=attn_mask.unsqueeze(0))
        # Note: to prevent keeping track of intermediate values which
        # can lead to cuda of memory runtime error logits should be detached
        return logits.detach()

    def save_state(self):
        """
        Returns model state
        """
        model_state = dict()
        model_state['classifier'] = self.classifier.state_dict()
        model_state['key_encoder'] = self.key_encoder.state_dict()

        return model_state
    
    
    def train(order, model, memory):
        """
        Train function
        """
        workers = 0
        if use_cuda:
            model.cuda()
            # Number of workers should be 4*num_gpu_available
            # https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/5
            workers = 4
        # time at the start of training
        start = time.time()

        train_data = DataSet(order, split='train')
        train_sampler = data.SequentialSampler(train_data)
        train_dataloader = data.DataLoader(
            train_data, sampler=train_sampler, batch_size=args.batch_size, num_workers=workers)
        param_optimizer = list(model.classifier.named_parameters())
        # parameters that need not be decayed
        no_decay = ['bias', 'gamma', 'beta']
        # Grouping the parameters based on whether each parameter undergoes decay or not.
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.0}]
        optimizer = transformers.AdamW(
            optimizer_grouped_parameters, lr=LEARNING_RATE)

        # Store our loss and accuracy for plotting
        train_loss_set = []
        # trange is a tqdm wrapper around the normal python range
        for epoch in trange(args.epochs, desc="Epoch"):
            # Training begins
            print("Training begins")
            # Set our model to training mode (as opposed to evaluation mode)
            model.classifier.train()
            # Tracking variables
            tr_loss = 0
            nb_tr_examples, nb_tr_steps, num_curr_exs = 0, 0, 0
            # Train the data for one epoch
            for step, batch in enumerate(tqdm(train_dataloader)):
                # Release file descriptors which function as shared
                # memory handles otherwise it will hit the limit when
                # there are too many batches at dataloader
                batch_cp = copy.deepcopy(batch)
                del batch
                # Perform sparse experience replay after every REPLAY_FREQ steps
                if (step+1) % REPLAY_FREQ == 0:
                    # sample 64 examples from memory
                    content, attn_masks, labels = memory.sample(sample_size=64)
                    if use_cuda:
                        content = content.cuda()
                        attn_masks = attn_masks.cuda()
                        labels = labels.cuda()
                    # Clear out the gradients (by default they accumulate)
                    optimizer.zero_grad()
                    # Forward pass
                    loss, logits = model.classify(content, attn_masks, labels)
                    train_loss_set.append(loss.item())
                    # Backward pass
                    loss.backward()
                    # Update parameters and take a step using the computed gradient
                    optimizer.step()

                    # Update tracking variables
                    tr_loss += loss.item()
                    nb_tr_examples += content.size(0)
                    nb_tr_steps += 1

                    del content
                    del attn_masks
                    del labels
                    del loss
                # Unpacking the batch items
                content, attn_masks, labels = batch_cp
                content = content.squeeze(1)
                attn_masks = attn_masks.squeeze(1)
                labels = labels.squeeze(1)
                # number of examples in the current batch
                num_curr_exs = content.size(0)
                # Place the batch items on the appropriate device: cuda if avaliable
                if use_cuda:
                    content = content.cuda()
                    attn_masks = attn_masks.cuda()
                    labels = labels.cuda()
                # Clear out the gradients (by default they accumulate)
                optimizer.zero_grad()
                # Forward pass
                loss, _ = model.classify(content, attn_masks, labels)
                train_loss_set.append(loss.item())
                # Get the key representation of documents
                keys = model.get_keys(content, attn_masks)
                # Push the examples into the replay memory
                memory.push(keys.cpu().numpy(), (content.cpu().numpy(),
                                                attn_masks.cpu().numpy(), labels.cpu().numpy()))
                # delete the batch data to freeup gpu memory
                del keys
                del content
                del attn_masks
                del labels
                # Backward pass
                loss.backward()
                # Update parameters and take a step using the computed gradient
                optimizer.step()
                # Update tracking variables
                tr_loss += loss.item()
                nb_tr_examples += num_curr_exs
                nb_tr_steps += 1

            now = time.time()
            print("Train loss: {}".format(tr_loss/nb_tr_steps))
            print("Time taken till now: {} hours".format((now-start)/3600))
            model_dict = model.save_state()