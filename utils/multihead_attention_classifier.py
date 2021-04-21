import torch
import torch.nn as nn
import torch.optim as optim
import spacy
nlp = spacy.load('en_core_web_sm')


class MultiHeadScaledDotProductAttention(nn.Module):
    """
    Multi Head Scaled Dot-Product Attention mechanism as described in "Attention Is All You Need".
    Ref : https://arxiv.org/pdf/1706.03762.pdf

    ARGS:
        - embedding_dim: Size of token representation.
        - heads: Number of parallel Scaled Dot Product Attention mechanisms.
    """

    def __init__(self, embedding_dim, heads):

        super(MultiHeadScaledDotProductAttention, self).__init__()

        self.embedding_dim = embedding_dim
        self.heads = heads
        # Floor division. head_dim should be an integer.
        self.head_dim = embedding_dim // heads

        # Make sure we can come back to the original shape later on.
        assert embedding_dim % heads == 0

        # Value matrix.
        self.V = nn.Linear(self.head_dim, self.head_dim, bias=False)
        # Key matrix.
        self.K = nn.Linear(self.head_dim, self.head_dim, bias=False)
        # Query matrix.
        self.Q = nn.Linear(self.head_dim, self.head_dim, bias=False)

        # After concatenatation we do the following
        self.linear1 = nn.Linear(
            self.head_dim * self.heads, self.embedding_dim)

    def forward(self, embeddings, pad_mask):

        N = embeddings.shape[0]  # Batch size

        D = embeddings.shape[1]  # Length

        # Split the embeddings for all heads so that the attention is paid to different pieces of the embeddings.
        Value = self.V(embeddings.reshape(N, D, self.heads, self.head_dim))
        Key = self.K(embeddings.reshape(N, D, self.heads, self.head_dim))
        Query = self.Q(embeddings.reshape(N, D, self.heads, self.head_dim))

        # Query & Key multiplication.
        score = torch.einsum("nqhd,nkhd->nhqk", [Query, Key])

        # Padding mask so that the attention is not paid to padding tokens.
        # The score is set to -inf if a padding token is envolved.
        # When we take the softmax it is set to 0.
        if pad_mask is not None:
            score = score.masked_fill(pad_mask == 0, float("-inf"))

        # D ivision for numerical stability and softmax so that the combinations sum up to one.
        attention = torch.softmax(score / (self.embedding_dim ** (1/2)), dim=3)

        # attention & Value multiplication + back to the original shape.
        full_attention = torch.einsum(
            "nhql,nlhd->nqhd", [attention, Value]).reshape(N, D, self.head_dim * self.heads)

        # Dence layer.
        output = self.linear1(full_attention)

        return output


class TransformerBlock(nn.Module):
    """
    Transformer Block as described in "Attention Is All You Need".
    Ref : https://arxiv.org/pdf/1706.03762.pdf

    ARGS:
        - embedding_dim: Size of token representation.
        - heads: Number of parallel Scaled Dot Product Attention mechanisms.
        - dropout: Dropout probability.
        - augmentation_factor : The embeddings are projected into
                                a space <augmentation_factor> times larger,
                                are relu activated and are projected back into the original space. 
    """

    def __init__(self, embedding_dim, heads, dropout, augmentation_factor):

        super(TransformerBlock, self).__init__()

        self.attention = MultiHeadScaledDotProductAttention(
            embedding_dim, heads)

        # To speed up the training process.
        self.LayerNormalization1 = nn.LayerNorm(embedding_dim)

        # To speed up the training process.
        self.LayerNormalization2 = nn.LayerNorm(embedding_dim)

        self.linear_augmentation = nn.Sequential(

            nn.Linear(embedding_dim, embedding_dim*augmentation_factor),
            nn.ReLU(),
            nn.Linear(embedding_dim*augmentation_factor, embedding_dim)

        )

        self.dropout = nn.Dropout(dropout)  # Regularization.

    def forward(self, embeddings, pad_mask):

        # Attention mechanism.
        attention = self.attention(embeddings, pad_mask)

        # Skip connection + norm + dropout.
        x = self.dropout(self.LayerNormalization1(attention + embeddings))

        forward = self.linear_augmentation(x)

        # Skip connection + norm + dropout.
        output = self.dropout(self.LayerNormalization2(forward + x))

        return output


class Encoder(nn.Module):
    """
    Encoder part as described in "Attention Is All You Need".
    The only difference is that positional embedding is used instead of positional encoding. 
    Ref : https://arxiv.org/pdf/1706.03762.pdf

    ARGS:
        - num_embeddings : Number of embeddings, used to initialize the embedding layer.
        - embedding_dim : Size of token representation.
        - num_layers : Number of stacked Transformer Blocks.
        - heads : Number of parallel Scaled Dot Product Attention mechanisms.
        - device : Specify on wich device we should do the computations.
        - augmentation_factor : The embeddings are projected into
                                a space <augmentation_factor> times larger,
                                are relu activated and are projected back into the original space.
        - dropout : Dropout probability.
        - max_length : Maximum length for the positional embedding as described in
                       "BERT". Ref : https://arxiv.org/pdf/1810.04805.pdf
    """

    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 num_layers,
                 heads,
                 device,
                 augmentation_factor,
                 dropout,
                 max_length):

        super(Encoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.device = device

        self.tok_embedding = nn.Embedding(num_embeddings, embedding_dim)

        self.pos_embedding = nn.Embedding(max_length, embedding_dim)

        self.layers = nn.ModuleList(

            [
                TransformerBlock(
                    embedding_dim=embedding_dim,
                    heads=heads,
                    dropout=dropout,
                    augmentation_factor=augmentation_factor) for _ in range(num_layers)
            ]

        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, tok_idx, pad_mask):

        N, D = tok_idx.shape  # Batch size & Length.

        # Token arange for all instances.
        pos = torch.arange(0, D).expand(N, D).to(self.device)

        new_embeddings = self.dropout(
            self.tok_embedding(tok_idx) + self.pos_embedding(pos))

        for layer in self.layers:
            # Go through all the blocks.
            new_embeddings = layer(new_embeddings, pad_mask)

        return new_embeddings


class MultiheadAttentionClassifier(nn.Module):
    """
    Architecture inspired by the "BERT: Pre-training of Deep Bidirectional
    Transformers for Language Understanding" paper. (Ref : https://arxiv.org/pdf/1810.04805.pdf)
    The following architecture is directly adapted for classification.
    This model is fully supervised and does not need to be pre-trained.
    In addition, the model takes the linear combination of the rows of the final representation,
    applies a tanh squeezing and a leaky relu multilayered perceptron is added on top of these transformations. 


    ARGS:
        - num_embeddings : Number of embeddings, used to initialize the embedding layer.
        - embedding_dim : Size of token representation.
        - num_layers : Number of stacked Transformer Blocks.
        - heads : Number of parallel Scaled Dot Product Attention mechanisms.
        - device : Specify on wich device we should do the computations.
        - augmentation_factor : The embeddings are projected into
                                a space <augmentation_factor> times larger,
                                are relu activated and are projected back into the original space.
        - dropout : Dropout probability.
        - max_length : Maximum length for the positional embedding as described in
                       "BERT". Ref : https://arxiv.org/pdf/1810.04805.pdf
        - pad_idx : The index of the padding token to mask the attention.
    """

    def __init__(self,
                 n_classes,
                 num_embeddings,
                 embedding_dim,
                 num_layers,
                 heads,
                 device,
                 augmentation_factor,
                 dropout,
                 max_length,
                 pad_idx):

        super(MultiheadAttentionClassifier, self).__init__()

        self.pad_idx = pad_idx
        self.device = device

        self.encoder = Encoder(

            num_embeddings,
            embedding_dim,
            num_layers,
            heads,
            device,
            augmentation_factor,
            dropout,
            max_length

        )

        # Combination parameters
        self.combination = nn.Parameter(torch.randn((1, max_length)))

        self.linear1 = nn.Linear(embedding_dim, embedding_dim)
        self.linear2 = nn.Linear(embedding_dim, embedding_dim)
        self.linear3 = nn.Linear(embedding_dim, n_classes)

    def get_pad_mask(self, mol):

        mask = (mol != self.pad_idx).unsqueeze(1).unsqueeze(2)  # Shape: (N, 1, 1, D)

        return mask.to(self.device)

    def forward(self, mol):

        pad_mak = self.get_pad_mask(mol)

        encoded_mol = self.encoder(mol, pad_mak)

        # Linear combination of rows + tanh squeezing.
        features = torch.tanh(torch.matmul(self.combination, encoded_mol))

        # leaky relu MLP.
        linear_combinations1 = nn.functional.leaky_relu(self.linear1(features))
        linear_combinations2 = nn.functional.leaky_relu(self.linear2(linear_combinations1))
        linear_combinations3 = self.linear3(linear_combinations2)

        return linear_combinations3


def MultiheadAttentionClassifier_one_train(model, iterator, optimizer, criterion, accuracy_function=None):
    """
    Function to train the MultiheadAttentionClassifier for one epoch.

    ARGS:
        - model: Pytorch MultiheadAttentionClassifier model.
        - iterator: A BucketIterator with fields = [('text', <field1>), ('label', <field2>)].
        - optimizer: A Pytorch optimizer object such as SGD or Adam.
        - criterion: A Pytorch loss function.
    """

    # Needed to store losses and accuracies.
    epoch_loss = 0
    epoch_acc = 0

    # Needed to display progress of an epoch.
    counter = 0

    # Model in training stage
    model.train()

    for batch in iterator:

        # Display progress.
        counter += 1
        perc = round((counter/len(iterator))*100)
        print(f"Epoch progress : {perc}%                 \r", end="")

        # At each batch, we backpropagate the signal.
        optimizer.zero_grad()

        # Retrieve smile string and length.
        text = batch.text[0]

        # Compute the prediction.
        predictions = model(text).squeeze()

        # Compute the loss.
        loss = criterion(predictions, batch.label)

        # Compute the gradients.
        loss.backward()

        # Backpropagate the signal.
        optimizer.step()

        # Store the loss.
        epoch_loss += loss.item()

        if accuracy_function is not None:

            # Compute the accuracy.
            acc = accuracy_function(predictions, batch.label)

            # Store the accuracy.
            epoch_acc += acc

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def MultiheadAttentionClassifier_evaluate(model, iterator, criterion, accuracy_function=None):
    """
    Function to evaluate the MultiheadAttentionClassifier for one epoch.

    ARGS:
        - model: Pytorch MultiheadAttentionClassifier model.
        - iterator: A BucketIterator with fields = [('text', <field1>), ('label', <field2>)].
        - criterion: A Pytorch loss function.
    """

    # Needed to store losses and accuracies.
    epoch_loss = 0
    epoch_acc = 0

    # Deactivating dropout layers.
    model.eval()

    # Deactivate gradient computation.
    with torch.no_grad():

        for batch in iterator:

            # Retrieve smile string and length.
            text = batch.text[0]

            # Compute the prediction.
            predictions = model(text).squeeze()

            # Compute the loss.
            loss = criterion(predictions, batch.label)

            # Store the loss and accuracy.
            epoch_loss += loss.item()

            if accuracy_function is not None:

                # Compute the accuracy.
                acc = accuracy_function(predictions, batch.label)

                # Store the accuracy.
                epoch_acc += acc

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def MultiheadAttentionClassifier_train(model,
                                       train_iterator,
                                       valid_iterator,
                                       n_epoch,
                                       optimizer,
                                       criterion,
                                       accuracy_function=None,
                                       save=False,
                                       saving_path="_.bin"):
    """
    Function to train the MultiheadAttentionClassifier for <n_epoch> epoch.

    ARGS:
        - model: Pytorch MultiheadAttentionClassifier model.
        - train_iterator: A BucketIterator with fields = [('text', <field1>), ('label', <field2>)] for training data.
        - valid_iterator: A BucketIterator with fields = [('text', <field1>), ('label', <field2>)] for validation data.
        - n_epoch: Number of epoch to train.
        - save: If True to model is saved in bin format in <saving_path>.
        - saving_path: Saving path for the model.
        - lr: Learning rate for the optimizer.
    """

    # Check on wich device we should do the computation.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Start by assuming the best validation loss is inf, so that it can be replaced at first epoch.
    best_valid_loss = float('inf')

    # Put model and criterion to the desired device.
    model = model.to(device)
    criterion = criterion.to(device)

    for epoch in range(n_epoch):

        train_loss, train_acc = MultiheadAttentionClassifier_one_train(model,
                                                                       train_iterator,
                                                                       optimizer,
                                                                       criterion,
                                                                       accuracy_function=accuracy_function)

        valid_loss, valid_acc = MultiheadAttentionClassifier_evaluate(model,
                                                                      valid_iterator,
                                                                      criterion,
                                                                      accuracy_function=accuracy_function)

        # Save if we beat the current best validation loss.
        if save:
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), saving_path)

        if accuracy_function is not None:

            # Quick summary.
            print(f"\tNo. epoch: {epoch+1}/{n_epoch}                  ")
            print(
                f'\t   Train Loss: {round(train_loss, 3)} | Train Acc: {round(train_acc, 3)}%')
            print(
                f'\t    Val. Loss: {round(valid_loss, 3)} |  Val. Acc: {round(valid_acc, 3)}%')
            print("\n")

        else:

            # Quick summary.
            print(f"\tNo. epoch: {epoch+1}/{n_epoch}                  ")
            print(f'\t   Train Loss: {round(train_loss, 3)}')
            print(f'\t    Val. Loss: {round(valid_loss, 3)}')
            print("\n")