import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F

CONTEXT_SIZE = 2
raw_text = "We are about to study the idea of a computational process. Computational processes are abstract beings that inhabit computers. As they evolve, processes manipulate other abstract things called data. The evolution of a process is directed by a pattern of rules called a program. People create programs to direct processes. In effect, we conjure the spirits of the computer with our spells.".split(
    ' ')

vocab = set(raw_text)

word_to_idx = {word: i for i, word in enumerate(vocab)}

data = []

for i in range(CONTEXT_SIZE, len(raw_text) - CONTEXT_SIZE):
    context = [raw_text[i - 2], raw_text[i - 1], raw_text[i + 1], raw_text[i + 2]]
    target = raw_text[i]
    data.append((context, target))


class CBOW(nn.Module):
    def __init__(self, n_word, n_dim, context_size):
        super(CBOW, self).__init__()
        self.embedding = nn.Embedding(n_word, n_dim)
        self.linear1 = nn.Linear(2 * context_size * n_dim, 128)
        self.linear2 = nn.Linear(128, n_word)

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(1, -1)
        x = self.linear1(x)
        x = F.relu(x, inplace=True)
        x = self.linear2(x)
        x = F.log_softmax(x)
        return x


model = CBOW(len(word_to_idx), 100, CONTEXT_SIZE)
if torch.cuda.is_available():
    model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)

for epoch in range(50):
    print('epoch{}'.format(epoch))
    print('*' * 10)
    running_loss = 0
    for word in data:
        context, target = word
        context = Variable(torch.LongTensor([word_to_idx[i] for i in context]))
        target = Variable(torch.LongTensor([word_to_idx[target]]))
        if torch.cuda.is_available():
            context = context.cuda()
            target = target.cuda()
        out = model(context)
        loss = criterion(out, target)
        running_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('loss:{:.6f}'.format(running_loss / len(data)))
    
    
    
""" 提取词向量 """
w =  model.embedding.weight
print(w.shape)


