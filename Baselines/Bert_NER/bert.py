from transformers import AutoModel
import torch.nn as nn

class Bert(nn.Module):
    def __init__(self, nb_labels):
        super(Bert, self).__init__()
        self.nb_labels = nb_labels
        self.bert = AutoModel.from_pretrained('m3rg-iitd/matscibert')
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(768, self.nb_labels)

    # Batch operation
    def forward(self, input_ids, token_type_ids, attention_mask, labels):
        outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        loss_fn = nn.CrossEntropyLoss()

        active_loss = attention_mask.view(-1) == 1
        active_logits = logits.view(-1, self.nb_labels)[active_loss]
        active_labels = labels.view(-1)[active_loss]
        loss = loss_fn(active_logits, active_labels)
        return loss, logits, outputs[0], labels 
