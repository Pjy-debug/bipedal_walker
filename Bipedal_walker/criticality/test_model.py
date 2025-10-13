"""
import torch
from criticality_model import TransformerEncoder

model = TransformerEncoder()
input = torch.rand(32,107)
output,_ = model(input)
print(output.shape)
"""
from tensorboard.backend.event_processing import event_accumulator

#加载日志数据
ea=event_accumulator.EventAccumulator(r'/root/brx/tta_new/tta/criticality/tf-logs/log/events.out.tfevents.1689138120.autodl-container-1ae111b408-45d344f6.4695.35') 
ea.Reload()
print(ea.scalars.Keys())

"""
['Loss/train', 'Loss/test', 'Accuracy/train', 'Accuracy/test', 'Precision/train', 'Precision/test', 'Recall/train', 'Recall/test', 'ROC/train', 'ROC/test']
"""


loss_train=ea.scalars.Items('Loss/train')
print(len(loss_train))
print([(i.step,i.value) for i in loss_train])

"""
import matplotlib.pyplot as plt
fig=plt.figure(figsize=(6,4))
ax1=fig.add_subplot(111)
val_acc=ea.scalars.Items('val_acc')
ax1.plot([i.step for i in val_acc],[i.value for i in val_acc],label='val_acc')
ax1.set_xlim(0)
acc=ea.scalars.Items('acc')
ax1.plot([i.step for i in acc],[i.value for i in acc],label='acc')
ax1.set_xlabel("step")
ax1.set_ylabel("")

plt.legend(loc='lower right')
plt.show()
"""