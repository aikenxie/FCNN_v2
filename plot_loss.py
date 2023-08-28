import matplotlib.pyplot as plt
import numpy as np
import glob

if __name__ == "__main__":
    loss_log_names = glob.glob('/hildafs/projects/phy230010p/xiea/checkpoints/FCNN_train8/'+'*.log')

    data = []
    for loss_log in loss_log_names:
        with open(loss_log,"r") as log:
            temp = log.read().splitlines()[1:]
            temp = temp[1:]
            data.extend(temp)
            #lines = loss_log.read().splitlines()[1:]
    for i in range (len(data)):
        data[i] = data[i].split(',')
    data = np.array(data)
    data = data.astype(float)

    
    epochs = data[:,0].astype(int)
    train_loss = data[:,1]
    eval_loss = data[:,2]

    epochsInds = epochs.argsort()
    epochs = epochs[epochsInds]
    train_loss = train_loss[epochsInds]
    eval_loss = eval_loss[epochsInds]
  
    print("epochs",epochs)
    print(train_loss)
    print(eval_loss)
    
    plt.xticks(epochs)

    plt.plot(epochs,train_loss,color="orange",label='train_loss')
    plt.gcf().set_size_inches(15,plt.gcf().get_size_inches()[1])
    plt.savefig("/hildafs/projects/phy230010p/xiea/checkpoints/FCNN_train8_pngs/FCNN_v2_train_loss.png") 
    plt.legend()
    plt.clf()
    
    plt.xticks(epochs)
    plt.plot(epochs,eval_loss,color="blue",label='eval_loss')
    plt.gcf().set_size_inches(15,plt.gcf().get_size_inches()[1])
    plt.savefig("/hildafs/projects/phy230010p/xiea/checkpoints/FCNN_train8_pngs/FCNN_v2_eval_loss.png")
    plt.legend()
    plt.clf()
    
    plt.xticks(epochs)
    plt.gcf().set_size_inches(15,plt.gcf().get_size_inches()[1])
    plt.plot(epochs,train_loss,color="orange",label='train_loss')
    plt.plot(epochs,eval_loss,color="blue",label='eval_loss') 
    plt.legend()
    plt.savefig("/hildafs/projects/phy230010p/xiea/checkpoints/FCNN_train8_pngs/FCNN_v2_loss_log.png")


