import pandas as pd

def read_txt_file(txt_file):
    data = []
    Epochs=[]
    Loss=[]
    Acc=[]
    F1=[]
    Perlexity=[]

    with open(txt_file, 'r') as file:
        for line in file:

            line_data = [float(num) for num in line.strip().split()]
            epoch=int(line_data[0])
            Epochs.append(epoch)
            loss=line_data[1]
            Loss.append(loss)
            acc=line_data[2]
            Acc.append(acc)

            f1=line_data[3]
            F1.append(f1)
            per=line_data[4]
            Perlexity.append(per)
            data.append(line_data)

    return Epochs,Loss,Acc,F1,Perlexity

txt_file = 'train_result_GPT2_alibi.txt'
excel_file = 'train_result_GPT2_alibi.xlsx'

Epochs_alibi,Loss_alibi,Acc_alibi,F1_alibi,Perlexity_alibi=read_txt_file(txt_file)

Epochs_pos,Loss_pos,Acc_pos,F1_pos,Perlexity_pos=read_txt_file('train_result_GPT2_pos_embeding.txt')

Epochs_RoPE,Loss_RoPE,Acc_RoPE,F1_RoPE,Perlexity_RoPE=read_txt_file('train_result_GPT2_RoPE.txt')
import matplotlib.pyplot as plt
def draw_two_line(x,y1,y2,y3,name,start=0,end=100):

    plt.plot(x[start:end], y1[start:end],label='Alibi')
    plt.plot(x[start:end], y2[start:end], label='Pos_embeding')
    plt.plot(x[start:end], y3[start:end], label='RoPE')


    plt.xlabel('Epoch')
    plt.ylabel(name)
    plt.show()
    plt.legend(bbox_to_anchor=(1.20, 1), loc=1, borderaxespad=0)
draw_two_line(Epochs_alibi, Perlexity_alibi, Perlexity_pos, Perlexity_RoPE,"Perlexity",start=10)
draw_two_line(Epochs_alibi,Acc_alibi,Acc_pos,Acc_RoPE,"Accurary")
draw_two_line(Epochs_alibi,Loss_alibi,Loss_pos,Loss_RoPE,"Loss")
draw_two_line(Epochs_alibi,F1_alibi,F1_pos,F1_RoPE,"F1")

print()