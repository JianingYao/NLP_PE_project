import argparse
from datasets import load_from_disk
from transformers import GPT2Tokenizer,AutoTokenizer
from torch.utils.data import DataLoader
from model import GPT2
from model_alibi import GPT2_alibi
from model_RoPE import GPT2_RoPE
import torch
from torch.nn import functional as F
import math
import time
import os
from sklearn.metrics import f1_score, accuracy_score
from datasets import load_dataset
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
def get_lr(it, warmup_iters, learning_rate, lr_decay_iters):
    min_lr = learning_rate/10
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

@torch.no_grad()
def accuracy(logits, targets):
    prediction = F.softmax(logits, dim=2)

    prediction = torch.argmax(prediction, dim=2)

    #print(prediction.shape)
    compare = torch.eq(prediction, targets).float()

    #print(targets.shape)

    accuracy = torch.mean(compare).item()
    return accuracy

def F1_score(logits, targets):
    prediction = F.softmax(logits, dim=2)
    prediction = torch.argmax(prediction, dim=2)
    # 将预测结果和目标值展平
    prediction_flat = prediction.view(-1).to('cpu').numpy()
    #print(prediction_flat)
    targets_flat = targets.view(-1).to('cpu').numpy()
    #print(targets_flat)

    # 计算 F1 分数
    f1 = f1_score(targets_flat, prediction_flat, average='weighted')


    return f1

def cal_perplixity(logits, targets):
    prediction = F.softmax(logits, dim=2)
    # 计算 perplexity
    perplexity = torch.exp(torch.mean(torch.cat(prediction)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='gpt1')
    parser.add_argument('--resume', type=str, default='', help='Specify the CKPT name for resume training')
    parser.add_argument('--start_epoch', type=int, default=0, help='If resume training then specify the epoch to continue')
    parser.add_argument('--num_epoch', type=int, default=100, help='Specify the number of epochs to train')
    parser.add_argument('--steps_epoch', type=int, default=5000, help='Specify the steps of epoch')
    parser.add_argument('--total_epochs', type=int, default=120, help='Specify the total target epochs to train')
    parser.add_argument('--no_mixed', action='store_true', help='Specify this to not use mixed precesion to train')
    parser.add_argument('--dataset', type=str, default='wikitext_10000_512tokens/', help='The dataset path')
    parser.add_argument('--block_size', type=int, default=512, help='512The sequence lenght of the tokens for trianing')
    parser.add_argument('--decoder_layers', type=int, default=6, help='Decoder layers, orginial gpt1 model contains 12 layers')
    parser.add_argument('--heads', type=int, default=12, help='Multi attention heads per decoder layer')
    parser.add_argument('--d_model', type=int, default=768, help='768Embedding dimension')
    parser.add_argument('--dff', type=int, default=3072, help='Feed forward layer feature dimension')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size, original model use 64')
    parser.add_argument('--attn_pdrop', type=float, default=0.1)
    parser.add_argument('--resid_pdrop', type=float, default=0.1)
    parser.add_argument('--embed_pdrop', type=float, default=0.1)
    parser.add_argument('--ff_pdrop', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--learning_rate', type=float, default=0.0006, help='Original gpt1 use 0.00025')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=2000)
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/')
    parser.add_argument('--logfile', type=str, default='train_result_gpt2.txt')
    parser.add_argument('--model', type=str, default='GPT2_RoPE',help='GPT2_pos_embeding, GPT2_alibi,GPT2_RoPE')
    args = parser.parse_args()

    dataset = load_from_disk(args.dataset)
    #dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    dataset = dataset.with_format("torch")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    tokenizer = AutoTokenizer.from_pretrained("./gpt2model")
    #tokenizer = GPT2Tokenizer.from_pretrained("gpt2model")
    vocab_size = len(tokenizer.get_vocab())

    mixed = False
    dtype = 'float32'
    if not args.no_mixed:
        mixed = True
        dtype = 'float16'
    if args.model=='GPT2_pos_embeding':
        model = GPT2(
            vocab_size=vocab_size,
            d_model=args.d_model,
            block_size=args.block_size,
            embed_pdrop=args.embed_pdrop,
            num_heads=args.heads,
            dff=args.dff,
            attn_pdrop=args.attn_pdrop,
            resid_pdrop=args.resid_pdrop,
            dropout=args.ff_pdrop,
            num_layer=args.decoder_layers)
    if args.model=='GPT2_alibi':
        model = GPT2_alibi(
            vocab_size=vocab_size,
            d_model=args.d_model,
            block_size=args.block_size,
            embed_pdrop=args.embed_pdrop,
            num_heads=args.heads,
            dff=args.dff,
            attn_pdrop=args.attn_pdrop,
            resid_pdrop=args.resid_pdrop,
            dropout=args.ff_pdrop,
            num_layer=args.decoder_layers)
    if args.model=='GPT2_RoPE':
        model = GPT2_RoPE(
            vocab_size=vocab_size,
            d_model=args.d_model,
            block_size=args.block_size,
            embed_pdrop=args.embed_pdrop,
            num_heads=args.heads,
            dff=args.dff,
            attn_pdrop=args.attn_pdrop,
            resid_pdrop=args.resid_pdrop,
            dropout=args.ff_pdrop,
            num_layer=args.decoder_layers)
    
    model.to(args.device)
    #model = torch.compile(model)

    optimizer = model.configure_optimizers(args.weight_decay, args.learning_rate, (0.9, 0.95), args.device)

    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

    start_epoch = args.start_epoch
    if args.resume != '':
        checkpoint = torch.load(args.checkpoint_path+args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print('Checkpoint '+args.resume+' loaded! Resume from epoch '+str(start_epoch))

    model.train()
    print("Epoch  Loss  Acc   F1   Perplexity")

    for epoch in range(start_epoch, start_epoch+args.num_epoch):
        total_loss = 0
        total_accuracy = 0
        total_F1=0
        total_perplexity=0
        start = time.time()
        for batch, data in enumerate(dataloader):
            optimizer.zero_grad()
            lr = get_lr(batch+epoch*args.steps_epoch, args.warmup_steps, args.learning_rate, args.steps_epoch*args.total_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            x = data['token_ids'][...,:-1].contiguous().to(args.device)
            y = data['token_ids'][...,1:].contiguous().to(args.device)
            if mixed:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    logits, loss = model(x, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, loss = model(x, y)
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            total_accuracy += accuracy(logits, y)
            total_F1+=F1_score(logits,y)
            total_perplexity+=math.exp(loss)

            if batch%100 == 0 and batch>0:
                line = f'Batch: {batch+epoch*args.steps_epoch}, Loss: {total_loss/100:.4f}, Accuracy: {total_accuracy/100:.4f}, Learning_rate: {lr:.5f}'
                with open(args.logfile, 'a') as logfile:
                    logfile.write(line+'\n')
                print(line)
                total_loss = 0
                total_accuracy = 0
                if batch%args.steps_epoch == 0:
                    break

        with open(f'train_result_{args.model}.txt', 'a') as logfile:
            #line = f'Saving checkpoint for epoch {epoch+1} in {args.checkpoint_path}'
            #logfile.write(line+'\n')
            #print(line)
            #line = f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n'
            #logfile.write(line+'\n')

            #print(line)
            line=f'{epoch} {total_loss/float(len(dataloader)):.4f} {total_accuracy/float(len(dataloader)):.4f} {total_F1/float(len(dataloader)):.4f} {total_perplexity/float(len(dataloader)):.4f}'
            print(line)
            logfile.write(line + '\n')
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, args.checkpoint_path+args.model+'_'+str(epoch)+'.pt')


