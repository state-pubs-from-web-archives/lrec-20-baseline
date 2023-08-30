# lrec-20-baseline
Repository for generating datasets and results from Dynamic Classification in Web Archiving Collections (Patel, Caragea, Phillips)


## Requirements
```
$pip install -r requirements.txt
```

## Word2Vec Model Download for CNN training
https://drive.google.com/u/1/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download
gunzip GoogleNews-vectors-negative300.bin.gz


## Script for training BERT
```

export DEVICE=0
export MODEL="bert-base-uncased"  
export TASK="UNTEdu"  
export MAX_SEQ_LENGTH=256

if [ $MODEL = "bert-base-uncased" ]; then
    BATCH_SIZE=16
    LEARNING_RATE=2e-5
    WEIGHT_DECAY=0
fi


python3 train_bert.py \
    --device $DEVICE \
    --model $MODEL \
    --task $TASK \
    --ckpt_path "ckpt/${TASK}_${MODEL}.pt" \
    --output_path "output/${TASK}_${MODEL}.json" \
    --train_path "./untedu/train.tsv" \
    --dev_path "./untedu/dev.tsv" \
    --test_path "./untedu/test.tsv" \
    --epochs 3 \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --max_seq_length $MAX_SEQ_LENGTH \
    --do_train \
    --do_evaluate 

```

## Script for training CNN 

```

export DEVICE=0
export MODEL="CNN"  
export TASK="UNTEdu"  
export MAX_SEQ_LENGTH=300

if [ $MODEL = "CNN" ]; then
    BATCH_SIZE=16
    LEARNING_RATE=2e-5
fi

python3 train_cnn.py \
    --device $DEVICE \
    --model $MODEL \
    --task $TASK \
    --train_path "./untedu/rain.tsv" \
    --dev_path "./untedu/val.tsv" \
    --test_path "./untedu/test.tsv" \
    --epochs 3 \
    --batch_size $BATCH_SIZE \
    --max_seq_length $MAX_SEQ_LENGTH \
    --do_train \
    --do_evaluate 

```
