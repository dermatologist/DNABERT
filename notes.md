## Codon-LLM training

* --model_type gt (Genomic Tokenizer)

```
cd examples

export KMER=6
export TRAIN_FILE=/home/orion-lab/data/cosmic/Cosmic_Genes_v101_GRCh37.fasta
export TEST_FILE=sample_data/pre/6_3k.txt
export SOURCE=/home/orion-lab/repos/DNABERT
export OUTPUT_PATH=output_codon_llm

python run_pretrain.py \
    --output_dir $OUTPUT_PATH \
    --model_type=gt \
    --config_name=$SOURCE/src/transformers/dnabert-config/codon-llm-config/config.json \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --mlm \
    --gradient_accumulation_steps 25 \
    --per_gpu_train_batch_size 4 \
    --per_gpu_eval_batch_size 4 \
    --save_steps 5 \
    --save_total_limit 2 \
    --max_steps 100 \
    --evaluate_during_training \
    --logging_steps 5 \
    --learning_rate 4e-4 \
    --block_size 512 \
    --adam_epsilon 1e-6 \
    --weight_decay 0.01 \
    --beta1 0.9 \
    --beta2 0.98 \
    --mlm_probability 0.025 \
    --warmup_steps 10 \
    --overwrite_output_dir \
    --n_process 8
```




#### 2.2 Model Training


```
cd examples

export KMER=6
export TRAIN_FILE=sample_data/pre/6_3k.txt
export TEST_FILE=sample_data/pre/6_3k.txt
export SOURCE=/home/orion-lab/repos/DNABERT
export OUTPUT_PATH=output$KMER

python run_pretrain.py \
    --output_dir $OUTPUT_PATH \
    --model_type=dna \
    --tokenizer_name=dna$KMER \
    --config_name=$SOURCE/src/transformers/dnabert-config/bert-config-$KMER/config.json \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --mlm \
    --gradient_accumulation_steps 25 \
    --per_gpu_train_batch_size 4 \
    --per_gpu_eval_batch_size 4 \
    --save_steps 5 \
    --save_total_limit 2 \
    --max_steps 100 \
    --evaluate_during_training \
    --logging_steps 5 \
    --line_by_line \
    --learning_rate 4e-4 \
    --block_size 512 \
    --adam_epsilon 1e-6 \
    --weight_decay 0.01 \
    --beta1 0.9 \
    --beta2 0.98 \
    --mlm_probability 0.025 \
    --warmup_steps 10 \
    --overwrite_output_dir \
    --n_process 8
```


#### 3.3 Fine-tune with pre-trained model

In the following example,  we use DNABERT with kmer=6 as example. We use `prom-core`, a 2-class classification task as example.

export DATA_PATH=sample_data/ft/$KMER


```
cd examples

export KMER=6
export SOURCE=/home/orion-lab/repos/DNABERT
export MODEL_PATH=/home/orion-lab/.cache/huggingface/hub/models--zhihan1996--DNA_bert_6/snapshots/55e0c0eb7b734c8b9b77bc083bf89eb6fbda1341
export DATA_PATH=/home/orion-lab/data/genome/seq
export OUTPUT_PATH=./ft/$KMER

python run_finetune.py \
    --model_type dna \
    --tokenizer_name=dna$KMER \
    --model_name_or_path $MODEL_PATH \
    --task_name dnaprom \
    --do_train \
    --do_eval \
    --data_dir $DATA_PATH \
    --max_seq_length 512 \
    --per_gpu_eval_batch_size=8   \
    --per_gpu_train_batch_size=8   \
    --learning_rate 2e-4 \
    --num_train_epochs 2 \
    --output_dir $OUTPUT_PATH \
    --evaluate_during_training \
    --logging_steps 10 \
    --save_steps 40 \
    --warmup_percent 0.1 \
    --hidden_dropout_prob 0.1 \
    --overwrite_output \
    --weight_decay 0.01 \
    --n_process 8
```

## 4. Prediction

After the model is fine-tuned, we can get predictions by running

```$
export KMER=6
export MODEL_PATH=./ft/$KMER
export DATA_PATH=sample_data/ft/$KMER
export PREDICTION_PATH=./result/$KMER

python run_finetune.py \
    --model_type dna \
    --tokenizer_name=dna$KMER \
    --model_name_or_path $MODEL_PATH \
    --task_name dnaprom \
    --do_predict \
    --data_dir $DATA_PATH  \
    --max_seq_length 512 \
    --per_gpu_pred_batch_size=8   \
    --output_dir $MODEL_PATH \
    --predict_dir $PREDICTION_PATH \
    --n_process 8
```