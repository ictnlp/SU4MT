TOTAL_NUM_UPDATES=18000
WARMUP_UPDATES=4000
LR=7e-04
MAX_TOKENS=4096
UPDATE_FREQ=1
save_dir=#add your saving directory#

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup fairseq-train #add your training data path# \
	--max-tokens $MAX_TOKENS \
	--layernorm-embedding \
	--remind-valid-step \
	--task translation_with_phrase_compression \
	--source-lang en --target-lang ro \
	--offline-phrase #add your phrase data path# \
	--truncate-source --share-all-embeddings \
	--ddp-backend=legacy_ddp \
	--share-all-embeddings \
	--required-batch-size-multiple 1 \
	--arch transformer_su4mt --criterion phrase_compress \
	--label-smoothing 0.1 \
	--dropout 0.1 \
	--weight-decay 0.0001 \
	--optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-08 \
	--save-interval 1 --keep-interval-updates 40 \
	--seed 222 \
	--log-format simple --log-interval 100 \
	--clip-norm 0.0 \
	--patience 5 \
	--lr-scheduler inverse_sqrt --lr $LR \
	--max-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
	--fp16 --update-freq $UPDATE_FREQ \
	--skip-invalid-size-inputs-valid-test \
	--copy-source-sent \
	--save-dir $save_dir | tee -a $save_dir/log.out &
