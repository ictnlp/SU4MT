modelfile=#add your model checkpoints#
phrasefile=#add your path to phrase file#
python #directory to your fairseq#/fairseq-master/scripts/average_checkpoints.py --inputs $modelfile/ --num-epoch-checkpoints 5 --output $modelfile/checkpoint_averepoch5.pt

test=#add your path to test data#
subset=test
outfile=#add the output path#

mkdir -p $outfile

for num in _averepoch5 _best _last
do
        model=checkpoint$num.pt
        CUDA_VISIBLE_DEVICES=0 fairseq-generate $test --path $modelfile/$model --task translation_with_phrase_compression --pe-plugin --source-lang en --target-lang ro --gen-subset $subset --beam 4 --batch-size 512 --remove-bpe --lenpen 1 --no-progress-bar --fp16 --log-format json > $outfile/pred$num.bpe.ro 
        echo "finish bpe version epoch $num"
        CUDA_VISIBLE_DEVICES=0 fairseq-generate $test --path $modelfile/$model --task translation_with_phrase_compression --pe-plugin --source-lang en --target-lang ro --gen-subset $subset --phrase-path $phrasefile --beam 4 --batch-size 512 --remove-bpe --lenpen 1 --no-progress-bar --fp16 --log-format json > $outfile/pred$num.frsbpe.1w.ro
        echo "finish phrase+bpe version epoch $num"
done
