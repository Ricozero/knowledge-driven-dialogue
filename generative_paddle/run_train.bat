::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::
:: Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
::
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
: @echo off

: set gpu id to use
set CUDA_VISIBLE_DEVICES=0

: generalizes target_a/target_b of goal for all outputs, replaces them with slot mark
set TOPIC_GENERALIZATION=1

: set python path according to your actual environment
set pythonpath='python'

: the prefix of the file name used by the model, must be consistent with the configuration in network.py
set prefix=demo

: put all data set that used and generated for training under this folder: datapath
: for more details, please refer to the following data processing instructions
: 不用反斜杠会导致copy时报错
set datapath=.\data

set vocabpath=%datapath%/vocab.txt

: in train stage, use "train.txt" to train model, and use "dev.txt" to eval model
: the "train.txt" and "dev.txt" are the original data provided by the organizer and
: need to be placed in this folder: datapath/resource/
: the following preprocessing will generate the actual data needed for model training
: datatype = "train" or "dev"
set datatype=train dev

: data preprocessing
: 形式变量为单个字符
for %%d in (%datatype%) do (
    : ensure that each file is in the correct path
    :     1. put the data provided by the organizers under this folder: datapath/resource/
    :            - the data provided consists of three parts: train.txt dev.txt test.txt
    :            - the train.txt and dev.txt are session data, the test.txt is sample data
    :            - in train stage, we just use the train.txt and dev.txt
    :     2. the sample data extracted from session data is in this folder: datapath/resource/
    :     3. the text file required by the model is in this folder: datapath
    :     4. the topic file used to generalize data is in this directory: datapath
    set corpus_file=%datapath%/resource/%%d.txt
    set sample_file=%datapath%/resource/sample.%%d.txt
    set text_file=%datapath%/%prefix%.%%d
    set topic_file=%datapath%/%prefix%.%%d.topic

    : step 1: firstly have to convert session data to sample data
    python ./tools/convert_session_to_sample.py %corpus_file% %sample_file%

    : step 2: convert sample data to text data required by the model
    python ./tools/convert_conversation_corpus_to_model_text.py %sample_file% %text_file% %topic_file% %TOPIC_GENERALIZATION%

    : step 3: build vocabulary from the training data
    if %%d == train (
        python ./tools/build_vocabulary.py %text_file% %vocabpath%
    )
)

: step 4: in train stage, we just use train.txt and dev.txt, so we copy dev.txt to test.txt for model training
copy /Y %datapath%\%prefix%.dev %datapath%\%prefix%.test

: step 5: train model in two stage, you can find the model file in ./models/ after training
: step 5.1: stage 0, you can get model_stage_0.npz and opt_state_stage_0.npz in save_dir after stage 0
python -u network.py --run_type train ^
                            --stage 0 ^
                            --use_gpu True ^
                            --pretrain_epoch 5 ^
                            --batch_size 32 ^
                            --use_posterior True ^
                            --save_dir ./models ^
                            --vocab_path %vocabpath% ^
                            --embed_file ./data/sgns.weibo.300d.txt

: step 5.2: stage 1, init the model and opt state using the result of stage 0 and train the model
python -u network.py --run_type train ^
                            --stage 1 ^
                            --use_gpu True ^
                            --init_model ./models/model_stage_0.npz ^
                            --init_opt_state ./models/opt_state_stage_0.npz ^
                            --num_epochs 12 ^
                            --batch_size 24 ^
                            --use_posterior True ^
                            --save_dir ./models ^
                            --vocab_path %vocabpath%
