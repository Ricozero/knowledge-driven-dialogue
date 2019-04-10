: set gpu id to use
set CUDA_VISIBLE_DEVICES=1

: task_name can select from ["match", "match_kn", "match_kn_gene"]
: match task: do not use knowledge info (goal and knowledge) for retrieval model
: match_kn task: use knowledge info (goal and knowledge) for retrieval model
: match_kn_gene task: 1) use knowledge info (goal and knowledge) for retrieval model;
:                     2) generalizes target_a/target_b of goal, replaces them with slot mark
: more information about generalization in match_kn_gene,
: you can refer to ./tools/convert_conversation_corpus_to_model_text.py
set TASK_NAME=%1%

if %TASK_NAME% == match (
  set DICT_NAME=./dict/char.dict
  set USE_KNOWLEDGE=0
  set TOPIC_GENERALIZATION=0
) else if %TASK_NAME% == match_kn (
  set DICT_NAME=./dict/char.dict
  set USE_KNOWLEDGE=1
  set TOPIC_GENERALIZATION=0
else if %TASK_NAME% == match_kn_gene (
  set DICT_NAME=./dict/gene.dict
  set USE_KNOWLEDGE=1
  set TOPIC_GENERALIZATION=1
else (
  echo "task name error, should be match|match_kn|match_kn_gene"
)

: in train stage, FOR_PREDICT=1
set FOR_PREDICT=1

: put all data set that used and generated for testing under this folder: INPUT_PATH
: for more details, please refer to the following data processing instructions
set INPUT_PATH=./data

: put the model files needed for testing under this folder: OUTPUT_PATH
set OUTPUT_PATH=./models

: set python path according to your actual environment
set PYTHON_PATH=python

: in test stage, you can eval dev.txt or test.txt
: the "dev.txt" and "test.txt" are the original data provided by the organizer and
: need to be placed in this folder: INPUT_PATH/resource/
: the following preprocessing will generate the actual data needed for model testing
: after testing dev.txt, you can run eval.py to get the final eval score,
: because dev.txt is session data, you have all the utterances both of bot and user
: after testing test.txt, you can upload the predict to the competition website to get result
: DATA_TYPE = "dev" or "test"
set DATA_TYPE=dev

: ensure that each file is in the correct path
:     1. put the data provided by the organizers under this folder: INPUT_PATH/resource/
:            - the data provided consists of three parts: train.txt dev.txt test.txt
:            - the train.txt and dev.txt are session data, the test.txt is sample data
:            - in test stage, we just use the dev.txt or test.txt
:     2. the sample data extracted from session data is in this folder: INPUT_PATH/resource/
:     3. the candidate data constructed from sample data is in this folder: INPUT_PATH/resource/
:     4. the text file required by the model is in this folder: INPUT_PATH
set corpus_file=%INPUT_PATH%/resource/%DATA_TYPE%.txt
set sample_file=%INPUT_PATH%/resource/sample.%DATA_TYPE%.txt
set candidate_file=%INPUT_PATH%/resource/candidate.%DATA_TYPE%.txt
set text_file=%INPUT_PATH%/test.txt
set score_file=./output/score.txt
set predict_file=./output/predict.txt

: step 1: if eval dev.txt, firstly have to convert session data to sample data
: if eval test.txt, we can use test.txt provided by the organizer directly.
if %DATA_TYPE% == test (
    sample_file=%corpus_file%
) else (
    %PYTHON_PATH% ./tools/convert_session_to_sample.py %corpus_file% %sample_file%
)

: step 2: construct candidate for sample data
: here do not construct real candidate, you have to do it by yourself !!!
%PYTHON_PATH% ./tools/construct_candidate.py %sample_file% %candidate_file% 9

: step 3: convert sample data with candidates to text data required by the model
%PYTHON_PATH% ./tools/convert_conversation_corpus_to_model_text.py %candidate_file% %text_file% %{USE_KNOWLEDGE% %TOPIC_GENERALIZATION% %FOR_PREDICT%

: inference_model can used for interact.py
set inference_model=./models/inference_model

: step 4: predict score by model
%PYTHON_PATH% -u predict.py --task_name %TASK_NAME%^
                   --use_cuda^
                   --batch_size 10^
                   --init_checkpoint %OUTPUT_PATH%/50^
                   --data_dir %INPUT_PATH%^
                   --vocab_path %DICT_NAME%^
                   --save_inference_model_path %inference_model%^
                   --max_seq_len 128^
                   --output %score_file%

: step 5: extract predict utterance by candidate_file and score_file
: if you eval dev.txt, the predict_file format is "predict \t gold \n predict \t gold \n ......"
: if you eval test.txt, the predict_file format is "predict \n predict \n predict \n predict \n ......"
%PYTHON_PATH% ./tools/extract_predict_utterance.py %candidate_file% %score_file% %predict_file%

: step 6: if you eval dev.txt, you can run the following command to get result
: if you eval test.txt, you can upload the predict_file to the competition website to get result
if not %DATA_TYPE% == test (
    %PYTHON_PATH% ./tools/eval.py %predict_file%
)