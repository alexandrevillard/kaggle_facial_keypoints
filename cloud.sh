read -p 'job name?' JOB_NAME

CONVS='32_64_128'
KERNELS='3_3_3'
PKEEP=0.7
BATCH=200
LR=0.003
MODEL='c_'$CONVS'_k_'$KERNELS'_p_'$PKEEP'_lr_'$LR
MODEL="${MODEL//./}"
BASE=gs://train_data_alwld_eu/jobs/
JOB_DIR=$BASE'outs/'$MODEL
TRAIN_FILE=$BASE'data/train.tfrecords'
EVAL_FILE=$BASE'data/pred.tfrecords'
/home/alwld/cloud_sdk/google-cloud-sdk/bin/gcloud ml-engine jobs submit training $JOB_NAME \
                                    --stream-logs \
                                    --runtime-version 1.0 \
                                    --job-dir $JOB_DIR \
                                    --module-name trainer.task \
                                    --package-path trainer/ \
                                    --region europe-west1 \
                                    --scale-tier BASIC_GPU \
                                    -- \
                                    --train-files $TRAIN_FILE \
                                    --eval-files $EVAL_FILE \
                                    --batch-size $BATCH \
                                    --num-epochs 3000 \
                                    --eval-frequency 1 \
                        	    --p-keep $PKEEP \
                                    --convs $CONVS \
                            	    --ckpts-save-freq-sec 600 \
                                    --learning-rate $LR \
                        	    --kernels $KERNELS \
                                    --seed 67
